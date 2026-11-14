#!/usr/bin/env python3
"""
E5: three-domain sequential continual learning A -> B -> C.

Grow the base model ONCE with the gate-FP (function-preserving) operator and
Isolation CL protection, then CL-train sequentially: first on domain B, then on
domain C. After each phase, perplexity is evaluated on ALL THREE domains
(A, B, C), giving the full forgetting picture that the 2-domain trainer path
does not surface on its own.

Domains (fixed letter -> dataset mapping, independent of --order position):
    A = wikitext  (the base model's pretraining domain; protected)
    B = bookcorpus
    C = code

Isolation freezes the original blocks and keeps the grown blocks trainable, so
"grow once, then learn B then C" trains the SAME grown capacity across both
phases while the original (domain-A) capacity stays frozen.

Usage (from repo root):
    python experiments/_shared/run_sequential_domains.py \\
        --base-checkpoint checkpoints/exp1_shared_base/base_final.pt \\
        --growth-factor 2 --cl-epochs 10 --order A,B,C --seed 42
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from gatezero.reproducibility import set_reproducible
from gatezero.config import (ExperimentConfig, ModelConfig, TrainingConfig,
                             GrowthConfig, CLConfig, GrowthMethod, CLMethod)
from gatezero.data import build_datasets, get_tokenizer
from gatezero.trainer import Trainer
from gatezero.evaluation import compute_perplexity

# Fixed letter -> dataset name. A is the wikitext-pretrained base domain.
DOMAIN_DATASETS = {"A": "wikitext", "B": "bookcorpus", "C": "code"}
# Token cap for the non-base domains (match dataset sizes); wikitext uncapped.
DOMAIN_MAX_TOKENS = {"A": None, "B": 118_000_000, "C": 118_000_000}


def parse_order(order_str: str):
    """Parse '--order A,B,C' into a validated list of three domain letters."""
    letters = [s.strip().upper() for s in order_str.split(",") if s.strip()]
    if sorted(letters) != ["A", "B", "C"]:
        raise ValueError(
            f"--order must be a permutation of A,B,C (got {order_str!r})"
        )
    return letters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--growth-factor", type=int, default=2,
                        help="g: 12 -> g*12 layers (the paper's 3-domain runs use g=4)")
    parser.add_argument("--growth-method", type=str, default="gate_fp",
                        choices=["gate_fp", "net2net", "gstack"],
                        help="FP mechanism (E7): gate_fp=scalar zero-gate (ours), "
                             "net2net=zero output projections, gstack=non-FP "
                             "counter-example. Default gate_fp.")
    parser.add_argument("--cl-epochs", type=int, default=10)
    parser.add_argument("--order", type=str, default="A,B,C",
                        help="Comma list over {A,B,C}; A=wikitext, B=bookcorpus, "
                             "C=code. order[0] is the protected/base domain; the "
                             "remaining two are learned sequentially. Intended: "
                             "'A,B,C' and 'A,C,B'.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Disable full determinism (recover cuDNN autotuning "
                             "speed on large runs).")
    args = parser.parse_args()

    # Seed every RNG surface FIRST, before any dataset/model construction.
    set_reproducible(args.seed, deterministic=not args.fast)

    order = parse_order(args.order)
    order_names = [DOMAIN_DATASETS[l] for l in order]
    gf = args.growth_factor
    growth_method = GrowthMethod(args.growth_method)
    name = f"seq_{'-'.join(order_names)}_g{gf}_seed{args.seed}"
    # Encode method only for non-default operators, so existing gate_fp result
    # paths are unchanged and net2net/gstack land in their own dirs.
    if growth_method != GrowthMethod.GATE_ZERO:
        name = f"{name}_{args.growth_method}"

    cfg = ExperimentConfig(
        name=name,
        model=ModelConfig(),
        training=TrainingConfig(seed=args.seed, deterministic=not args.fast),
        growth=GrowthConfig(
            method=growth_method,
            growth_factor=gf,
        ),
        cl=CLConfig(method=CLMethod.ISOLATION, cl_epochs=args.cl_epochs),
        resume_base_checkpoint=args.base_checkpoint,
    )

    tokenizer = get_tokenizer()
    seq_len = cfg.model.max_seq_len

    # Build all three domains once. Keyed by fixed letter (A/B/C), not by order.
    print("Loading datasets...")
    train_ds = {}
    val_ds = {}
    for letter in ("A", "B", "C"):
        ds_name = DOMAIN_DATASETS[letter]
        tr, va = build_datasets(
            ds_name, tokenizer, seq_len, max_tokens=DOMAIN_MAX_TOKENS[letter]
        )
        train_ds[letter] = tr
        val_ds[letter] = va

    base_letter = order[0]           # protected old domain (base = wikitext)
    phase1_letter = order[1]         # first sequential new domain
    phase2_letter = order[2]         # second sequential new domain
    if base_letter != "A":
        print(f"NOTE: order[0]={base_letter} ({DOMAIN_DATASETS[base_letter]}) but "
              f"the base checkpoint was pretrained on wikitext (A). The protected "
              f"domain and the base-pretrained domain differ for this run.")

    trainer = Trainer(cfg, args.device)

    def eval3(model):
        """Perplexity on all three domains, labelled by fixed letter A/B/C."""
        return {
            "ppl_A": compute_perplexity(
                model, val_ds["A"], cfg.training.batch_size,
                args.device, cfg.training.num_workers,
            ),
            "ppl_B": compute_perplexity(
                model, val_ds["B"], cfg.training.batch_size,
                args.device, cfg.training.num_workers,
            ),
            "ppl_C": compute_perplexity(
                model, val_ds["C"], cfg.training.batch_size,
                args.device, cfg.training.num_workers,
            ),
        }

    t_start = time.time()

    # Load base, then grow ONCE (gate-FP). apply_growth also builds the frozen
    # pre-growth teacher that Isolation uses to protect the old domain; the same
    # teacher is reused across both sequential phases.
    model = trainer.build_model()
    trainer._load_checkpoint(model, args.base_checkpoint)
    model, teacher = trainer.apply_growth(model, gate_init_value=0.0)

    snapshots = {}
    snapshots["post_growth"] = eval3(model)
    print(f"Post-growth (pre-CL): {snapshots['post_growth']}")

    # PHASE 1: learn order[1]. Protect base domain (val/train A given to train_cl).
    print(f"\n### SEQUENTIAL PHASE 1: learn {phase1_letter} "
          f"({DOMAIN_DATASETS[phase1_letter]}) ###")
    model = trainer.train_cl(
        model,
        train_ds[phase1_letter],
        val_ds[base_letter],
        val_ds[phase1_letter],
        train_ds_a=train_ds[base_letter],
        teacher=teacher,
    )
    snapshots["after_phase1"] = eval3(model)
    print(f"After phase 1: {snapshots['after_phase1']}")

    # Save the after-phase-1 (post-B, pre-C) checkpoint BEFORE phase 2 runs.
    # train_cl writes cl_final.pt each call, so phase 2 would otherwise overwrite
    # the intermediate state. Keeping it enables per-phase (B-vs-C) analysis.
    p1_ckpt = trainer.ckpt_dir / "cl_after_phase1.pt"
    p1_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "phase": 1, "phase1_domain": phase1_letter}, p1_ckpt)
    print(f"Saved after-phase-1 checkpoint: {p1_ckpt}")

    # PHASE 2: learn order[2] on the same grown model (grown blocks keep training,
    # original blocks stay frozen under Isolation).
    print(f"\n### SEQUENTIAL PHASE 2: learn {phase2_letter} "
          f"({DOMAIN_DATASETS[phase2_letter]}) ###")
    model = trainer.train_cl(
        model,
        train_ds[phase2_letter],
        val_ds[base_letter],
        val_ds[phase2_letter],
        train_ds_a=train_ds[base_letter],
        teacher=teacher,
    )
    snapshots["after_phase2"] = eval3(model)
    print(f"After phase 2: {snapshots['after_phase2']}")

    elapsed = time.time() - t_start

    # Record the sequential run. train_cl's own metrics (final_ppl_a/b, delta_a)
    # reflect ONLY the last phase and only two domains, so the full three-domain
    # trajectory lives here under "sequential".
    trainer.results["metrics"]["sequential"] = {
        "order": order,
        "order_datasets": order_names,
        "base_domain": base_letter,
        "phase1_domain": phase1_letter,
        "phase2_domain": phase2_letter,
        "growth_factor": gf,
        "cl_epochs": args.cl_epochs,
        "seed": args.seed,
        "deterministic": not args.fast,
        "snapshots": snapshots,
    }
    trainer.results["metrics"]["total_time_seconds"] = elapsed
    trainer.results["metrics"]["total_time_hours"] = elapsed / 3600

    out_dir = Path(cfg.output_dir) / cfg.name
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(trainer.results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'results.json'}")
    print(f"Total time: {elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()

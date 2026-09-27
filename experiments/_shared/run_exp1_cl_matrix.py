#!/usr/bin/env python3
"""
Experiment 1: CL Matrix — The Headline Result

Runs all combinations of:
  - growth operators (gate-zero, Net2Net, G_stack)
  - CL methods (replay, distillation, isolation)
  + upper bound (scratch on A+B) and lower bound (base model, no growth)

Usage:
  # Run all configs
  python experiments/_shared/run_exp1_cl_matrix.py

  # Run a specific config by index
  python experiments/_shared/run_exp1_cl_matrix.py --run-index 0

  # Resume from a base checkpoint (skip Phase 1 for all runs)
  python experiments/_shared/run_exp1_cl_matrix.py --base-checkpoint checkpoints/exp1_shared_base/base_final.pt

  # Train base only (shared across all runs)
  python experiments/_shared/run_exp1_cl_matrix.py --base-only
"""

import sys
import os
import argparse


from gatezero.config import exp1_configs, ExperimentConfig, ModelConfig, TrainingConfig, GrowthMethod
from gatezero.data import build_datasets, get_tokenizer
from gatezero.trainer import Trainer


def train_shared_base(device="cuda"):
    """Train the 300M base model once, shared by all Exp 1 runs."""
    cfg = ExperimentConfig(
        name="exp1_shared_base",
        model=ModelConfig(),
        training=TrainingConfig(epochs=10),
    )
    tokenizer = get_tokenizer()
    train_a, val_a = build_datasets("wikitext", tokenizer, cfg.model.max_seq_len)

    trainer = Trainer(cfg, device)
    model = trainer.build_model()
    model = trainer.train_base(model, train_a, val_a)

    ckpt_path = f"checkpoints/{cfg.name}/base_final.pt"
    print(f"\nShared base model saved to: {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: CL Matrix")
    parser.add_argument("--run-index", type=int, default=-1,
                        help="Run a specific config (-1 = all)")
    parser.add_argument("--base-checkpoint", type=str, default=None,
                        help="Path to shared base model checkpoint")
    parser.add_argument("--base-only", action="store_true",
                        help="Only train the shared base model")
    parser.add_argument("--gate-init", type=float, default=0.0,
                        help="Gate init for gate-zero growth runs")
    parser.add_argument("--distill-lambda", type=float, default=None,
                        help="Override CL distill_lambda on the selected config. "
                             "Use 0 for a PURE-ISOLATION control (freeze only, no "
                             "preservation KL on replayed old data) to separate "
                             "the geometric freeze from the LwF distillation.")
    parser.add_argument("--growth-method", type=str, default=None,
                        choices=["gate_fp", "net2net", "gstack"],
                        help="Override the selected config's growth operator (E7). "
                             "e.g. --run-index 2 --clean-fp --growth-method net2net "
                             "matches exp1_gate_fp_isolation_seed42_cleanfp but with Net2Net.")
    parser.add_argument("--growth-factor", type=int, default=4,
                        help="Growth factor g (12 -> 12*g layers, K = 12*(g-1) new "
                             "blocks). Default 4 (K=36). The scale check uses g=2 "
                             "(K=12) and g=3 (K=24); non-default g adds a _g{N} tag.")
    parser.add_argument("--clean-fp", action="store_true",
                        help="Fully clean function preservation: distill_lambda=0,"
                             "gate_warmup_eps=0, clone_noise_std=0 (no preservation "
                             "tricks at all). New-block weights get no gradient until "
                             "the scalar gate opens — the honest clean dynamics.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42,
                        help="TrainingConfig.seed; appended as _seed{N} to the run "
                             "name so runs across seeds never collide (default 42).")
    parser.add_argument("--fast", action="store_true",
                        help="Disable full determinism (cuDNN autotuning) to speed up "
                             "large runs. Use for extra multi-seed points where exact "
                             "bit-reproducibility isn't needed.")
    args = parser.parse_args()

    # Step 1: Train or load shared base
    if args.base_only:
        train_shared_base(args.device)
        return

    base_ckpt = args.base_checkpoint
    if base_ckpt is None:
        print("No base checkpoint provided. Training shared base model...")
        base_ckpt = train_shared_base(args.device)

    # Step 2: Load datasets
    tokenizer = get_tokenizer()
    seq_len = ModelConfig().max_seq_len
    print("\nLoading datasets...")
    train_a, val_a = build_datasets("wikitext", tokenizer, seq_len)
    train_b, val_b = build_datasets("bookcorpus", tokenizer, seq_len,
                                     max_tokens=118_000_000)

    # Step 3: Run experiments
    configs = exp1_configs()

    if args.run_index >= 0:
        configs = [configs[args.run_index]]
        print(f"\nRunning single config: {configs[0].name}")
    else:
        print(f"\nRunning {len(configs)} configs")

    # Growth-factor override (scale check). g=4 keeps the original run names.
    if args.growth_factor != 4:
        for cfg in configs:
            cfg.growth.growth_factor = args.growth_factor
            cfg.name = f"{cfg.name}_g{args.growth_factor}"
        print(f"  Override: growth_factor={args.growth_factor} "
              f"-> output tagged (_g{args.growth_factor})")

    # Tag every run with its seed (default 42) so runs across seeds never collide.
    for cfg in configs:
        cfg.training.seed = args.seed
        cfg.name = f"{cfg.name}_seed{args.seed}"
    print(f"  seed={args.seed} -> output tagged (_seed{args.seed})")
    if args.fast:
        for cfg in configs:
            cfg.training.deterministic = False
        print("  FAST: deterministic=False (cuDNN autotuning; not bit-reproducible)")

    # Optional growth-operator override (E7: run the same CL recipe with a
    # different FP mechanism). Rename gate_fp -> <method> so results don't collide.
    if args.growth_method is not None:
        for cfg in configs:
            cfg.growth.method = GrowthMethod(args.growth_method)
            cfg.name = cfg.name.replace("gate_fp", args.growth_method)
        print(f"  Override: growth_method={args.growth_method} "
              f"-> output renamed (gate_fp -> {args.growth_method})")

    # Optional distill_lambda override (e.g. --distill-lambda 0 for a
    # pure-isolation control). Rename the run so its results.json does not
    # overwrite the default-lambda run's output.
    if args.distill_lambda is not None:
        for cfg in configs:
            cfg.cl.distill_lambda = args.distill_lambda
            cfg.name = f"{cfg.name}_lam{args.distill_lambda:g}"
        print(f"  Override: distill_lambda={args.distill_lambda:g} "
              f"(pure isolation if 0); output renamed with _lam suffix")

    if args.clean_fp:
        for cfg in configs:
            cfg.cl.distill_lambda = 0.0
            cfg.cl.gate_warmup_eps = 0.0
            cfg.growth.clone_noise_std = 0.0
            cfg.name = f"{cfg.name}_cleanfp"
        print("  CLEAN-FP: distill_lambda=0, gate_warmup_eps=0, clone_noise_std=0 "
              "-> output renamed with _cleanfp suffix")

    all_results = []
    for i, cfg in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"# Config {i+1}/{len(configs)}: {cfg.name}")
        print(f"{'#'*60}")

        # Upper bound: train from scratch on A ∪ B jointly.
        # Fair comparison vs CL: CL sees 10|A| + 10|B| tokens total (10 ep
        # base on A + 10 ep CL on B), so joint training runs 10 epochs over
        # A∪B = 10(|A|+|B|) tokens — equivalent token budget.
        if "upper_bound" in cfg.name:
            import json
            from pathlib import Path
            from torch.utils.data import ConcatDataset
            from gatezero.evaluation import evaluate_cl

            print("  Upper bound: training from scratch on A ∪ B jointly")
            train_ab = ConcatDataset([train_a, train_b])
            val_ab = ConcatDataset([val_a, val_b])

            trainer = Trainer(cfg, args.device)
            model = trainer.build_model()
            model = trainer.train_base(model, train_ab, val_ab)
            best_ckpt = trainer.ckpt_dir / "base_best.pt"
            if best_ckpt.exists():
                trainer._load_checkpoint(model, best_ckpt)
                print(f"  Upper bound: evaluating best checkpoint ({best_ckpt.name})")
            metrics = evaluate_cl(model, val_a, val_b, device=args.device)
            print(f"  Upper bound: PPL_A={metrics['ppl_a']:.2f}, "
                  f"PPL_B={metrics['ppl_b']:.2f}")

            # Upper bound bypasses run_full_pipeline, so write results.json
            # ourselves to keep downstream analysis consistent.
            out_dir = Path(cfg.output_dir) / cfg.name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "results.json", "w") as f:
                json.dump({
                    "config": cfg.name,
                    "metrics": {
                        "final_ppl_a": metrics["ppl_a"],
                        "final_ppl_b": metrics["ppl_b"],
                        "delta_a": "N/A",
                        "training_data": "wikitext ∪ bookcorpus (joint)",
                        "epochs": cfg.training.epochs,
                    },
                }, f, indent=2)
            all_results.append({"name": cfg.name, **metrics})
            continue

        if cfg.growth.method.value == "gate_fp":
            cfg.growth.gate_init_value = args.gate_init
            if args.gate_init != 0.0:
                cfg.name = f"{cfg.name}_gateinit_{args.gate_init}"

        cfg.resume_base_checkpoint = base_ckpt
        trainer = Trainer(cfg, args.device)
        results = trainer.run_full_pipeline(train_a, val_a, train_b, val_b)
        all_results.append(results)

    # Summary table
    print(f"\n{'='*80}")
    print("EXPERIMENT 1 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<40} {'PPL_A':>8} {'PPL_B':>8} {'Delta_A':>8}")
    print("-" * 66)

    def fmt_float(val, signed=False):
        if isinstance(val, (int, float)):
            return f"{val:+8.2f}" if signed else f"{val:>8.2f}"
        return f"{str(val):>8}"

    for r in all_results:
        name = r.get("config", r.get("name", "?"))
        m = r.get("metrics", r)
        ppl_a = m.get("final_ppl_a", m.get("ppl_a", "?"))
        ppl_b = m.get("final_ppl_b", m.get("ppl_b", "?"))
        delta = m.get("delta_a", "?")
        print(
            f"{name:<40} {fmt_float(ppl_a)} {fmt_float(ppl_b)} {fmt_float(delta, signed=True)}"
        )


if __name__ == "__main__":
    main()

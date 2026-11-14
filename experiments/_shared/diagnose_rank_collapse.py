#!/usr/bin/env python3
"""
E1/E3 — the functional rank added by growth (E1) and whether it survives
continual learning or collapses (E3). Measured at three stages: base -> grown
(alpha=0) -> after-CL.

Two matrix-free measures at each stage:

  PRIMARY  (param->output Jacobian, theory-matching):
    Effective rank of the K new-block gate directions of the functional
    Jacobian J = d(outputs)/d(theta), via forward-mode JVP columns (reused from
    diagnose_transversality). At growth the K columns are ~independent (rank K,
    the transversality result); collapse under CL => effective rank < K.
    Applies to grown & after-CL (base has no new blocks).

  ROBUSTNESS (representation matrix rank, Dong-2021 style, forward-only):
    Effective rank of the final hidden-state matrix H [n_tokens, d_model].
    This is the rank-collapse-literature measure and the "usable capacity"
    notion. Applies to all three stages.

Effective rank = exp(H) where H is the Shannon entropy of the L1-normalised
singular-value spectrum (Roy & Vetterli 2007) — a soft, threshold-free rank.

Run --self-test first (tiny CPU model, no checkpoints): validates the estimator
against a KNOWN-independent case (rank == K) and a KNOWN-collapsed case
(rank << K) before spending any GPU time.
"""
import sys
import os
import json
import copy
import math
import argparse

import torch
import torch.func as tf


# ── effective rank ────────────────────────────────────────────────────────
def effective_rank(singular_values: torch.Tensor) -> float:
    """exp(entropy of L1-normalised singular values). Roy & Vetterli 2007."""
    s = singular_values.float()
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return math.exp(entropy)


def threshold_rank(singular_values: torch.Tensor, rel: float = 1e-3) -> int:
    s = singular_values.float()
    if s.numel() == 0:
        return 0
    return int((s > rel * s[0]).sum().item())


def spectrum_stats(M: torch.Tensor) -> dict:
    """SVD-based rank stats for a [D, K] column matrix (economy SVD)."""
    S = torch.linalg.svdvals(M.float())
    return {
        "effective_rank": effective_rank(S),
        "threshold_rank_1e-3": threshold_rank(S, 1e-3),
        "n_columns": int(M.shape[1]),
        "sigma_max": float(S[0]),
        "sigma_min": float(S[-1]),
        "top_svs": S[:min(10, S.numel())].tolist(),
    }


# ── PRIMARY: param->output Jacobian columns for new-block gates ────────────
def jvp_column(model, x, params, tangent_dict):
    def fn(p):
        return tf.functional_call(model, p, (x,))
    _out, jvp_value = tf.jvp(fn, (params,), (tangent_dict,))
    return jvp_value.detach().flatten()


def new_gate_param_names(model):
    new_idx = list(range(model.n_original_blocks, len(model.blocks)))
    return [f"blocks.{i}.block_gate" for i in new_idx]


def gate_jacobian_columns(model, x):
    """[D, K] — one JVP column per new-block scalar gate direction."""
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    names = new_gate_param_names(model)
    cols = []
    for gname in names:
        tangent = {k: torch.zeros_like(v) for k, v in params.items()}
        tangent[gname] = torch.ones_like(params[gname])
        cols.append(jvp_column(model, x, params, tangent))
    if not cols:
        return torch.empty(0, 0), 0
    return torch.stack(cols, dim=1), len(names)


# ── ROBUSTNESS: final hidden-state representation matrix rank ───────────────
def representation_matrix(model, x):
    """Final hidden states H [n_tokens, d_model] via a hook on final_norm."""
    captured = {}

    def hook(_m, _inp, out):
        captured["h"] = out.detach()

    handle = model.final_norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    h = captured["h"]                       # [B, T, d_model]
    return h.reshape(-1, h.shape[-1])       # [B*T, d_model]


def representation_rank(model, x) -> dict:
    H = representation_matrix(model, x).float()
    H = H - H.mean(dim=0, keepdim=True)     # center (Dong: remove rank-1 mean)
    S = torch.linalg.svdvals(H)
    return {
        "effective_rank": effective_rank(S),
        "threshold_rank_1e-3": threshold_rank(S, 1e-3),
        "n_tokens": int(H.shape[0]),
        "d_model": int(H.shape[1]),
        "sigma_max": float(S[0]),
        "top_svs": S[:min(10, S.numel())].tolist(),
    }


def measure_stage(model, x, label, do_gate=True) -> dict:
    model.eval()
    out = {"stage": label, "representation": representation_rank(model, x)}
    print(f"  [{label}] representation eff-rank = "
          f"{out['representation']['effective_rank']:.2f} "
          f"(thr-rank {out['representation']['threshold_rank_1e-3']}, "
          f"n_tokens={out['representation']['n_tokens']})")
    if do_gate:
        J, K = gate_jacobian_columns(model, x)
        if K > 0:
            gs = spectrum_stats(J)
            out["gate_jacobian"] = {**gs, "K_new_gates": K}
            print(f"  [{label}] gate-Jacobian eff-rank = {gs['effective_rank']:.2f}"
                  f" / K={K}  (thr-rank {gs['threshold_rank_1e-3']})")
    return out


# ── self-test: validate the estimator on a tiny model ──────────────────────
def self_test():
    """Estimator must: (a) report rank==K when K gate columns are independent,
    (b) report rank<<K when the new blocks are made near-identical (collapsed)."""
    import torch.nn as nn
    from gatezero.config import ModelConfig, GrowthConfig, GrowthMethod
    from gatezero.models.transformer import GrowableTransformer, GatedTransformerBlock
    from gatezero.growth import grow_transformer

    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=4, d_ffn=64, n_layers=2,
                      max_seq_len=64, vocab_size=100)
    x = torch.randint(0, 100, (2, 32))

    # independent case: gate-zero growth, then replace the new blocks with fresh
    # random blocks (exact clones are degenerate by design) and open the gates to
    # small DISTINCT values so the columns are non-degenerate
    m = GrowableTransformer(copy.deepcopy(cfg)).eval()
    gc = GrowthConfig(method=GrowthMethod.GATE_ZERO, growth_factor=4)
    m = grow_transformer(m, gc, gate_init_value=0.0)
    new_idx = list(range(m.n_original_blocks, len(m.blocks)))
    for i in new_idx:
        m.blocks[i] = GatedTransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ffn,
                                            cfg.max_seq_len).eval()
    with torch.no_grad():
        for j, i in enumerate(new_idx):
            m.blocks[i].block_gate.data.fill_(0.02 + 0.001 * j)
    J, K = gate_jacobian_columns(m, x)
    indep = spectrum_stats(J)
    print(f"\n[self-test] INDEPENDENT: K={K}, eff-rank={indep['effective_rank']:.2f}, "
          f"thr-rank={indep['threshold_rank_1e-3']}")

    # collapsed case: every new block an identical copy of block 0, all gates at
    # 0 (the growth point) -> identical inputs, identical columns -> rank 1
    src = copy.deepcopy(m.blocks[new_idx[0]])
    with torch.no_grad():
        for i in new_idx:
            m.blocks[i].load_state_dict(src.state_dict())
            m.blocks[i].block_gate.data.zero_()
    Jc, _ = gate_jacobian_columns(m, x)
    coll = spectrum_stats(Jc)
    print(f"[self-test] COLLAPSED:   K={K}, eff-rank={coll['effective_rank']:.2f}, "
          f"thr-rank={coll['threshold_rank_1e-3']}")

    # representation rank sanity: finite, <= d_model
    rr = representation_rank(m, x)
    print(f"[self-test] representation eff-rank={rr['effective_rank']:.2f} "
          f"(<= d_model={rr['d_model']})")

    ok = (indep["effective_rank"] > coll["effective_rank"] + 1.0
          and indep["threshold_rank_1e-3"] > coll["threshold_rank_1e-3"]
          and coll["threshold_rank_1e-3"] <= 3
          and rr["effective_rank"] <= rr["d_model"] + 1e-6)
    print(f"\n[self-test] {'PASS' if ok else 'FAIL'}: "
          f"independent rank ({indep['effective_rank']:.1f}) >> "
          f"collapsed rank ({coll['effective_rank']:.1f})")
    return ok


# ── real-checkpoint driver ─────────────────────────────────────────────────
def build_grown_skeleton(growth_factor, device):
    """Base architecture grown to g*L layers, gate-zero (alpha=0)."""
    from gatezero.config import ExperimentConfig, ModelConfig, GrowthConfig, GrowthMethod, CLConfig
    from gatezero.models.transformer import GrowableTransformer
    from gatezero.growth import grow_transformer
    cfg = ExperimentConfig(
        name="diag_rank_collapse", model=ModelConfig(),
        growth=GrowthConfig(method=GrowthMethod.GATE_ZERO,
                            growth_factor=growth_factor, gate_init_value=0.0),
        cl=CLConfig(),
    )
    model = GrowableTransformer(cfg.model)
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="Validate the estimator on a tiny CPU model; no checkpoints.")
    ap.add_argument("--base-checkpoint", type=str, default=None)
    ap.add_argument("--cl-checkpoint", type=str, default=None,
                    help="after-CL cl_final.pt (grown architecture)")
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=128,
                    help="Probe sequence length; JVP column dim D = batch*seq*vocab.")
    ap.add_argument("--probe-data", type=str, default="random",
                    choices=["random", "wikitext", "bookcorpus", "code"],
                    help="Probe input: random token ids, or a real batch from a "
                         "domain (representation rank on real text is the "
                         "Dong-2021 setting; random tokens are a weak proxy).")
    ap.add_argument("--out", type=str, default="results/diagnose_rank_collapse/results.json")
    args = ap.parse_args()

    if args.self_test:
        ok = self_test()
        sys.exit(0 if ok else 1)

    assert args.base_checkpoint and args.cl_checkpoint, \
        "need --base-checkpoint and --cl-checkpoint (or --self-test)"

    from gatezero.growth import grow_transformer
    device = args.device
    torch.manual_seed(0)
    _, cfg0 = build_grown_skeleton(args.growth_factor, device)
    vocab = cfg0.model.vocab_size
    seq = min(args.probe_seq, cfg0.model.max_seq_len)
    if args.probe_data == "random":
        x_base = torch.randint(0, vocab, (args.probe_batch, seq)).to(device)
    else:
        from gatezero.data import build_datasets, get_tokenizer
        from torch.utils.data import DataLoader
        tok = get_tokenizer()
        tr, _va = build_datasets(args.probe_data, tok, seq, max_tokens=2_000_000)
        loader = DataLoader(tr, batch_size=args.probe_batch, shuffle=False)
        xb, _yb = next(iter(loader))
        x_base = xb.to(device)
        print(f"Probe: real '{args.probe_data}' batch {tuple(x_base.shape)}")

    results = {"probe": {"batch": args.probe_batch, "seq": seq, "vocab": vocab,
                         "data": args.probe_data},
               "growth_factor": args.growth_factor, "stages": []}

    # STAGE base
    model, cfg = build_grown_skeleton(args.growth_factor, device)
    ck = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()
    print("STAGE base:")
    results["stages"].append(measure_stage(model, x_base, "base", do_gate=False))

    # STAGE grown (alpha=0) — reconstruct from base, deterministic
    model_g, cfg = build_grown_skeleton(args.growth_factor, device)
    ck = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model_g.load_state_dict(ck["model_state_dict"])
    model_g = grow_transformer(model_g, cfg.growth, gate_init_value=0.0).to(device).eval()
    print("STAGE grown (alpha=0):")
    results["stages"].append(measure_stage(model_g, x_base, "grown", do_gate=True))

    # STAGE after-CL — grown skeleton + cl_final weights
    model_cl, cfg = build_grown_skeleton(args.growth_factor, device)
    model_cl = grow_transformer(model_cl, cfg.growth, gate_init_value=0.0)
    ckc = torch.load(args.cl_checkpoint, map_location="cpu", weights_only=False)
    state = ckc.get("model_state_dict", ckc)
    model_cl.load_state_dict(state)
    model_cl = model_cl.to(device).eval()
    print("STAGE after-CL:")
    results["stages"].append(measure_stage(model_cl, x_base, "after_cl", do_gate=True))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    # headline
    def rr(stage):
        s = next(s for s in results["stages"] if s["stage"] == stage)
        return s["representation"]["effective_rank"]
    def gr(stage):
        s = next(s for s in results["stages"] if s["stage"] == stage)
        return s.get("gate_jacobian", {}).get("effective_rank")
    print("\n=== E1/E3 headline ===")
    print(f"representation eff-rank:  base {rr('base'):.2f} -> grown {rr('grown'):.2f} "
          f"-> after-CL {rr('after_cl'):.2f}")
    print(f"gate-Jacobian eff-rank:   grown {gr('grown'):.2f} -> after-CL {gr('after_cl'):.2f}")


if __name__ == "__main__":
    main()

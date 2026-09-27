#!/usr/bin/env python3
"""Appendix B robustness check: reproduce (or not) Dong-2021 representation-rank
collapse vs depth in our model. Measures the
effective rank of the hidden-state matrix H_l [tokens, d_model] after EACH block
l, on real WikiText, and reports rank vs depth. Dong: pure attention collapses
toward rank-1 with depth; skip+MLP prevent it. Our model has skip+MLP, so the
outcome is informative either way (collapse -> the prior applies; no collapse ->
our architecture resists it, and so does the added functional rank).
"""
import sys, os, json, argparse, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import diagnose_rank_gaps as g

def eff_rank(S):
    S = S.float(); S = S[S > 0]
    if S.numel() == 0: return 0.0
    p = S / S.sum(); return math.exp(-(p * torch.log(p)).sum().item())

def per_layer_repr_rank(model, x):
    caps = {}
    handles = []
    for i, blk in enumerate(model.blocks):
        def mk(idx):
            def hook(_m, _in, out): caps[idx] = out.detach()
            return hook
        handles.append(blk.register_forward_hook(mk(i)))
    try:
        with torch.no_grad(): model(x)
    finally:
        for h in handles: h.remove()
    ranks = []
    for i in range(len(model.blocks)):
        H = caps[i].reshape(-1, caps[i].shape[-1]).float()
        H = H - H.mean(0, keepdim=True)
        S = torch.linalg.svdvals(H)
        ranks.append({"layer": i, "eff_rank": eff_rank(S),
                      "n_tokens": H.shape[0], "d_model": H.shape[1]})
    return ranks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--cl-checkpoint", default=None, help="after-CL grown; if omitted, use base only")
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--growth-method", default="gate_fp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-batch", type=int, default=4)
    ap.add_argument("--probe-seq", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    from gatezero.data import build_datasets, get_tokenizer
    from torch.utils.data import DataLoader

    if a.cl_checkpoint:
        model, cfg = g.load_after_cl(a.base_checkpoint, a.cl_checkpoint,
                                     a.growth_factor, a.growth_method, dev)
        tag = "after_cl"
    else:
        model, cfg = g.build_grown(a.base_checkpoint, 1, "gate_fp", dev)  # g=1 -> base only
        tag = "base"
    seq = min(a.probe_seq, cfg.model.max_seq_len)
    tok = get_tokenizer(); tr, _ = build_datasets("wikitext", tok, seq, max_tokens=2_000_000)
    x = next(iter(DataLoader(tr, batch_size=a.probe_batch, shuffle=False)))[0].to(dev)

    ranks = per_layer_repr_rank(model, x)
    d_model = ranks[0]["d_model"]; n_tokens = ranks[0]["n_tokens"]
    print(f"[{tag}] {len(ranks)} layers, d_model={d_model}, n_tokens={n_tokens}")
    print("layer:  " + " ".join(f"{r['layer']:>4}" for r in ranks))
    print("erank:  " + " ".join(f"{r['eff_rank']:>4.0f}" for r in ranks))
    first, last = ranks[0]["eff_rank"], ranks[-1]["eff_rank"]
    mn = min(r["eff_rank"] for r in ranks)
    print(f"first-layer erank={first:.1f}  last-layer erank={last:.1f}  min={mn:.1f}  "
          f"({'COLLAPSE toward depth' if last < 0.5*first else 'no strong collapse (skip+MLP)'})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"tag": tag, "d_model": d_model, "n_tokens": n_tokens, "ranks": ranks},
              open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()

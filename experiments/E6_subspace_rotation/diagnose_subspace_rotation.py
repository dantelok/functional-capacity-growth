#!/usr/bin/env python3
"""
Appendix-grade subspace-rotation table: principal-angle distribution of the
K=36 gate directions, binned, averaged over probe batches, compared to a random
null. Reuses helpers from diagnose_rank_gaps.py.

Two comparisons per run:
  grown->afterCL : rotation of the added subspace over CL (single checkpoint).
  B-vs-C         : if --compare-cl-checkpoint given, angles between the
                   B-specialized subspace (--compare) and this after-CL (C) one.

NULL: two independent random K-dim subspaces in R^D (same D as the probe),
averaged over --null-draws. In high D these concentrate at ~90 deg, so the null
row shows ~all 36 angles in the top bin — the baseline the observed rows beat.
"""
import sys, os, json, argparse, math
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))
import diagnose_rank_gaps as g

BINS = [(0, 45), (45, 60), (60, 70), (70, 80), (80, 85), (85, 90.0001)]
BIN_LABELS = ["[0,45)", "[45,60)", "[60,70)", "[70,80)", "[80,85)", "[85,90]"]

def binned(angles):
    counts = [0] * len(BINS)
    for a in angles:
        for i, (lo, hi) in enumerate(BINS):
            if lo <= a < hi:
                counts[i] += 1; break
    return counts

def mean_std(rows):
    import statistics as st
    n = len(rows[0])
    return ([st.mean([r[i] for r in rows]) for i in range(n)],
            [st.pstdev([r[i] for r in rows]) if len(rows) > 1 else 0.0 for i in range(n)])

def null_dist(D, K, draws, device):
    rows = []
    for _ in range(draws):
        A = torch.randn(D, K, device=device); B = torch.randn(D, K, device=device)
        Qa, _ = torch.linalg.qr(A, mode="reduced"); Qb, _ = torch.linalg.qr(B, mode="reduced")
        c = torch.linalg.svdvals(Qa.T @ Qb).clamp(0, 1)
        ang = torch.rad2deg(torch.acos(c)).cpu().tolist()
        rows.append(binned(ang))
    return mean_std(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--cl-checkpoint", required=True)
    ap.add_argument("--compare-cl-checkpoint", default=None)
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--growth-method", default="gate_fp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-probe-batches", type=int, default=5)
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=128)
    ap.add_argument("--null-draws", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device

    grown, cfg = g.build_grown(args.base_checkpoint, args.growth_factor, args.growth_method, dev)
    aftercl, _ = g.load_after_cl(args.base_checkpoint, args.cl_checkpoint,
                                 args.growth_factor, args.growth_method, dev)
    compare = None
    if args.compare_cl_checkpoint:
        compare, _ = g.load_after_cl(args.base_checkpoint, args.compare_cl_checkpoint,
                                     args.growth_factor, args.growth_method, dev)
    vocab = cfg.model.vocab_size
    seq = min(args.probe_seq, cfg.model.max_seq_len)
    D = args.probe_batch * seq * vocab

    growncl_rows, bvsc_rows = [], []
    for b in range(args.n_probe_batches):
        torch.manual_seed(1000 + b)
        x = torch.randint(0, vocab, (args.probe_batch, seq)).to(dev)
        Jg = g.gate_columns(grown, x); Jc = g.gate_columns(aftercl, x)
        growncl_rows.append(binned(g.principal_angles(Jg, Jc)))
        if compare is not None:
            Jcmp = g.gate_columns(compare, x)
            bvsc_rows.append(binned(g.principal_angles(Jc, Jcmp)))

    null_m, null_s = null_dist(D, len(g.new_gate_names(grown)), args.null_draws, dev)
    gc_m, gc_s = mean_std(growncl_rows)
    out = {"D": D, "K": len(g.new_gate_names(grown)),
           "n_probe_batches": args.n_probe_batches, "null_draws": args.null_draws,
           "bins": BIN_LABELS,
           "null": {"mean": null_m, "std": null_s},
           "grown_to_aftercl": {"mean": gc_m, "std": gc_s}}
    if bvsc_rows:
        bc_m, bc_s = mean_std(bvsc_rows)
        out["b_vs_c"] = {"mean": bc_m, "std": bc_s}

    def fmt(m, s): return "  ".join(f"{m[i]:4.1f}±{s[i]:3.1f}" for i in range(len(m)))
    print(f"\nD={D:,}  K={out['K']}  batches={args.n_probe_batches}  null_draws={args.null_draws}")
    print("bin:            " + "  ".join(f"{l:>9}" for l in BIN_LABELS))
    print("null:           " + fmt(null_m, null_s))
    print("grown->afterCL: " + fmt(gc_m, gc_s))
    if bvsc_rows:
        print("B-vs-C:         " + fmt(out['b_vs_c']['mean'], out['b_vs_c']['std']))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

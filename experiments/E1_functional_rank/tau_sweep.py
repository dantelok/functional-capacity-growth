#!/usr/bin/env python3
"""Threshold (tau) sensitivity of the hard rank, from stored singular values.

Reads every results.json that stores `residual_singular_values` (the spectrum of
the transversal residual (I - P_old) J_alpha written by diagnose_transversality.py
and multigpu_transversality.py) and reports, per measurement:

  * sigma_min / sigma_max and the hard rank at each tau, with the threshold taken
    relative to the residual's own sigma_max (the convention the stored
    `effective_rank_at_thr_1e_3` used; exact);
  * the same quantities with the threshold relative to sigma_max(J_alpha) (the
    convention described in Appendix B). sigma_max(J_alpha) is not stored, so it
    is bracketed rigorously and the hard rank is reported as [lower, upper]:
        lower bounds: sigma_max(resid)            (projection cannot increase norms)
                      max_i ||J_alpha[:, i]||     (if column norms are stored)
        upper bounds: sigma_max(resid) / sin(theta_min)
                      ||J_alpha||_F               (if column norms are stored)
    where theta_min is the smallest principal angle between col(J_alpha) and the
    sampled old span: every v in col(J_alpha) keeps at least sin(theta_min)*||v||
    after projection, so sigma_max(resid) >= sin(theta_min) * sigma_max(J_alpha);
  * the largest consecutive spectral gap sigma_i / sigma_{i+1}.

CPU only, standard library only; no checkpoints needed.

Usage (from repo root):
    python3 experiments/E1_functional_rank/tau_sweep.py
    python3 experiments/E1_functional_rank/tau_sweep.py --files results/transversality_grid/*/results.json
"""
import argparse
import glob
import json
import math
import os

DEFAULT_GLOBS = [
    "results/diagnose_transversality/results.json",
    "results/transversality_sweep/*/results.json",
    "results/transversality_grid/*/results.json",
]
DEFAULT_TAUS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def hard_rank(S, thr):
    return sum(1 for s in S if s > thr)


def analyze(path, taus):
    with open(path) as f:
        d = json.load(f)
    m = d.get("metrics", {})
    S = m.get("residual_singular_values")
    if not S:
        return None
    S = sorted((float(s) for s in S), reverse=True)
    args = d.get("args", {})
    K = m.get("K_new_gates", len(S))
    s_max, s_min = S[0], S[-1]

    # Bracket sigma_max(J_alpha).
    col_norms = m.get("J_alpha_norm_per_column")
    lo = [s_max]
    hi = []
    theta_min = m.get("smallest_angle_deg")
    if theta_min is not None and theta_min > 0:
        hi.append(s_max / math.sin(math.radians(theta_min)))
    if col_norms:
        lo.append(max(col_norms))
        hi.append(math.sqrt(sum(c * c for c in col_norms)))
    ja_lo = max(lo)
    ja_hi = min(hi) if hi else None

    gaps = [(S[i] / S[i + 1], i + 1) for i in range(len(S) - 1) if S[i + 1] > 0]
    gap_ratio, gap_at = max(gaps) if gaps else (float("nan"), None)

    rows = []
    for t in taus:
        rows.append({
            "tau": t,
            "hard_rank_own": hard_rank(S, t * s_max),
            "hard_rank_jalpha_lower": (hard_rank(S, t * ja_hi)
                                       if ja_hi is not None else None),
            "hard_rank_jalpha_upper": hard_rank(S, t * ja_lo),
        })
    return {
        "file": path,
        "seq_len": args.get("seq_len"),
        "n_old_samples": m.get("M_old_samples", args.get("n_old_samples")),
        "output_dim": m.get("output_dim"),
        "K": K,
        "sigma_max_resid": s_max,
        "sigma_min_resid": s_min,
        "ratio_own": s_min / s_max,
        "sigma_max_jalpha_bounds": [ja_lo, ja_hi],
        "ratio_jalpha_bounds": [s_min / ja_hi if ja_hi else None, s_min / ja_lo],
        "smallest_angle_deg": theta_min,
        "largest_gap_ratio": gap_ratio,
        "largest_gap_after_index": gap_at,
        "tau_sweep": rows,
    }


def fmt_rank(lo, hi):
    if lo is None:
        return f"≤{hi}"
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=None,
                    help="results.json files (default: the shipped transversality runs)")
    ap.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAUS)
    ap.add_argument("--out", default="results/tau_sweep/results.json")
    a = ap.parse_args()

    files = a.files or sorted({f for g in DEFAULT_GLOBS for f in glob.glob(g)})
    reports = [r for r in (analyze(f, a.taus) for f in files) if r is not None]
    if not reports:
        raise SystemExit("no results.json with residual_singular_values found")

    taus_hdr = "  ".join(f"τ={t:.0e}" for t in a.taus)
    print(f"{'measurement':<44} {'seq':>4} {'m':>5}  {'σmin/σmax(own)':>14}  "
          f"{'σmin/σmax(Jα) ∈':>17}  {'gap@':>5}  hard rank (own | Jα bracket): {taus_hdr}")
    for r in reports:
        name = os.path.relpath(os.path.dirname(r["file"]), "results")
        lo, hi = r["ratio_jalpha_bounds"]
        ranks = "  ".join(
            f"{x['hard_rank_own']}|{fmt_rank(x['hard_rank_jalpha_lower'], x['hard_rank_jalpha_upper'])}"
            for x in r["tau_sweep"])
        lo_s = f"{lo:.3f}" if lo is not None else "  ?  "
        print(f"{name:<44} {r['seq_len']!s:>4} {r['n_old_samples']!s:>5}  "
              f"{r['ratio_own']:>14.3f}  [{lo_s}, {hi:.3f}]  "
              f"{r['largest_gap_after_index']!s:>5}  {ranks}")

    # Largest tau at which every measurement provably keeps full rank K.
    def robust_tau(key):
        ok = [t for t in a.taus
              if all(next(x for x in r["tau_sweep"] if x["tau"] == t)[key] == r["K"]
                     for r in reports)]
        return max(ok) if ok else None

    worst_own = min(r["ratio_own"] for r in reports)
    jlo = [r["ratio_jalpha_bounds"][0] for r in reports]
    worst_j = min(jlo) if all(v is not None for v in jlo) else None
    summary = {
        "n_measurements": len(reports),
        "min_ratio_own": worst_own,
        "min_ratio_jalpha_lower_bound": worst_j,
        "largest_tau_full_rank_own": robust_tau("hard_rank_own"),
        "largest_tau_full_rank_jalpha_guaranteed": (robust_tau("hard_rank_jalpha_lower")
                                                    if worst_j is not None else None),
    }
    print("\nsummary:", json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"taus": a.taus, "summary": summary, "measurements": reports}, f, indent=2)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()

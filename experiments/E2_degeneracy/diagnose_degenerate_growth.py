#!/usr/bin/env python3
"""E2: show rank additivity (the paper's Theorem 1) has CONTENT at scale — when new directions are
degenerate, functional rank adds < K. We grow g=4 (K=36) then zero the OUTPUT
projections (o_proj, down_proj) of a fraction of the new blocks, so their
gate-Jacobian columns vanish (block output = 0). Hard rank of J_alpha should then
be ~ K*(1-fraction), not K. This demonstrates transversality/non-degeneracy is a
real, satisfiable-but-not-automatic condition — not a tautology.
"""
import sys, os, json, argparse, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))
import diagnose_rank_gaps as g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--n-probe-batches", type=int, default=3)
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device
    rows = []
    print(f"{'frac':>5} {'n_zeroed':>8} {'K':>4} {'expect':>7} {'hard_rank':>9} {'eff_rank':>13}")
    for frac in a.fractions:
        model, cfg = g.build_grown(a.base_checkpoint, a.growth_factor, "gate_fp", dev)
        new_idx = list(range(model.n_original_blocks, len(model.blocks)))
        K = len(new_idx); n_zero = int(round(frac * K))
        with torch.no_grad():
            for i in new_idx[:n_zero]:
                model.blocks[i].attn.o_proj.weight.zero_()
                model.blocks[i].ffn.down_proj.weight.zero_()
        vocab = cfg.model.vocab_size; seq = min(a.probe_seq, cfg.model.max_seq_len)
        effs, hards = [], []
        for b in range(a.n_probe_batches):
            torch.manual_seed(1000 + b)
            x = torch.randint(0, vocab, (a.probe_batch, seq)).to(dev)
            S = torch.linalg.svdvals(g.gate_columns(model, x).float())
            effs.append(g.effective_rank(S)); hards.append(g.hard_rank(S))
        hm = st.mean(hards); em = st.mean(effs)
        rows.append({"fraction": frac, "n_zeroed": n_zero, "K": K,
                     "expect_rank": K - n_zero, "hard_rank": hm, "eff_rank": em})
        print(f"{frac:>5.2f} {n_zero:>8} {K:>4} {K-n_zero:>7} {hm:>9.1f} {em:>13.2f}")
        del model; torch.cuda.empty_cache()
    ok = all(abs(r["hard_rank"] - r["expect_rank"]) <= 1 for r in rows)
    print(f"\nT2 content: hard rank tracks K - (degenerate count)  [{'PASS' if ok else 'CHECK'}]")
    print("=> rank additivity is NOT a tautology: degenerate new directions add 0 rank.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": rows, "tracks_expected": ok}, open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R3.3 core: does functional rank scale sub-linearly with growth factor g while
params scale linearly? Hard rank = K = (g-1)*n_orig is collinear with params, so
the test is whether EFFECTIVE (soft) rank saturates -> params overstate usable
capacity ('counting directions, not parameters'). Grown models are deterministic
from the base, so no CL checkpoints needed.
"""
import sys, os, json, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))
import diagnose_rank_gaps as g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--growth-factors", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-probe-batches", type=int, default=5)
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=128)
    ap.add_argument("--probe-data", default="random", choices=["random", "wikitext"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    real_x = None
    if a.probe_data == "wikitext":
        from gatezero.data import build_datasets, get_tokenizer
        from torch.utils.data import DataLoader
        _m, _cfg = g.build_grown(a.base_checkpoint, a.growth_factors[0], "gate_fp", "cpu")
        seq0 = min(a.probe_seq, _cfg.model.max_seq_len)
        tok = get_tokenizer(); tr, _ = build_datasets("wikitext", tok, seq0, max_tokens=2_000_000)
        dl = DataLoader(tr, batch_size=a.probe_batch, shuffle=False)
        real_x = [b[0] for i, b in zip(range(a.n_probe_batches), dl)]
        del _m
    rows = []
    print(f"{'g':>2} {'K':>4} {'params':>12} {'hard_rank':>9} {'eff_rank':>13} {'eff/K':>6}")
    for gf in a.growth_factors:
        model, cfg = g.build_grown(a.base_checkpoint, gf, "gate_fp", dev)
        K = len(g.new_gate_names(model))
        params = model.count_params() if hasattr(model, "count_params") else \
                 sum(p.numel() for p in model.parameters())
        vocab = cfg.model.vocab_size; seq = min(a.probe_seq, cfg.model.max_seq_len)
        effs, hards = [], []
        for b in range(a.n_probe_batches):
            torch.manual_seed(1000 + b)
            if real_x is not None:
                x = real_x[b].to(dev)
            else:
                x = torch.randint(0, vocab, (a.probe_batch, seq)).to(dev)
            J = g.gate_columns(model, x)
            S = torch.linalg.svdvals(J.float())
            effs.append(g.effective_rank(S)); hards.append(g.hard_rank(S))
        import statistics as st
        eff_m, eff_s = st.mean(effs), (st.pstdev(effs) if len(effs) > 1 else 0.0)
        hard_m = st.mean(hards)
        rows.append({"g": gf, "K": K, "params": params, "hard_rank": hard_m,
                     "eff_rank_mean": eff_m, "eff_rank_std": eff_s, "eff_over_K": eff_m / K})
        print(f"{gf:>2} {K:>4} {params:>12,} {hard_m:>9.1f} {eff_m:>7.2f}±{eff_s:<4.2f} {eff_m/K:>6.3f}")
        del model; torch.cuda.empty_cache()
    # Saturation check: effective rank PER ADDED BLOCK (eff/K). Params and hard
    # rank both scale exactly with K (new blocks), so the right question is whether
    # EFFECTIVE rank keeps up with K. eff/K falling => added directions increasingly
    # redundant => params overstate usable capacity ('directions, not parameters').
    print("\nsaturation check (effective rank per added block, eff/K):")
    for r in rows:
        print(f"  g={r['g']}: eff/K = {r['eff_over_K']:.3f}"
              f"  ({'saturating -> params overstate capacity' if r['eff_over_K'] < rows[0]['eff_over_K'] - 0.01 else 'no saturation'})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()

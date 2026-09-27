#!/usr/bin/env python3
"""Dump pooled raw principal angles (deg) for the CDF figure:
  b_vs_c    : B-specialized (exp1 after-B) vs C-specialized (seq seed s) over seeds×batches
  grown_cl  : grown -> after-CL over seeds×batches
  null      : random 36-dim subspace pairs in R^D
Saves one JSON of raw angle lists; plotting is done locally.
"""
import sys, os, json, argparse, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))
import diagnose_rank_gaps as g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--compare-cl-checkpoint", required=True, help="B-specialized (exp1 after-B)")
    ap.add_argument("--seq-checkpoints", nargs="+", required=True, help="C-specialized seq cl_final.pt")
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-probe-batches", type=int, default=5)
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=128)
    ap.add_argument("--null-draws", type=int, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device

    grown, cfg = g.build_grown(a.base_checkpoint, a.growth_factor, "gate_fp", dev)
    Bmodel, _ = g.load_after_cl(a.base_checkpoint, a.compare_cl_checkpoint,
                                a.growth_factor, "gate_fp", dev)
    vocab = cfg.model.vocab_size; seq = min(a.probe_seq, cfg.model.max_seq_len)
    D = a.probe_batch * seq * vocab; K = len(g.new_gate_names(grown))

    xs = []
    for b in range(a.n_probe_batches):
        torch.manual_seed(1000 + b)
        xs.append(torch.randint(0, vocab, (a.probe_batch, seq)).to(dev))

    grown_cl, b_vs_c = [], []
    Jg_cache = [g.gate_columns(grown, x) for x in xs]
    JB_cache = [g.gate_columns(Bmodel, x) for x in xs]
    for ck in a.seq_checkpoints:
        C, _ = g.load_after_cl(a.base_checkpoint, ck, a.growth_factor, "gate_fp", dev)
        for i, x in enumerate(xs):
            Jc = g.gate_columns(C, x)
            grown_cl += g.principal_angles(Jg_cache[i], Jc)
            b_vs_c += g.principal_angles(JB_cache[i], Jc)
        del C; torch.cuda.empty_cache()

    null = []
    for _ in range(a.null_draws):
        A = torch.randn(D, K, device=dev); Bm = torch.randn(D, K, device=dev)
        Qa, _ = torch.linalg.qr(A, mode="reduced"); Qb, _ = torch.linalg.qr(Bm, mode="reduced")
        c = torch.linalg.svdvals(Qa.T @ Qb).clamp(0, 1)
        null += torch.rad2deg(torch.acos(c)).cpu().tolist()

    out = {"D": D, "K": K, "n_seeds": len(a.seq_checkpoints),
           "n_probe_batches": a.n_probe_batches, "null_draws": a.null_draws,
           "b_vs_c": b_vs_c, "grown_cl": grown_cl, "null": null}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"Wrote {a.out}: b_vs_c={len(b_vs_c)} grown_cl={len(grown_cl)} null={len(null)} angles")

if __name__ == "__main__":
    main()

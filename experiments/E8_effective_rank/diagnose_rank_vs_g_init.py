#!/usr/bin/env python3
"""E8: is the effective-rank saturation a ROUND-ROBIN-CLONING artifact, or
intrinsic? For each g, measure gate-Jacobian effective rank for two inits of the
NEW blocks:
  clone  : gate-zero's round-robin clones (the real operator) -> has duplicates at high g
  random : fresh randomly-initialized blocks (block_gate still 0, so still FP) -> no duplicates
If 'random' also saturates (eff/K falls with g) -> saturation is intrinsic, E8 claim
generalizes beyond cloning. If 'random' stays ~flat/high -> saturation IS the cloning
artifact, and the E8 claim must be scoped to gate-zero's cloning.
"""
import sys, os, json, argparse, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))
import diagnose_rank_gaps as g
from gatezero.models.transformer import GatedTransformerBlock
from gatezero.reproducibility import set_reproducible

def randomize_new_blocks(model, cfg, dev):
    c = cfg.model
    for i in range(model.n_original_blocks, len(model.blocks)):
        blk = GatedTransformerBlock(c.d_model, c.n_heads, c.d_ffn, c.max_seq_len)
        blk.block_gate.data.zero_()          # keep FP at init (gate=0)
        model.blocks[i] = blk.to(dev)
    return model

def measure(model, cfg, dev, nb, pbatch, pseq):
    vocab = cfg.model.vocab_size; seq = min(pseq, cfg.model.max_seq_len)
    K = len(g.new_gate_names(model)); effs, hards = [], []
    for b in range(nb):
        torch.manual_seed(1000 + b)
        x = torch.randint(0, vocab, (pbatch, seq)).to(dev)
        S = torch.linalg.svdvals(g.gate_columns(model, x).float())
        effs.append(g.effective_rank(S)); hards.append(g.hard_rank(S))
    return K, st.mean(effs), (st.pstdev(effs) if len(effs) > 1 else 0.0), st.mean(hards)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--growth-factors", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-probe-batches", type=int, default=5)
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seeds the random init of the new blocks (clones are exact copies) "
                         "(reset before each model is grown). Probes stay fixed.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device
    rows = []
    print(f"{'g':>2} {'K':>4} {'init':>7} {'hard':>5} {'eff_rank':>13} {'eff/K':>6}")
    for gf in a.growth_factors:
        for init in ("clone", "random"):
            set_reproducible(a.seed)
            model, cfg = g.build_grown(a.base_checkpoint, gf, "gate_fp", dev)
            if init == "random":
                model = randomize_new_blocks(model, cfg, dev)
            K, em, es, hm = measure(model, cfg, dev, a.n_probe_batches, a.probe_batch, a.probe_seq)
            rows.append({"g": gf, "K": K, "init": init, "hard": hm,
                         "eff_mean": em, "eff_std": es, "eff_over_K": em / K})
            print(f"{gf:>2} {K:>4} {init:>7} {hm:>5.0f} {em:>7.2f}±{es:<4.2f} {em/K:>6.3f}")
            del model; torch.cuda.empty_cache()
    print("\nclone vs random eff/K by g:")
    for gf in a.growth_factors:
        c = next(r for r in rows if r["g"] == gf and r["init"] == "clone")["eff_over_K"]
        r = next(r for r in rows if r["g"] == gf and r["init"] == "random")["eff_over_K"]
        print(f"  g={gf}: clone {c:.3f}  random {r:.3f}  (gap {r-c:+.3f})")
    cl = [next(r for r in rows if r["g"]==gf and r["init"]=="clone")["eff_over_K"] for gf in a.growth_factors]
    rd = [next(r for r in rows if r["g"]==gf and r["init"]=="random")["eff_over_K"] for gf in a.growth_factors]
    print(f"\nclone eff/K trend:  {cl[0]:.3f} -> {cl[-1]:.3f}  (drop {cl[0]-cl[-1]:+.3f})")
    print(f"random eff/K trend: {rd[0]:.3f} -> {rd[-1]:.3f}  (drop {rd[0]-rd[-1]:+.3f})")
    verdict = ("random ALSO saturates -> intrinsic, E8 generalizes" if (rd[0]-rd[-1]) > 0.03
               else "random stays flat -> saturation is a CLONING artifact; scope E8 to cloning")
    print("VERDICT:", verdict)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"seed": a.seed, "n_probe_batches": a.n_probe_batches, "probe_batch": a.probe_batch,
               "probe_seq": a.probe_seq, "rows": rows, "verdict": verdict}, open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()

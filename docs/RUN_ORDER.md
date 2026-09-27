# Reproduction Guide — Functional-Rank Decoupling paper

Reproduces the experiments behind the current paper (**growth adds K functional
directions that survive CL — geometric capacity is decoupled from retained
function — and this is a property of growth, not of function preservation**).
Each experiment (E1–E8) maps to a claim in the paper; see its Experiments and Appendix sections.

```bash
pip install -e .          # once, so experiments/ can import gatezero
```
Run from the **repo root**; `results/` and `checkpoints/` resolve against cwd. Analysis
scripts are matrix-free (JVP) and run in minutes on one L40; training runs are multi-day.

---

## Part A — Training runs (produce the checkpoints)

```bash
BASE=checkpoints/exp1_shared_base/base_final.pt
```

| # | Command | Produces |
|---|---|---|
| A0 base (one-time) | `python3 experiments/_shared/run_exp1_cl_matrix.py --base-only` | `exp1_shared_base/base_final.pt` (253M, WikiText=A) |
| A1 gate-zero g4 2-domain ×3 seeds | `python3 experiments/_shared/run_exp1_cl_matrix.py --run-index 2 --base-checkpoint $BASE --clean-fp --seed {42,43,44}` | `exp1_gate_fp_isolation_seed{42,43,44}_cleanfp/cl_final.pt` (857M, A→B) |
| A2 gate-zero g4 3-domain ×3 seeds | `python3 experiments/_shared/run_sequential_domains.py --base-checkpoint $BASE --growth-factor 4 --order A,B,C --clean-fp --seed {42,43,44}` | `seq_wikitext-bookcorpus-code_g4_seed{42,43,44}/cl_final.pt` (A→B→C; saves `cl_after_phase1.pt` too) |
| A3 gate-zero g2/g3 2-domain (scale check) | `python3 experiments/_shared/run_exp1_cl_matrix.py --run-index 2 --base-checkpoint $BASE --clean-fp --growth-factor {2,3}` | `exp1_gate_fp_isolation_g{2,3}_seed42_cleanfp/cl_final.pt` (454M/656M) |
| A4 Net2Net g4 2-domain ×3 seeds | `python3 experiments/_shared/run_exp1_cl_matrix.py --run-index 2 --base-checkpoint $BASE --clean-fp --growth-method net2net --seed {42,43,44} [--fast]` | `exp1_net2net_isolation_seed{42,43,44}_cleanfp/cl_final.pt` |
| A5 replay onto a conflicting objective ×3 seeds | `python3 experiments/_shared/run_exp1_cl_matrix.py --run-index 0 --base-checkpoint $BASE --seed {42,43,44}` | `exp1_gate_fp_replay_seed{42,43,44}/cl_final.pt` (old-domain function destroyed; E5) |

A5 is the one run **without** `--clean-fp`: it deliberately keeps the production
stabilizing knobs, as described in the paper.

*`--fast` disables cuDNN determinism (faster; use for extra multi-seed points — the
hard-rank measure has zero observed variance). Seed 42 kept deterministic as the
reproducible reference.*

---

## Part B — Analysis (the paper's measurements; minutes each, on the checkpoints)

`diagnose_rank_gaps.py` and `diagnose_rank_collapse.py` accept `--self-test` (no
checkpoints needed), which checks the estimator against exact rank on a synthetic
problem; run it once before spending GPU time. Output file names below are the ones
shipped in `results/`, so each command regenerates the file behind the paper's number.

Shared helper for the rank/gap measurements (3 probe batches, 64 old-direction samples):
```bash
gaps() {  # $1 = checkpoint dir, $2 = output name, $3 = growth factor (default 4), $4 = growth method (default gate_fp)
  python3 experiments/_shared/diagnose_rank_gaps.py --base-checkpoint $BASE \
    --cl-checkpoint checkpoints/$1/cl_final.pt --growth-factor ${3:-4} --growth-method ${4:-gate_fp} \
    --n-probe-batches 3 --m-old 64 --out results/diagnose_rank_collapse/$2.json
}
```

### E1 — growth adds rank K, transversal to the old parameters (Table 1)
```bash
python3 experiments/E1_functional_rank/diagnose_transversality.py --base-checkpoint $BASE \
    --growth-factor 4 --batch-size 2 --seq-len 16 --n-old-samples 50 --seed 0 \
    --output-dir results/diagnose_transversality                       # hard rank 36 = K
```

Transversality stress test (Table 4, Appendix B): batch 2, seed 42, fp32, one cell per
(sequence length, number of old-direction samples m).
```bash
T=experiments/E1_functional_rank/diagnose_transversality.py
cell() {  # $1 = results subdir, $2 = seq len, $3 = m
  python3 $T --base-checkpoint $BASE --growth-factor 4 --batch-size 2 --seed 42 \
    --seq-len $2 --n-old-samples $3 --output-dir results/$1/seq$2_m$3
}
for m in 64 128 256 512 1024; do cell transversality_sweep 16 $m; cell transversality_grid 32 $m; done
for m in 64 128 256 512;      do cell transversality_grid 64 $m; done
for m in 64 128 256;          do cell transversality_sweep 128 $m; done

# Cells too large for one GPU: the same projector, D-sharded across GPUs
M=experiments/E1_functional_rank/multigpu_transversality.py
CUDA_VISIBLE_DEVICES=0,1   python3 $M --base-checkpoint $BASE --seed 42 --seq-len 64  --n-old-samples 1024 --output-dir results/transversality_grid/seq64_m1024
CUDA_VISIBLE_DEVICES=0,1   python3 $M --base-checkpoint $BASE --seed 42 --seq-len 128 --n-old-samples 512  --output-dir results/transversality_grid/seq128_m512
CUDA_VISIBLE_DEVICES=0,1,2 python3 $M --base-checkpoint $BASE --seed 42 --seq-len 128 --n-old-samples 1024 --output-dir results/transversality_grid/seq128_m1024
# Cross-check: the multi-GPU projector reproduces the single-GPU seq128 m256 cell
CUDA_VISIBLE_DEVICES=0,1,2 python3 $M --base-checkpoint $BASE --seed 42 --seq-len 128 --n-old-samples 256  --output-dir results/transversality_grid/mgpu_verify3_m256
```
The `run_transv_*.sh` files in `E1_functional_rank/` are the launchers used for the
original runs (GPU ids are machine-specific).

### E2 — degeneracy stress test (Table 1)
```bash
python3 experiments/E2_degeneracy/diagnose_degenerate_growth.py --base-checkpoint $BASE --growth-factor 4 \
    --out results/diagnose_rank_collapse/ea3b_degenerate.json          # hard rank = K − #degenerate (36/27/18/9)
```

### E3–E5 — the added capacity survives continual learning (Table 2, Appendix C)
```bash
# 2-domain isolation (E3 hard rank, E4 transversal residual)
gaps exp1_gate_fp_isolation_seed42_cleanfp gaps_exp1_gatefp
gaps exp1_gate_fp_isolation_seed43_cleanfp gaps_gatefp_isolation_seed43
gaps exp1_gate_fp_isolation_seed44_cleanfp gaps_gatefp_isolation_seed44
# 3-domain sequence (E3, E4)
for s in 42 43 44; do gaps seq_wikitext-bookcorpus-code_g4_seed$s gaps_seq_seed$s; done
# Replay onto a conflicting objective: function destroyed, rank kept (E5)
for s in 42 43 44; do gaps exp1_gate_fp_replay_seed$s gaps_gatefp_replay_seed$s; done
# Scale check, K = 12 / 24 (g = 2 / 3)
gaps exp1_gate_fp_isolation_g2_seed42_cleanfp scale_g2 2
gaps exp1_gate_fp_isolation_g3_seed42_cleanfp scale_g3 3
```
`ea4a_replay.json` is an earlier run of the same seed-42 replay measurement
(`gaps exp1_gate_fp_replay_seed42 ea4a_replay`).

Rank at each stage, base → grown → after-CL, with a random and a WikiText probe:
```bash
C=experiments/_shared/diagnose_rank_collapse.py
python3 $C --base-checkpoint $BASE --cl-checkpoint checkpoints/exp1_gate_fp_isolation_seed42_cleanfp/cl_final.pt \
    --growth-factor 4 --probe-batch 2 --probe-seq 256 --out results/diagnose_rank_collapse/exp1_gatefp_cleanfp.json
python3 $C --base-checkpoint $BASE --cl-checkpoint checkpoints/exp1_gate_fp_isolation_seed42_cleanfp/cl_final.pt \
    --growth-factor 4 --probe-batch 4 --probe-seq 256 --probe-data wikitext --out results/diagnose_rank_collapse/exp1_gatefp_cleanfp_wikitext.json
python3 $C --base-checkpoint $BASE --cl-checkpoint checkpoints/seq_wikitext-bookcorpus-code_g4_seed42/cl_final.pt \
    --growth-factor 4 --probe-batch 2 --probe-seq 256 --out results/diagnose_rank_collapse/seq_g4_seed42.json
python3 $C --base-checkpoint $BASE --cl-checkpoint checkpoints/seq_wikitext-bookcorpus-code_g4_seed42/cl_final.pt \
    --growth-factor 4 --probe-batch 4 --probe-seq 256 --probe-data wikitext --out results/diagnose_rank_collapse/seq_g4_seed42_wikitext.json
```

### E6 — the preserved subspace is re-tasked across domains (Figure 1, Table 6, Appendix D)
The B-specialized reference is the 2-domain seed-42 checkpoint (after A→B); the
C-specialized side is each 3-domain seed.
```bash
B=checkpoints/exp1_gate_fp_isolation_seed42_cleanfp/cl_final.pt
for s in 42 43 44; do
  python3 experiments/E6_subspace_rotation/diagnose_subspace_rotation.py --base-checkpoint $BASE \
      --cl-checkpoint checkpoints/seq_wikitext-bookcorpus-code_g4_seed$s/cl_final.pt --compare-cl-checkpoint $B \
      --n-probe-batches 5 --null-draws 5 --out results/diagnose_rank_collapse/rot_seq$s.json
done
python3 experiments/E6_subspace_rotation/dump_angles_for_cdf.py --base-checkpoint $BASE --compare-cl-checkpoint $B \
    --seq-checkpoints checkpoints/seq_wikitext-bookcorpus-code_g4_seed{42,43,44}/cl_final.pt \
    --n-probe-batches 5 --null-draws 15 --out results/diagnose_rank_collapse/cdf_angles.json
python3 experiments/E6_subspace_rotation/plot_subspace_rotation.py        # -> figures/subspace_rotation_cdf.pdf
```

### E7 — generality across FP mechanisms: Net2Net (Table 3, top)
```bash
for s in 42 43 44; do gaps exp1_net2net_isolation_seed${s}_cleanfp gaps_net2net_seed$s 4 net2net; done
# The gate-zero comparison row reuses the 2-domain isolation files from E3–E5.
```

### E8 — effective rank at fixed parameter count, clone vs random (Table 3, bottom)
```bash
for s in 42 43 44; do                                                           # clone vs random at g = 2/3/4 (Appendix E)
  python3 experiments/E8_effective_rank/diagnose_rank_vs_g_init.py --base-checkpoint $BASE --seed $s \
      --out results/diagnose_rank_collapse/rank_vs_g_init_seed$s.json
done
python3 experiments/E8_effective_rank/diagnose_rank_vs_g.py --base-checkpoint $BASE --probe-data random \
    --out results/diagnose_rank_collapse/rank_vs_g.json                         # effective rank / K vs g
python3 experiments/E8_effective_rank/diagnose_rank_vs_g.py --base-checkpoint $BASE --probe-data wikitext \
    --out results/diagnose_rank_collapse/rank_vs_g_wikitext.json
```

### Appendix B — representation rank vs depth (robustness check)
```bash
python3 experiments/_shared/diagnose_repr_rank_vs_depth.py --base-checkpoint $BASE \
    --cl-checkpoint checkpoints/exp1_gate_fp_isolation_seed42_cleanfp/cl_final.pt --growth-factor 4 \
    --out results/diagnose_rank_collapse/ea4b_repr_depth_aftercl.json
```

---

## Notes
- **Estimators are matrix-free** (forward-mode JVP; never materialize the 343GB Jacobian).
  Hard rank (threshold) is probe-independent and is the headline measure; effective rank is
  probe-dependent and secondary.
- Full experiment→claim mapping and caveats are in the paper (Experiments and Appendix sections).

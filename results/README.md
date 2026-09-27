# results/

One `results.json` (plus rank/gap analysis JSONs) per run — the numbers behind
the paper's tables and figures. This file maps each directory to the paper
result it backs and the command that reproduces it.

**To reproduce:** `pip install -e .`, then run from the repo root. The full
ordered sequence (with the `$BASE` checkpoint and every command) is in
`docs/RUN_ORDER.md`.

## Naming conventions (read first)

- **`gate_fp` in a directory name is the paper's `gate-zero` operator.** The
  on-disk tag `gate_fp` is kept for provenance; in code it is
  `GrowthMethod.GATE_ZERO`.
- **`_cleanfp`** = the *clean-FP configuration* (paper Section 4.1): all
  stabilizing knobs disabled. This is the paper's standard reported setting.
- **Every run directory carries an explicit `seedNN` tag** (`seed42`, `seed43`,
  `seed44`).
- **`archive/`** holds superseded runs and legacy diagnostics that no paper
  result depends on.

## Training runs (checkpoints + perplexities)

| Directory | Paper result | What it is |
|---|---|---|
| `exp1_gate_fp_isolation_seed{42,43,44}_cleanfp` | Table 2, "2-domain isolation" (E3, E4); per-seed PPL in Appendix C | gate-zero, isolation, seeds 42/43/44 |
| `exp1_gate_fp_replay_seed{42,43,44}` | Table 2, "replay onto conflict" (E5, function-destruction counterexample); Appendix C | gate-zero, replay, seeds 42/43/44 |
| `exp1_net2net_isolation_seed{42,43,44}_cleanfp` | Table 3 top, "Net2Net" (E7, generality across FP mechanisms) | Net2Net, isolation, seeds 42/43/44 |
| `seq_wikitext-bookcorpus-code_g4_seed{42,43,44}` | Table 2, "3-domain sequence"; subspace rotation (E6); Appendix C | gate-zero, 3-domain sequential CL, g=4, seeds 42/43/44 |

Commands: `run_exp1_cl_matrix.py` (2-domain matrix) and
`run_sequential_domains.py` (3-domain), per `docs/RUN_ORDER.md`.

## Analysis runs (rank measurements on the checkpoints above)

| Directory | Paper result | What it is |
|---|---|---|
| `diagnose_transversality/` | Table 1 (E1): hard rank 36 = K and transversal at the moment of growth | `diagnose_transversality.py` |
| `diagnose_rank_collapse/` | Rank/gap JSONs behind Tables 1–3 and Figure 1 (E1–E8): functional-Jacobian rank at base → grown → after-CL, per seed and operator | `diagnose_rank_*.py`, `diagnose_subspace_rotation.py` |
| `transversality_grid/`, `transversality_sweep/` | Table 4 (Appendix B): transversality stress test across the sequence-length × probe-width grid (single- and multi-GPU) | `diagnose_transversality.py` via the grid scripts |

## E-label index (paper terminology)

E1 rank = K at growth · E2 degeneracy stress test · E3 hard rank persists
through CL · E4 stays transversal · E5 persists even when the function is
destroyed · E6 subspace re-tasked across domains · E7 generality across FP
mechanisms · E8 effective rank at fixed parameter count.

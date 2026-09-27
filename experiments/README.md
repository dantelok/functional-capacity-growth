# experiments/

Entry-point scripts, organized by the paper's experiment labels (E1–E8). The
installable library lives separately in `src/gatezero/`.

Run everything from the **repo root** after `pip install -e .` (so `import
gatezero` works); outputs resolve against the working directory. Full ordered
commands with arguments are in `docs/RUN_ORDER.md`.

## Layout

- **`E1_functional_rank/`** — E1: growth adds K directions, transversal at init
  (`diagnose_transversality.py`); the multi-GPU transversality sweep behind
  Table 4 (`multigpu_transversality.py`, `transversality_gram.py`,
  `validate_projector.py`, `run_transv_*.sh`); closed-form self-test
  (`verify_rank_theorem.py`).
- **`E2_degeneracy/`** — E2: degeneracy stress test (`diagnose_degenerate_growth.py`).
- **`E6_subspace_rotation/`** — E6: subspace re-tasking across domains
  (`diagnose_subspace_rotation.py`, `dump_angles_for_cdf.py`,
  `plot_subspace_rotation.py`).
- **`E8_effective_rank/`** — E8: effective rank vs growth factor, clone vs random
  (`diagnose_rank_vs_g.py`, `diagnose_rank_vs_g_init.py`).
- **`_shared/`** — scripts used by several experiments:
  - `run_exp1_cl_matrix.py` — 2-domain training runs (E3/E4/E5, and E7 via `--growth-method net2net`).
  - `run_sequential_domains.py` — 3-domain training runs (E6).
  - `diagnose_rank_gaps.py` — hard-rank / transversal-residual estimator on after-CL checkpoints (E3/E4/E5/E7).
  - `diagnose_rank_collapse.py` — rank at base → grown → after-CL (E1/E3).
  - `diagnose_repr_rank_vs_depth.py` — representation-rank robustness (Appendix B).
- **`tests/`** — function-preservation and Net2Net integration checks.

## E → scripts

| E | Question | Scripts |
|---|---|---|
| E1 | rank = K at growth, transversal | `E1_functional_rank/diagnose_transversality.py`; `_shared/diagnose_rank_collapse.py` |
| E2 | degeneracy stress test | `E2_degeneracy/diagnose_degenerate_growth.py` |
| E3 | hard rank persists through CL | `_shared/run_exp1_cl_matrix.py`, `_shared/diagnose_rank_gaps.py`, `_shared/diagnose_rank_collapse.py` |
| E4 | stays transversal | `_shared/run_exp1_cl_matrix.py`, `_shared/diagnose_rank_gaps.py` |
| E5 | persists even when the function is destroyed (replay) | `_shared/run_exp1_cl_matrix.py`, `_shared/diagnose_rank_gaps.py` |
| E6 | subspace re-tasked across domains | `_shared/run_sequential_domains.py`; `E6_subspace_rotation/*` |
| E7 | generality across FP mechanisms (Net2Net) | `_shared/run_exp1_cl_matrix.py --growth-method net2net`, `_shared/diagnose_rank_gaps.py` |
| E8 | effective rank at fixed parameter count | `E8_effective_rank/*` |

Table 4 (transversality sweep, Appendix B) is produced by
`E1_functional_rank/multigpu_transversality.py`.

**Note:** `_shared/` scripts are imported by several experiment scripts (via a
`sys.path` entry pointing at `_shared/`). Keep the folder layout intact so those
cross-imports resolve.

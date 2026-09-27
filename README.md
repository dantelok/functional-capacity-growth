# What Does Model Growth Really Add? Functional Capacity Beyond Parameter Count

[![arXiv](https://img.shields.io/badge/arXiv-2607.14571-b31b1b.svg)](https://arxiv.org/abs/2607.14571)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Dante Lok** · Votee AI & Beever AI

Code and results for the paper
[*What Does Model Growth Really Add? Functional Capacity Beyond Parameter Count*](https://arxiv.org/abs/2607.14571).

## TL;DR

Model growth adds parameters to a trained checkpoint and keeps training. We ask what
it *actually* adds, and measure it directly: the number of new, functionally
independent directions a growth step contributes, i.e. the rank increment of the
model's functional Jacobian. For zero-gated function-preserving growth this increment
is exactly `K` (the number of new blocks) under a transversality condition. Growing a
Transformer from 253M to 857M parameters (`K = 36`) and continually training it across
three domains, **all 36 added directions stay independent**, even when the model
catastrophically forgets (perplexity 28 → 481) or its old function is deliberately
destroyed (25.9 → 127). **Geometric capacity and retained function are decoupled.**

## The question

"Capacity" means three different things in the literature: parameter count and FLOPs
(the growth literature), effective rank of representations (the plasticity literature),
and the ability to represent tasks (continual-learning theory). None of them says how
many *new functional directions* a growth step opens, or whether those directions
survive subsequent training.

We measure that quantity directly. For a model `f_θ` evaluated on a fixed probe set,
the **functional Jacobian** is `J = ∂f/∂θ`, and the **functional rank** is `rank J`:
how many independent directions in parameter space the model can use to change its
output. The capacity a growth step adds is the rank increment it produces.

- **Hard rank**: the number of singular values above `10⁻³ · σ_max`. An integer,
  probe-independent in practice, and our headline measure.
- **Effective (soft) rank**: `exp(H(σ))`, the exponentiated entropy of the normalized
  singular values. Probe-dependent and secondary; it separates redundant from
  efficient directions.

Both are computed **matrix-free** with forward-mode Jacobian–vector products, never
materializing the Jacobian; a full measurement takes minutes on one GPU at the
857M-parameter scale.

## Theory

- **Function preservation and the live direction.** With the gate at zero, the grown
  model computes exactly the base function, the new blocks' weights have zero Jacobian,
  and the `K` scalar gates are the only first-order directions growth opens.
- **Rank additivity under transversality.** If the `K` gate columns are linearly
  independent of the old column space, then `rank(J_grown) = rank(J_old) + K`; if
  transversality fails, the rank grows by strictly less than `K`.
- **Parameter count is not functional dimension.** The added rank is at most `K`, far
  fewer than the number of new parameters; the rest lie in the Jacobian's null space
  at the growth point.
- **Functional rank is a tangent dimension, not a position.** Rank is the dimension of
  the local function manifold, and equal rank does not imply equal function, which is
  why rank can persist while the function changes.

## Two function-preserving growth mechanisms

On the depth axis there are two ways to make a new residual block contribute exactly
zero at initialization; we implement one representative of each.

| Mechanism | How the new block starts at zero | Added directions at growth |
|---|---|---|
| **Gate-zero** (Class 1, zero the gate) | `x ↦ x + α·B(x)` with scalar gate `α = 0` | live: rank `K` |
| **Net2Net** (Class 2, zero the output weights) | attention output and MLP down projections set to zero, gate fixed at 1 | dormant: rank 0, activating as the output weights train |

In code, gate-zero is `GrowthMethod.GATE_ZERO` (on-disk tag `gate_fp`) and Net2Net is
`GrowthMethod.NET2NET`.

## Setup

- **Base:** a 253M-parameter, 12-layer decoder-only Transformer trained on
  WikiText-103 (domain A), shared by every experiment.
- **Growth:** growth factor `g = 4` takes it from 12 to 48 layers, adding `K = 36`
  blocks (857M parameters); `g = 2, 3` (`K = 12, 24`) serve as a scale check. New
  blocks are appended after the original stack and cloned from it in round-robin order.
- **Continual learning:** two domains (A → BookCorpus, B) and three domains
  (A → B → code, C); isolation (old parameters frozen) and replay onto a conflicting
  objective; seeds 42/43/44.
- **Clean function preservation:** persistence and generality runs disable all
  stabilizing aids (distillation toward the base, gate warm-up, clone noise).

## Findings (E1–E8)

| | Claim | Result |
|---|---|---|
| **E1** | Growth adds rank `K`, transversal to the old parameters | hard rank 36 = `K` at growth; transversal in all 20 cells of a sequence-length × sample-count stress test |
| **E2** | The rank counts *live* directions, not blocks | making 0/25/50/75% of new blocks degenerate gives rank 36/27/18/9 |
| **E3** | The added rank persists through continual learning | 36 → 36 after CL, in every condition and seed, with zero variance |
| **E4** | It stays transversal, not absorbed into the old span | transversal residual 36 → 36 |
| **E5** | It persists even when the old function is destroyed | replay onto a conflicting objective: old-domain perplexity 25.9 → 125–127, rank still 36 → 36 |
| **E6** | The preserved subspace is re-tasked, not frozen | the domain-B and domain-C subspaces are mostly near-orthogonal (principal angles ≥ 85° for most directions) |
| **E7** | Persistence is a property of growth, not of the gate | Net2Net reaches the same capacity from dormant directions: 0 → 36 |
| **E8** | Effective rank separates redundant from efficient growth | at `g = 4`: clone-initialized 28.8 vs random-initialized 35.7 (same parameter count, same hard rank) |

Catastrophic forgetting therefore need not reflect a loss of functional capacity, and
preserving dimensionality alone is not enough to preserve the learned function.

## Repository layout

| Path | What lives there |
|---|---|
| `src/gatezero/` | The installable library: growth operators, CL methods, trainer, models |
| `experiments/` | Entry-point scripts, organized by experiment E1–E8 (see [`experiments/README.md`](experiments/README.md)) |
| `results/` | One `results.json` per run plus rank/gap JSONs (see [`results/README.md`](results/README.md) for the directory → paper-result map) |
| `docs/RUN_ORDER.md` | The reproduction sequence: commands, seeds, and order |

## Setup and reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run everything **from the repository root**; `results/` and `checkpoints/` resolve
relative to the working directory. Start with [`docs/RUN_ORDER.md`](docs/RUN_ORDER.md):
Part A trains the checkpoints (multi-day), and Part B runs the paper's measurements on
them (minutes each). Each Part B command regenerates the shipped result file behind a
number in the paper.

```bash
# Train the shared base checkpoint (everything else reuses this)
python3 experiments/_shared/run_exp1_cl_matrix.py --base-only

# One 2-domain continual-learning run (gate-zero, isolation, clean-FP)
python3 experiments/_shared/run_exp1_cl_matrix.py --run-index 2 --clean-fp \
    --base-checkpoint checkpoints/exp1_shared_base/base_final.pt

# Estimator self-test (no checkpoints needed)
python3 experiments/_shared/diagnose_rank_gaps.py --self-test
```

Trained checkpoints are not included (about 8 GB each).

## Citation

```bibtex
@article{lok2026growth,
  title   = {What Does Model Growth Really Add? Functional Capacity Beyond Parameter Count},
  author  = {Lok, Dante},
  journal = {arXiv preprint arXiv:2607.14571},
  year    = {2026}
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

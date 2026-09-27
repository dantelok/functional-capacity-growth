"""
Centralized reproducibility control: seed every RNG surface (CPU + CUDA) and,
optionally, force deterministic GPU kernels.

Why this module exists
----------------------
A fixed torch seed alone does NOT make runs reproducible here. Three surfaces
were previously seeded ad hoc, inside ``Trainer.__init__``, or not at all:

  * numpy   — ``ReplayBuffer`` (data.py) draws its subset with
              ``np.random.choice``; an unseeded numpy stream makes the
              Replay / Isolation recipes non-reproducible even at a
              fixed torch seed.
  * CUDA    — cuDNN autotunes algorithms and uses non-deterministic atomics in
              backprop, so GPU runs vary run-to-run at a fixed torch seed
              unless CUDA is seeded and deterministic kernels are forced.
  * DataLoader workers — with ``num_workers > 0`` each worker process needs an
              explicit, seed-derived RNG (``worker_init_fn``) or batch order and
              any per-sample randomness is nondeterministic across workers.

``set_reproducible(seed)`` closes all three from a single place that every
entry point (the scripts under ``experiments/``) can call BEFORE
touching CUDA, data, or models. It mirrors the bit-reproducible
``set_reproducible`` proven in ``preliminary/`` (verified for Table E.2) and the
inline seeding previously in ``trainer.py``.

Determinism is per-machine: identical results require the same GPU architecture
and pinned cuDNN / torch versions.
"""

import os
import random

import numpy as np
import torch

# Deterministic cuBLAS requires this set BEFORE the CUDA context initialises,
# otherwise torch.use_deterministic_algorithms(True) raises at the first matmul.
# Set at import (harmless when determinism is off) so it precedes any CUDA use
# in a process that imports gatezero.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_reproducible(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, torch (CPU) and all CUDA devices.

    Call at the very top of every entry point, before building datasets,
    models, or moving anything to GPU.

    Args:
        seed: RNG seed applied to every surface (Python, NumPy, torch, CUDA).
        deterministic: when True (default) also force deterministic cuDNN /
            cuBLAS kernels. Set False only for throwaway speed runs — reported
            and reviewer-facing runs must keep this True. On matmul-bound
            Transformers the speed cost is small (~4% measured), so determinism
            is effectively free here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must precede CUDA init (see module-level CUBLAS_WORKSPACE_CONFIG).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: degrade gracefully if an op lacks a deterministic kernel
        # instead of killing a multi-day run at the first offending matmul.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Recover cuDNN autotuning speed; results are no longer bit-reproducible.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn``: derive each worker's Python / NumPy
    seed from torch's per-worker base seed so ``num_workers > 0`` stays
    reproducible.

    Use together with :func:`make_generator`::

        DataLoader(..., worker_init_fn=seed_worker,
                   generator=make_generator(seed))
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """A seeded ``torch.Generator`` for ``DataLoader(generator=...)`` so that
    shuffling order and the per-worker base seeds are deterministic."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

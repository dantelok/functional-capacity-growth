"""Non-function-preserving growth operators.

G_stack (Du et al., NeurIPS 2024): stack g copies of the trained model. Unlike the
function-preserving operators in growth.py (gate-zero and Net2Net), this does not
preserve the base function at initialization; it is kept as a non-FP baseline.
"""

import copy
import torch.nn as nn

from .config import GrowthConfig
from .models.transformer import GrowableTransformer


def gstack_growth(model: GrowableTransformer,
                  config: GrowthConfig) -> GrowableTransformer:
    """
    G_stack: stack g copies of the entire model (Du et al. NeurIPS 2024).

    NOT function-preserving. Simply repeats all blocks g times.
    Each group of n_orig blocks is an identical copy of the trained model.
    """
    g = config.growth_factor
    n_orig = len(model.blocks)
    n_target = n_orig * g

    print(f"G_stack depth growth: {n_orig} -> {n_target} layers "
          f"(stacking {g} copies)")

    new_blocks = nn.ModuleList()
    original_indices = list(range(n_orig))  # first n_orig are originals

    for copy_idx in range(g):
        for block_idx in range(n_orig):
            if copy_idx == 0:
                # First copy: keep original blocks
                new_blocks.append(model.blocks[block_idx])
            else:
                # Subsequent copies: deep copy
                new_blocks.append(copy.deepcopy(model.blocks[block_idx]))

    model.blocks = new_blocks
    model.config.n_layers = n_target
    model.original_block_indices = original_indices

    print(f"  Total params after growth: {model.count_params():,}")
    return model

"""
Function-preserving growth operators: gate-zero and Net2Net depth growth.

Both insert new blocks that contribute exactly zero at initialization, so the grown
model reproduces the base function. The non-function-preserving G_stack baseline lives
in nonfp_growth.py.
"""

import copy
import torch
import torch.nn as nn

from .config import GrowthConfig, GrowthMethod
from .models.transformer import GrowableTransformer, GatedTransformerBlock
from .nonfp_growth import gstack_growth


# ─────────────────────────────────────────────────────────────
# Transformer growth
# ─────────────────────────────────────────────────────────────

def grow_transformer(model: GrowableTransformer, config: GrowthConfig,
                     gate_init_value: float = 0.0) -> GrowableTransformer:
    """Apply growth operator to a GrowableTransformer."""
    if config.method == GrowthMethod.GATE_ZERO:
        return _gate_depth_growth(model, config, gate_init_value)
    elif config.method == GrowthMethod.GSTACK:
        return gstack_growth(model, config)
    elif config.method == GrowthMethod.NET2NET:
        return _net2net_growth(model, config)
    else:
        raise ValueError(f"Unknown growth method: {config.method}")


def _net2net_growth(model: GrowableTransformer,
                    config: GrowthConfig) -> GrowableTransformer:
    """Net2Net / Fixup-style FP depth growth (a DIFFERENT mechanism from gate-zero).

    New blocks are function-preserving because their OUTPUT projections are zeroed
    (o_proj = down_proj = 0), so the block adds 0 to the residual stream regardless
    of block_gate. block_gate is left at 1.0. Contrast with gate-zero, where FP
    comes from the scalar block_gate=0.

    Consequence for the rank story: here the block "opens" through its output
    weight MATRICES (many DOF, non-zero Jacobian at init), not a single scalar gate.
    This is the E7 generality point — a genuinely different FP construction.

    Note: this construction is identical to LEMON's depth Type-1 expansion
    (Wang et al. 2024, §4.3: "zero the last FC layer"). LEMON Type-2 (cancelling
    neuron duplication) is distinct but widens blocks; not implemented.

    Uses the 'append' layout (originals first) so CL isolation freezes the old path.
    """
    g = config.growth_factor
    n_orig = len(model.blocks)
    n_target = n_orig * g
    n_new = n_target - n_orig
    print(f"Net2Net FP depth growth: {n_orig} -> {n_target} layers "
          f"(zero output projections, block_gate=1)")

    new_blocks = nn.ModuleList()
    original_indices = []
    for i in range(n_orig):
        original_indices.append(len(new_blocks))
        new_blocks.append(model.blocks[i])

    for j in range(n_new):
        src_idx = j % n_orig
        nb = copy.deepcopy(model.blocks[src_idx])
        with torch.no_grad():
            # FP: zero the two output projections -> block contributes exactly 0.
            nb.attn.o_proj.weight.zero_()
            nb.ffn.down_proj.weight.zero_()
            # Gate is a FIXED structural 1.0, NOT a learned knob: in Net2Net the
            # block opens through its output matrices, so block_gate must stay at
            # 1.0 (x + 1.0*Block(x) = x + Block(x), the literature form). Freeze it
            # so it is excluded from the optimizer and never drifts into a
            # gate+matrix hybrid — that would contaminate the E7 mechanism split.
            nb.block_gate = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        new_blocks.append(nb)

    model.blocks = new_blocks
    model.config.n_layers = n_target
    model.original_block_indices = original_indices
    print(f"  Total params after growth: {model.count_params():,}")
    return model


def _gate_depth_growth(model: GrowableTransformer, config: GrowthConfig,
                       gate_init_value: float = 0.0) -> GrowableTransformer:
    """
    Gate-based function-preserving depth growth.

    For growth_factor=4 with 12 original layers:
      - Target: 48 layers
      - Insert 36 new blocks
      - Each new block cloned from an existing block + small noise
      - block_gate = 0.0 (or gate_init_value) for exact/near-exact FP

    Two insertion patterns (controlled by config.growth_pattern):

    - "append" (default, CL-safe):
        [old1, old2, ..., oldN, NEW, NEW, ..., NEW]
        All new blocks stacked AFTER old blocks. Old computation path is
        literally unchanged → PPL_A stays preserved during CL with isolation.

    - "interleaved" (pre-training speedup):
        [old1, NEW, NEW, NEW, old2, NEW, NEW, NEW, ...]
        New blocks between old blocks. Good for pre-training acceleration but
        pollutes the residual stream, causing PPL_A drift in CL experiments.
    """
    g = config.growth_factor
    n_orig = len(model.blocks)
    n_target = n_orig * g
    n_new = n_target - n_orig
    pattern = getattr(config, 'growth_pattern', 'append')

    print(f"gate-zero depth growth: {n_orig} -> {n_target} layers "
          f"(inserting {n_new} blocks, gate_init={gate_init_value}, "
          f"pattern={pattern})")

    new_blocks = nn.ModuleList()
    original_indices = []  # Track positions of original blocks

    if pattern == "append":
        # Keep all originals first, then stack new blocks after
        for i in range(n_orig):
            original_indices.append(len(new_blocks))
            new_blocks.append(model.blocks[i])

        # Append new blocks at the end (clone from originals, round-robin)
        for j in range(n_new):
            src_idx = j % n_orig
            new_block = _clone_block(model.blocks[src_idx], config.clone_noise_std)
            new_block.block_gate = nn.Parameter(
                torch.tensor(gate_init_value)
            )
            new_blocks.append(new_block)

    elif pattern == "interleaved":
        for i in range(n_orig):
            # Keep original block
            original_indices.append(len(new_blocks))
            new_blocks.append(model.blocks[i])

            # Insert (g-1) new blocks after each original
            for j in range(g - 1):
                new_block = _clone_block(model.blocks[i], config.clone_noise_std)
                # ONLY block_gate=0 for FP. Internal gates keep cloned values.
                # block_gate=0 makes the entire block identity via residual:
                #   x + 0 * Block(x) = x
                # Internal head_gate/ffn_gate stay at trained values so that
                # Block(x) is non-zero, giving block_gate a non-zero gradient:
                #   dL/d(block_gate) = dL/dx * Block(x) != 0
                # If we also zeroed internal gates, Block(x)=0 and gradient dies.
                new_block.block_gate = nn.Parameter(
                    torch.tensor(gate_init_value)
                )
                new_blocks.append(new_block)

    else:
        raise ValueError(f"Unknown growth_pattern: {pattern}. "
                         f"Must be 'append' or 'interleaved'.")

    model.blocks = new_blocks
    model.config.n_layers = n_target
    # CRITICAL: store which indices are original (for isolation CL setup)
    model.original_block_indices = original_indices

    # Verify function preservation if gate_init=0
    if gate_init_value == 0.0:
        _verify_new_blocks_identity(model, n_orig, g)

    print(f"  Total params after growth: {model.count_params():,}")
    return model


def _clone_block(block: GatedTransformerBlock,
                 noise_std: float = 0.01) -> GatedTransformerBlock:
    """Deep copy a block and add small noise to weights."""
    new_block = copy.deepcopy(block)

    # Add noise to all weight parameters (not gates)
    with torch.no_grad():
        for name, param in new_block.named_parameters():
            if "gate" not in name and "norm" not in name:
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)

    return new_block




def _verify_new_blocks_identity(model, n_orig, g):
    """Verify that new blocks with gate=0 are identity (for logging)."""
    n_identity = 0
    for i, block in enumerate(model.blocks):
        if block.block_gate.item() == 0.0:
            n_identity += 1
    expected = n_orig * (g - 1)
    status = "PASS" if n_identity == expected else "FAIL"
    print(f"  FP verification: {n_identity}/{expected} blocks have gate=0 [{status}]")


# ─────────────────────────────────────────────────────────────
# FP verification utility
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def verify_function_preservation(model_before, model_after, input_ids):
    """
    Check that model_after produces identical logits to model_before.
    Only valid for gate-based FP growth (not G_stack).

    Returns max absolute difference in logits.
    """
    model_before.eval()
    model_after.eval()

    before_device = next(model_before.parameters()).device
    after_device = next(model_after.parameters()).device

    logits_before = model_before(input_ids.to(before_device)).float().cpu()
    logits_after = model_after(input_ids.to(after_device)).float().cpu()

    max_diff = (logits_before - logits_after).abs().max().item()
    mean_diff = (logits_before - logits_after).abs().mean().item()

    # Cross-device comparisons can introduce tiny numerical drift even when the
    # growth construction is function-preserving; use a slightly looser tolerance
    # for cross-device verification.
    threshold = 1e-5 if before_device == after_device else 1e-4

    passed = max_diff < threshold

    print(f"  FP check — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
    if passed:
        if before_device == after_device:
            print(f"  FP check: PASS (exact within float precision)")
        else:
            print(f"  FP check: PASS (near-exact; cross-device numerical drift)")
    else:
        print(f"  FP check: FAIL (significant logit difference)")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "threshold": threshold,
        "passed": passed,
    }

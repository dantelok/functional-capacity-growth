#!/usr/bin/env python3
"""
Direct empirical test of the transversality assumption underlying
Theorem 3.1 (Part 2).

Computes the projected gate-Jacobian rank
    rank((I - P_old) J_alpha)
where:
    J_alpha = K columns, partial f / partial alpha_l for each new gate
    P_old   = orthogonal projection onto im(J_old), approximated by sampling
              M random unit directions in old-parameter space and stacking
              their Jacobian images.

Reports:
    - K (number of new gates)
    - M (sample size)
    - Singular values of (I - P_old) J_alpha
    - Effective rank at threshold 1e-3 * sigma_max
    - Smallest principal angle (degrees) between im(J_alpha) and the
      sampled subspace im(J_old)

Output:
    results/diagnose_transversality/results.json

Usage:
    cd experiments
    python experiments/E1_functional_rank/diagnose_transversality.py \\
        --base-checkpoint checkpoints/exp1_shared_base/base_final.pt \\
        --growth-factor 4 --batch-size 2 --seq-len 16 --n-old-samples 200
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.func as tf


from gatezero.config import (ExperimentConfig, ModelConfig, GrowthConfig,
                    GrowthMethod, TrainingConfig, CLConfig)
from gatezero.data import build_datasets, get_tokenizer
from gatezero.growth import grow_transformer
from gatezero.models.transformer import GrowableTransformer
from gatezero.reproducibility import set_reproducible


def load_grown_model(base_ckpt: str, growth_factor: int, device: str):
    """Build base, load checkpoint, apply gate-zero growth, return model."""
    cfg = ExperimentConfig(
        name="diag_transversality",
        model=ModelConfig(),
        growth=GrowthConfig(method=GrowthMethod.GATE_ZERO,
                            growth_factor=growth_factor,
                            gate_init_value=0.0),
        cl=CLConfig(),
    )
    model = GrowableTransformer(cfg.model)
    print(f"Loading base from {base_ckpt}")
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = grow_transformer(model, cfg.growth, gate_init_value=0.0)
    model = model.to(device).eval()
    return model


def get_new_gate_param_names(model):
    """Names of new-block scalar block_gate parameters in the model state."""
    new_indices = list(range(model.n_original_blocks, len(model.blocks)))
    return [f"blocks.{i}.block_gate" for i in new_indices]


def jvp_column(model, x, params, tangent_dict):
    """Compute one Jacobian-vector product column."""
    def fn(p):
        return tf.functional_call(model, p, (x,))
    _output, jvp_value = tf.jvp(fn, (params,), (tangent_dict,))
    return jvp_value.detach().flatten()


def compute_gate_columns(model, x, params, gate_names):
    """Stack J_alpha columns: one column per scalar gate."""
    cols = []
    for gname in gate_names:
        tangent = {k: torch.zeros_like(v) for k, v in params.items()}
        tangent[gname] = torch.ones_like(params[gname])
        col = jvp_column(model, x, params, tangent)
        cols.append(col)
        del tangent
    return torch.stack(cols, dim=1)  # [D, K]


def compute_old_sample_columns(model, x, params, old_param_names,
                               n_samples: int, generator: torch.Generator):
    """
    Sample n_samples random unit-norm directions in the joint old-parameter
    space and return their Jacobian images.

    Each sample is a unit vector across all old-param coordinates, with
    component magnitudes drawn from N(0, 1) then normalised to unit norm in
    aggregate.
    """
    device = next(iter(params.values())).device
    cols = []
    # Pre-compute total old-parameter dimension for normalisation.
    total_dim = sum(int(params[name].numel()) for name in old_param_names)
    for _ in range(n_samples):
        # Build one random unit-norm tangent supported only on old params.
        tangent = {k: torch.zeros_like(v) for k, v in params.items()}
        sq = 0.0
        for name in old_param_names:
            t = torch.randn(params[name].shape,
                            generator=generator, device=device,
                            dtype=params[name].dtype)
            tangent[name] = t
            sq += float(t.float().pow(2).sum().item())
        # Normalise so the joint tangent has unit Euclidean norm.
        norm = math.sqrt(sq)
        if norm > 0:
            for name in old_param_names:
                tangent[name] = tangent[name] / norm
        col = jvp_column(model, x, params, tangent)
        cols.append(col)
        del tangent
    return torch.stack(cols, dim=1)  # [D, M]


def transversality_diagnostic(model, x, n_old_samples=200, seed=0):
    """Run the transversality probe and return a diagnostic dict."""
    device = next(model.parameters()).device
    params = {k: v.detach().clone() for k, v in model.named_parameters()}

    gate_names = get_new_gate_param_names(model)
    K = len(gate_names)

    old_block_names = [f"blocks.{i}." for i in range(model.n_original_blocks)]
    old_param_names = [
        n for n in params
        if (any(n.startswith(b) for b in old_block_names)
            or n.startswith("token_emb")
            or n.startswith("final_norm")
            or n == "lm_head.weight")
    ]

    print(f"  K = {K} new-block gates")
    print(f"  Old-parameter names: {len(old_param_names)} tensors")
    print(f"  M = {n_old_samples} random old-direction samples")

    print("Computing gate Jacobian columns (J_alpha)...")
    J_alpha = compute_gate_columns(model, x, params, gate_names)
    print(f"  J_alpha shape: {tuple(J_alpha.shape)}")

    gen = torch.Generator(device=device).manual_seed(seed)
    print("Computing sampled old-direction Jacobian columns (J_old)...")
    J_old = compute_old_sample_columns(model, x, params, old_param_names,
                                       n_old_samples, gen)
    print(f"  J_old shape: {tuple(J_old.shape)}")

    # Orthonormal basis for sampled im(J_old)
    print("QR decomposition of J_old...")
    Q, _R = torch.linalg.qr(J_old.float(), mode="reduced")  # [D, M]
    # Project J_alpha onto orthogonal complement of Q
    J_alpha_f = J_alpha.float()
    proj = Q @ (Q.T @ J_alpha_f)
    residual = J_alpha_f - proj
    print(f"  residual shape: {tuple(residual.shape)}")

    # SVD of residual gate columns
    print("SVD of residual gate columns...")
    _U, S, _V = torch.linalg.svd(residual, full_matrices=False)
    sigma_max = float(S[0].item())
    sigma_min = float(S[-1].item())
    rank_thr = 1e-3 * sigma_max
    eff_rank = int((S > rank_thr).sum().item())

    # Principal angles between im(J_alpha) and sampled im(J_old)
    Q_a, _ = torch.linalg.qr(J_alpha_f, mode="reduced")  # [D, K]
    M_inner = Q.T @ Q_a  # [M, K]
    cos_vals = torch.linalg.svdvals(M_inner)
    cos_vals = cos_vals.clamp(0, 1)
    angles_deg = torch.rad2deg(torch.acos(cos_vals)).cpu().tolist()
    angles_deg = sorted(angles_deg)
    smallest_angle_deg = float(angles_deg[0]) if angles_deg else float("nan")
    largest_angle_deg = float(angles_deg[-1]) if angles_deg else float("nan")

    return {
        "K_new_gates": K,
        "M_old_samples": n_old_samples,
        "output_dim": int(J_alpha.shape[0]),
        "J_alpha_norm_per_column": J_alpha.float().norm(dim=0).cpu().tolist(),
        "residual_singular_values": S.cpu().tolist(),
        "residual_sigma_max": sigma_max,
        "residual_sigma_min": sigma_min,
        "residual_condition_number": (sigma_max / sigma_min
                                      if sigma_min > 0 else float("inf")),
        "effective_rank_at_thr_1e_3": eff_rank,
        "transversality_holds_at_thr": (eff_rank == K),
        "principal_angles_deg_ascending": angles_deg,
        "smallest_angle_deg": smallest_angle_deg,
        "largest_angle_deg": largest_angle_deg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--growth-factor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--n-old-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default="results/diagnose_transversality")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    # Seed every RNG surface (CPU + CUDA) before growth; the projected-Jacobian
    # sampling below also draws random old-parameter directions.
    set_reproducible(args.seed)

    model = load_grown_model(args.base_checkpoint, args.growth_factor, device)

    tokenizer = get_tokenizer()
    print("\nLoading WikiText for the input batch...")
    _train_a, val_a = build_datasets("wikitext", tokenizer,
                                     seq_len=args.seq_len)
    sample_ids = []
    for i, item in enumerate(val_a):
        if i >= args.batch_size:
            break
        # Dataset returns (input_ids, targets); we only need input_ids.
        ids = item[0] if isinstance(item, (tuple, list)) else item
        sample_ids.append(ids)
    x = torch.stack(sample_ids, dim=0).to(device)
    print(f"Input batch shape: {tuple(x.shape)}")

    print("\nRunning transversality diagnostic...")
    diag = transversality_diagnostic(model, x, args.n_old_samples, args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / "results.json"
    payload = {
        "config": "diagnose_transversality",
        "args": vars(args),
        "metrics": diag,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"  effective_rank_at_thr_1e_3 = {diag['effective_rank_at_thr_1e_3']}"
          f" / K = {diag['K_new_gates']}")
    print(f"  smallest principal angle  = {diag['smallest_angle_deg']:.2f} deg")
    print(f"  residual sigma_min        = {diag['residual_sigma_min']:.4e}")


if __name__ == "__main__":
    main()

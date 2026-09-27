"""Transversal-residual diagnostic via the (validated) Gram normal-equations
projector. Memory-cheaper than QR (no [D,M] orthonormal-basis duplicate), so
large M fits at seq_len 128. Optionally stores J_old in fp16 so M=1024 fits on
one 46GB GPU. Math validated against QR by experiments/E1_functional_rank/validate_projector.py.

Writes results.json compatible with diagnose_transversality.

  CUDA_VISIBLE_DEVICES=0 python3 -u experiments/E1_functional_rank/transversality_gram.py \
    --base-checkpoint checkpoints/exp1_shared_base/base_final.pt \
    --seq-len 128 --seed 42 --n-old-samples 512 --jold-dtype fp32 \
    --device cuda --output-dir results/transversality_sweep/seq128_m512
"""
import argparse
import json
import math
import os
import torch
import diagnose_transversality as dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-old-samples", type=int, default=512)
    ap.add_argument("--jold-dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="results/transversality_gram")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    M = args.n_old_samples
    jdt = torch.float16 if args.jold_dtype == "fp16" else torch.float32

    dt.set_reproducible(args.seed)
    model = dt.load_grown_model(args.base_checkpoint, args.growth_factor, device)
    tok = dt.get_tokenizer()
    _t, val_a = dt.build_datasets("wikitext", tok, seq_len=args.seq_len)
    ids = []
    for i, item in enumerate(val_a):
        if i >= args.batch_size:
            break
        ids.append(item[0] if isinstance(item, (tuple, list)) else item)
    x = torch.stack(ids, dim=0).to(device)

    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    gate_names = dt.get_new_gate_param_names(model)
    K = len(gate_names)
    old_blocks = [f"blocks.{i}." for i in range(model.n_original_blocks)]
    old_names = [n for n in params
                 if (any(n.startswith(b) for b in old_blocks)
                     or n.startswith("token_emb")
                     or n.startswith("final_norm")
                     or n == "lm_head.weight")]

    J_alpha = dt.compute_gate_columns(model, x, params, gate_names).float()  # [D,K]
    D = J_alpha.shape[0]
    # Build J_old pre-allocated in the target dtype and fill column by column.
    # (compute_old_sample_columns lists all M fp32 columns then stacks, doubling
    # memory and OOMing for large M.) Same seeded unit directions as that helper,
    # so results match the QR sweep.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    J_old = torch.empty(D, M, dtype=jdt, device=device)
    for j in range(M):
        tangent = {k: torch.zeros_like(v) for k, v in params.items()}
        sq = 0.0
        for name in old_names:
            t = torch.randn(params[name].shape, generator=gen,
                            device=device, dtype=params[name].dtype)
            tangent[name] = t
            sq += float(t.float().pow(2).sum().item())
        norm = math.sqrt(sq)
        if norm > 0:
            for name in old_names:
                tangent[name] = tangent[name] / norm
        J_old[:, j] = dt.jvp_column(model, x, params, tangent).to(jdt)
        del tangent
    print(f"D={D} K={K} M={M} jold_dtype={args.jold_dtype} device={device}", flush=True)

    def hard_rank(S):
        return int((S > 1e-3 * S[0]).sum().item())

    # Orthogonal projection onto col(J_old) via normal equations (Gram).
    Jo = J_old
    Ja = J_alpha.to(jdt)
    G = (Jo.transpose(0, 1) @ Jo).float()                      # [M,M]
    G = G + (1e-10 * G.diagonal().mean()) * torch.eye(M, device=device)
    B = (Jo.transpose(0, 1) @ Ja).float()                      # [M,K]
    xB = torch.linalg.solve(G, B)                              # [M,K]
    proj = (Jo @ xB.to(jdt)).float()                           # [D,K]
    residual = J_alpha - proj
    S = torch.linalg.svdvals(residual)

    Qa, _ = torch.linalg.qr(J_alpha, mode="reduced")           # [D,K]
    A = (Jo.transpose(0, 1) @ Qa.to(jdt)).float()              # [M,K]
    xa = torch.linalg.solve(G, A)
    cos2 = torch.linalg.eigvalsh(A.transpose(0, 1) @ xa).clamp(0, 1)
    angles = torch.rad2deg(torch.arccos(cos2.sqrt())).sort().values

    rank = hard_rank(S)
    smallest, largest = float(angles[0]), float(angles[-1])
    print(f"residual hard rank = {rank}/{K}  "
          f"min_angle={smallest:.2f}  max_angle={largest:.2f}", flush=True)

    out = {
        "config": "transversality_gram",
        "args": vars(args),
        "metrics": {
            "K_new_gates": K, "M_old_samples": M, "output_dim": int(D),
            "effective_rank_at_thr_1e_3": rank,
            "transversality_holds_at_thr": rank == K,
            "residual_singular_values": S.cpu().tolist(),
            "principal_angles_deg_ascending": angles.cpu().tolist(),
            "smallest_angle_deg": smallest, "largest_angle_deg": largest,
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", os.path.join(args.output_dir, "results.json"), flush=True)


if __name__ == "__main__":
    main()

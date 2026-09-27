"""N-GPU transversal-residual diagnostic, QR projector, D-sharded (TSQR).

Same math as experiments/E1_functional_rank/diagnose_transversality.py (reference QR projector), but the
output dimension D is split across ALL visible GPUs so seq_len=128 with large m
fits. With N shards J_old = [J_0; ...; J_{N-1}], the orthonormal basis of
col(J_old) is built with tall-skinny QR (TSQR):
    J_k = Q_k R_k,  [R_0; ...; R_{N-1}] = Qr Rf,
    Qg_k = Q_k @ Qr[k*M:(k+1)*M]        (orthonormal, sharded).
Residual rank and principal angles are then formed from small K x K / M x K
Gram matrices summed across the shards. N = number of visible GPUs; run with the
GPUs made visible, e.g. CUDA_VISIBLE_DEVICES=0,1,2 for the 3-GPU points.

The N=2 sharding (bounds = D//2) reproduces the earlier two-GPU results exactly.
Verify against a trusted point (e.g. seq128 m=256) before trusting a new N.

  CUDA_VISIBLE_DEVICES=0,1,2 python3 -u experiments/E1_functional_rank/multigpu_transversality.py \
    --base-checkpoint checkpoints/exp1_shared_base/base_final.pt \
    --seq-len 128 --batch-size 2 --seed 42 --n-old-samples 1024 \
    --output-dir results/transversality_grid/seq128_m1024
"""
import argparse
import json
import math
import os
import torch
import diagnose_transversality as dt


def old_unit_tangent(params, old_names, gen):
    tangent = {k: torch.zeros_like(v) for k, v in params.items()}
    sq = 0.0
    for name in old_names:
        t = torch.randn(params[name].shape, generator=gen,
                        device=params[name].device, dtype=params[name].dtype)
        tangent[name] = t
        sq += float(t.float().pow(2).sum().item())
    norm = math.sqrt(sq)
    if norm > 0:
        for name in old_names:
            tangent[name] = tangent[name] / norm
    return tangent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-old-samples", type=int, default=512)
    ap.add_argument("--output-dir", default="results/multigpu_transversality")
    args = ap.parse_args()
    N = torch.cuda.device_count()
    assert N >= 2, "need >= 2 visible GPUs"
    DEVS = [f"cuda:{i}" for i in range(N)]
    DEV0 = DEVS[0]
    M = args.n_old_samples

    dt.set_reproducible(args.seed)
    model = dt.load_grown_model(args.base_checkpoint, args.growth_factor, DEV0)
    tok = dt.get_tokenizer()
    _t, val_a = dt.build_datasets("wikitext", tok, seq_len=args.seq_len)
    ids = []
    for i, item in enumerate(val_a):
        if i >= args.batch_size:
            break
        ids.append(item[0] if isinstance(item, (tuple, list)) else item)
    x = torch.stack(ids, dim=0).to(DEV0)

    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    gate_names = dt.get_new_gate_param_names(model)
    K = len(gate_names)
    old_blocks = [f"blocks.{i}." for i in range(model.n_original_blocks)]
    old_names = [n for n in params
                 if (any(n.startswith(b) for b in old_blocks)
                     or n.startswith("token_emb")
                     or n.startswith("final_norm")
                     or n == "lm_head.weight")]

    # J_alpha: kept whole on GPU0 (K=36 columns, small), used sharded below.
    J_alpha = dt.compute_gate_columns(model, x, params, gate_names).float()  # [D,K] on dev0
    D = J_alpha.shape[0]
    # Contiguous D-shards, one per GPU. bounds[k]..bounds[k+1] is shard k.
    # For N=2 this gives D//2, matching the original two-GPU split exactly.
    bounds = [(D * k) // N for k in range(N + 1)]
    sizes = [bounds[k + 1] - bounds[k] for k in range(N)]
    print(f"D={D} N={N} shard_sizes={sizes} K={K} M={M}", flush=True)

    # J_old sharded across the GPUs, filled column by column.
    Jo = [torch.empty(sizes[k], M, device=DEVS[k]) for k in range(N)]
    gen = torch.Generator(device=DEV0).manual_seed(args.seed)
    for j in range(M):
        col = dt.jvp_column(model, x, params, old_unit_tangent(params, old_names, gen)).float()
        for k in range(N):
            Jo[k][:, j] = col[bounds[k]:bounds[k + 1]].to(DEVS[k])
        del col

    # JVPs are done; free the model, its param clone, and the input batch so the
    # per-GPU shard has room for the TSQR Q allocation (J_alpha is kept).
    del model, params, x
    torch.cuda.empty_cache()

    # TSQR: local QR on each shard, then QR of the stacked R's, then
    # Qg_k = Q_k @ Qr_k. Free Jo_k right after its QR and Q_k right after building
    # Qg_k, so the per-GPU peak is ~2 shard-sized tensors instead of 3; this is
    # what lets seq128 m1024 fit across 3 GPUs (naive freeing OOMs at Qg).
    Qs, Rs = [], []
    for k in range(N):
        q, r = torch.linalg.qr(Jo[k], mode="reduced")    # Q_k [D_k,M], R_k [M,M]
        Qs.append(q); Rs.append(r)
        Jo[k] = None                                     # free the shard now
    del Jo
    torch.cuda.empty_cache()
    Rstack = torch.cat([Rs[k].to(DEV0) for k in range(N)], dim=0)  # [N*M,M] dev0
    Qr, _Rf = torch.linalg.qr(Rstack, mode="reduced")             # [N*M,M] dev0
    del Rstack, Rs
    Qg = []
    for k in range(N):
        Qg.append(Qs[k] @ Qr[k * M:(k + 1) * M].to(DEVS[k]))      # Qg_k [D_k,M]
        Qs[k] = None                                              # free Q_k now
    del Qs, Qr
    torch.cuda.empty_cache()

    # Residual of J_alpha off col(J_old): resid = J_alpha - Qg (Qg^T J_alpha).
    Ja = [J_alpha[bounds[k]:bounds[k + 1]].to(DEVS[k]) for k in range(N)]
    QtJa = sum((Qg[k].transpose(0, 1) @ Ja[k]).to(DEV0) for k in range(N))     # [M,K] dev0
    RtR = None
    for k in range(N):
        resid_k = Ja[k] - Qg[k] @ QtJa.to(DEVS[k])          # [D_k,K]
        term = (resid_k.transpose(0, 1) @ resid_k).to(DEV0)  # [K,K]
        RtR = term if RtR is None else RtR + term
        del resid_k
    s2 = torch.linalg.eigvalsh(RtR).clamp(min=0)
    S = s2.sqrt().flip(0)                                    # descending singular values
    rank = int((S > 1e-3 * S[0]).sum().item())

    # Principal angles between col(J_alpha) and col(J_old): cos = svd(Qg^T Qa).
    Qa = torch.linalg.qr(J_alpha, mode="reduced")[0]        # [D,K] dev0
    Minner = sum((Qg[k].transpose(0, 1) @ Qa[bounds[k]:bounds[k + 1]].to(DEVS[k])).to(DEV0)
                 for k in range(N))                          # [M,K] dev0
    cos = torch.linalg.svdvals(Minner).clamp(0, 1)
    angles = torch.rad2deg(torch.arccos(cos)).sort().values

    smallest, largest = float(angles[0]), float(angles[-1])
    print(f"residual hard rank = {rank}/{K}  min_angle={smallest:.2f}  max_angle={largest:.2f}",
          flush=True)
    out = {
        "config": "multigpu_transversality",
        "args": vars(args),
        "metrics": {
            "K_new_gates": K, "M_old_samples": M, "output_dim": int(D),
            "n_devices": N, "shard_sizes": sizes,
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

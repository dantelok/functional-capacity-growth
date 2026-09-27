"""Validate the memory-efficient Gram (normal-equations) transversal-residual
projector against the reference QR projector, on IDENTICAL J_alpha / J_old.

Both compute the same orthogonal projection onto col(J_old):
    Q Q^T  ==  J_old (J_old^T J_old)^{-1} J_old^T.
QR is numerically stable; Gram squares the condition number. This script checks
that, at m=64/seq=128, the two agree on the hard rank and match the singular
values / principal angles to ~4 digits -- i.e. that Gram (which enables the
chunked, memory-cheap sweep to m=1024) is safe to use.

Run on L40 (a free GPU):
  CUDA_VISIBLE_DEVICES=0 python3 -u experiments/E1_functional_rank/validate_projector.py \
    --base-checkpoint checkpoints/exp1_shared_base/base_final.pt \
    --seq-len 128 --seed 42 --n-old-samples 64 --device cuda
"""
import argparse
import torch
import diagnose_transversality as dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-old-samples", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    dt.set_reproducible(args.seed)
    model = dt.load_grown_model(args.base_checkpoint, args.growth_factor, device)
    tok = dt.get_tokenizer()
    _train, val_a = dt.build_datasets("wikitext", tok, seq_len=args.seq_len)
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

    J_alpha = dt.compute_gate_columns(model, x, params, gate_names).float()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    J_old = dt.compute_old_sample_columns(model, x, params, old_names,
                                          args.n_old_samples, gen).float()
    D, M = J_old.shape
    print(f"D={D} K={K} M={M} device={device}")

    def hard_rank(S):
        return int((S > 1e-3 * S[0]).sum().item())

    # ---------- QR reference ----------
    Q, _ = torch.linalg.qr(J_old, mode="reduced")
    resid_qr = J_alpha - Q @ (Q.T @ J_alpha)
    S_qr = torch.linalg.svdvals(resid_qr)
    Qa, _ = torch.linalg.qr(J_alpha, mode="reduced")
    cos_qr = torch.linalg.svdvals(Q.T @ Qa).clamp(0, 1)
    ang_qr = torch.rad2deg(torch.arccos(cos_qr)).sort().values

    # ---------- Gram (normal equations) candidate ----------
    G = J_old.T @ J_old
    ridge = 1e-10 * G.diagonal().mean()
    G = G + ridge * torch.eye(M, device=G.device, dtype=G.dtype)
    xB = torch.linalg.solve(G, J_old.T @ J_alpha)
    resid_gram = J_alpha - J_old @ xB
    S_gram = torch.linalg.svdvals(resid_gram)
    JtQa = J_old.T @ Qa
    xa = torch.linalg.solve(G, JtQa)
    cos2 = torch.linalg.eigvalsh(JtQa.T @ xa).clamp(0, 1)
    ang_gram = torch.rad2deg(torch.arccos(cos2.sqrt())).sort().values

    dS = (S_qr - S_gram).abs().max().item()
    dA = (ang_qr - ang_gram).abs().max().item()
    print("----- projector comparison (identical J_alpha, J_old) -----")
    print(f"hard rank:         QR={hard_rank(S_qr)}   Gram={hard_rank(S_gram)}")
    print(f"residual sigma_max: QR={S_qr[0].item():.6f}   Gram={S_gram[0].item():.6f}")
    print(f"residual sigma_min: QR={S_qr[-1].item():.6f}   Gram={S_gram[-1].item():.6f}")
    print(f"min angle (deg):   QR={ang_qr[0].item():.4f}   Gram={ang_gram[0].item():.4f}")
    print(f"med angle (deg):   QR={ang_qr[len(ang_qr)//2].item():.4f}   Gram={ang_gram[len(ang_gram)//2].item():.4f}")
    print(f"max angle (deg):   QR={ang_qr[-1].item():.4f}   Gram={ang_gram[-1].item():.4f}")
    print(f"max |d sigma|:     {dS:.3e}   (rank threshold = {1e-3 * S_qr[0].item():.3e})")
    print(f"max |d angle| deg: {dA:.3e}")
    ok = (hard_rank(S_qr) == hard_rank(S_gram)
          and dS < 1e-3 * S_qr[0].item()
          and dA < 1e-2)
    print("VERDICT:", "MATCH (Gram is safe)" if ok else "DIFFER (use streaming QR)")


if __name__ == "__main__":
    main()

"""
Evaluation: perplexity metrics and manifold dimension measurement.
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# Perplexity evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, dataset, batch_size=16, device="cuda",
                       num_workers=4) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def evaluate_cl(model, val_a, val_b, batch_size=16, device="cuda",
                num_workers=4) -> Dict[str, float]:
    """
    Full CL evaluation: perplexity on both datasets.

    Returns:
        ppl_a: perplexity on Dataset A (forgetting metric)
        ppl_b: perplexity on Dataset B (plasticity metric)
    """
    ppl_a = compute_perplexity(model, val_a, batch_size, device, num_workers)
    ppl_b = compute_perplexity(model, val_b, batch_size, device, num_workers)

    return {"ppl_a": ppl_a, "ppl_b": ppl_b}


# ─────────────────────────────────────────────────────────────
# Preservation diagnostics (E3) and departure geometry (E4)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def preservation_diagnostics(model_ref, model_test, dataset, batch_size=8,
                             device="cuda", num_workers=4,
                             max_batches=None) -> Dict[str, float]:
    """Empirical function-preservation between two models on identical inputs.

    Compares ``model_test`` against a reference ``model_ref`` (e.g. pre-growth
    base vs post-growth, or pre-CL vs post-CL) over ``dataset``, measuring how
    far the output distribution drifts. This complements the *exact* FP check in
    growth.py (which only certifies the block_gate=0 identity at growth time) by
    quantifying residual drift on real data after CL training — the number that
    actually maps to old-domain forgetting.

    Returns (all averaged per token):
        kl:             mean KL(ref || test) in nats
        logit_l2:       mean L2 distance between logit vectors
        top1_agreement: fraction of tokens where argmax(ref) == argmax(test)
        n_tokens:       tokens compared
    """
    model_ref.eval()
    model_test.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    total_kl = 0.0
    total_l2 = 0.0
    total_agree = 0
    total_tokens = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        logits_ref = model_ref(x).float().view(-1, model_ref.config.vocab_size)
        logits_test = model_test(x).float().view(-1, model_test.config.vocab_size)

        logp_ref = F.log_softmax(logits_ref, dim=-1)
        logp_test = F.log_softmax(logits_test, dim=-1)
        # KL(ref || test) = sum_x p_ref (log p_ref - log p_test); summed over
        # vocab and tokens, then averaged per token below.
        total_kl += F.kl_div(
            logp_test, logp_ref, log_target=True, reduction="sum"
        ).item()
        total_l2 += (logits_test - logits_ref).norm(dim=-1).sum().item()
        total_agree += (logits_ref.argmax(-1) == logits_test.argmax(-1)).sum().item()
        total_tokens += logits_ref.size(0)

    return {
        "kl": total_kl / total_tokens,
        "logit_l2": total_l2 / total_tokens,
        "top1_agreement": total_agree / total_tokens,
        "n_tokens": total_tokens,
    }


@torch.no_grad()
def gate_departure_stats(model) -> Dict:
    """Departure geometry from the block gates of the newly grown blocks.

    ``alpha`` is the vector of block_gate magnitudes over blocks NOT in
    ``model.original_block_indices`` (i.e. the inserted blocks). The theory ties
    old-domain forgetting to its norms: drift is O(||alpha||_2^2) and leakage is
    O(||alpha||_inf). Reported so E4 can correlate departure size with the
    measured Delta_A (old-domain PPL increase).

    Returns:
        n_new_blocks, alpha_l2, alpha_linf, alpha_mean, gates (per-block list)
    """
    orig = set(getattr(model, "original_block_indices", []))
    gates = []
    for i, block in enumerate(model.blocks):
        if i in orig:
            continue
        if hasattr(block, "block_gate"):
            gates.append(abs(float(block.block_gate.detach().item())))

    if not gates:
        return {"n_new_blocks": 0, "alpha_l2": 0.0, "alpha_linf": 0.0,
                "alpha_mean": 0.0, "gates": []}

    return {
        "n_new_blocks": len(gates),
        "alpha_l2": math.sqrt(sum(g * g for g in gates)),
        "alpha_linf": max(gates),
        "alpha_mean": sum(gates) / len(gates),
        "gates": gates,
    }


# ─────────────────────────────────────────────────────────────
# Manifold dimension measurement (Experiment 3)
# ─────────────────────────────────────────────────────────────

class HessianAnalyzer:
    """
    Estimate effective dimensionality of the loss landscape
    via top-k eigenvalues of the Hessian using Lanczos iteration.

    The effective dimension = number of eigenvalues above a threshold,
    which approximates the manifold dimension in the parameter space.
    """

    def __init__(self, model, dataset, device="cuda", batch_size=8):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

    def _hvp(self, loss_fn, params, v):
        """Hessian-vector product via double backprop."""
        grads = torch.autograd.grad(loss_fn, params, create_graph=True)
        flat_grads = torch.cat([g.reshape(-1) for g in grads])
        hvp = torch.autograd.grad(flat_grads, params, grad_outputs=v,
                                  retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hvp])

    def _sample_loss(self, n_batches=4):
        """Compute loss on a few batches for Hessian estimation."""
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
        )
        total_loss = 0.0
        count = 0

        self.model.train()
        for x, y in loader:
            if count >= n_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            total_loss = total_loss + loss
            count += 1

        return total_loss / count

    def estimate_gradient_covariance_rank(self, n_batches=50,
                                          n_components=100) -> Dict:
        """
        Estimate the rank of the gradient covariance matrix.

        Collect per-batch gradients and compute their covariance.
        The effective rank approximates the number of active dimensions.
        """
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
        )

        # Collect gradient snapshots
        grad_snapshots = []
        self.model.train()

        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(self.device), y.to(self.device)

            # This routine explicitly measures training gradients, so autograd
            # must remain enabled here.
            self.model.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            loss.backward()

            # Collect gradients from a subset of parameters (for memory)
            grads = []
            for p in self.model.parameters():
                if p.grad is not None and p.requires_grad:
                    grads.append(p.grad.detach().cpu().reshape(-1))
            grad_vec = torch.cat(grads)

            # Random projection to manageable size
            if len(grad_snapshots) == 0:
                n_params = grad_vec.numel()
                proj_dim = min(n_components, n_params)
                self.proj_matrix = torch.randn(n_params, proj_dim) / math.sqrt(proj_dim)

            projected = grad_vec @ self.proj_matrix
            grad_snapshots.append(projected)

        # Stack and compute SVD
        G = torch.stack(grad_snapshots)  # [n_batches, proj_dim]
        G = G - G.mean(dim=0, keepdim=True)  # center

        _, S, _ = torch.linalg.svd(G, full_matrices=False)

        # Effective rank = exp(entropy of normalized singular values)
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]
        entropy = -(S_norm * torch.log(S_norm)).sum().item()
        effective_rank = math.exp(entropy)

        # Count significant singular values (> 1% of max)
        threshold = S[0] * 0.01
        n_significant = (S > threshold).sum().item()

        return {
            "effective_rank": effective_rank,
            "n_significant_svs": n_significant,
            "top_10_svs": S[:10].tolist(),
            "sv_ratio_1_10": (S[0] / S[min(9, len(S)-1)]).item(),
        }


def lanczos_top_eigenvalues(model, dataset, device="cuda",
                            n_eigenvalues=20, n_batches=4,
                            batch_size=8) -> Dict:
    """
    Compute top-k eigenvalues of the Hessian using Lanczos iteration.

    This is expensive — use sparingly (Experiment 3 only).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    def compute_loss():
        total = 0.0
        count = 0
        for x, y in loader:
            if count >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total = total + F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            count += 1
        return total / count

    def hvp(v_list):
        """Hessian-vector product."""
        model.zero_grad(set_to_none=True)
        loss = compute_loss()
        grads = torch.autograd.grad(loss, params, create_graph=True)

        # Compute g^T v
        gv = sum((g * v).sum() for g, v in zip(grads, v_list))

        # Compute H @ v = grad(g^T v)
        hvp_result = torch.autograd.grad(gv, params, retain_graph=False)
        return [h.detach() for h in hvp_result]

    # Lanczos iteration
    n_iter = min(n_eigenvalues * 2, 50)

    # Random initial vector (normalized)
    q = [torch.randn_like(p) for p in params]
    norm = math.sqrt(sum((qi ** 2).sum().item() for qi in q))
    q = [qi / norm for qi in q]

    alphas = []
    betas = []
    q_prev = None
    q_curr = q

    model.train()

    for j in range(n_iter):
        # w = H @ q_j
        w = hvp(q_curr)

        # alpha_j = q_j^T H q_j
        alpha = sum((wi * qi).sum().item() for wi, qi in zip(w, q_curr))
        alphas.append(alpha)

        # w = w - alpha_j * q_j
        w = [wi - alpha * qi for wi, qi in zip(w, q_curr)]

        # w = w - beta_{j-1} * q_{j-1}
        if q_prev is not None:
            w = [wi - betas[-1] * qi for wi, qi in zip(w, q_prev)]

        # beta_j = ||w||
        beta = math.sqrt(sum((wi ** 2).sum().item() for wi in w))
        if beta < 1e-10:
            break
        betas.append(beta)

        # q_{j+1} = w / beta_j
        q_prev = q_curr
        q_curr = [wi / beta for wi in w]

    # Build tridiagonal matrix and compute eigenvalues
    n = len(alphas)
    T = torch.zeros(n, n)
    for i in range(n):
        T[i, i] = alphas[i]
        if i < n - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    eigenvalues = torch.linalg.eigvalsh(T)
    eigenvalues = eigenvalues.sort(descending=True).values

    top_k = eigenvalues[:n_eigenvalues]

    return {
        "top_eigenvalues": top_k.tolist(),
        "max_eigenvalue": top_k[0].item(),
        "trace_estimate": sum(alphas),
        "n_positive": (eigenvalues > 0).sum().item(),
        "n_negative": (eigenvalues < 0).sum().item(),
    }

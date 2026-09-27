#!/usr/bin/env python3
"""
Toy verification of the rank-separation theorem + validation of the matrix-free
rank estimator — in one self-contained script (no gatezero imports).

It checks, on a TINY model where the exact functional Jacobian is computable:

  Part 1 (unconditional):  d f / d W_new  = 0  at gate alpha = 0.
  Part 2 (conditional):    rank(J_grown) = rank(J_old) + K   UNDER transversality;
                           and it FAILS when transversality is broken.
  Estimator:               a matrix-free JVP randomized-rank estimate recovers the
                           exact rank (this is the estimator you'll scale to 857M).

Run:  python experiments/E1_functional_rank/verify_rank_theorem.py
Read it end-to-end — it's the theorem, in code.
"""
import torch

torch.manual_seed(0)
DT = torch.float64  # double precision so rank thresholds are unambiguous


# ---------------------------------------------------------------- model
def init_params(d_in, d_h, d_out, n_old, n_new, break_transversality=False):
    """Params as a flat dict. New residual blocks are gated by scalar alpha."""
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator=g, dtype=DT) * 0.5
    p = {"W_in": r(d_in, d_h)}
    for i in range(n_old):
        p[f"Wold{i}"] = r(d_h, d_h)
    for j in range(n_new):
        if break_transversality:
            # Make the new block a near-duplicate of old block 0's map: its
            # functional direction then lies (nearly) in im(J_old) -> not transverse.
            p[f"Wnew{j}"] = p["Wold0"].clone() + 1e-6 * r(d_h, d_h)
        else:
            p[f"Wnew{j}"] = r(d_h, d_h)
        p[f"alpha{j}"] = torch.zeros((), dtype=DT)  # gate-zero init
    p["W_out"] = r(d_h, d_out)
    return {k: v.clone().requires_grad_(False) for k, v in p.items()}


def forward(p, x, n_old, n_new):
    h = x @ p["W_in"]
    for i in range(n_old):
        h = h + torch.tanh(h @ p[f"Wold{i}"])
    for j in range(n_new):
        h = h + p[f"alpha{j}"] * torch.tanh(h @ p[f"Wnew{j}"])  # gated new block
    return h @ p["W_out"]


# ------------------------------------------------- exact Jacobian machinery
def flat_jacobian(p, x, n_old, n_new, param_keys):
    """Exact d(vec f)/d(concat of param_keys) via autograd, at the current p."""
    leaves = {k: p[k].clone().requires_grad_(True) for k in param_keys}
    full = {**p, **leaves}
    y = forward(full, x, n_old, n_new).reshape(-1)  # [batch*d_out]
    cols = []
    for k in param_keys:
        # jacobian of y wrt this leaf, flattened to [out_dim, numel(k)]
        J = torch.autograd.functional.jacobian(
            lambda t, k=k: forward({**full, k: t}, x, n_old, n_new).reshape(-1),
            leaves[k], vectorize=True)
        cols.append(J.reshape(y.numel(), -1))
    return torch.cat(cols, dim=1)  # [out_dim, sum numel]


def rank(M, tol=1e-6):
    if M.numel() == 0:
        return 0
    s = torch.linalg.svdvals(M)
    return int((s > tol * s[0]).sum()) if s[0] > 0 else 0


# ---------------------------------------------- matrix-free JVP rank estimator
def jvp_rank_estimate(p, x, n_old, n_new, param_keys, n_probes=128, tol=1e-6):
    """Estimate rank(J) WITHOUT materializing J: apply J to random tangents via
    torch.func.jvp, stack the images, and take the rank of the sketch.
    This is the 857M-scalable estimator, validated here against the exact rank."""
    import torch.func as tf
    keys = list(param_keys)
    sketch = []
    gen = torch.Generator().manual_seed(7)
    for _ in range(n_probes):
        tangent = {k: torch.zeros_like(v) for k, v in p.items()}
        for k in keys:  # random tangent supported on the measured params only
            tangent[k] = torch.randn(p[k].shape, generator=gen, dtype=DT)
        _, jv = tf.jvp(lambda pp: forward(pp, x, n_old, n_new).reshape(-1),
                       (p,), (tangent,))
        sketch.append(jv.detach())
    S = torch.stack(sketch, dim=1)  # [out_dim, n_probes]
    return rank(S, tol)


# --------------------------------------------------------------------- run
def report(break_tv):
    # out_dim = batch*d_out must exceed r_old + K, else the output space
    # saturates and the gates have no room to add rank (a real cap, not a bug).
    d_in, d_h, d_out, n_old, n_new, batch = 6, 4, 4, 1, 3, 40
    p = init_params(d_in, d_h, d_out, n_old, n_new, break_transversality=break_tv)
    x = torch.randn(batch, d_in, dtype=DT, generator=torch.Generator().manual_seed(2))
    K = n_new

    old_keys = ["W_in"] + [f"Wold{i}" for i in range(n_old)] + ["W_out"]
    new_w_keys = [f"Wnew{j}" for j in range(n_new)]
    gate_keys = [f"alpha{j}" for j in range(n_new)]

    # Part 1: new-weight Jacobian at alpha=0
    Jnew = flat_jacobian(p, x, n_old, n_new, new_w_keys)
    part1 = Jnew.abs().max().item()

    # Part 2: rank additivity
    r_old = rank(flat_jacobian(p, x, n_old, n_new, old_keys))
    r_grown = rank(flat_jacobian(p, x, n_old, n_new, old_keys + gate_keys))
    # estimator (matrix-free) on the same [old + gates] block
    r_est = jvp_rank_estimate(p, x, n_old, n_new, old_keys + gate_keys)

    tag = "BROKEN transversality" if break_tv else "transversality holds"
    print(f"\n=== {tag} ===")
    print(f"  Part 1:  max|df/dW_new| at alpha=0 = {part1:.2e}   "
          f"({'PASS ~0' if part1 < 1e-9 else 'NONZERO'})")
    print(f"  Part 2:  rank(J_old) = {r_old},  rank(J_old|J_alpha) = {r_grown},  K = {K}")
    print(f"           additivity r_old + K = {r_old + K}  ->  "
          f"{'PASS' if r_grown == r_old + K else f'FAIL (added {r_grown - r_old}, not {K})'}")
    print(f"  Estimator (matrix-free JVP): rank = {r_est}  ->  "
          f"{'MATCHES exact' if r_est == r_grown else f'MISMATCH vs exact {r_grown}'}")


if __name__ == "__main__":
    print("Toy verification of rank-separation theorem + estimator")
    print("(double precision; tiny model with exact Jacobian)")
    report(break_tv=False)   # theorem should hold: additivity = +K
    report(break_tv=True)    # transversality broken: additivity should FAIL (< +K)
    print("\nExpected: (1) Part-1 ~0 in both; (2) additivity PASS when transverse, "
          "FAIL when broken; (3) estimator matches exact rank in both.")

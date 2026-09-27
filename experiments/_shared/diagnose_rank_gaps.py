#!/usr/bin/env python3
"""
Rank and transversality of the added gate directions on after-CL checkpoints
(E3, E4, E5, E7). Complements experiments/_shared/diagnose_rank_collapse.py.

  E4 (transversality after CL): is the added gate subspace still independent
    of the OLD-parameter directions after CL, not just self-full-rank? Measure
    residual rank of (I - P_old) J_alpha at grown AND after-CL.
  E3 (stability): report every quantity as mean +/- std over >=N probe
    batches; run across seeds 42/43/44.
  E6 (subspace rotation): principal angles between the grown gate subspace
    col(J_alpha^grown) and the after-CL subspace col(J_alpha^afterCL). Large
    angles + preserved rank => capacity preserved but REPURPOSED (supports "forgetting
    = overwriting within a preserved subspace"). This is grown->after-CL over
    the whole CL run; for B-vs-C, pass --compare-cl-checkpoint.

All matrix-free via forward-mode JVP (torch.func). --self-test validates the
transversality-residual and principal-angle math on a tiny CPU model first.
"""
import sys, os, json, copy, math, argparse
import torch
import torch.func as tf


# ── JVP + rank helpers (shared with diagnose_rank_collapse) ─────────────────
def effective_rank(S):
    s = S.float(); s = s[s > 0]
    if s.numel() == 0: return 0.0
    p = s / s.sum()
    return math.exp(-(p * torch.log(p)).sum().item())

def hard_rank(S, rel=1e-3):
    s = S.float()
    return int((s > rel * s[0]).sum().item()) if s.numel() else 0

def jvp_column(model, x, params, tangent):
    def fn(p): return tf.functional_call(model, p, (x,))
    _o, j = tf.jvp(fn, (params,), (tangent,))
    return j.detach().flatten()

def new_gate_names(model):
    return [f"blocks.{i}.block_gate"
            for i in range(model.n_original_blocks, len(model.blocks))]

def old_param_names(model, params):
    old_blocks = [f"blocks.{i}." for i in range(model.n_original_blocks)]
    return [n for n in params
            if (any(n.startswith(b) for b in old_blocks)
                or n.startswith("token_emb") or n.startswith("final_norm")
                or n == "lm_head.weight")]

def gate_columns(model, x):
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    cols = []
    for g in new_gate_names(model):
        t = {k: torch.zeros_like(v) for k, v in params.items()}
        t[g] = torch.ones_like(params[g])
        cols.append(jvp_column(model, x, params, t))
    return torch.stack(cols, dim=1) if cols else torch.empty(0, 0)  # [D,K]

def old_columns(model, x, M, gen):
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    names = old_param_names(model, params)
    dev = next(iter(params.values())).device
    cols = []
    for _ in range(M):
        t = {k: torch.zeros_like(v) for k, v in params.items()}
        sq = 0.0
        for n in names:
            r = torch.randn(params[n].shape, generator=gen, device=dev,
                            dtype=params[n].dtype)
            t[n] = r; sq += float(r.float().pow(2).sum())
        nrm = math.sqrt(sq)
        if nrm > 0:
            for n in names: t[n] = t[n] / nrm
        cols.append(jvp_column(model, x, params, t))
    return torch.stack(cols, dim=1)  # [D,M]

def residual_rank(J_alpha, J_old):
    """rank of (I - P_old) J_alpha : gate directions independent of old subspace.
    Hard rank is thresholded against J_alpha's OWN sigma_max (not the residual's),
    so a residual that is ~0 (gate dirs lie in the old span) correctly reads rank 0."""
    Ja = J_alpha.float()
    ref = float(torch.linalg.svdvals(Ja)[0])
    Q, _ = torch.linalg.qr(J_old.float(), mode="reduced")  # [D,M]
    resid = Ja - Q @ (Q.T @ Ja)
    S = torch.linalg.svdvals(resid)
    hard = int((S > 1e-3 * ref).sum().item()) if ref > 0 else 0
    return effective_rank(S), hard, float(S[-1] / ref if ref > 0 else 0.0)

def principal_angles(J1, J2):
    """Angles (deg) between col(J1) and col(J2); both [D,K]."""
    Q1, _ = torch.linalg.qr(J1.float(), mode="reduced")
    Q2, _ = torch.linalg.qr(J2.float(), mode="reduced")
    c = torch.linalg.svdvals(Q1.T @ Q2).clamp(0, 1)
    return torch.rad2deg(torch.acos(c)).cpu().tolist()


def summarize(vals):
    import statistics as st
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": st.mean(vals),
            "std": (st.pstdev(vals) if len(vals) > 1 else 0.0)}


# ── self-test ───────────────────────────────────────────────────────────────
def self_test():
    torch.manual_seed(0)
    D, K = 400, 6
    # independent gate cols orthogonal to old cols -> residual hard rank = K
    U = torch.linalg.qr(torch.randn(D, K + K), mode="reduced")[0]
    J_old, J_alpha = U[:, :K], U[:, K:]
    er, hr, _ = residual_rank(J_alpha, J_old)
    print(f"[self-test] transversal residual: hard {hr} (expect {K}), eff {er:.2f}")
    # gate cols lying inside old span -> residual ~ 0
    J_in = J_old @ torch.randn(K, K)
    er0, hr0, _ = residual_rank(J_in, J_old)
    print(f"[self-test] non-transversal residual: hard {hr0} (expect ~0), eff {er0:.2f}")
    # angles: subspace vs itself = 0; vs orthogonal = 90
    a_same = principal_angles(J_alpha, J_alpha.clone())
    a_orth = principal_angles(J_alpha, J_old)
    print(f"[self-test] angles self max={max(a_same):.2f} (expect ~0), "
          f"orth min={min(a_orth):.2f} (expect ~90)")
    ok = (hr == K and hr0 <= 1 and max(a_same) < 1.0 and min(a_orth) > 89.0)
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return ok


# ── driver ──────────────────────────────────────────────────────────────────
def build_grown(base_ckpt, gf, method, device):
    from gatezero.config import (ExperimentConfig, ModelConfig, GrowthConfig,
                                 GrowthMethod, CLConfig)
    from gatezero.models.transformer import GrowableTransformer
    from gatezero.growth import grow_transformer
    cfg = ExperimentConfig(name="gaps", model=ModelConfig(),
        growth=GrowthConfig(method=GrowthMethod(method), growth_factor=gf,
                            gate_init_value=0.0), cl=CLConfig())
    m = GrowableTransformer(cfg.model)
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    m = grow_transformer(m, cfg.growth, gate_init_value=0.0).to(device).eval()
    return m, cfg

def load_after_cl(base_ckpt, cl_ckpt, gf, method, device):
    m, cfg = build_grown(base_ckpt, gf, method, device)
    ck = torch.load(cl_ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    return m.to(device).eval(), cfg

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base-checkpoint"); ap.add_argument("--cl-checkpoint")
    ap.add_argument("--compare-cl-checkpoint", default=None,
                    help="A SECOND after-CL checkpoint. If set, also report the "
                         "principal angles between --cl-checkpoint's gate subspace "
                         "and this one's (e.g. B-specialized vs C-specialized) — "
                         "the direct 'overwriting within a preserved subspace' test.")
    ap.add_argument("--growth-factor", type=int, default=4)
    ap.add_argument("--growth-method", default="gate_fp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-probe-batches", type=int, default=3)
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--probe-seq", type=int, default=128)
    ap.add_argument("--m-old", type=int, default=64, help="# random old-direction samples")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    assert args.base_checkpoint and args.cl_checkpoint

    dev = args.device
    grown, cfg = build_grown(args.base_checkpoint, args.growth_factor,
                             args.growth_method, dev)
    aftercl, _ = load_after_cl(args.base_checkpoint, args.cl_checkpoint,
                               args.growth_factor, args.growth_method, dev)
    compare = None
    if args.compare_cl_checkpoint:
        compare, _ = load_after_cl(args.base_checkpoint, args.compare_cl_checkpoint,
                                   args.growth_factor, args.growth_method, dev)
    vocab = cfg.model.vocab_size
    seq = min(args.probe_seq, cfg.model.max_seq_len)

    acc = {"grown_gate_eff": [], "grown_gate_hard": [],
           "aftercl_gate_eff": [], "aftercl_gate_hard": [],
           "grown_resid_hard": [], "aftercl_resid_hard": [],
           "aftercl_resid_eff": [],
           "angle_min": [], "angle_median": [], "angle_max": [],
           "cmp_angle_min": [], "cmp_angle_median": [], "cmp_angle_max": []}
    for b in range(args.n_probe_batches):
        torch.manual_seed(1000 + b)
        x = torch.randint(0, vocab, (args.probe_batch, seq)).to(dev)
        gen = torch.Generator(device=dev).manual_seed(2000 + b)

        Jg = gate_columns(grown, x)
        Jc = gate_columns(aftercl, x)
        Sg = torch.linalg.svdvals(Jg.float()); Sc = torch.linalg.svdvals(Jc.float())
        acc["grown_gate_eff"].append(effective_rank(Sg));   acc["grown_gate_hard"].append(hard_rank(Sg))
        acc["aftercl_gate_eff"].append(effective_rank(Sc)); acc["aftercl_gate_hard"].append(hard_rank(Sc))

        # E4: transversality residual at each stage
        Jog = old_columns(grown, x, args.m_old, gen)
        _, hrg, _ = residual_rank(Jg, Jog)
        acc["grown_resid_hard"].append(hrg)
        Joc = old_columns(aftercl, x, args.m_old, gen)
        er_c, hr_c, _ = residual_rank(Jc, Joc)
        acc["aftercl_resid_hard"].append(hr_c); acc["aftercl_resid_eff"].append(er_c)

        # E6: subspace rotation grown -> after-CL
        ang = principal_angles(Jg, Jc)
        acc["angle_min"].append(min(ang))
        acc["angle_median"].append(sorted(ang)[len(ang)//2])
        acc["angle_max"].append(max(ang))
        cmp_str = ""
        if compare is not None:
            Jcmp = gate_columns(compare, x)
            ca = principal_angles(Jc, Jcmp)
            acc["cmp_angle_min"].append(min(ca))
            acc["cmp_angle_median"].append(sorted(ca)[len(ca)//2])
            acc["cmp_angle_max"].append(max(ca))
            cmp_str = (f"; CMP angles [{min(ca):.1f},"
                       f"{sorted(ca)[len(ca)//2]:.1f},{max(ca):.1f}]")
        print(f"  batch {b}: gate hard grown {acc['grown_gate_hard'][-1]} / "
              f"afterCL {acc['aftercl_gate_hard'][-1]}; "
              f"resid hard grown {hrg} / afterCL {hr_c}; "
              f"angles [{min(ang):.1f},{sorted(ang)[len(ang)//2]:.1f},{max(ang):.1f}]{cmp_str}")

    out = {"config": {"cl_checkpoint": args.cl_checkpoint,
                      "growth_method": args.growth_method, "gf": args.growth_factor,
                      "n_probe_batches": args.n_probe_batches,
                      "probe": [args.probe_batch, seq], "m_old": args.m_old},
           "summary": {k: summarize(v) for k, v in acc.items()},
           "K": len(new_gate_names(grown))}
    print("\n=== GAP summary (mean +/- std over batches) ===")
    K = out["K"]
    s = out["summary"]
    print(f"K = {K}")
    print(f"E3 gate hard rank:  grown {s['grown_gate_hard']['mean']:.1f} -> "
          f"afterCL {s['aftercl_gate_hard']['mean']:.1f}")
    print(f"       gate eff rank:  grown {s['grown_gate_eff']['mean']:.1f}±{s['grown_gate_eff']['std']:.1f} -> "
          f"afterCL {s['aftercl_gate_eff']['mean']:.1f}±{s['aftercl_gate_eff']['std']:.1f}")
    print(f"E4 transversal residual hard rank (want = K): grown "
          f"{s['grown_resid_hard']['mean']:.1f} -> afterCL {s['aftercl_resid_hard']['mean']:.1f}")
    print(f"E6 grown->afterCL principal angles (deg): min "
          f"{s['angle_min']['mean']:.1f}, median {s['angle_median']['mean']:.1f}, "
          f"max {s['angle_max']['mean']:.1f}")
    if compare is not None and s.get('cmp_angle_median', {}).get('mean') is not None \
            and len(acc['cmp_angle_median']):
        print(f"E6 (B-vs-C) angles between this after-CL and "
              f"--compare-cl-checkpoint (deg): min {s['cmp_angle_min']['mean']:.1f}, "
              f"median {s['cmp_angle_median']['mean']:.1f}, max {s['cmp_angle_max']['mean']:.1f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

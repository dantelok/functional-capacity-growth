#!/usr/bin/env python3
"""Integration check: Net2Net survives the REAL trainer CL path (isolation +
clean-fp) with its mechanism intact. This is what the standalone FP test cannot
catch — the trainer's per-step gate clamp (_sanitize_gate_params) and the
isolation gate-warmup are gate-zero-specific and previously would have throttled
Net2Net's block_gate from 1.0 to ~1/n_new, silently converting it to a hybrid.

Asserts, on a tiny CPU model, for the clean-fp isolation recipe the L40 run uses:
  (1) Net2Net new-block block_gate is frozen at 1.0 (requires_grad=False).
  (2) Isolation.setup() does NOT warm up / disturb the gate (clean-fp, gate=1).
  (3) _sanitize_gate_params leaves Net2Net gate at 1.0 (the clamp is skipped)...
  (4) ...while it DOES clamp gate-zero (guard is targeted, not blanket).
  (5) Over several optimizer steps WITH _sanitize_gate_params called each step
      (as train_cl does), the gate stays 1.0 and the output matrices open.
"""
import copy
import torch
import torch.nn.functional as F

from gatezero.config import (ExperimentConfig, ModelConfig, TrainingConfig,
                             GrowthConfig, CLConfig, GrowthMethod, CLMethod)
from gatezero.models.transformer import GrowableTransformer
from gatezero.trainer import Trainer
from gatezero.cl_methods import create_cl_method

torch.manual_seed(0)
mcfg = ModelConfig(d_model=32, n_heads=4, d_ffn=64, n_layers=2,
                   max_seq_len=64, vocab_size=100)  # >=64: apply_growth FP-verify uses a (2,64) probe


def build_grown(method):
    """Grow a tiny model exactly as the run script would (g=2, clean-fp)."""
    cfg = ExperimentConfig(
        name=f"itest_{method.value}",
        model=copy.deepcopy(mcfg),
        training=TrainingConfig(seed=0, deterministic=False),
        growth=GrowthConfig(method=method, growth_factor=2, clone_noise_std=0.0),
        cl=CLConfig(method=CLMethod.ISOLATION, cl_epochs=1,
                    distill_lambda=0.0, gate_warmup_eps=0.0),  # clean-fp
    )
    trainer = Trainer(cfg, "cpu")
    model = trainer.build_model()
    gi = 0.0 if method == GrowthMethod.GATE_ZERO else 1.0
    model, teacher = trainer.apply_growth(model, gate_init_value=gi)
    return trainer, model, teacher


def new_gate_vals(model):
    orig = set(model.original_block_indices)
    return [model.blocks[i].block_gate.item()
            for i in range(len(model.blocks)) if i not in orig]


# ---- Net2Net path ----------------------------------------------------------
tr, model, teacher = build_grown(GrowthMethod.NET2NET)
orig = set(model.original_block_indices)
new_idx = [i for i in range(len(model.blocks)) if i not in orig]

# (1) gate frozen at 1.0
for i in new_idx:
    bg = model.blocks[i].block_gate
    assert bg.item() == 1.0, f"net2net gate != 1.0 at init: {bg.item()}"
    assert bg.requires_grad is False, "net2net block_gate should be frozen"
print(f"(1) net2net new-block gates frozen at 1.0  ✓  (n_new={len(new_idx)})")

# (2) isolation setup must not warm up the gate
cl = create_cl_method(tr.config.cl, model, "cpu", teacher)
cl.setup()
assert all(abs(g - 1.0) < 1e-9 for g in new_gate_vals(model)), \
    "isolation.setup disturbed net2net gate"
print("(2) isolation.setup leaves gate at 1.0 (no warmup)  ✓")

# (3) the per-step sanitize clamp must NOT touch net2net (n_new=2 -> max_gate=0.15)
max_gate = min(0.15, 1.0 / len(new_idx))
tr._sanitize_gate_params(model, max_gate)
gates = new_gate_vals(model)
assert all(abs(g - 1.0) < 1e-9 for g in gates), \
    f"CLAMP BUG: net2net gate throttled to {gates} (expected 1.0)"
print(f"(3) _sanitize_gate_params leaves net2net gate at 1.0 (clamp skipped)  ✓")

# (5) several steps with sanitize each step: gate stays 1.0, matrices open
x = torch.randint(0, 100, (4, 16))
y = torch.randint(0, 100, (4, 16))
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
nb = model.blocks[new_idx[0]]
o0 = nb.attn.o_proj.weight.detach().abs().max().item()
d0 = nb.ffn.down_proj.weight.detach().abs().max().item()
model.train()
for _ in range(15):
    opt.zero_grad()
    loss = F.cross_entropy(model(x).reshape(-1, 100), y.reshape(-1))
    loss.backward()
    opt.step()
    tr._sanitize_gate_params(model, max_gate)  # exactly what train_cl does
gates = new_gate_vals(model)
o1 = nb.attn.o_proj.weight.detach().abs().max().item()
d1 = nb.ffn.down_proj.weight.detach().abs().max().item()
assert all(abs(g - 1.0) < 1e-9 for g in gates), f"gate drifted: {gates}"
assert o1 > o0 and d1 > d0, "matrices did not open under training"
print(f"(5) after 15 steps+sanitize: gate=1.0 held; o_proj {o0:.3f}->{o1:.3f}, "
      f"down_proj {d0:.3f}->{d1:.3f} (opens via matrices)  ✓")

# ---- Gate-zero path: the clamp MUST still fire (guard is targeted) ----------
tr2, model2, _ = build_grown(GrowthMethod.GATE_ZERO)
orig2 = set(model2.original_block_indices)
new2 = [i for i in range(len(model2.blocks)) if i not in orig2]
# force a gate above the clamp, then sanitize; it must be pulled back down
for i in new2:
    model2.blocks[i].block_gate.data.fill_(0.9)
mg2 = min(0.15, 1.0 / len(new2))
tr2._sanitize_gate_params(model2, mg2)
g2 = [model2.blocks[i].block_gate.item() for i in new2]
assert all(abs(g) <= mg2 + 1e-6 for g in g2), f"gate_fp clamp failed: {g2}"
print(f"(4) gate_fp clamp still active: 0.9 -> <= {mg2:.3f}  ✓")

print("\nAll integration checks passed: Net2Net keeps gate=1.0 through the real "
      "clean-fp isolation CL path; gate-zero clamp untouched. Safe for L40.")

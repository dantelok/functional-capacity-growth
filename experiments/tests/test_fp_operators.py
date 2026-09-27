#!/usr/bin/env python3
"""Local FP check for growth operators on a TINY model (CPU, no GPU/box needed).
A growth operator is function-preserving iff logits are unchanged after growth."""
import copy
import torch

from gatezero.config import ModelConfig, GrowthConfig, GrowthMethod
from gatezero.models.transformer import GrowableTransformer
from gatezero.growth import grow_transformer, verify_function_preservation

torch.manual_seed(0)
cfg = ModelConfig(d_model=32, n_heads=4, d_ffn=64, n_layers=2,
                  max_seq_len=16, vocab_size=100)
x = torch.randint(0, 100, (2, 16))

for method in [GrowthMethod.GATE_ZERO, GrowthMethod.NET2NET, GrowthMethod.GSTACK]:
    m = GrowableTransformer(copy.deepcopy(cfg)).eval()
    before = copy.deepcopy(m).eval()
    gc = GrowthConfig(method=method, growth_factor=4, clone_noise_std=0.0)
    gi = 0.0 if method == GrowthMethod.GATE_ZERO else 1.0
    try:
        grown = grow_transformer(m, gc, gate_init_value=gi).eval()
        s = verify_function_preservation(before, grown, x)
        verdict = "FP ✓" if s["passed"] else "NOT FP ✗"
        print(f">>> {method.value:8s}: max|Δlogit|={s['max_diff']:.2e}  ->  {verdict}\n")
    except Exception as e:
        print(f">>> {method.value:8s}: ERROR {e}\n")

print("Expected: gate_fp, net2net all FP; gstack NOT FP.")

"""
Configuration dataclasses for the experiments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GrowthMethod(Enum):
    # Paper's "gate-zero" operator. The on-disk value is kept as "gate_fp" so
    # existing results/ and checkpoints/ directory names stay valid provenance.
    GATE_ZERO = "gate_fp"     # Our method (paper: gate-zero): gate=0, function-preserving
    GSTACK = "gstack"         # Baseline: stack g copies (Du et al. 2024)
    NET2NET = "net2net"       # FP via ZERO output projections, block_gate=1
                              # (Fixup / Net2DeeperNet; == LEMON depth Type-1,
                              #  Wang et al. 2024 — opens via matrices, not the scalar gate)


class CLMethod(Enum):
    REPLAY = "replay"
    DISTILLATION = "distillation"
    ISOLATION = "isolation"
    EWC = "ewc"                 # elastic weight consolidation (no-growth classic CL baseline)


@dataclass
class ModelConfig:
    """Base Transformer configuration."""
    d_model: int = 1024
    n_heads: int = 16
    d_ffn: int = 4096            # 4x d_model
    n_layers: int = 12
    max_seq_len: int = 1024
    vocab_size: int = 50257      # GPT-2 tokenizer
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 2
    grad_accum: int = 64         # effective batch = 128
    lr: float = 3e-4
    gate_lr: float = 1e-3
    weight_decay: float = 0.1
    gate_weight_decay: float = 0.0
    epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    max_gate_grad_norm: float = 0.1
    fp16: bool = True
    log_interval: int = 50       # log every N steps
    eval_interval: int = 500     # eval every N steps
    save_interval: int = 1000    # checkpoint every N steps
    seed: int = 42
    num_workers: int = 4
    activation_checkpointing: bool = False
    # Full GPU determinism (cuDNN + cuBLAS). On by default for reproducible
    # runs; set False to recover cuDNN autotuning speed on large runs.
    deterministic: bool = True


@dataclass
class GrowthConfig:
    """Growth operator configuration."""
    method: GrowthMethod = GrowthMethod.GATE_ZERO
    growth_factor: int = 4       # n_layers: 12 -> 48
    gate_init_value: float = 0.0  # block_gate init for newly inserted blocks
    clone_noise_std: float = 0.01  # noise added to cloned blocks
    # Growth pattern:
    # - "interleaved": insert new blocks between old blocks (can pollute
    #   the residual stream, causing PPL_A drift in CL)
    # - "append": stack all new blocks AFTER old blocks (preserves old
    #   computation path exactly, required for manifold-preserving CL)
    growth_pattern: str = "append"


@dataclass
class CLConfig:
    """Continual learning method configuration."""
    method: CLMethod = CLMethod.ISOLATION
    # Post-growth training LR (lower than base)
    cl_lr: float = 1e-4
    # Gate LR: must be low enough that (n_new_blocks × gate_value) doesn't
    # explode the residual stream. With 36 new blocks, gate_lr=1e-3 causes
    # gates to reach ~0.9 after 1 epoch → 36×0.9 = 32× perturbation → NaN.
    # Safe rule: gate_lr ≈ 1e-3 / sqrt(n_new_blocks).
    cl_gate_lr: float = 5e-5
    cl_epochs: int = 10
    # Replay params
    replay_alpha: float = 0.1           # fraction of old data per batch
    replay_buffer_fraction: float = 0.1  # fraction of Dataset A to store
    # Distillation params
    distill_lambda: float = 0.5
    distill_temperature: float = 2.0
    # EWC params. lambda is Fisher-scale-dependent — SWEEP it for a fair baseline
    # (same lesson as replay_alpha): an under-tuned lambda makes EWC look bad.
    ewc_lambda: float = 1e4
    ewc_fisher_batches: int = 50   # old-data batches to estimate the diagonal Fisher
    # Isolation params — what to freeze
    freeze_old_weights: bool = True
    freeze_old_gates: bool = True
    freeze_old_block_gates: bool = True
    # Gate warmup: when gate_init=0 and isolation is used, new block gates are
    # nudged from 0 -> eps so internal weights get non-zero gradients.
    # Without this, dL/dW = block_gate * (...) = 0 and nothing learns.
    gate_warmup_eps: float = 1e-3


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    cl: CLConfig = field(default_factory=CLConfig)
    # Paths
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    dataset_a: str = "wikitext"
    dataset_b: str = "bookcorpus"
    # Resume
    resume_base_checkpoint: Optional[str] = None  # skip base training if set


def _replay_training_config() -> TrainingConfig:
    """
    Replay methods use the stable default microbatch regime (batch_size=2,
    grad_accum=64). Replay ratio is realised at microbatch granularity, so any
    alpha works here regardless of batch size (see CombinedDataLoader).
    """
    return TrainingConfig(batch_size=2, grad_accum=64)


def _replay_cl_config(method: CLMethod) -> CLConfig:
    """Shared CL settings for replay-based methods.

    replay_alpha=0.1 is now the intended fair ratio (realisable at any batch size
    via microbatch-level mixing in CombinedDataLoader). Sweep it as a hyperparameter.
    """
    return CLConfig(method=method, replay_alpha=0.1)


# ─────────────────────────────────────────────────────────────
# Preset configurations for each experiment
# ─────────────────────────────────────────────────────────────

def exp1_configs():
    """Experiment 1: CL matrix — replay/distillation/isolation x 2 growth ops + baseline."""
    configs = []
    for growth in [GrowthMethod.GATE_ZERO, GrowthMethod.GSTACK]:
        methods = [CLMethod.REPLAY, CLMethod.DISTILLATION, CLMethod.ISOLATION]
        # G_stack has no gates to freeze — skip isolation
        if growth == GrowthMethod.GSTACK:
            methods = [CLMethod.REPLAY, CLMethod.DISTILLATION]
        for method in methods:
            training = TrainingConfig()
            cl = CLConfig(method=method)
            if method == CLMethod.REPLAY:
                training = _replay_training_config()
                cl = _replay_cl_config(method)
            cfg = ExperimentConfig(
                name=f"exp1_{growth.value}_{method.value}",
                training=training,
                growth=GrowthConfig(method=growth),
                cl=cl,
            )
            configs.append(cfg)

    # Upper bound: train from scratch on A ∪ B jointly.
    # - epochs=10 over A∪B = 10|A| + 10|B| tokens, matching the CL runs'
    #   total budget (10 base epochs on A + 10 CL epochs on B).
    # - dropout + higher weight_decay because the previous 20-epoch run on
    #   A-only peaked val PPL at epoch 3 and overfit for 17 more epochs;
    #   mild regularization keeps the from-scratch 857M model honest.
    cfg_upper = ExperimentConfig(
        name="exp1_upper_bound_scratch",
        model=ModelConfig(n_layers=48, dropout=0.05),
        training=TrainingConfig(epochs=10, weight_decay=0.2),
    )
    configs.append(cfg_upper)
    return configs

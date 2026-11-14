"""
Main training loop for all experiments.

Handles:
  1. Base model training on Dataset A
  2. Growth operator application
  3. CL training on Dataset B with forgetting prevention
  4. Evaluation on both datasets
  5. Checkpointing and logging
"""

import os
import json
import math
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional

from .config import ExperimentConfig, GrowthMethod, CLMethod
from .models.transformer import GrowableTransformer
from .growth import grow_transformer, verify_function_preservation
from .cl_methods import create_cl_method
from .evaluation import compute_perplexity, evaluate_cl
from .reproducibility import set_reproducible, seed_worker, make_generator

# Deterministic cuBLAS requires this set BEFORE the CUDA context initialises,
# otherwise torch.use_deterministic_algorithms(True) raises at the first matmul.
# Set at import (harmless when determinism is off); Trainer applies the rest.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class Trainer:
    """Unified trainer for the experiments."""

    def __init__(self, config: ExperimentConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        # Record the ACTUAL knobs used. Growth is clean-FP by construction
        # (exact clones, no gate warm-up); distill_lambda is the only remaining
        # preservation knob and is 0.0 except for the Distillation baseline.
        self.results = {
            "config": config.name,
            "config_params": {
                "cl_method": config.cl.method.value,
                "growth_method": config.growth.method.value,
                "growth_factor": config.growth.growth_factor,
                "growth_pattern": config.growth.growth_pattern,
                "gate_init_value": config.growth.gate_init_value,
                "distill_lambda": config.cl.distill_lambda,
                "distill_temperature": config.cl.distill_temperature,
                "replay_alpha": config.cl.replay_alpha,
                "replay_buffer_fraction": config.cl.replay_buffer_fraction,
                "freeze_old_weights": config.cl.freeze_old_weights,
                "freeze_old_gates": config.cl.freeze_old_gates,
                "freeze_old_block_gates": config.cl.freeze_old_block_gates,
                "cl_epochs": config.cl.cl_epochs,
                "seed": config.training.seed,
                "deterministic": config.training.deterministic,
            },
            "metrics": {},
        }

        # Create output dirs
        self.output_dir = Path(config.output_dir) / config.name
        self.ckpt_dir = Path(config.checkpoint_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Seed every RNG surface (Python, NumPy, torch, all CUDA devices) and
        # apply GPU determinism from one place. See gatezero.reproducibility for
        # why a torch seed alone is insufficient (numpy ReplayBuffer draws +
        # cuDNN autotuning/atomics). Opt out via config.training.deterministic
        # =False to recover cuDNN autotuning speed on throwaway runs.
        set_reproducible(config.training.seed,
                         deterministic=config.training.deterministic)

    def build_model(self):
        """Create the model based on config."""
        model = GrowableTransformer(self.config.model)

        model.activation_checkpointing = (
            self.config.training.activation_checkpointing
        )
        model = model.to(self.device)
        print(f"Model: {model.count_params():,} params")
        return model

    def _build_cpu_teacher(self, model):
        """
        Build a frozen teacher copy directly on CPU.

        Deep-copying a large GPU model can temporarily require a second full
        VRAM copy. Rebuild the module on CPU and load a CPU state dict instead.
        """
        teacher_cfg = copy.deepcopy(model.config)
        teacher = GrowableTransformer(teacher_cfg)

        state_dict_cpu = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        }
        teacher.load_state_dict(state_dict_cpu)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    def build_optimizer(self, model, lr=None, gate_lr=None):
        """Create AdamW optimizer with separate gate parameter group."""
        cfg = self.config.training
        lr = lr or cfg.lr
        gate_lr = gate_lr or cfg.gate_lr

        gate_params = model.get_gate_params()
        gate_ids = {id(p) for p in gate_params}
        non_gate_params = [p for p in model.parameters()
                          if p.requires_grad and id(p) not in gate_ids]
        gate_params = [p for p in gate_params if p.requires_grad]

        param_groups = []
        if non_gate_params:
            param_groups.append({
                "params": non_gate_params,
                "lr": lr,
                "weight_decay": cfg.weight_decay,
            })
        if gate_params:
            param_groups.append({
                "params": gate_params,
                "lr": gate_lr,
                "weight_decay": cfg.gate_weight_decay,
            })

        return torch.optim.AdamW(
            param_groups, betas=(0.9, 0.999)
        )

    def build_scheduler(self, optimizer, total_steps, warmup_steps=None):
        """Cosine LR schedule with linear warmup."""
        warmup = warmup_steps or self.config.training.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _clip_grads(self, model, max_grad_norm, max_gate_grad_norm):
        """Clip gate gradients more aggressively than other parameters."""
        if not hasattr(model, "get_gate_params"):
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            return

        gate_params = [
            p for p in model.get_gate_params()
            if p.requires_grad and p.grad is not None
        ]
        gate_ids = {id(p) for p in gate_params}
        non_gate_params = [
            p for p in model.parameters()
            if p.requires_grad and p.grad is not None and id(p) not in gate_ids
        ]

        if non_gate_params:
            nn.utils.clip_grad_norm_(non_gate_params, max_grad_norm)
        if gate_params:
            nn.utils.clip_grad_norm_(gate_params, max_gate_grad_norm)

    def _sanitize_gate_params(self, model, max_block_gate):
        """
        Zero out non-finite gate updates, then clamp new block gates.

        If a gate turns NaN after optimizer.step(), a plain clamp will not
        recover it. Cleaning the gate tensors here prevents a single unstable
        update from poisoning the next forward pass.
        """
        if not hasattr(model, "get_gate_params"):
            return

        orig_indices = set(getattr(model, "original_block_indices", []))
        with torch.no_grad():
            for p in model.get_gate_params():
                if p.requires_grad:
                    p.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            # The block_gate clamp is residual-overflow protection for GATE-based
            # growth, where block_gate IS the opening mechanism. Net2Net opens
            # through its output matrices with block_gate fixed at 1.0 — clamping
            # it to ~1/n_new would throttle the whole mechanism to a few percent
            # and silently turn Net2Net into a gate-limited hybrid. Skip it there.
            if self.config.growth.method == GrowthMethod.NET2NET:
                return
            for i, block in enumerate(model.blocks):
                if hasattr(block, "block_gate") and i not in orig_indices:
                    block.block_gate.data.clamp_(-max_block_gate, max_block_gate)

    def _collect_diagnostics(self, model) -> dict:
        """
        Per-epoch diagnostic snapshot for the cross-architecture analysis.

        Captures block_gate values on new blocks — diagnoses gate-explosion /
        residual-stream pollution — with summary stats (raw per-block vectors
        under "block_gates_new"). For pre-growth models without
        `original_block_indices`, returns empty dict.
        """
        orig_indices = set(getattr(model, "original_block_indices", []))
        if not orig_indices:
            return {}

        # (a) block_gate values on new blocks
        block_gates_new = []
        for i, block in enumerate(model.blocks):
            if i in orig_indices:
                continue
            if hasattr(block, "block_gate"):
                block_gates_new.append(float(block.block_gate.detach().item()))

        diag: dict = {}
        if block_gates_new:
            abs_gates = [abs(g) for g in block_gates_new]
            diag["block_gates_new"] = block_gates_new
            diag["block_gate_max_abs"] = max(abs_gates)
            diag["block_gate_mean_abs"] = sum(abs_gates) / len(abs_gates)

        return diag

    # ─────────────────────────────────────────────────────────
    # Phase 1: Base training on Dataset A
    # ─────────────────────────────────────────────────────────

    def train_base(self, model, train_ds, val_ds):
        """Train the base model on Dataset A."""
        cfg = self.config.training
        print(f"\n{'='*60}")
        print(f"PHASE 1: Base training on Dataset A")
        print(f"  Epochs: {cfg.epochs}, LR: {cfg.lr}, Batch: {cfg.batch_size}x{cfg.grad_accum}")
        print(f"{'='*60}")

        loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        )
        optimizer = self.build_optimizer(model, lr=cfg.lr, gate_lr=cfg.gate_lr)
        total_steps = len(loader) * cfg.epochs // cfg.grad_accum
        scheduler = self.build_scheduler(optimizer, total_steps)
        scaler = GradScaler('cuda', enabled=cfg.fp16)

        global_step = 0
        best_ppl = float("inf")

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_tokens = 0

            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                with autocast('cuda', enabled=cfg.fp16):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                    loss = loss / cfg.grad_accum

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * cfg.grad_accum * y.numel()
                epoch_tokens += y.numel()

                if (batch_idx + 1) % cfg.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    self._clip_grads(
                        model,
                        cfg.max_grad_norm,
                        cfg.max_gate_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % cfg.log_interval == 0:
                        avg_loss = epoch_loss / epoch_tokens
                        ppl = math.exp(min(avg_loss, 20))
                        lr_now = optimizer.param_groups[0]["lr"]
                        print(f"  [Step {global_step}] loss={avg_loss:.4f} "
                              f"ppl={ppl:.2f} lr={lr_now:.2e}")

            # End-of-epoch eval
            val_ppl = compute_perplexity(
                model, val_ds, cfg.batch_size, self.device, cfg.num_workers
            )
            train_ppl = math.exp(min(epoch_loss / epoch_tokens, 20))
            print(f"  Epoch {epoch+1}/{cfg.epochs}: "
                  f"train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}")

            if val_ppl < best_ppl:
                best_ppl = val_ppl
                self._save_checkpoint(model, optimizer, epoch, "base_best")

        self._save_checkpoint(model, optimizer, cfg.epochs - 1, "base_final")

        self.results["metrics"]["base_train_ppl"] = train_ppl
        self.results["metrics"]["base_val_ppl"] = val_ppl
        self.results["metrics"]["base_best_ppl"] = best_ppl
        print(f"\nBase training complete. Best val PPL: {best_ppl:.2f}")
        return model

    # ─────────────────────────────────────────────────────────
    # Phase 2: Growth
    # ─────────────────────────────────────────────────────────

    def apply_growth(self, model, gate_init_value=0.0):
        """Apply growth operator and return (grown_model, teacher_copy)."""
        cfg = self.config.growth
        print(f"\n{'='*60}")
        print(f"PHASE 2: Growth ({cfg.method.value}, factor={cfg.growth_factor})")
        print(f"{'='*60}")

        # Run growth on CPU. This avoids a second large VRAM spike during
        # cloning/expansion, and keeps FP verification on one device.
        model = model.cpu()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # Save teacher copy BEFORE growth only for methods that actually use it.
        teacher = None
        if self.config.cl.method in {
            CLMethod.DISTILLATION, CLMethod.ISOLATION
        }:
            teacher = self._build_cpu_teacher(model)

        params_before = model.count_params()

        # Apply growth operator
        model = grow_transformer(model, cfg, gate_init_value)

        params_after = model.count_params()
        print(f"  Params: {params_before:,} -> {params_after:,} "
              f"({params_after/params_before:.1f}x)")

        # Verify FP at growth. Gate-zero is FP only when gate_init=0; Net2Net is
        # FP unconditionally (zeroed output projections), so verify it too — this
        # confirms the grown checkpoint equals the base before any CL, at hour 0.
        fp_stats = None
        if (cfg.method == GrowthMethod.GATE_ZERO and gate_init_value == 0.0) \
                or cfg.method == GrowthMethod.NET2NET:
            dummy_input = torch.randint(
                0, self.config.model.vocab_size, (2, 64)
            )
            fp_stats = verify_function_preservation(
                teacher if teacher is not None else model, model, dummy_input
            )

        # CL training still runs on the configured device.
        model = model.to(self.device)

        # The frozen teacher is only needed by distillation/isolation.
        # Keep it off GPU until a CL method explicitly loads it to avoid
        # carrying unused VRAM through replay runs (which need no teacher).
        if teacher is not None:
            teacher = teacher.cpu()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        self.results["metrics"]["params_before_growth"] = params_before
        self.results["metrics"]["params_after_growth"] = params_after
        if fp_stats is not None:
            self.results["metrics"]["growth_fp_max_diff"] = fp_stats["max_diff"]
            self.results["metrics"]["growth_fp_mean_diff"] = fp_stats["mean_diff"]
            self.results["metrics"]["growth_fp_threshold"] = fp_stats["threshold"]
            self.results["metrics"]["growth_fp_passed"] = fp_stats["passed"]

        return model, teacher

    # ─────────────────────────────────────────────────────────
    # Phase 3: CL training on Dataset B
    # ─────────────────────────────────────────────────────────

    def train_cl(self, model, train_ds_b, val_ds_a, val_ds_b,
                 train_ds_a=None, teacher=None):
        """Train on Dataset B with CL method, evaluate on both A and B."""
        cfg_cl = self.config.cl
        cfg_tr = self.config.training
        print(f"\n{'='*60}")
        print(f"PHASE 3: CL training ({cfg_cl.method.value})")
        print(f"  Epochs: {cfg_cl.cl_epochs}, LR: {cfg_cl.cl_lr}")
        print(f"{'='*60}")

        # Evaluate before CL training (post-growth baseline)
        pre_cl = evaluate_cl(
            model, val_ds_a, val_ds_b, cfg_tr.batch_size,
            self.device, cfg_tr.num_workers,
        )
        print(f"  Pre-CL: PPL_A={pre_cl['ppl_a']:.2f}, PPL_B={pre_cl['ppl_b']:.2f}")
        self.results["metrics"]["pre_cl_ppl_a"] = pre_cl["ppl_a"]
        self.results["metrics"]["pre_cl_ppl_b"] = pre_cl["ppl_b"]

        # Set up CL method
        cl_method = create_cl_method(cfg_cl, model, self.device, teacher)
        cl_method.setup()

        # Build data loader
        loader = cl_method.build_dataloader(
            train_ds_b, train_ds_a, cfg_tr.batch_size, cfg_tr.num_workers
        )

        # Build optimizer (only trainable params)
        optimizer = self.build_optimizer(
            model, lr=cfg_cl.cl_lr, gate_lr=cfg_cl.cl_gate_lr
        )
        total_steps = len(loader) * cfg_cl.cl_epochs // cfg_tr.grad_accum
        scheduler = self.build_scheduler(optimizer, total_steps)
        scaler = GradScaler('cuda', enabled=cfg_tr.fp16)

        global_step = 0
        history = []

        for epoch in range(cfg_cl.cl_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_tokens = 0

            for batch_idx, raw_batch in enumerate(loader):
                batch = cl_method.move_batch_to_device(raw_batch)
                x, y = cl_method.get_student_batch(batch)

                batch_loss = 0.0
                with autocast('cuda', enabled=cfg_tr.fp16):
                    logits = model(x)
                    loss = cl_method.compute_main_loss(
                        logits, y, input_ids=x, batch=batch
                    )
                    if hasattr(model, "get_aux_loss"):
                        loss = loss + model.get_aux_loss()
                    loss = loss / cfg_tr.grad_accum

                scaler.scale(loss).backward()
                batch_loss += loss.item() * cfg_tr.grad_accum

                if cl_method.has_old_forward_loss():
                    old_x = batch["old_input_ids"]
                    with autocast('cuda', enabled=cfg_tr.fp16):
                        old_logits = model(old_x)
                        old_loss = cl_method.compute_old_forward_loss(
                            old_logits, batch
                        )
                        if hasattr(model, "get_aux_loss"):
                            old_loss = old_loss + model.get_aux_loss()
                        old_loss = old_loss / cfg_tr.grad_accum

                    scaler.scale(old_loss).backward()
                    batch_loss += old_loss.item() * cfg_tr.grad_accum

                epoch_loss += batch_loss * y.numel()
                epoch_tokens += y.numel()

                if (batch_idx + 1) % cfg_tr.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    self._clip_grads(
                        model,
                        cfg_tr.max_grad_norm,
                        cfg_tr.max_gate_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    # Clamp gate values to prevent residual stream overflow.
                    # 36 new blocks × gate × Block(x) compounds through sequential blocks.
                    # Even gate=0.15 can overflow fp16 after 36 blocks.
                    # gate=0.05 → each block adds 5% → total ~1.8× after 36 (safe).
                    orig_indices = set(getattr(model, 'original_block_indices', []))
                    n_new = len([
                        b for i, b in enumerate(model.blocks)
                        if hasattr(b, 'block_gate') and i not in orig_indices
                    ])
                    # Adaptive clamp: tighter with more new blocks
                    max_gate = min(0.15, 1.0 / max(n_new, 1))
                    self._sanitize_gate_params(model, max_gate)

                    if global_step % cfg_tr.log_interval == 0:
                        avg_loss = epoch_loss / epoch_tokens
                        print(f"  [Step {global_step}] loss={avg_loss:.4f}")

                    # Early NaN detection — fail fast instead of wasting hours
                    if global_step % 50 == 0:
                        check_loss = epoch_loss / max(epoch_tokens, 1)
                        if not math.isfinite(check_loss):
                            # Log gate values for debugging
                            gate_vals = []
                            orig_idx = set(getattr(model, 'original_block_indices', []))
                            for i, block in enumerate(model.blocks):
                                if hasattr(block, 'block_gate') and i not in orig_idx:
                                    gate_vals.append(f"{block.block_gate.item():.4f}")
                            print(f"  NaN detected at step {global_step}!")
                            print(f"  New block gates: [{', '.join(gate_vals[:10])}{'...' if len(gate_vals) > 10 else ''}]")
                            raise RuntimeError(
                                f"NaN loss at step {global_step}. "
                                f"Gate clamp={max_gate:.4f}, "
                                f"n_new_blocks={n_new}. "
                                f"Try: reduce cl_lr, or disable fp16."
                            )

            # End-of-epoch evaluation on BOTH datasets
            metrics = evaluate_cl(
                model, val_ds_a, val_ds_b, cfg_tr.batch_size,
                self.device, cfg_tr.num_workers,
            )
            train_loss = epoch_loss / epoch_tokens
            print(f"  Epoch {epoch+1}/{cfg_cl.cl_epochs}: "
                  f"PPL_A={metrics['ppl_a']:.2f}, PPL_B={metrics['ppl_b']:.2f}")

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "ppl_a": metrics["ppl_a"],
                "ppl_b": metrics["ppl_b"],
            }
            # Per-epoch diagnostics: gate values on new blocks.
            # Captures the failure-mode signals (gate explosion, expert
            # routing collapse) for the cross-architecture analysis.
            try:
                epoch_record["diagnostics"] = self._collect_diagnostics(model)
            except Exception as e:
                # Diagnostics are best-effort; never let logging fail training.
                print(f"  (diagnostics skipped: {type(e).__name__}: {e})")
            history.append(epoch_record)

        # Final results
        final = history[-1]
        # delta_a measures CL-induced drift only: post-CL PPL_A minus
        # post-growth pre-CL PPL_A. Using base_val_ppl was wrong because
        # (a) it defaults to 0 when resuming from a checkpoint, and
        # (b) for non-FP growth (e.g. G_stack), growth itself changes PPL_A
        # so base PPL is not the right CL-baseline.
        pre_cl_ppl_a = self.results["metrics"].get("pre_cl_ppl_a")
        if pre_cl_ppl_a is None:
            pre_cl_ppl_a = final["ppl_a"]  # degenerate: no CL drift to report
        delta_a = final["ppl_a"] - pre_cl_ppl_a

        self.results["metrics"]["final_ppl_a"] = final["ppl_a"]
        self.results["metrics"]["final_ppl_b"] = final["ppl_b"]
        self.results["metrics"]["delta_a"] = delta_a
        self.results["metrics"]["cl_history"] = history

        print(f"\nCL training complete:")
        print(f"  PPL_A: {final['ppl_a']:.2f} (delta={delta_a:+.2f})")
        print(f"  PPL_B: {final['ppl_b']:.2f}")

        self._save_checkpoint(model, optimizer, cfg_cl.cl_epochs - 1, "cl_final")
        return model

    # ─────────────────────────────────────────────────────────
    # Full pipeline
    # ─────────────────────────────────────────────────────────

    def run_full_pipeline(self, train_ds_a, val_ds_a, train_ds_b, val_ds_b):
        """Run the complete experiment: base train -> grow -> CL train -> eval."""
        t_start = time.time()

        # Phase 1: Base training (or load checkpoint)
        model = self.build_model()
        if self.config.resume_base_checkpoint:
            print(f"Loading base checkpoint: {self.config.resume_base_checkpoint}")
            self._load_checkpoint(model, self.config.resume_base_checkpoint)
        else:
            model = self.train_base(model, train_ds_a, val_ds_a)

        # Phase 2: Growth
        gate_init = self.config.growth.gate_init_value
        # Backwards compatibility for older configs that encoded gate init only
        # in the experiment name.
        if gate_init == 0.0 and "gate_init_" in self.config.name:
            gate_init = float(self.config.name.split("gate_init_")[1])
        model, teacher = self.apply_growth(model, gate_init)

        # Phase 3: CL training
        model = self.train_cl(
            model, train_ds_b, val_ds_a, val_ds_b,
            train_ds_a=train_ds_a, teacher=teacher,
        )

        # Save final results
        elapsed = time.time() - t_start
        self.results["metrics"]["total_time_seconds"] = elapsed
        self.results["metrics"]["total_time_hours"] = elapsed / 3600

        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")
        print(f"Total time: {elapsed/3600:.1f} hours")

        return self.results

    # ─────────────────────────────────────────────────────────
    # Checkpoint utilities
    # ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, model, optimizer, epoch, tag):
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": self.config.name,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, model, path):
        # Load through CPU first so checkpoint restore does not require a full
        # second copy of the model on GPU during deserialization.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
        return model

"""
Continual Learning methods — forgetting-prevention strategies.

Each method defines:
  1. How to set up the model (freeze params, create teacher, etc.)
  2. How to compute the loss for each training step
  3. How to build the data loader (replay vs standard)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CLConfig, CLMethod, GrowthMethod
from .data import ReplayBuffer, CombinedDataLoader, PairedDataLoader


def teacher_student_kl(student_logits: torch.Tensor,
                       teacher_logits: torch.Tensor,
                       temperature: float) -> torch.Tensor:
    """
    KL(teacher || student) computed in fp32 for numerical stability.

    The preservation / distillation path is the first place fp16 tends to blow
    up late in training because softmax(logits / T) can get very peaky once the
    new blocks start learning aggressively. Keeping this in float32 is cheap
    relative to the full forward and removes a common NaN source.
    """
    T = temperature
    student = (student_logits.float() / T).view(-1, student_logits.size(-1))
    teacher = (teacher_logits.float() / T).view(-1, teacher_logits.size(-1))
    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_probs = F.softmax(teacher, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


class CLMethodBase:
    """Base class for continual learning methods."""

    def __init__(self, config: CLConfig, model, device):
        self.config = config
        self.model = model
        self.device = device

    def setup(self):
        """Called once before CL training begins."""
        pass

    def compute_loss(self, logits, labels, input_ids=None, batch=None):
        """Compute the loss for one training step."""
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )

    def compute_main_loss(self, logits, labels, input_ids=None, batch=None):
        """Loss term attached to the main new-task student forward."""
        return self.compute_loss(logits, labels, input_ids=input_ids, batch=batch)

    def has_old_forward_loss(self) -> bool:
        """Whether this method needs a separate student forward on old data."""
        return False

    def compute_old_forward_loss(self, old_logits, batch):
        """Loss term attached to the old-data student forward."""
        raise NotImplementedError

    def move_batch_to_device(self, batch):
        """Move a batch structure to the configured device."""
        if isinstance(batch, dict):
            return {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

        x, y = batch
        return {"input_ids": x.to(self.device), "labels": y.to(self.device)}

    def get_student_batch(self, batch):
        """Return the student forward inputs and labels for the main loss."""
        return batch["input_ids"], batch["labels"]

    def build_dataloader(self, new_dataset, old_dataset, batch_size, num_workers):
        """Build the data loader for CL training."""
        return DataLoader(
            new_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    def _replay_batch_size(self, batch_size: int) -> int:
        """Batch size for the replay / preservation side-channel."""
        return batch_size


class Replay(CLMethodBase):
    """
    Method 1: Mix alpha% of old data into each training batch.

    Loss = CE on combined batch (old + new samples together).
    """

    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        self.replay_buffer = None

    def build_dataloader(self, new_dataset, old_dataset, batch_size, num_workers):
        # Create replay buffer from old dataset
        self.replay_buffer = ReplayBuffer(
            old_dataset, self.config.replay_buffer_fraction
        )

        new_loader = DataLoader(
            new_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        replay_loader = DataLoader(
            self.replay_buffer, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

        return CombinedDataLoader(
            new_loader, replay_loader, alpha=self.config.replay_alpha
        )


class Distillation(CLMethodBase):
    """
    Method 2: Knowledge distillation from the pre-growth teacher model.

    Loss = CE(student, labels) + lambda * T^2 * KL(teacher || student)

    Teacher = frozen copy of the ORIGINAL (pre-growth) model.
    """

    def __init__(self, config, model, device, teacher_model=None):
        super().__init__(config, model, device)
        self.teacher = teacher_model
        self.temperature = config.distill_temperature
        self.lam = config.distill_lambda

    def setup(self):
        if self.teacher is None:
            raise ValueError("Distillation requires a teacher model")
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        print(f"Distillation: teacher loaded ({sum(p.numel() for p in self.teacher.parameters()):,} params, frozen)")
        print(f"  lambda={self.lam}, temperature={self.temperature}")

    def compute_loss(self, logits, labels, input_ids=None, batch=None):
        # Standard CE loss on new data
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )

        # KL divergence from teacher
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)

        T = self.temperature
        kl_loss = teacher_student_kl(logits, teacher_logits, T)

        total_loss = ce_loss + self.lam * (T ** 2) * kl_loss
        return total_loss


class Isolation(CLMethodBase):
    """
    Method 3: Architectural isolation — freeze ALL old parameters.

    Only newly grown parameters are trainable. The old computation path
    is literally unchanged -> exact preservation by construction.

    This is the manifold-theoretic method: old manifold dimensions frozen,
    learning only in new orthogonal dimensions.
    """

    def __init__(self, config, model, device, teacher_model=None):
        super().__init__(config, model, device)
        self.teacher = teacher_model
        self.temperature = config.distill_temperature
        self.lam = config.distill_lambda
        self.replay_buffer = None

    @property
    def uses_preservation(self) -> bool:
        """Old-data replay + teacher KL only exist when distill_lambda > 0.
        At the clean-FP default (0.0) isolation trains on new data alone."""
        return self.lam > 0.0

    def setup(self):
        cfg = self.config

        if self.uses_preservation:
            if self.teacher is None:
                raise ValueError("Isolation with distill_lambda > 0 requires a "
                                 "teacher model for old-task preservation")
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        # Use original_block_indices set by growth (interleaved insertion scatters
        # originals at positions 0, g, 2g, ... not 0..n_orig-1). Fall back to
        # first n_orig indices for backwards compat.
        if hasattr(self.model, 'original_block_indices'):
            orig_indices = set(self.model.original_block_indices)
        else:
            orig_indices = set(range(self.model.n_original_blocks))

        frozen_count = 0
        trainable_count = 0

        # Freeze embedding + final norm (always original)
        if cfg.freeze_old_weights:
            for p in self.model.token_emb.parameters():
                p.requires_grad = False
                frozen_count += p.numel()
            for p in self.model.final_norm.parameters():
                p.requires_grad = False
                frozen_count += p.numel()
            # lm_head is tied to token_emb, already frozen

        # Growth gate parameter names — these are the scalar/vector gates
        # we added for growth, NOT the SwiGLU gate_proj weight matrix.
        GROWTH_GATE_NAMES = {"head_gate", "ffn_gate", "block_gate"}

        def _is_growth_gate(param_name: str) -> bool:
            """Check if a named parameter is a growth gate (not gate_proj)."""
            return any(param_name.endswith(g) for g in GROWTH_GATE_NAMES)

        # Freeze original blocks (by actual index, not first n_orig)
        for i in sorted(orig_indices):
            block = self.model.blocks[i]

            if cfg.freeze_old_weights:
                # Freeze all NON-growth-gate parameters in original blocks
                # This includes gate_proj (SwiGLU), all linear weights, norms
                for name, p in block.named_parameters():
                    if not _is_growth_gate(name):
                        p.requires_grad = False
                        frozen_count += p.numel()

            if cfg.freeze_old_gates:
                # Freeze growth gates (head_gate, ffn_gate)
                block.attn.head_gate.requires_grad = False
                frozen_count += block.attn.head_gate.numel()
                if hasattr(block, 'ffn'):
                    block.ffn.ffn_gate.requires_grad = False
                    frozen_count += block.ffn.ffn_gate.numel()

            if cfg.freeze_old_block_gates:
                block.block_gate.requires_grad = False
                frozen_count += block.block_gate.numel()

        # New blocks remain fully trainable (all indices not in orig_indices)
        for i in range(len(self.model.blocks)):
            if i in orig_indices:
                continue
            for p in self.model.blocks[i].parameters():
                trainable_count += p.numel()

        # If nothing was frozen and there are no "new" blocks (e.g. a no-growth
        # baseline), count all model params as
        # trainable so the print statement is still informative and we avoid
        # a div-by-zero. Old/new partitioning is meaningless in that regime.
        if frozen_count == 0 and trainable_count == 0:
            trainable_count = sum(p.numel() for p in self.model.parameters()
                                  if p.requires_grad)

        total = frozen_count + trainable_count
        if total > 0:
            print(f"Isolation: frozen={frozen_count:,} ({frozen_count/total:.1%}), "
                  f"trainable={trainable_count:,} ({trainable_count/total:.1%})")
        else:
            print("Isolation: no parameters (degenerate setup)")

        # Clean FP: new gate-zero blocks start at block_gate=0, so only the
        # scalar gate gets a first-step gradient; block weights start learning
        # once their gate opens. No warm-up nudge is applied.

        print(f"  preserve_lambda={self.lam}, preserve_temperature={self.temperature}")

    def build_dataloader(self, new_dataset, old_dataset, batch_size, num_workers):
        if not self.uses_preservation:
            # No preservation term: no replay buffer, new-domain data only.
            return super().build_dataloader(new_dataset, old_dataset,
                                            batch_size, num_workers)
        self.replay_buffer = ReplayBuffer(
            old_dataset, self.config.replay_buffer_fraction
        )
        replay_batch_size = self._replay_batch_size(batch_size)

        new_loader = DataLoader(
            new_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        replay_loader = DataLoader(
            self.replay_buffer, batch_size=replay_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

        if replay_batch_size != batch_size:
            print(f"  Isolation replay batch size capped: new={batch_size}, old={replay_batch_size}")

        return PairedDataLoader(new_loader, replay_loader)

    def compute_main_loss(self, logits, labels, input_ids=None, batch=None):
        # Learn Dataset B through the new path.
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )

    def has_old_forward_loss(self) -> bool:
        return self.uses_preservation

    def compute_old_forward_loss(self, old_logits, batch):
        # Preserve the old function on Dataset A replay samples.
        old_input_ids = batch["old_input_ids"]
        with torch.no_grad():
            teacher_old_logits = self.teacher(old_input_ids)

        T = self.temperature
        preserve_kl = teacher_student_kl(old_logits, teacher_old_logits, T)

        return self.lam * (T ** 2) * preserve_kl


class EWC(CLMethodBase):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Regularization-based CL: penalize movement of each parameter away from its
    pre-CL value, weighted by the diagonal Fisher information estimated on old
    data. Used here as the classic *no-growth* baseline (protect a fixed model),
    answering "why grow instead of just regularize?".

    Note on the grown model: new-weight parameters have ~0 Fisher (they do not
    affect the output at gate=0), so EWC reduces to an old-parameter / gate-only
    penalty — the sparse-Fisher subsumption the theory predicts.
    """

    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        self.lam = config.ewc_lambda
        self.n_fisher_batches = config.ewc_fisher_batches
        self.fisher = None       # {name: diagonal Fisher}
        self.ref_params = None   # {name: theta* (pre-CL reference)}

    def build_dataloader(self, new_dataset, old_dataset, batch_size, num_workers):
        # Estimate the Fisher + snapshot reference params on OLD data, once,
        # before returning the new-data loader for CL training.
        self._estimate_fisher(old_dataset, batch_size, num_workers)
        return DataLoader(
            new_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    def _estimate_fisher(self, old_dataset, batch_size, num_workers):
        model = self.model
        was_training = model.training
        model.eval()
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.ref_params = {n: p.detach().clone() for n, p in params}
        fisher = {n: torch.zeros_like(p) for n, p in params}

        loader = DataLoader(
            old_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        n = 0
        for x, y in loader:
            if n >= self.n_fisher_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            # empirical Fisher: CE on the data labels, accumulate grad^2
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            for name, p in params:
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
            n += 1
        for name in fisher:
            fisher[name] /= max(n, 1)
        self.fisher = fisher
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        print(f"EWC: estimated diagonal Fisher over {n} old-data batches "
              f"(lambda={self.lam})")

    def compute_loss(self, logits, labels, input_ids=None, batch=None):
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        penalty = logits.new_zeros(())
        for name, p in self.model.named_parameters():
            if self.fisher is not None and name in self.fisher and p.requires_grad:
                penalty = penalty + (
                    self.fisher[name] * (p - self.ref_params[name]) ** 2
                ).sum()
        return ce_loss + 0.5 * self.lam * penalty


def create_cl_method(config: CLConfig, model, device,
                     teacher_model=None) -> CLMethodBase:
    """Create the appropriate CL method from config."""
    method_map = {
        CLMethod.REPLAY: Replay,
        CLMethod.EWC: EWC,
    }

    if config.method == CLMethod.DISTILLATION:
        return Distillation(config, model, device, teacher_model)
    elif config.method == CLMethod.ISOLATION:
        return Isolation(config, model, device, teacher_model)
    elif config.method in method_map:
        return method_map[config.method](config, model, device)
    else:
        raise ValueError(f"Unknown CL method: {config.method}")

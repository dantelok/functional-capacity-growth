"""
GrowableTransformer — GPT-style decoder-only Transformer with gates at 3 levels.

Gate levels:
  1. Per-head gates (attention)   — head_gate in R^H
  2. Per-hidden-unit gates (FFN)  — ffn_gate in R^d_ffn
  3. Per-block gates (layer)      — block_gate in R^1

Growth operations:
  - Depth: insert new blocks with block_gate=0 (exact FP via residual)
  - Width not used in main experiments (RMSNorm issue)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .components import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


class GatedMultiHeadAttention(nn.Module):
    """Multi-head attention with per-head learnable gates."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-head gate: gate[h] scales head h's output
        self.head_gate = nn.Parameter(torch.ones(n_heads))

        self.rotary = RotaryEmbedding(self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, H, T, D]

        # Apply per-head gates: gate[h] * head_h
        out = out * self.head_gate.view(1, H, 1, 1)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class GatedFFN(nn.Module):
    """SwiGLU FFN with per-hidden-unit learnable gates for growth."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)  # SwiGLU gate
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

        # Per-hidden-unit gate for growth (NOT the SwiGLU gate)
        self.ffn_gate = nn.Parameter(torch.ones(d_ffn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        hidden = hidden * self.ffn_gate  # growth gate
        return self.down_proj(hidden)


class GatedTransformerBlock(nn.Module):
    """Transformer block with block-level gate for depth growth."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int,
                 max_seq_len: int = 1024):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GatedMultiHeadAttention(d_model, n_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = GatedFFN(d_model, d_ffn)

        # Block-level gate (scalar): when 0, block is identity via residual
        self.block_gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.block_gate * self.attn(self.attn_norm(x), mask)
        x = x + self.block_gate * self.ffn(self.ffn_norm(x))
        return x


class GrowableTransformer(nn.Module):
    """
    Decoder-only causal LM with learnable gates at all three levels.

    Base config (~303M params):
        d_model=1024, n_heads=16, d_ffn=4096, n_layers=12

    After depth growth (g=4, ~1.14B params):
        n_layers=48, everything else unchanged
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding (tied with LM head)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GatedTransformerBlock(
                config.d_model, config.n_heads, config.d_ffn, config.max_seq_len
            )
            for _ in range(config.n_layers)
        ])

        # Final norm + LM head
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embedding weights
        self.lm_head.weight = self.token_emb.weight

        # Causal attention mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

        # Track original block count for CL isolation
        self.n_original_blocks = config.n_layers

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        mask = self.causal_mask[:, :, :T, :T]
        use_ckpt = self.training and getattr(self, "activation_checkpointing", False)
        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gate_params(self):
        """Return all gate parameters (for separate optimizer group)."""
        gate_params = []
        for block in self.blocks:
            gate_params.append(block.block_gate)
            gate_params.append(block.attn.head_gate)
            gate_params.append(block.ffn.ffn_gate)
        return gate_params

    def get_non_gate_params(self):
        """Return all non-gate parameters."""
        gate_ids = {id(p) for p in self.get_gate_params()}
        return [p for p in self.parameters() if id(p) not in gate_ids]

    def get_gate_stats(self) -> dict:
        """Return gate statistics for logging."""
        stats = {}
        for i, block in enumerate(self.blocks):
            stats[f"block_{i}/block_gate"] = block.block_gate.item()
            stats[f"block_{i}/head_gate_mean"] = block.attn.head_gate.mean().item()
            stats[f"block_{i}/head_gate_min"] = block.attn.head_gate.min().item()
            stats[f"block_{i}/ffn_gate_mean"] = block.ffn.ffn_gate.mean().item()
        return stats

    def get_original_params(self):
        """Parameters belonging to original (pre-growth) blocks."""
        params = []
        # Embedding + final norm + LM head are always 'original'
        params.extend(self.token_emb.parameters())
        params.extend(self.final_norm.parameters())
        # LM head weight is tied to embedding, skip to avoid double count
        for i in range(self.n_original_blocks):
            params.extend(self.blocks[i].parameters())
        return params

    def get_new_params(self):
        """Parameters belonging to newly grown blocks."""
        params = []
        for i in range(self.n_original_blocks, len(self.blocks)):
            params.extend(self.blocks[i].parameters())
        return params

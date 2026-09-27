"""
Dataset loading for WikiText-103 (Dataset A) and BookCorpus (Dataset B).
Includes replay buffer and combined data loader for CL experiments.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path


class TokenizedLMDataset(Dataset):
    """Pre-tokenized language modeling dataset stored as memory-mapped tensor."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        # Number of complete sequences
        self.n_sequences = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end].long()
        y = self.tokens[start + 1:end + 1].long()
        return x, y


def load_and_tokenize(dataset_name, split, tokenizer, cache_dir="data_cache"):
    """Load a HuggingFace dataset, tokenize, and cache as a flat tensor."""
    cache_path = Path(cache_dir) / f"{dataset_name}_{split}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached tokens from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    from datasets import load_dataset

    print(f"Loading {dataset_name}/{split} from HuggingFace...")

    if dataset_name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        text_field = "text"
    elif dataset_name == "bookcorpus":
        ds = load_dataset("Saibo-creator/bookcorpus_deduplicated", split=split)
        text_field = "text"
    elif dataset_name == "pg19":
        ds = load_dataset("pg19", split=split)
        text_field = "text"
    elif dataset_name == "code":
        # Third CL domain (E5 three-domain sequence A->B->C). codeparrot-clean
        # is parquet on the Hub; the 'content' field holds source files. Only a
        # 'train' split exists, so (like bookcorpus) build_datasets carves
        # validation off the tail.
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split=split)
        text_field = "content"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Tokenize in batches
    print(f"Tokenizing {len(ds)} examples...")
    all_tokens = []
    batch_size = 1000
    for i in range(0, len(ds), batch_size):
        batch_texts = ds[i:i + batch_size][text_field]
        # Filter empty strings
        batch_texts = [t for t in batch_texts if t.strip()]
        if batch_texts:
            encoded = tokenizer(
                batch_texts,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            for ids in encoded["input_ids"]:
                all_tokens.extend(ids)

    tokens = torch.tensor(all_tokens, dtype=torch.int32)
    print(f"Total tokens: {len(tokens):,}")

    torch.save(tokens, cache_path)
    print(f"Cached to {cache_path}")
    return tokens


def build_datasets(dataset_name, tokenizer, seq_len, cache_dir="data_cache",
                   max_tokens=None):
    """Build train and validation datasets."""
    train_tokens = load_and_tokenize(dataset_name, "train", tokenizer, cache_dir)
    if dataset_name == "wikitext":
        val_tokens = load_and_tokenize(dataset_name, "validation", tokenizer, cache_dir)
    else:
        # BookCorpus / PG19: split off last 500K tokens as validation
        val_tokens = train_tokens[-500_000:]
        train_tokens = train_tokens[:-500_000]

    # Optionally cap token count (for matching dataset sizes)
    if max_tokens is not None and len(train_tokens) > max_tokens:
        train_tokens = train_tokens[:max_tokens]

    train_ds = TokenizedLMDataset(train_tokens, seq_len)
    val_ds = TokenizedLMDataset(val_tokens, seq_len)

    print(f"  {dataset_name} train: {len(train_ds)} sequences "
          f"({len(train_tokens):,} tokens)")
    print(f"  {dataset_name} val:   {len(val_ds)} sequences "
          f"({len(val_tokens):,} tokens)")
    return train_ds, val_ds


class ReplayBuffer:
    """Stores a subset of Dataset A for replay during CL training."""

    def __init__(self, dataset: TokenizedLMDataset, buffer_fraction: float = 0.1):
        n_buffer = int(len(dataset) * buffer_fraction)
        indices = np.random.choice(len(dataset), n_buffer, replace=False)
        self.subset = Subset(dataset, indices)
        print(f"Replay buffer: {len(self.subset)} sequences "
              f"({buffer_fraction:.0%} of training set)")

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


class CombinedDataLoader:
    """
    Interleave new-domain and replay batches so a fraction ``alpha`` of the
    microbatches are pure replay (old data) and the rest are pure new data.

    Mixing is at the *microbatch* granularity, not within a microbatch, so the
    effective replay fraction is exactly ``alpha`` regardless of batch size.
    The previous within-microbatch split forced alpha up to 0.5 at
    ``batch_size=2`` (``int(2 * 0.1) = 0`` rounded up to >=1 replay sample),
    which made small replay ratios unrealisable on the large model. With
    gradient accumulation, ~``alpha`` of the accumulated gradient is old data.
    """

    def __init__(self, new_loader, replay_loader, alpha=0.1):
        self.new_loader = new_loader
        self.replay_loader = replay_loader
        self.alpha = alpha

    def __iter__(self):
        # Every k-th microbatch is a pure-replay microbatch -> replay fraction 1/k.
        k = round(1.0 / self.alpha) if self.alpha > 0 else 0
        replay_iter = iter(self.replay_loader)
        for step, new_batch in enumerate(self.new_loader, start=1):
            if k and step % k == 0:
                try:
                    yield next(replay_iter)
                except StopIteration:               # cycle the (smaller) replay buffer
                    replay_iter = iter(self.replay_loader)
                    yield next(replay_iter)
            else:
                yield new_batch

    def __len__(self):
        return len(self.new_loader)


class PairedDataLoader:
    """
    Yields a fresh Dataset B batch together with a replay batch from Dataset A.

    Unlike CombinedDataLoader, this keeps the two objectives separate so methods
    can form an exact weighted loss:
      L = (1-r) * L_B + r * L_A / KL_A
    regardless of microbatch size.
    """

    def __init__(self, new_loader, replay_loader):
        self.new_loader = new_loader
        self.replay_loader = replay_loader

    def __iter__(self):
        new_iter = iter(self.new_loader)
        replay_iter = iter(self.replay_loader)

        for new_x, new_y in new_iter:
            try:
                old_x, old_y = next(replay_iter)
            except StopIteration:
                replay_iter = iter(self.replay_loader)
                old_x, old_y = next(replay_iter)

            yield {
                "input_ids": new_x,
                "labels": new_y,
                "old_input_ids": old_x,
                "old_labels": old_y,
            }

    def __len__(self):
        return len(self.new_loader)


def get_tokenizer():
    """Load GPT-2 tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

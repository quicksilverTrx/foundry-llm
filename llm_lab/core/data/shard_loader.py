# llm_lab/core/data/shard_loader.py
"""
ShardLoader — raw .npy shard data loader for FineWeb-Edu style pre-tokenized datasets.

Compatible with both direct next_batch() usage (step loop)
and torch.utils.data.IterableDataset (for use with existing Trainer / DataLoader).

Data format expected:
  <data_dir>/
    edufineweb_train_000001.npy  ... (uint16 token arrays, 100M tokens each)
    edufineweb_val_000000.npy

File pattern is configurable so the loader works with any naming convention
that follows glob-matchable shard files.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ShardLoader:
    """
    Stateful sequential loader for pre-tokenized .npy shard files.

    Loads one shard at a time into CPU RAM, yields (x, y) batches of shape
    [B, T] and [B, T] respectively.  When a shard is exhausted the next shard
    (wrapping around) is loaded automatically.

    Args:
        data_dir:    Directory containing .npy shard files.
        split:       "train" or "val".
        B:           Micro-batch size (sequences per batch).
        T:           Sequence length (tokens per sequence).
        file_pattern:  Glob pattern inside data_dir.  ``{split}`` is replaced
                     with the value of ``split``.  Defaults to the FineWeb-Edu
                     naming convention used for FineWeb-Edu pretraining.
        device:      Target device for returned tensors ("cpu", "cuda", "mps").
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        B: int,
        T: int,
        file_pattern: str = "edufineweb_{split}_*.npy",
        device: str = "cpu",
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.B = B
        self.T = T
        self.device = device

        pattern = os.path.join(data_dir, file_pattern.replace("{split}", split))
        shards = sorted(glob.glob(pattern))
        if not shards:
            raise FileNotFoundError(
                f"No shards found for split={split!r} in {data_dir!r}\n"
                f"Pattern tried: {pattern}"
            )
        self.shards = shards
        self.reset()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Rewind to shard 0, position 0.  Call before every val eval pass."""
        self.shard_idx = 0
        self._load_shard(self.shard_idx)
        self.pos = 0

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the next (x, y) micro-batch.

        x, y: torch.long tensors of shape [B, T] on self.device.
        Automatically advances to the next shard when the current one
        is exhausted.
        """
        need = self.B * self.T + 1
        buf = self.tokens[self.pos : self.pos + need]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.pos += self.B * self.T

        # advance shard if needed
        if self.pos + need > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shards)
            self._load_shard(self.shard_idx)
            self.pos = 0

        return x.to(self.device), y.to(self.device)

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    @property
    def tokens_per_shard(self) -> int:
        """Token count in the *currently loaded* shard."""
        return len(self.tokens)

    # ── IterableDataset wrapper ───────────────────────────────────────────────

    def as_iterable_dataset(self) -> "ShardIterableDataset":
        """
        Wrap this loader as a torch.utils.data.IterableDataset so it can be
        passed to an existing DataLoader / Trainer.

        Note: the dataset yields individual (x, y) pairs (not batches); set
        DataLoader batch_size=None (or batch_size=1 + squeeze) when using this.
        For most shard-training use-cases the ShardTrainer is simpler.
        """
        return ShardIterableDataset(self)

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_shard(self, idx: int) -> None:
        data = np.load(self.shards[idx])
        # Cast uint16 → int32 then to torch.long (int64); avoids overflow on
        # vocab sizes > 32767 while staying memory-efficient.
        self.tokens = torch.tensor(data.astype(np.int32), dtype=torch.long)

    def __repr__(self) -> str:
        return (
            f"ShardLoader(split={self.split!r}, B={self.B}, T={self.T}, "
            f"n_shards={self.num_shards}, device={self.device!r})"
        )


class ShardIterableDataset(IterableDataset):
    """
    torch.utils.data.IterableDataset wrapper around ShardLoader.

    Yields individual (x_row, y_row) tensor pairs of shape [T] each,
    suitable for use with DataLoader(batch_size=B, ...).

    Each epoch resets the loader to shard 0 / position 0.
    """

    def __init__(self, loader: ShardLoader) -> None:
        super().__init__()
        self._loader = loader

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        self._loader.reset()
        # Yield rows rather than batches so DataLoader handles batching.
        while True:
            x_batch, y_batch = self._loader.next_batch()   # [B, T]
            for i in range(x_batch.shape[0]):
                yield x_batch[i], y_batch[i]               # [T], [T]

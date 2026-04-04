# tests/core/test_shard_loader.py
"""
Tests for llm_lab.core.data.shard_loader.ShardLoader and ShardIterableDataset.

Uses synthetic in-memory shards (numpy .npy files) so no real FineWeb data needed.
"""
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from llm_lab.core.data.shard_loader import ShardIterableDataset, ShardLoader


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_fake_shards(tmpdir: str, split: str, n_shards: int, tokens_per_shard: int) -> None:
    """Write n_shards .npy files with sequential uint16 tokens."""
    for i in range(n_shards):
        fname = os.path.join(tmpdir, f"edufineweb_{split}_{i:06d}.npy")
        # sequential token IDs 0..tokens_per_shard-1 wrapped into uint16
        data = np.arange(tokens_per_shard, dtype=np.uint16)
        np.save(fname, data)


# ── ShardLoader tests ─────────────────────────────────────────────────────────

def test_shardloader_basic():
    """ShardLoader returns (x, y) tensors of correct shape."""
    B, T = 4, 8
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "train", n_shards=2, tokens_per_shard=500)
        loader = ShardLoader(tmpdir, "train", B=B, T=T, device="cpu")
        x, y = loader.next_batch()
        assert x.shape == (B, T), f"expected ({B}, {T}), got {x.shape}"
        assert y.shape == (B, T), f"expected ({B}, {T}), got {y.shape}"
        assert x.dtype == torch.long
        assert y.dtype == torch.long


def test_shardloader_y_is_x_shifted():
    """y must be x shifted right by one position (language-model targets)."""
    B, T = 2, 4
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "train", n_shards=1, tokens_per_shard=200)
        loader = ShardLoader(tmpdir, "train", B=B, T=T, device="cpu")
        x, y = loader.next_batch()
        # x[i, j] + 1 == y[i, j] for sequential data
        assert torch.all(y == x + 1), "y should equal x + 1 for sequential token data"


def test_shardloader_reset_returns_same_batch():
    """After reset(), next_batch() returns the same batch as the first call."""
    B, T = 2, 4
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "train", n_shards=2, tokens_per_shard=200)
        loader = ShardLoader(tmpdir, "train", B=B, T=T, device="cpu")
        x1, y1 = loader.next_batch()
        loader.next_batch()        # advance
        loader.reset()
        x2, y2 = loader.next_batch()
        assert torch.equal(x1, x2), "reset should bring back same first batch"
        assert torch.equal(y1, y2)


def test_shardloader_shard_wrap():
    """Loader wraps around to shard 0 when it exhausts all shards."""
    B, T = 2, 4
    tokens_per_shard = B * T + 1   # barely enough for one batch per shard
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "train", n_shards=3, tokens_per_shard=tokens_per_shard)
        loader = ShardLoader(tmpdir, "train", B=B, T=T, device="cpu")
        # exhaust all shards; 4th call must not raise
        for _ in range(4):
            loader.next_batch()


def test_shardloader_missing_dir_raises():
    """FileNotFoundError raised when no shards match the pattern."""
    with pytest.raises(FileNotFoundError):
        ShardLoader("/nonexistent/path", "train", B=2, T=4, device="cpu")


def test_shardloader_num_shards():
    """num_shards property reflects the glob result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "val", n_shards=3, tokens_per_shard=200)
        loader = ShardLoader(tmpdir, "val", B=2, T=4, device="cpu")
        assert loader.num_shards == 3


def test_shardloader_repr():
    """__repr__ doesn't crash and contains key fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "train", n_shards=1, tokens_per_shard=100)
        loader = ShardLoader(tmpdir, "train", B=4, T=8, device="cpu")
        r = repr(loader)
        assert "train" in r
        assert "B=4" in r
        assert "T=8" in r


# ── ShardIterableDataset tests ────────────────────────────────────────────────

def test_shard_iterable_dataset_shapes():
    """ShardIterableDataset yields [T] tensors (rows, not batches)."""
    B, T = 2, 4
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "train", n_shards=1, tokens_per_shard=200)
        loader = ShardLoader(tmpdir, "train", B=B, T=T, device="cpu")
        ds = ShardIterableDataset(loader)
        it = iter(ds)
        x_row, y_row = next(it)
        assert x_row.shape == (T,), f"expected ({T},), got {x_row.shape}"
        assert y_row.shape == (T,), f"expected ({T},), got {y_row.shape}"


def test_shard_iterable_dataset_dataloader_batching():
    """DataLoader(batch_size=B) reassembles ShardIterableDataset rows into [B, T]."""
    B, T = 4, 8
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_fake_shards(tmpdir, "train", n_shards=2, tokens_per_shard=1000)
        loader = ShardLoader(tmpdir, "train", B=B, T=T, device="cpu")
        ds = ShardIterableDataset(loader)
        dl = DataLoader(ds, batch_size=B)
        x_batch, y_batch = next(iter(dl))
        assert x_batch.shape == (B, T)
        assert y_batch.shape == (B, T)

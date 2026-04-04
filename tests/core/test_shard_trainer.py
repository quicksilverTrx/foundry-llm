# tests/core/test_shard_trainer.py
"""
Tests for llm_lab.core.train.shard_trainer.ShardTrainer and ShardTrainerConfig.

Uses a tiny model and synthetic shards so tests run fast on CPU.
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.shard_trainer import ShardTrainer, ShardTrainerConfig


# ── helpers ───────────────────────────────────────────────────────────────────

VOCAB = 256   # small vocab for test model

def _tiny_model_cfg():
    return MiniGPTConfig(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
        d_ff=64, block_size=16, dropout=0.0,
        norm_type="rmsnorm", mlp_type="swiglu",
        attention_type="gqa", num_kv_heads=1,
        pos_encoding_type="rope", arch_family="nanollama",
    )


def _write_shards(tmpdir: str, split: str, n_shards: int = 3,
                  tokens_per_shard: int = 2048) -> None:
    for i in range(n_shards):
        fname = os.path.join(tmpdir, f"edufineweb_{split}_{i:06d}.npy")
        data = np.random.randint(0, VOCAB, size=tokens_per_shard, dtype=np.uint16)
        np.save(fname, data)


def _make_trainer(tmpdir: str, out_dir: str, max_steps: int = 5) -> ShardTrainer:
    cfg = ShardTrainerConfig(
        out_dir=out_dir,
        data_dir=tmpdir,
        B=2, T=8, grad_accum=2,
        max_lr=1e-3, min_lr=1e-4,
        warmup_steps=2, max_steps=max_steps,
        grad_clip=1.0, weight_decay=0.1,
        val_every=3, val_steps=2,
        ckpt_every=3, log_every=2,
        device="cpu",
    )
    model_cfg = _tiny_model_cfg()
    model = MiniGPT(model_cfg)
    return ShardTrainer(model, model_cfg.__dict__, cfg)


# ── config tests ──────────────────────────────────────────────────────────────

def test_config_total_batch():
    """total_batch property equals B * T * grad_accum."""
    cfg = ShardTrainerConfig(B=4, T=16, grad_accum=8)
    assert cfg.total_batch == 4 * 16 * 8


def test_config_total_tokens():
    """total_tokens = total_batch * max_steps."""
    cfg = ShardTrainerConfig(B=4, T=16, grad_accum=8, max_steps=100)
    assert cfg.total_tokens == cfg.total_batch * 100


# ── ShardTrainer.fit() smoke tests ────────────────────────────────────────────

def test_fit_completes_without_error():
    """fit() should complete 5 steps without raising."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_shards(data_dir, "train")
        _write_shards(data_dir, "val")
        trainer = _make_trainer(data_dir, out_dir, max_steps=5)
        trainer.fit()   # should not raise


def test_trajectory_csv_created():
    """fit() should write a trajectory.csv with correct header."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_shards(data_dir, "train")
        _write_shards(data_dir, "val")
        trainer = _make_trainer(data_dir, out_dir, max_steps=4)
        trainer.fit()
        csv_path = os.path.join(out_dir, "trajectory.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "step,train_loss,lr,grad_norm,tok_per_sec,val_loss"


def test_config_json_created():
    """fit() should write a config.json."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_shards(data_dir, "train")
        _write_shards(data_dir, "val")
        trainer = _make_trainer(data_dir, out_dir, max_steps=3)
        trainer.fit()
        import json
        config_path = os.path.join(out_dir, "config.json")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert "B" in config
        assert "max_steps" in config


def test_checkpoint_saved():
    """Checkpoint file should exist after ckpt_every steps."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_shards(data_dir, "train")
        _write_shards(data_dir, "val")
        trainer = _make_trainer(data_dir, out_dir, max_steps=3)
        trainer.fit()
        ckpts_dir = os.path.join(out_dir, "ckpts")
        ckpts = os.listdir(ckpts_dir)
        assert len(ckpts) > 0, "expected at least one checkpoint file"


def _write_structured_shards(tmpdir: str, split: str, n_shards: int = 2,
                              tokens_per_shard: int = 4096) -> None:
    """Write shards with a fixed repeating pattern so the model can overfit."""
    rng = np.random.RandomState(42)
    # short repeating pattern: same 32-token sequence tiled
    pattern = rng.randint(0, VOCAB, size=32, dtype=np.uint16)
    for i in range(n_shards):
        fname = os.path.join(tmpdir, f"edufineweb_{split}_{i:06d}.npy")
        data = np.tile(pattern, tokens_per_shard // len(pattern) + 1)[:tokens_per_shard]
        np.save(fname, data.astype(np.uint16))


def test_loss_decreases_over_steps():
    """Train loss should decrease when overfitting a tiny repeating sequence."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_structured_shards(data_dir, "train")
        _write_structured_shards(data_dir, "val")
        import csv as csv_mod
        cfg = ShardTrainerConfig(
            out_dir=out_dir, data_dir=data_dir,
            B=2, T=8, grad_accum=1,
            max_lr=5e-3, min_lr=5e-4,
            warmup_steps=2, max_steps=60,
            grad_clip=1.0, weight_decay=0.0,
            val_every=100, val_steps=2,   # skip val to speed up
            ckpt_every=100, log_every=20,
            device="cpu",
        )
        model_cfg = _tiny_model_cfg()
        torch.manual_seed(0)
        model = MiniGPT(model_cfg)
        trainer = ShardTrainer(model, model_cfg.__dict__, cfg)
        trainer.fit()

        csv_path = os.path.join(out_dir, "trajectory.csv")
        with open(csv_path) as f:
            rows = list(csv_mod.DictReader(f))
        losses = [float(r["train_loss"]) for r in rows if r["train_loss"]]
        # compare average of first 5 steps vs last 5 steps
        avg_early = sum(losses[:5]) / 5
        avg_late  = sum(losses[-5:]) / 5
        assert avg_late < avg_early, (
            f"Expected loss to decrease: early_avg={avg_early:.4f} late_avg={avg_late:.4f}"
        )


# ── checkpoint save/load tests ────────────────────────────────────────────────

def test_save_and_load_checkpoint():
    """save_checkpoint() then load_checkpoint() restores model + optimizer state."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_shards(data_dir, "train")
        _write_shards(data_dir, "val")
        trainer = _make_trainer(data_dir, out_dir, max_steps=3)
        trainer.fit()

        ckpt_path = os.path.join(out_dir, "ckpts", "step_00003.pt")
        assert os.path.exists(ckpt_path)

        # Load into a fresh trainer
        trainer2 = _make_trainer(data_dir, out_dir + "_resumed", max_steps=3)
        step = trainer2.load_checkpoint(ckpt_path)
        assert step == 3


def test_save_model_only_smaller_than_full():
    """Model-only checkpoint must be smaller than full checkpoint."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_shards(data_dir, "train")
        _write_shards(data_dir, "val")
        trainer = _make_trainer(data_dir, out_dir, max_steps=3)
        trainer.fit()

        full_path  = os.path.join(out_dir, "full.pt")
        slim_path  = os.path.join(out_dir, "slim.pt")
        trainer.save_checkpoint(full_path, step=3)
        trainer.save_model_only(slim_path, step=3)
        assert os.path.getsize(slim_path) < os.path.getsize(full_path)


def test_checkpoint_contains_config():
    """Full checkpoint must contain the model config dict."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as out_dir:
        _write_shards(data_dir, "train")
        _write_shards(data_dir, "val")
        trainer = _make_trainer(data_dir, out_dir, max_steps=2)
        ckpt_path = os.path.join(out_dir, "test.pt")
        trainer.save_checkpoint(ckpt_path, step=2)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert "config" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "step" in ckpt

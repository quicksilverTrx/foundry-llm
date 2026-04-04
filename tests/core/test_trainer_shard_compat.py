# tests/core/test_trainer_shard_compat.py
"""
Compatibility tests: Trainer + ShardIterableDataset.

Verifies that the five targeted fixes to Trainer enable it to accept
IterableDataset loaders with LR schedule behaviour identical to ShardTrainer.

All tests run on CPU with a tiny model (2 layers, d_model=32) so no GPU
is required and CI stays fast (<30 s total).
"""
from __future__ import annotations

import math
import tempfile
from typing import Iterator, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.lr_schedule import cosine_with_warmup
from llm_lab.core.train.trainer import Trainer, TrainerConfig


# ── shared fixtures ───────────────────────────────────────────────────────────

VOCAB = 64
BLOCK = 8
B     = 2    # batch size used by DataLoader


def _tiny_model() -> MiniGPT:
    cfg = MiniGPTConfig(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
        d_ff=64, block_size=BLOCK, dropout=0.0,
    )
    return MiniGPT(cfg)


class _InfiniteIterableDataset(IterableDataset):
    """Yields an infinite stream of (x, y) token rows of shape [BLOCK]."""

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            x = torch.randint(0, VOCAB, (BLOCK,))
            y = torch.randint(0, VOCAB, (BLOCK,))
            yield x, y


class _FiniteIterableDataset(IterableDataset):
    """Yields exactly n_items (x, y) pairs before stopping."""

    def __init__(self, n_items: int) -> None:
        self._n = n_items

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(self._n):
            x = torch.randint(0, VOCAB, (BLOCK,))
            y = torch.randint(0, VOCAB, (BLOCK,))
            yield x, y


def _make_loader(ds: IterableDataset) -> DataLoader:
    return DataLoader(ds, batch_size=B)


def _base_cfg(**overrides) -> TrainerConfig:
    defaults = dict(
        device="cpu",
        dtype="fp32",
        lr=1e-3,
        lr_min=1e-4,
        lr_warmup_steps=2,
        max_steps=5,
        lr_schedule_type="cosine",
        grad_accum_steps=1,
        log_every_n_steps=1,
        eval_every_n_steps=3,
        eval_max_batches=2,
        num_epochs=1,
    )
    defaults.update(overrides)
    return TrainerConfig(**defaults)


# ── test 1: Trainer accepts IterableDataset without error ─────────────────────

def test_iterable_dataset_accepted():
    """Trainer.train_epoch should complete without raising on IterableDataset."""
    model = _tiny_model()
    train_dl = _make_loader(_InfiniteIterableDataset())
    val_dl   = _make_loader(_FiniteIterableDataset(n_items=B * 2))
    cfg = _base_cfg(max_steps=3, eval_max_batches=2)
    trainer = Trainer(model, train_dl, val_dl, cfg)
    # Should not raise
    trainer.train_epoch(epoch_index=0)


# ── test 2: loss decreases over N steps with IterableDataset ─────────────────

def test_loss_decreases_with_iterable_dataset():
    """After several steps, train loss should trend downward (overfit a tiny dataset)."""
    torch.manual_seed(0)
    model = _tiny_model()
    # Use the same finite dataset repeatedly so the model can overfit
    train_ds = _FiniteIterableDataset(n_items=B * 20)
    train_dl = _make_loader(train_ds)
    cfg = _base_cfg(max_steps=10, lr=5e-3, eval_every_n_steps=999)

    trainer = Trainer(model, train_dl, None, cfg)
    # Two epochs; loss in second should be lower than first
    loss1 = trainer.train_epoch(epoch_index=0)
    # Reset dataset iterator for second epoch
    train_dl2 = _make_loader(_FiniteIterableDataset(n_items=B * 20))
    trainer.train_loader = train_dl2
    loss2 = trainer.train_epoch(epoch_index=1)
    assert loss2 < loss1, f"Expected loss to decrease: {loss1:.4f} → {loss2:.4f}"


# ── test 3: checkpoint round-trip restores global_step ────────────────────────

def test_checkpoint_resume_restores_global_step():
    """save_checkpoint + load_checkpoint must preserve global_step."""
    torch.manual_seed(1)
    model = _tiny_model()
    train_dl = _make_loader(_InfiniteIterableDataset())
    cfg = _base_cfg(max_steps=4, eval_every_n_steps=999)

    trainer = Trainer(model, train_dl, None, cfg)
    trainer.train_epoch(epoch_index=0)
    step_after_training = trainer.global_step
    assert step_after_training > 0, "global_step should have advanced"

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        trainer.save_checkpoint(f.name)

        # Fresh trainer, load checkpoint
        model2  = _tiny_model()
        train_dl2 = _make_loader(_InfiniteIterableDataset())
        trainer2 = Trainer(model2, train_dl2, None, cfg)
        assert trainer2.global_step == 0
        trainer2.load_checkpoint(f.name)
        assert trainer2.global_step == step_after_training, (
            f"global_step not restored: expected {step_after_training}, "
            f"got {trainer2.global_step}"
        )


# ── test 4: cosine LR matches cosine_with_warmup (ShardTrainer parity) ────────

def test_cosine_lr_matches_cosine_with_warmup():
    """
    _lr_for_step(step) must be bit-for-bit equal to cosine_with_warmup(step, ...)
    so that Trainer and ShardTrainer produce identical LR sequences.
    """
    WARMUP, MAX_STEPS, MAX_LR, MIN_LR = 10, 50, 6e-4, 6e-5

    model = _tiny_model()
    train_dl = _make_loader(_InfiniteIterableDataset())
    cfg = TrainerConfig(
        device="cpu", lr=MAX_LR, lr_min=MIN_LR,
        lr_warmup_steps=WARMUP, max_steps=MAX_STEPS,
        lr_schedule_type="cosine",
    )
    trainer = Trainer(model, train_dl, None, cfg)

    probe_steps = [0, 1, WARMUP - 1, WARMUP, WARMUP + 1, MAX_STEPS - 1, MAX_STEPS, MAX_STEPS + 5]
    for step in probe_steps:
        got      = trainer._lr_for_step(step)
        expected = cosine_with_warmup(
            step, warmup_steps=WARMUP, max_steps=MAX_STEPS,
            max_lr=MAX_LR, min_lr=MIN_LR,
        )
        assert math.isclose(got, expected, rel_tol=1e-9), (
            f"LR mismatch at step={step}: Trainer={got}, cosine_with_warmup={expected}"
        )


# ── test 5: missing eval_max_batches raises for IterableDataset val loader ────

def test_iterable_val_without_cap_raises():
    """
    evaluate() must raise ValueError if val_loader wraps an IterableDataset
    and neither eval_max_batches nor val_steps is set.
    """
    model = _tiny_model()
    train_dl = _make_loader(_InfiniteIterableDataset())
    val_dl   = _make_loader(_InfiniteIterableDataset())

    cfg = TrainerConfig(
        device="cpu", lr=1e-3, max_steps=2,
        lr_schedule_type="constant",
        # eval_max_batches and val_steps intentionally omitted
    )
    trainer = Trainer(model, train_dl, val_dl, cfg)
    with pytest.raises(ValueError, match="eval_max_batches"):
        trainer.evaluate(epoch_index=0)

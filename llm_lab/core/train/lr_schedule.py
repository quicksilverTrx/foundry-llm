# llm_lab/core/train/lr_schedule.py
"""
Learning-rate schedule functions for pretraining.

Currently provides:
  cosine_with_warmup — linear warmup followed by cosine decay.

This is the canonical implementation used by Phase 5 swaps and Phase 6
NanoLlama pretraining.  Previously it was duplicated inline in run_swap.py
and run_phase6.py; it is now a single importable function.
"""
from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine LR schedule with linear warmup.

    Regime breakdown:
      [0, warmup_steps)     : linear ramp from 0 → max_lr
      [warmup_steps, max_steps] : cosine decay from max_lr → min_lr
      (max_steps, ...)      : constant min_lr

    Args:
        step:         Current optimiser step (0-indexed).
        warmup_steps: Number of warm-up steps (200 for Phase 6).
        max_steps:    Total training steps (4768 for Phase 6).
        max_lr:       Peak learning rate (6e-4 for Phase 6).
        min_lr:       Floor learning rate (6e-5 = 0.1 × max_lr for Phase 6).

    Returns:
        Learning rate (float) for the current step.

    Examples:
        >>> cosine_with_warmup(0, warmup_steps=200, max_steps=4768,
        ...                    max_lr=6e-4, min_lr=6e-5)
        3e-06
        >>> cosine_with_warmup(200, warmup_steps=200, max_steps=4768,
        ...                    max_lr=6e-4, min_lr=6e-5)
        0.0006
        >>> cosine_with_warmup(5000, warmup_steps=200, max_steps=4768,
        ...                    max_lr=6e-4, min_lr=6e-5)
        6e-05
    """
    if step < warmup_steps:
        # linear warm-up: at step=0 returns max_lr/warmup_steps (not 0)
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # cosine decay
    decay_steps = max_steps - warmup_steps
    ratio = (step - warmup_steps) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)

# llm_lab/core/train/lr_schedule.py
"""
Learning-rate schedule functions for pretraining.

Provides:
  cosine_with_warmup  — linear warmup followed by cosine decay.
                        Used by NanoLlama v1 and SwiftLlama ablation AdamW probes.
  constant_warmdown   — linear warmup, constant plateau, linear warmdown to 0.
                        Used by NanoLlama v2 nanochat Muon recipe.

Canonical implementation used across ablation runs and NanoLlama pretraining.
Previously duplicated inline in experiment scripts; extracted here as a
single importable function.
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
      [0, warmup_steps)       : linear ramp from 0 → max_lr
      [warmup_steps, max_steps] : cosine decay from max_lr → min_lr
      (max_steps, ...)        : constant min_lr

    Args:
        step:         Current optimiser step (0-indexed).
        warmup_steps: Number of warm-up steps.
        max_steps:    Total training steps.
        max_lr:       Peak learning rate.
        min_lr:       Floor learning rate.

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


def constant_warmdown(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
    warmdown_ratio: float = 0.4,
) -> float:
    """
    Constant LR schedule with linear warmup and linear warmdown.

    Regime breakdown:
      [0, warmup_steps)           : linear ramp from 0 → max_lr
      [warmup_steps, warmdown_start) : constant max_lr
      [warmdown_start, max_steps]    : linear decay from max_lr → min_lr
      (max_steps, ...)               : constant min_lr

    warmdown_start = floor(max_steps * (1 - warmdown_ratio))

    Designed for the nanochat Muon recipe.  Called with max_steps equal to
    the *production* schedule length (e.g. 9537 for 5B tokens) even when the
    actual run executes fewer steps (e.g. 1500 for a probe).  This ensures
    probe LRs sit at the correct point in the production schedule.

    At a probe step of 1500 with max_steps=9537 and warmdown_ratio=0.4:
      warmdown_start = floor(9537 * 0.6) = 5722
      step 1500 < 5722  →  constant max_lr  (probe is at peak LR)  ✓

    Args:
        step:            Current optimiser step (0-indexed).
        warmup_steps:    Linear warm-up length.
        max_steps:       Full schedule length for LR computation.
        max_lr:          Peak (plateau) learning rate.
        min_lr:          Floor LR at end of warmdown (default 0.0).
        warmdown_ratio:  Fraction of max_steps dedicated to warmdown (default 0.4).

    Returns:
        Learning rate (float) for the current step.

    Examples:
        >>> constant_warmdown(0, warmup_steps=100, max_steps=9537,
        ...                   max_lr=0.02, warmdown_ratio=0.4)
        0.0002
        >>> constant_warmdown(100, warmup_steps=100, max_steps=9537,
        ...                   max_lr=0.02, warmdown_ratio=0.4)
        0.02
        >>> constant_warmdown(1500, warmup_steps=100, max_steps=9537,
        ...                   max_lr=0.02, warmdown_ratio=0.4)
        0.02
        >>> constant_warmdown(9537, warmup_steps=100, max_steps=9537,
        ...                   max_lr=0.02, warmdown_ratio=0.4)
        0.0
    """
    warmdown_start = int(max_steps * (1.0 - warmdown_ratio))

    if step < warmup_steps:
        # linear ramp: step=0 → max_lr/warmup_steps (not 0)
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    if step >= warmdown_start:
        # linear decay: warmdown_start → max_lr, max_steps → min_lr
        decay_steps = max_steps - warmdown_start
        ratio = (step - warmdown_start) / decay_steps
        return min_lr + (1.0 - ratio) * (max_lr - min_lr)
    # constant plateau
    return max_lr

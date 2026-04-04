# tests/core/test_lr_schedule.py
"""
Tests for llm_lab.core.train.lr_schedule.cosine_with_warmup.
"""
import math
import pytest
from llm_lab.core.train.lr_schedule import cosine_with_warmup

MAX_LR   = 6e-4
MIN_LR   = 6e-5
WARMUP   = 200
MAX_STEPS = 4768


# ── warmup regime ─────────────────────────────────────────────────────────────

def test_warmup_step0_is_not_zero():
    """Step 0 returns max_lr / warmup_steps (first ramp value, not 0)."""
    lr = cosine_with_warmup(0, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                            max_lr=MAX_LR, min_lr=MIN_LR)
    expected = MAX_LR / WARMUP
    assert abs(lr - expected) < 1e-10


def test_warmup_final_step_is_max_lr():
    """At step=warmup_steps-1 the LR should equal max_lr (warmup complete)."""
    lr = cosine_with_warmup(WARMUP - 1, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                            max_lr=MAX_LR, min_lr=MIN_LR)
    assert abs(lr - MAX_LR) < 1e-10


def test_warmup_is_linear():
    """LR during warmup is linear in step."""
    lrs = [
        cosine_with_warmup(s, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                           max_lr=MAX_LR, min_lr=MIN_LR)
        for s in range(WARMUP)
    ]
    diffs = [lrs[i+1] - lrs[i] for i in range(len(lrs)-1)]
    # all increments should be equal (linear ramp)
    assert max(diffs) - min(diffs) < 1e-12


# ── cosine decay regime ───────────────────────────────────────────────────────

def test_at_warmup_boundary_equals_max_lr():
    """At exactly step=warmup_steps, cosine coeff=1 → LR=max_lr."""
    lr = cosine_with_warmup(WARMUP, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                            max_lr=MAX_LR, min_lr=MIN_LR)
    assert abs(lr - MAX_LR) < 1e-10


def test_at_max_steps_equals_min_lr():
    """At step=max_steps the schedule returns min_lr."""
    lr = cosine_with_warmup(MAX_STEPS, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                            max_lr=MAX_LR, min_lr=MIN_LR)
    assert abs(lr - MIN_LR) < 1e-10


def test_midpoint_halfway_between_max_and_min():
    """At the midpoint of decay, cosine coeff=0.5 → LR=(max+min)/2."""
    mid = WARMUP + (MAX_STEPS - WARMUP) // 2
    lr = cosine_with_warmup(mid, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                            max_lr=MAX_LR, min_lr=MIN_LR)
    expected = (MAX_LR + MIN_LR) / 2
    assert abs(lr - expected) < 1e-6


def test_decay_is_monotone_decreasing():
    """LR must never increase during the cosine decay phase."""
    lrs = [
        cosine_with_warmup(s, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                           max_lr=MAX_LR, min_lr=MIN_LR)
        for s in range(WARMUP, MAX_STEPS + 1)
    ]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i+1] - 1e-12, f"LR increased at step {WARMUP+i+1}"


# ── post-max-steps regime ─────────────────────────────────────────────────────

def test_beyond_max_steps_returns_min_lr():
    """Any step > max_steps returns exactly min_lr."""
    for s in [MAX_STEPS + 1, MAX_STEPS + 100, MAX_STEPS * 2]:
        lr = cosine_with_warmup(s, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                                max_lr=MAX_LR, min_lr=MIN_LR)
        assert abs(lr - MIN_LR) < 1e-10, f"step={s}"


# ── edge cases ────────────────────────────────────────────────────────────────

def test_no_warmup():
    """warmup_steps=0 — step 0 should equal max_lr (no warmup)."""
    lr = cosine_with_warmup(0, warmup_steps=0, max_steps=100,
                            max_lr=MAX_LR, min_lr=MIN_LR)
    assert abs(lr - MAX_LR) < 1e-10


def test_all_lrs_in_valid_range():
    """
    Every LR value must be in [0, max_lr].
    Note: warmup steps are allowed to go below min_lr (linear ramp from ~0).
    After warmup, LR decays from max_lr down to min_lr.
    """
    for step in range(0, MAX_STEPS + 10):
        lr = cosine_with_warmup(step, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                                max_lr=MAX_LR, min_lr=MIN_LR)
        assert 0.0 <= lr <= MAX_LR + 1e-10, \
            f"LR={lr} out of range [0, max_lr] at step={step}"
    # After warmup, LR must stay at or above min_lr
    for step in range(WARMUP, MAX_STEPS + 10):
        lr = cosine_with_warmup(step, warmup_steps=WARMUP, max_steps=MAX_STEPS,
                                max_lr=MAX_LR, min_lr=MIN_LR)
        assert MIN_LR - 1e-10 <= lr <= MAX_LR + 1e-10, \
            f"Post-warmup LR={lr} below min_lr at step={step}"

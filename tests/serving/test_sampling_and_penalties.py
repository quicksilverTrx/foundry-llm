# tests/serving/test_sampling_and_penalties.py
from __future__ import annotations

import torch

from llm_lab.serving.sampling import (
    apply_frequency_penalty,
    apply_repetition_penalty,
    apply_temperature,
    sample_next_token_id,
    select_next_token_id,
    top_k_filter,
    top_p_filter,
)


def _support_size(x: torch.Tensor) -> int:
    return int((x > -1e8).sum().item())


def test_seeded_sampling_is_deterministic_cpu():
    logits = torch.tensor([1.0, 1.01, 1.02, 1.03], dtype=torch.float32)
    a = select_next_token_id(
        logits,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        repetition_penalty=None,
        frequency_penalty=None,
        generated_token_ids=[],
        token_counts={},
        seed=123,
        greedy=False,
    )
    b = select_next_token_id(
        logits,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        repetition_penalty=None,
        frequency_penalty=None,
        generated_token_ids=[],
        token_counts={},
        seed=123,
        greedy=False,
    )
    assert a == b

    draws = {
        select_next_token_id(
            logits,
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            repetition_penalty=None,
            frequency_penalty=None,
            generated_token_ids=[],
            token_counts={},
            seed=s,
            greedy=False,
        )
        for s in range(200, 220)
    }
    assert len(draws) > 1


def test_temperature_monotonic_effect_on_entropy():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    low = []
    high = []
    for s in range(150):
        low.append(
            select_next_token_id(
                logits,
                temperature=0.4,
                top_k=None,
                top_p=1.0,
                repetition_penalty=None,
                frequency_penalty=None,
                generated_token_ids=[],
                token_counts={},
                seed=s,
                greedy=False,
            )
        )
        high.append(
            select_next_token_id(
                logits,
                temperature=1.5,
                top_k=None,
                top_p=1.0,
                repetition_penalty=None,
                frequency_penalty=None,
                generated_token_ids=[],
                token_counts={},
                seed=s,
                greedy=False,
            )
        )
    low_counts = torch.bincount(torch.tensor(low), minlength=4)
    high_counts = torch.bincount(torch.tensor(high), minlength=4)
    assert int(high_counts.max().item()) < int(low_counts.max().item())


def test_top_k_support_size_never_exceeds_k():
    logits = torch.tensor([0.1, 0.2, 0.5, 3.0, 1.5], dtype=torch.float32)
    filtered = top_k_filter(logits, k=2)
    assert _support_size(filtered) <= 2
    top_idx = torch.topk(logits, k=2).indices.tolist()
    for i, v in enumerate(filtered):
        if i not in top_idx:
            assert float(v.item()) <= -1e8


def test_top_p_support_monotonic_in_p():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1], dtype=torch.float32)
    small = top_p_filter(logits, p=0.5)
    large = top_p_filter(logits, p=0.9)
    assert _support_size(small) <= _support_size(large)


def test_repetition_penalty_reduces_prob_of_repeated_token():
    logits = torch.tensor([0.2, 3.0, 0.1], dtype=torch.float32)
    out = apply_repetition_penalty(logits, generated_token_ids=[1], penalty=1.2)
    assert float(out[1].item()) < float(logits[1].item())


def test_frequency_penalty_scales_with_count():
    logits = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
    out = apply_frequency_penalty(logits, token_counts={0: 3, 1: 1}, penalty=0.5)
    delta0 = float(logits[0].item() - out[0].item())
    delta1 = float(logits[1].item() - out[1].item())
    assert delta0 > delta1


def test_sampling_pipeline_order_is_stable():
    logits = torch.tensor([2.0, 1.8, 1.2, 0.9], dtype=torch.float32)
    seed = 77

    manual = apply_repetition_penalty(logits, generated_token_ids=[0], penalty=1.1)
    manual = apply_frequency_penalty(manual, token_counts={0: 2, 2: 1}, penalty=0.2)
    manual = apply_temperature(manual, temperature=0.8)
    manual = top_k_filter(manual, k=3)
    manual = top_p_filter(manual, p=0.95)
    manual_choice = sample_next_token_id(manual, seed=seed)

    choice = select_next_token_id(
        logits,
        temperature=0.8,
        top_k=3,
        top_p=0.95,
        repetition_penalty=1.1,
        frequency_penalty=0.2,
        generated_token_ids=[0],
        token_counts={0: 2, 2: 1},
        seed=seed,
        greedy=False,
    )
    assert choice == manual_choice


def test_sampling_invalid_args_raise():
    logits = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    try:
        _ = top_k_filter(logits, k=0)
        assert False, "expected ValueError for top_k=0"
    except ValueError:
        pass
    try:
        _ = top_p_filter(logits, p=1.2)
        assert False, "expected ValueError for top_p>1"
    except ValueError:
        pass
    try:
        _ = apply_temperature(logits, temperature=-0.1)
        assert False, "expected ValueError for negative temperature"
    except ValueError:
        pass

# llm_lab/serving/sampling.py
from __future__ import annotations

from typing import Dict, List

import torch


def _as_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError(f"logits must be [V] or [1,V], got {tuple(logits.shape)}")
        return logits[0]
    if logits.ndim != 1:
        raise ValueError(f"logits must be [V] or [1,V], got {tuple(logits.shape)}")
    return logits


def _large_negative_like(logits: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(logits) + torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)


def _build_keep_mask(logits_1d: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    mask = _large_negative_like(logits_1d)
    mask.scatter_(-1, keep_idx, 0.0)
    return mask


def apply_temperature(logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Scale logits by temperature. temperature=0 is handled by greedy path."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    x = _as_1d_logits(logits).clone()
    if temperature == 0:
        return x
    out = x.float() / float(temperature)
    return out.to(dtype=x.dtype)


def top_k_filter(logits: torch.Tensor, *, k: int) -> torch.Tensor:
    """Set non-topk logits to a large negative value."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    x = _as_1d_logits(logits).clone()
    vocab_size = int(x.shape[0])
    k_eff = min(int(k), vocab_size)
    _, keep_idx = torch.topk(x, k=k_eff, dim=-1)
    mask = _build_keep_mask(x, keep_idx)
    masked_logits = x + mask
    return masked_logits


def top_p_filter(logits: torch.Tensor, *, p: float) -> torch.Tensor:
    """Keep the smallest sorted prefix whose cumulative probability >= p."""
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    x = _as_1d_logits(logits).clone()
    probs = torch.softmax(x.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumprobs > float(p)
    if cutoff.numel() > 1:
        cutoff[1:] = cutoff[:-1].clone()
    cutoff[0] = False

    keep_idx = sorted_idx[~cutoff]
    mask = _build_keep_mask(x, keep_idx)
    masked_logits = x + mask
    return masked_logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    *,
    generated_token_ids: List[int],
    penalty: float,
) -> torch.Tensor:
    """Classic repetition penalty from CTRL/GPT-style decoding."""
    if penalty <= 0:
        raise ValueError("penalty must be > 0")
    x = _as_1d_logits(logits).clone()
    if penalty == 1.0 or not generated_token_ids:
        return x
    for tid in set(generated_token_ids):
        if tid < 0 or tid >= x.shape[0]:
            continue
        if x[tid] > 0:
            x[tid] = x[tid] / float(penalty)
        else:
            x[tid] = x[tid] * float(penalty)
    return x


def apply_frequency_penalty(
    logits: torch.Tensor,
    *,
    token_counts: Dict[int, int],
    penalty: float,
) -> torch.Tensor:
    """Subtract penalty * count(token) from each seen token logit."""
    if penalty < 0:
        raise ValueError("penalty must be >= 0")
    x = _as_1d_logits(logits).clone()
    if penalty == 0 or not token_counts:
        return x
    for tid, count in token_counts.items():
        if tid < 0 or tid >= x.shape[0] or count <= 0:
            continue
        x[tid] = x[tid] - float(penalty) * float(count)
    return x


def sample_next_token_id(logits: torch.Tensor, *, seed: int | None) -> int:
    """Draw one token id from softmax(logits) with optional local seed."""
    x = _as_1d_logits(logits)
    probs = torch.softmax(x.float().cpu(), dim=-1)
    if seed is None:
        out = torch.multinomial(probs, num_samples=1)
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        out = torch.multinomial(probs, num_samples=1, generator=g)
    return int(out.item())


def select_next_token_id(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    frequency_penalty: float | None,
    generated_token_ids: List[int],
    token_counts: Dict[int, int],
    seed: int | None,
    greedy: bool,
) -> int:
    """Canonical sampling pipeline for serving decode."""
    x = _as_1d_logits(logits)
    if greedy or temperature == 0:
        return int(torch.argmax(x, dim=-1).item())

    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    x2 = x.clone()
    if repetition_penalty is not None:
        x2 = apply_repetition_penalty(x2, generated_token_ids=generated_token_ids, penalty=float(repetition_penalty))
    if frequency_penalty is not None:
        x2 = apply_frequency_penalty(x2, token_counts=token_counts, penalty=float(frequency_penalty))

    x2 = apply_temperature(x2, temperature=float(temperature))

    if top_k is not None:
        x2 = top_k_filter(x2, k=int(top_k))
    if top_p is not None:
        x2 = top_p_filter(x2, p=float(top_p))

    return sample_next_token_id(x2, seed=seed)

# llm_lab/serving/_shared.py
"""Shared helpers used across engine, stream, api, and safety modules."""
from __future__ import annotations

import hashlib
import torch

from llm_lab.serving.sampling import (
    apply_frequency_penalty,
    apply_repetition_penalty,
    apply_temperature,
    top_k_filter,
    top_p_filter,
)


def device_family(device: str) -> str:
    dev = str(device).lower()
    if dev.startswith("cuda"):
        return "cuda"
    if dev.startswith("mps"):
        return "mps"
    if dev.startswith("cpu"):
        return "cpu"
    return "other"


def resolve_eos_token_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "token_to_id"):
        try:
            return int(tokenizer.token_to_id("<|endoftext|>"))
        except Exception:
            pass
    value = getattr(tokenizer, "eos_token_id", None)
    return int(value) if value is not None else None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_step_distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    frequency_penalty: float | None,
    generated_token_ids: list[int],
    token_counts: dict[int, int],
) -> torch.Tensor:
    x = logits[0].clone() if logits.ndim == 2 else logits.clone()
    if x.ndim != 1:
        raise ValueError(f"logits must be [V] or [1,V], got shape={tuple(logits.shape)}")
    if repetition_penalty is not None:
        x = apply_repetition_penalty(x, generated_token_ids=generated_token_ids, penalty=float(repetition_penalty))
    if frequency_penalty is not None:
        x = apply_frequency_penalty(x, token_counts=token_counts, penalty=float(frequency_penalty))
    x = apply_temperature(x, temperature=float(temperature))
    if top_k is not None:
        x = top_k_filter(x, k=int(top_k))
    if top_p is not None:
        x = top_p_filter(x, p=float(top_p))
    x2f = torch.nan_to_num(x.float(), nan=0.0, posinf=1e9, neginf=-1e9)
    probs = torch.softmax(x2f, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum()
    if not torch.isfinite(denom) or denom.item() <= 0.0:
        probs = torch.zeros_like(probs)
        probs[x2f.argmax()] = 1.0
    return probs


def token_logprobability(distribution: torch.Tensor, token_id: int) -> float:
    p = max(float(distribution[token_id].item()), 1e-30)
    return float(torch.log(torch.tensor(p, dtype=torch.float32)).item())

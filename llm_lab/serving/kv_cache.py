# llm_lab/serving/kv_cache.py
from __future__ import annotations

from typing import List, Tuple

import torch

PastKV = Tuple[torch.Tensor, torch.Tensor]
PastKVs = List[PastKV]

# Canonical cache layout: [B, n_kv_heads, T, head_dim]
TIME_DIM = 2


def _validate_kv_pair(kv: PastKV, *, name: str) -> None:
    k, v = kv
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError(f"{name} must be a tuple[Tensor, Tensor]")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"{name} tensors must be rank-4, got k={k.ndim}, v={v.ndim}")
    if k.shape != v.shape:
        raise ValueError(f"{name} shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}")


def _validate_compatible_pairs(past: PastKV, new: PastKV) -> None:
    k_past, v_past = past
    k_new, v_new = new
    if k_past.dtype != k_new.dtype or v_past.dtype != v_new.dtype:
        raise ValueError("past/new dtype mismatch")
    if k_past.device != k_new.device or v_past.device != v_new.device:
        raise ValueError("past/new device mismatch")

    # B, H, D must match; T may differ.
    if k_past.shape[0] != k_new.shape[0] or k_past.shape[1] != k_new.shape[1] or k_past.shape[3] != k_new.shape[3]:
        raise ValueError(
            "past/new shape mismatch outside time dim: "
            f"past={tuple(k_past.shape)} new={tuple(k_new.shape)}"
        )


def kv_append(past: PastKV, new: PastKV) -> PastKV:
    """Append new KV to past on TIME_DIM."""
    _validate_kv_pair(past, name="past")
    _validate_kv_pair(new, name="new")
    _validate_compatible_pairs(past, new)
    k_past, v_past = past
    k_new, v_new = new
    # Preserve chronological order by appending new tokens after existing cache.
    k_new = torch.cat([k_past, k_new], dim=TIME_DIM)
    v_new = torch.cat([v_past, v_new], dim=TIME_DIM)
    return k_new, v_new


def kv_truncate(past: PastKV, keep_last_n: int) -> PastKV:
    """Keep last N tokens along TIME_DIM."""
    if keep_last_n < 0:
        raise ValueError("keep_last_n must be >= 0")
    k_past, v_past = past
    _, _, t_cur, _ = k_past.shape
    if keep_last_n >= t_cur:
        return k_past, v_past
    if keep_last_n == 0:
        return k_past[:, :, 0:0, :], v_past[:, :, 0:0, :]
    return k_past[:, :, -keep_last_n:, :], v_past[:, :, -keep_last_n:, :]


def apply_sliding_window(past_kvs: PastKVs, max_len: int) -> PastKVs:
    """Truncate each layer cache to max_len along TIME_DIM."""
    if max_len < 0:
        raise ValueError("max_len must be >= 0")
    return [kv_truncate(kv, keep_last_n=max_len) for kv in past_kvs]

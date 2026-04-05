# tests/serving/test_kv_cache_ops.py
from __future__ import annotations

import pytest
import torch

from llm_lab.serving.kv_cache import TIME_DIM, kv_append, kv_truncate


def _devices_for_test() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def _dtypes_for_device(device: torch.device) -> list[torch.dtype]:
    dtypes = [torch.float32]
    if device.type in {"cuda", "mps"}:
        dtypes.append(torch.float16)
    return dtypes


def _make_kv(*, B: int = 2, H: int = 4, T: int = 5, D: int = 8, device: torch.device, dtype: torch.dtype):
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    return k, v


def _time_len(k: torch.Tensor) -> int:
    return int(k.shape[TIME_DIM])


def test_kv_append_grows_T():
    device = torch.device("cpu")
    dtype = torch.float32

    past = _make_kv(T=7, device=device, dtype=dtype)
    new = _make_kv(T=3, device=device, dtype=dtype)

    k_past, v_past = past
    k_new, v_new = new

    k_out, v_out = kv_append(past, new)

    assert _time_len(k_out) == _time_len(k_past) + _time_len(k_new)
    assert _time_len(v_out) == _time_len(v_past) + _time_len(v_new)
    assert torch.allclose(k_out[:, :, -_time_len(k_new) :, :], k_new)
    assert torch.allclose(v_out[:, :, -_time_len(v_new) :, :], v_new)


def test_kv_truncate_keeps_last_N():
    device = torch.device("cpu")
    dtype = torch.float32

    past = _make_kv(T=10, device=device, dtype=dtype)
    k_past, v_past = past

    keep_n = 4
    k_out, v_out = kv_truncate(past, keep_n)

    assert _time_len(k_out) == keep_n
    assert _time_len(v_out) == keep_n
    assert torch.allclose(k_out, k_past[:, :, -keep_n:, :])
    assert torch.allclose(v_out, v_past[:, :, -keep_n:, :])


def test_kv_dtype_device_preserved():
    for device in _devices_for_test():
        for dtype in _dtypes_for_device(device):
            past = _make_kv(T=6, device=device, dtype=dtype)
            new = _make_kv(T=2, device=device, dtype=dtype)

            k_out, v_out = kv_append(past, new)
            assert k_out.dtype == dtype and v_out.dtype == dtype
            # Compare device types to avoid backend-specific index formatting differences (e.g., mps vs mps:0).
            assert k_out.device.type == device.type and v_out.device.type == device.type


def test_kv_truncate_edge_cases_n_ge_t_and_zero():
    device = torch.device("cpu")
    dtype = torch.float32

    past = _make_kv(T=6, device=device, dtype=dtype)
    k_past, v_past = past

    k_full, v_full = kv_truncate(past, 6)
    assert torch.allclose(k_full, k_past)
    assert torch.allclose(v_full, v_past)

    k_big, v_big = kv_truncate(past, 9)
    assert torch.allclose(k_big, k_past)
    assert torch.allclose(v_big, v_past)

    k_empty, v_empty = kv_truncate(past, 0)
    assert _time_len(k_empty) == 0
    assert _time_len(v_empty) == 0
    assert k_empty.shape[:2] == k_past.shape[:2]
    assert k_empty.shape[-1] == k_past.shape[-1]


def test_kv_no_shape_drift_heads_and_head_dim():
    device = torch.device("cpu")
    dtype = torch.float32

    past = _make_kv(B=2, H=3, T=4, D=5, device=device, dtype=dtype)
    new = _make_kv(B=2, H=3, T=2, D=5, device=device, dtype=dtype)

    k_out, v_out = kv_append(past, new)

    assert k_out.shape[0] == 2 and v_out.shape[0] == 2
    assert k_out.shape[1] == 3 and v_out.shape[1] == 3
    assert k_out.shape[3] == 5 and v_out.shape[3] == 5


def test_kv_truncate_negative_raises():
    device = torch.device("cpu")
    dtype = torch.float32
    past = _make_kv(T=3, device=device, dtype=dtype)

    with pytest.raises(ValueError):
        _ = kv_truncate(past, -1)

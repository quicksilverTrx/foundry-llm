# llm_lab/serving/precision.py
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

from llm_lab.serving._shared import device_family

_SUPPORTED_DTYPES = {"fp32", "fp16", "bf16"}
_TORCH_DTYPE_BY_NAME = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_DTYPE_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "torch.float32": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "torch.float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
}


def normalize_requested_dtype(requested_dtype: str) -> str:
    if not isinstance(requested_dtype, str) or not requested_dtype.strip():
        raise ValueError("requested_dtype must be a non-empty string")
    key = requested_dtype.strip().lower()
    canonical = _DTYPE_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(_SUPPORTED_DTYPES))
        raise ValueError(f"unsupported dtype '{requested_dtype}'; supported={supported}")
    return canonical


def _dtype_supported_on_device(dtype: str, device: str) -> bool:
    family = device_family(device)
    if dtype == "fp32":
        return True
    if family == "cuda":
        if dtype == "fp16":
            return torch.cuda.is_available()
        if dtype == "bf16":
            return torch.cuda.is_available() and bool(torch.cuda.is_bf16_supported())
    if family == "mps":
        if dtype == "fp16":
            return bool(torch.backends.mps.is_available())
        if dtype == "bf16":
            return False
    if family == "cpu":
        if dtype == "fp16":
            return False
        if dtype == "bf16":
            return bool(getattr(torch.backends.cpu, "has_bf16", False))
    return False


def validate_precision_request(requested_dtype: str, device: str) -> tuple[bool, str | None]:
    try:
        canonical = normalize_requested_dtype(requested_dtype)
    except ValueError as exc:
        return False, str(exc)

    if _dtype_supported_on_device(canonical, device):
        return True, None
    return False, f"dtype '{canonical}' is not supported on device '{device}'"


def runtime_precision_decision(requested_dtype: str, device: str) -> tuple[str, str | None]:
    canonical = normalize_requested_dtype(requested_dtype)
    if _dtype_supported_on_device(canonical, device):
        return canonical, None
    runtime = "fp32"
    reason = f"requested dtype '{canonical}' unavailable on '{device}', fell back to '{runtime}'"
    return runtime, reason


def resolve_runtime_precision(requested_dtype: str, device: str) -> str:
    runtime, _ = runtime_precision_decision(requested_dtype, device)
    return runtime


def _is_precision_sensitive_module(module: "nn.Module") -> bool:
    if isinstance(module, torch.nn.LayerNorm):
        return True
    return module.__class__.__name__.lower() == "rmsnorm"


def cast_model_for_inference(model: "nn.Module", runtime_dtype: str) -> "nn.Module":
    dtype = normalize_requested_dtype(runtime_dtype)
    target_torch_dtype = _TORCH_DTYPE_BY_NAME[dtype]
    model = model.to(dtype=target_torch_dtype)
    if dtype == "fp32":
        return model

    for _, module in model.named_modules():
        if _is_precision_sensitive_module(module):
            module.to(dtype=torch.float32)
    return model

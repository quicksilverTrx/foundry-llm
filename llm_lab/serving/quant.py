# llm_lab/serving/quant.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from llm_lab.core.package.io import load_model_package
from llm_lab.serving._shared import device_family
from llm_lab.serving.precision import cast_model_for_inference, runtime_precision_decision

if TYPE_CHECKING:
    from torch import nn


def _normalize_quant_mode(quant_mode: str | None) -> str | None:
    if quant_mode is None:
        return None
    raw = str(quant_mode).strip().lower()
    if raw in {"", "none", "null"}:
        return None
    if raw != "int8":
        raise ValueError("unsupported quant_mode; expected one of: None, 'int8'")
    return "int8"


def quant_backend_is_available(quant_mode: str, device: str) -> tuple[bool, str | None]:
    mode = _normalize_quant_mode(quant_mode)
    if mode is None:
        return True, None

    family = device_family(device)
    if family != "cpu":
        return False, f"int8 backend unavailable on device '{device}'; cpu dynamic quantization only"

    engines = set(torch.backends.quantized.supported_engines)
    if "qnnpack" not in engines:
        return False, "int8 backend unavailable: qnnpack quantized engine is not supported"
    return True, None


def maybe_quantize_model(model: "nn.Module", quant_mode: str | None, device: str) -> "nn.Module":
    mode = _normalize_quant_mode(quant_mode)
    if mode is None:
        return model

    ok, reason = quant_backend_is_available(mode, device)
    if not ok:
        raise RuntimeError(reason or "quant backend unavailable")

    if torch.backends.quantized.engine != "qnnpack":
        torch.backends.quantized.engine = "qnnpack"

    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    setattr(q_model, "_quant_requested_mode", mode)
    setattr(q_model, "_quant_backend", torch.backends.quantized.engine)
    setattr(q_model, "_quant_linear_target_count", int(linear_count))
    setattr(q_model, "_quant_transform_applied", bool(linear_count > 0))
    return q_model


def load_quantized_model_package(
    package_path: str,
    device: str,
    dtype: str,
    quant_mode: str | None,
) -> tuple["nn.Module", object]:
    package = Path(package_path)
    config, tokenizer, model = load_model_package(package, device=device)
    runtime_dtype, _ = runtime_precision_decision(dtype, device)
    model = cast_model_for_inference(model.eval(), runtime_dtype)
    mode = _normalize_quant_mode(quant_mode)
    if mode is not None:
        ok, _ = quant_backend_is_available(mode, device)
        if ok:
            model = maybe_quantize_model(model, mode, device)
    return model, tokenizer


def describe_quant_runtime(quant_mode: str | None, device: str) -> dict[str, object]:
    mode = _normalize_quant_mode(quant_mode)
    if mode is None:
        return {
            "requested_quant_mode": None,
            "runtime_quant_mode": None,
            "quant_available": True,
            "quant_backend": None,
            "quant_fallback_reason": None,
            "quant_coverage": "none",
            "quant_honesty": "disabled",
        }

    ok, reason = quant_backend_is_available(mode, device)
    return {
        "requested_quant_mode": mode,
        "runtime_quant_mode": mode if ok else None,
        "quant_available": bool(ok),
        "quant_backend": "qnnpack" if ok else None,
        "quant_fallback_reason": None if ok else reason,
        # This descriptor is computed pre-transform; coverage is capability-only.
        "quant_coverage": "unknown" if ok else "none",
        "quant_honesty": "capability_only" if ok else "unavailable",
    }

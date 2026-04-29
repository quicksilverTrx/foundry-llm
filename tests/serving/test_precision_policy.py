# tests/serving/test_precision_policy.py
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from llm_lab.core.model.norms import RMSNorm
from llm_lab.serving.engine import build_engine_from_package
from llm_lab.serving.precision import (
    normalize_requested_dtype,
    runtime_precision_decision,
    resolve_runtime_precision,
    validate_precision_request,
    cast_model_for_inference,
)


def _package_dir() -> Path:
    return Path(os.environ.get("SERVING_PACKAGE_DIR", "experiments/p1_pos_enc/runs/rope/package"))


def test_resolve_runtime_precision_respects_device_support() -> None:
    assert resolve_runtime_precision("fp32", "cpu") == "fp32"
    assert resolve_runtime_precision("fp16", "cpu") == "fp32"
    assert resolve_runtime_precision("bf16", "mps") == "fp32"


def test_validate_precision_request_rejects_unknown_dtype() -> None:
    ok, msg = validate_precision_request("float16", "cpu")
    assert normalize_requested_dtype("float16") == "fp16"
    assert ok is False
    assert "unsupported dtype" not in (msg or "")

    ok2, msg2 = validate_precision_request("totally_invalid", "cpu")
    assert ok2 is False
    assert "unsupported dtype" in (msg2 or "")


def test_engine_loads_requested_precision_or_explains_fallback() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    engine = build_engine_from_package(
        package_path=str(pkg),
        device="cpu",
        dtype="fp16",
        quant_mode=None,
    )

    assert engine.requested_dtype == "fp16"
    assert engine.runtime_dtype == "fp32"
    assert engine.runtime_fallback_reason is not None


def test_service_surface_is_unchanged_across_fp32_fp16_bf16() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    surfaces: list[set[str]] = []
    metric_surfaces: list[set[str]] = []

    for req in ("fp32", "fp16", "bf16"):
        engine = build_engine_from_package(
            package_path=str(pkg),
            device="cpu",
            dtype=req,
            quant_mode=None,
        )
        out = engine.generate(
            prompt_ids=[1, 2],
            attention_mask=[1, 1],
            max_new_tokens=2,
            temperature=0.0,
            top_k=1,
        )
        surfaces.append(set(out.keys()))
        metric_surfaces.append(set(out["metrics"].keys()))

    assert surfaces[0] == surfaces[1] == surfaces[2]
    assert metric_surfaces[0] == metric_surfaces[1] == metric_surfaces[2]


def test_dtype_and_device_policy_contracts() -> None:
    assert normalize_requested_dtype("float16") == "fp16"
    with pytest.raises(ValueError):
        normalize_requested_dtype("not_a_dtype")

    runtime, reason = runtime_precision_decision("fp16", "cpu")
    assert runtime == "fp32"
    assert "fell back" in (reason or "")

    runtime_ok, reason_ok = runtime_precision_decision("fp32", "cpu")
    assert runtime_ok == "fp32"
    assert reason_ok is None

    with pytest.raises(ValueError):
        runtime_precision_decision("unknown", "cpu")


def test_precision_cast_exceptions_keep_norms_in_fp32() -> None:
    class TinyNormNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(8, 8)
            self.norm = torch.nn.LayerNorm(8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.fc(x))

    m = TinyNormNet().eval()
    casted = cast_model_for_inference(m, "fp16")
    assert casted.fc.weight.dtype == torch.float16
    assert casted.norm.weight.dtype == torch.float32


def test_precision_cast_exceptions_keep_rmsnorm_in_fp32() -> None:
    class TinyRmsNormNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(8, 8)
            self.norm = RMSNorm(8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.fc(x))

    m = TinyRmsNormNet().eval()
    casted = cast_model_for_inference(m, "fp16")
    assert casted.fc.weight.dtype == torch.float16
    assert casted.norm.weight.dtype == torch.float32


def test_precision_cast_exceptions_reject_invalid_dtype() -> None:
    m = torch.nn.Linear(4, 4).eval()
    with pytest.raises(ValueError):
        cast_model_for_inference(m, "invalid_dtype")

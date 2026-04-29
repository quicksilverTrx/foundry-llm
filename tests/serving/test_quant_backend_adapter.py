# tests/serving/test_quant_backend_adapter.py
from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_lab.core.package.io import load_model_package
from llm_lab.serving.engine import build_engine_from_package
from llm_lab.serving.quant import (
    describe_quant_runtime,
    load_quantized_model_package,
    maybe_quantize_model,
    quant_backend_is_available,
)


def _package_dir() -> Path:
    return Path(os.environ.get("SERVING_PACKAGE_DIR", "experiments/p1_pos_enc/runs/rope/package"))


def test_quant_backend_reports_unavailable_cleanly() -> None:
    ok, reason = quant_backend_is_available("int8", "mps")
    assert ok is False
    assert reason is not None and "unavailable" in reason

    meta = describe_quant_runtime("int8", "mps")
    assert meta["runtime_quant_mode"] is None
    assert meta["quant_fallback_reason"] is not None


def test_load_quantized_model_package_preserves_tokenizer_artifacts() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    _, tok_ref, _ = load_model_package(pkg, device="cpu")
    ok, _ = quant_backend_is_available("int8", "cpu")
    if not ok:
        pytest.skip("int8 backend unavailable on this CPU runtime")

    model_q, tok_q = load_quantized_model_package(
        package_path=str(pkg),
        device="cpu",
        dtype="fp32",
        quant_mode="int8",
    )

    assert type(tok_q) is type(tok_ref)
    assert getattr(tok_q, "pad_token_id", None) == getattr(tok_ref, "pad_token_id", None)
    assert getattr(tok_q, "eos_token_id", None) == getattr(tok_ref, "eos_token_id", None)
    assert hasattr(model_q, "forward")


def test_int8_generate_smoke_if_backend_available() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    ok, _ = quant_backend_is_available("int8", "cpu")
    if not ok:
        pytest.skip("int8 backend unavailable on this CPU runtime")

    engine = build_engine_from_package(
        package_path=str(pkg),
        device="cpu",
        dtype="fp32",
        quant_mode="int8",
    )
    out = engine.generate(
        prompt_ids=[1, 2],
        attention_mask=[1, 1],
        max_new_tokens=2,
        temperature=0.0,
        top_k=1,
    )
    assert len(out["completion_token_ids"]) >= 1


def test_quant_mode_none_behaves_like_unquantized_load() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    base = build_engine_from_package(str(pkg), device="cpu", dtype="fp32", quant_mode=None)
    none_mode = build_engine_from_package(str(pkg), device="cpu", dtype="fp32", quant_mode="none")

    out_a = base.generate(prompt_ids=[1], attention_mask=[1], max_new_tokens=2, temperature=0.0, top_k=1)
    out_b = none_mode.generate(prompt_ids=[1], attention_mask=[1], max_new_tokens=2, temperature=0.0, top_k=1)
    assert out_a["completion_token_ids"] == out_b["completion_token_ids"]


def test_quant_public_functions_preserve_adapter_contract() -> None:
    ok, _ = quant_backend_is_available("int8", "cpu")
    meta = describe_quant_runtime("int8", "cpu")
    assert "requested_quant_mode" in meta
    assert "runtime_quant_mode" in meta
    assert "quant_available" in meta
    assert "quant_backend" in meta
    assert "quant_fallback_reason" in meta
    assert "quant_coverage" in meta
    assert "quant_honesty" in meta

    if not ok:
        with pytest.raises(RuntimeError):
            maybe_quantize_model(object(), "int8", "cpu")


def test_quant_mode_invalid_fails_fast() -> None:
    with pytest.raises(ValueError):
        quant_backend_is_available("int4", "cpu")
    with pytest.raises(ValueError):
        describe_quant_runtime("bad_mode", "cpu")

# tests/eval/test_quant_eval_report.py
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from llm_lab.eval.ppl import evaluate_streaming_nll
from llm_lab.eval.report import (
    benchmark_inference_modes,
    estimate_runtime_memory_bytes,
    summarize_precision_recommendation,
    write_quant_report,
)
from llm_lab.serving.quant import quant_backend_is_available


class TinyTokenizer:
    def __init__(self, vocab_size: int = 16) -> None:
        self.vocab_size = int(vocab_size)

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % self.vocab_size for ch in text]


class UniformModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size, device=input_ids.device)
        return logits, None


def _text_file(tmp_path: Path) -> Path:
    p = tmp_path / "eval.txt"
    p.write_text("abcdefghijklmnoqrstuvwxyz" * 4, encoding="utf-8")
    return p


def test_streaming_nll_eval_outputs_required_schema(tmp_path: Path) -> None:
    text_path = _text_file(tmp_path)
    model = UniformModel(vocab_size=16)
    tok = TinyTokenizer(vocab_size=16)

    out = evaluate_streaming_nll(
        model=model,
        tokenizer=tok,
        text_path=str(text_path),
        device="cpu",
        max_seq_len=8,
        stride=1,
    )
    required = {"n_tokens", "total_nll", "avg_nll", "ppl", "device", "dtype", "quant_mode"}
    assert required.issubset(out.keys())


def test_streaming_nll_uses_consistent_target_alignment(tmp_path: Path) -> None:
    text_path = _text_file(tmp_path)
    model = UniformModel(vocab_size=32)
    tok = TinyTokenizer(vocab_size=32)

    out1 = evaluate_streaming_nll(
        model=model,
        tokenizer=tok,
        text_path=str(text_path),
        device="cpu",
        max_seq_len=8,
        stride=1,
    )
    out2 = evaluate_streaming_nll(
        model=model,
        tokenizer=tok,
        text_path=str(text_path),
        device="cpu",
        max_seq_len=8,
        stride=2,
    )

    assert math.isclose(float(out1["avg_nll"]), math.log(32.0), rel_tol=1e-6)
    assert math.isclose(float(out2["avg_nll"]), math.log(32.0), rel_tol=1e-6)


def test_benchmark_inference_modes_outputs_required_metrics() -> None:
    pkg = Path(os.environ.get("SERVING_PACKAGE_DIR", "experiments/p1_pos_enc/runs/rope/package"))
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    out = benchmark_inference_modes(
        package_path=str(pkg),
        device="cpu",
        dtype="fp32",
        quant_mode=None,
        prompt_len=8,
        gen_len=4,
        batch_size=1,
    )
    required = {
        "prefill_ms",
        "ttft_ms",
        "decode_ms_total",
        "decode_ms_per_token",
        "tokens_per_sec",
        "prompt_len",
        "gen_len",
        "batch_size",
        "dtype",
        "quant_mode",
    }
    assert required.issubset(out.keys())


def test_quant_report_contains_required_sections(tmp_path: Path) -> None:
    out = tmp_path / "quant_report.md"
    write_quant_report(
        results=[
            {"mode": "fp32", "supported": True, "reason": ""},
            {"mode": "int8", "supported": False, "reason": "backend unavailable"},
        ],
        out_path=str(out),
    )
    text = out.read_text(encoding="utf-8")
    for section in (
        "## Environment",
        "## Field Definitions",
        "## Run-State Semantics",
        "## Supported Modes",
        "## Memory Comparison",
        "## Latency Comparison",
        "## PPL Comparison",
        "## Drift / Caveats",
        "## What Failed",
        "## Recommendation",
    ):
        assert section in text


def test_quant_sweep_tiny_writes_required_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "serving_quant"
    script = Path("scripts/serving/quant_sweep.py")
    cmd = [
        sys.executable,
        str(script),
        "--tiny-mock",
        "--device",
        "cpu",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "precision_matrix.json").exists()
    assert (out_dir / "quant_results.json").exists()
    assert (out_dir / "bench_fp32.json").exists()
    assert (out_dir / "ppl_fp32.json").exists()
    assert (out_dir / "quant_report.md").exists()
    assert (out_dir / "recommendation.json").exists()

    matrix = json.loads((out_dir / "precision_matrix.json").read_text(encoding="utf-8"))
    assert "modes" in matrix
    assert isinstance(matrix["modes"], list)
    for row in matrix["modes"]:
        assert "run_state" in row
        assert "executed_mode" in row
        assert "metrics_collected" in row

    quant_results = json.loads((out_dir / "quant_results.json").read_text(encoding="utf-8"))
    assert "results" in quant_results
    assert quant_results["run_state_definition"] == "executed|normalized_not_run_separately|excluded"
    for row in quant_results["results"]:
        assert "run_state" in row
        assert "executed_mode" in row
        assert "metrics_collected" in row
        if row["run_state"] == "normalized_not_run_separately":
            assert row["metrics_collected"] is False
            assert row["prefill_ms"] is None
            assert row["ttft_ms"] is None
            assert row["decode_ms_total"] is None
            assert row["decode_ms_per_token"] is None
            assert row["tokens_per_sec"] is None
            assert row["avg_nll"] is None
            assert row["ppl"] is None

    int8_ok, _ = quant_backend_is_available("int8", "cpu")
    if int8_ok:
        assert (out_dir / "bench_int8.json").exists()
        assert (out_dir / "ppl_int8.json").exists()


def test_precision_recommendation_is_generated_from_results() -> None:
    results = [
        {"device": "cpu", "mode": "fp32", "ppl": 10.0, "tokens_per_sec": 100.0},
        {"device": "cpu", "mode": "int8", "ppl": 10.5, "tokens_per_sec": 180.0},
    ]
    rec = summarize_precision_recommendation(results)
    assert "cpu_default" in rec


def test_runtime_memory_estimate_is_positive() -> None:
    model = UniformModel(vocab_size=8)
    bytes_est = estimate_runtime_memory_bytes(model, cache_state=None)
    assert isinstance(bytes_est, int)
    assert bytes_est > 0

# tests/eval/test_eval_report.py
from __future__ import annotations

from pathlib import Path

from llm_lab.eval.report import (
    build_bucket_summary,
    build_latency_summary,
    build_eval_report,
    build_safety_summary,
    write_eval_report,
)


def _sample_results() -> list[dict]:
    return [
        {
            "case_id": "a",
            "bucket": "short_prompt",
            "stop_reason": "max_new_tokens",
            "ttft_ms": 1.0,
            "decode_ms_per_token": 2.0,
            "tokens_per_sec": 400.0,
            "safety_flags": [],
            "refusal_applied": False,
        },
        {
            "case_id": "b",
            "bucket": "safety_probe",
            "stop_reason": "safety_refusal",
            "ttft_ms": None,
            "decode_ms_per_token": None,
            "tokens_per_sec": None,
            "safety_flags": ["pii_email"],
            "refusal_applied": True,
        },
    ]


def test_build_bucket_summary_counts_correctly() -> None:
    s = build_bucket_summary(_sample_results())
    assert s["total"] == 2
    assert s["by_bucket"]["short_prompt"] == 1
    assert s["by_bucket"]["safety_probe"] == 1


def test_build_safety_summary_counts_flags_and_refusals() -> None:
    s = build_safety_summary(_sample_results())
    assert s["refusal_count"] == 1
    assert s["flagged_cases"] == 1
    assert s["reason_counts"]["pii_email"] == 1


def test_build_latency_summary_handles_missing_fields() -> None:
    s = build_latency_summary(_sample_results())
    assert s["count_with_ttft"] == 1
    assert s["count_with_tokens_per_sec"] == 1
    assert s["avg_ttft_ms"] == 1.0


def test_build_eval_report_contains_required_sections() -> None:
    results = _sample_results()
    summary = {
        "bucket_summary": build_bucket_summary(results),
        "safety_summary": build_safety_summary(results),
        "latency_summary": build_latency_summary(results),
        "stop_reason_counts": {"max_new_tokens": 1, "safety_refusal": 1},
        "notable_failures": ["b"],
    }
    text = build_eval_report(summary, ppl_artifact_path="experiments/serving_quant/ppl_fp32.json")

    for heading in (
        "## scope",
        "## suite size and bucket breakdown",
        "## stop reason breakdown",
        "## latency summary",
        "## safety summary",
        "## notable failures / anomalous cases",
        "## PPL / quantization artifacts",
        "## next-action items",
    ):
        assert heading in text


def test_write_eval_report_writes_markdown_file(tmp_path: Path) -> None:
    out = tmp_path / "eval_report.md"
    write_eval_report("# ok\n", str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() == "# ok"

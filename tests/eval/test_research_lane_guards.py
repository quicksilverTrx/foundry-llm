# tests/eval/test_research_lane_guards.py
from __future__ import annotations

from llm_lab.eval.research_lane import (
    check_provenance_integrity,
    check_row_comparability,
    summarize_stability,
    validate_prompt_suite_reconciliation,
)


def _base_row() -> dict:
    return {
        "package_path": "pkg",
        "model_config_hash": "cfg",
        "tokenizer_hash": "tok",
        "device": "cpu",
        "dtype": "fp32",
        "quant_mode": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "context_len": 256,
        "batch_size": 1,
        "gen_len": 64,
        "seed": 1,
        "benchmark_mode": "cache",
    }


def test_provenance_integrity_detects_missing_fields() -> None:
    row = _base_row()
    row.pop("seed")
    out = check_provenance_integrity([row])
    assert out["passed"] is False
    assert out["missing_rows"]
    assert "seed" in out["missing_rows"][0]["missing_fields"]


def test_row_comparability_flags_mismatch() -> None:
    a = _base_row()
    b = _base_row()
    b["tokenizer_hash"] = "other"
    out = check_row_comparability(a, b)
    assert out["non_comparable"] is True
    assert "tokenizer_hash" in out["mismatch_reasons"]


def test_summarize_stability_marks_high_cv_unstable() -> None:
    rows = [{"metric": 1.0}, {"metric": 2.0}, {"metric": 3.0}]
    out = summarize_stability(rows, metric_key="metric", cv_threshold=0.2)
    assert out["count"] == 3
    assert out["cv"] > 0.2
    assert out["unstable"] is True


def test_prompt_suite_reconciliation_checks_totals_and_refusals() -> None:
    rows = [
        {
            "case_id": "a",
            "bucket": "short_prompt",
            "stop_reason": "max_new_tokens",
            "refusal_applied": False,
        },
        {
            "case_id": "b",
            "bucket": "safety_probe",
            "stop_reason": "safety_refusal",
            "refusal_applied": True,
        },
    ]
    bad_summary = {
        "total_cases": 2,
        "refusal_count": 0,
        "bucket_counts": {"short_prompt": 1, "safety_probe": 1},
        "stop_reason_counts": {"max_new_tokens": 2},
    }
    out = validate_prompt_suite_reconciliation(rows=rows, summary=bad_summary)
    assert out["passed"] is False
    assert any("refusal_count mismatch" in e for e in out["errors"])

    good_summary = {
        "total_cases": 2,
        "refusal_count": 1,
        "bucket_counts": {"short_prompt": 1, "safety_probe": 1},
        "stop_reason_counts": {"max_new_tokens": 1, "safety_refusal": 1},
    }
    out2 = validate_prompt_suite_reconciliation(rows=rows, summary=good_summary)
    assert out2["passed"] is True
    assert out2["errors"] == []

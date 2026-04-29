# llm_lab/eval/research_lane.py
from __future__ import annotations

import math
from collections import Counter
from typing import Any


REQUIRED_PROVENANCE_FIELDS = {
    "package_path",
    "model_config_hash",
    "tokenizer_hash",
    "device",
    "dtype",
    "quant_mode",
    "timestamp",
    "context_len",
    "batch_size",
    "gen_len",
    "seed",
    "benchmark_mode",
}


def check_provenance_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_by_row: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = sorted(k for k in REQUIRED_PROVENANCE_FIELDS if k not in row)
        if missing:
            missing_by_row.append({"row_index": idx, "missing_fields": missing})
    return {
        "row_count": int(len(rows)),
        "passed": len(missing_by_row) == 0,
        "missing_rows": missing_by_row,
    }


def check_row_comparability(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "package_path": (a.get("package_path"), b.get("package_path")),
        "model_config_hash": (a.get("model_config_hash"), b.get("model_config_hash")),
        "tokenizer_hash": (a.get("tokenizer_hash"), b.get("tokenizer_hash")),
        "context_len": (a.get("context_len", a.get("prompt_len")), b.get("context_len", b.get("prompt_len"))),
        "batch_size": (a.get("batch_size"), b.get("batch_size")),
        "gen_len": (a.get("gen_len"), b.get("gen_len")),
        "dtype": (a.get("dtype"), b.get("dtype")),
        "quant_mode": (a.get("quant_mode"), b.get("quant_mode")),
    }
    mismatches = [k for k, (va, vb) in checks.items() if va != vb]
    return {
        "non_comparable": len(mismatches) > 0,
        "mismatch_reasons": mismatches,
        "checks": checks,
    }


def summarize_stability(rows: list[dict[str, Any]], *, metric_key: str, cv_threshold: float) -> dict[str, Any]:
    if not rows:
        return {
            "metric": metric_key,
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "cv": None,
            "unstable": True,
            "cv_threshold": float(cv_threshold),
        }

    vals = [float(r[metric_key]) for r in rows]
    mean = float(sum(vals) / float(len(vals)))
    if len(vals) >= 2:
        variance = sum((v - mean) ** 2 for v in vals) / float(len(vals) - 1)
        std = float(math.sqrt(variance))
    else:
        std = 0.0
    cv = float(std / mean) if abs(mean) > 1e-12 else float("inf")
    return {
        "metric": metric_key,
        "count": int(len(vals)),
        "mean": mean,
        "std": std,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "cv": cv,
        "unstable": bool(cv > float(cv_threshold)),
        "cv_threshold": float(cv_threshold),
    }


def summarize_stop_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        reason = row.get("stop_reason")
        if isinstance(reason, str) and reason:
            counts[reason] += 1
    return dict(counts)


def _tokenize_simple(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def repetition_ratio(text: str) -> float:
    tokens = _tokenize_simple(text)
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return float(1.0 - (unique / float(len(tokens))))


def distinct_ngrams_ratio(text: str, n: int = 2) -> float:
    tokens = _tokenize_simple(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return float(len(set(grams)) / float(len(grams)))


def validate_prompt_suite_reconciliation(*, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []

    case_ids = [str(r.get("case_id")) for r in rows]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case_id detected")

    total_cases = int(summary.get("total_cases", -1))
    if total_cases != len(rows):
        errors.append("summary total_cases does not match row count")

    refusal_rows = sum(1 for r in rows if bool(r.get("refusal_applied", False)))
    refusal_summary = int(summary.get("refusal_count", -1))
    if refusal_rows != refusal_summary:
        errors.append("refusal_count mismatch between summary and rows")

    bucket_counts = summary.get("bucket_counts") if isinstance(summary.get("bucket_counts"), dict) else {}
    if sum(int(v) for v in bucket_counts.values()) != len(rows):
        errors.append("bucket_counts total does not equal row count")

    stop_counts = summary.get("stop_reason_counts") if isinstance(summary.get("stop_reason_counts"), dict) else {}
    observed_stop = summarize_stop_reasons(rows)
    if dict(stop_counts) != observed_stop:
        errors.append("stop_reason_counts mismatch between summary and rows")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "row_count": int(len(rows)),
        "summary_total_cases": total_cases,
        "summary_refusal_count": refusal_summary,
        "observed_refusal_count": refusal_rows,
    }

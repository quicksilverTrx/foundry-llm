# tests/serving/test_cache_divergence_diagnostic.py
from __future__ import annotations

import json
from pathlib import Path

from llm_lab.serving.diagnostics import write_cache_divergence_report


def test_writes_cache_divergence_report_with_required_schema(tmp_path: Path):
    out = tmp_path / "cache_divergence_report.json"

    write_cache_divergence_report(
        str(out),
        step=7,
        max_logit_diff=0.00213,
        prompt_ids=[11, 12, 13],
        generated_ids=[21, 22],
        extra={
            "top_differing_tokens": [
                {"token_id": 123, "cached_logit": 9.1, "recompute_logit": 9.0, "abs_diff": 0.1}
            ],
            "past_len": 5,
        },
    )

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))

    required = {
        "step",
        "max_logit_diff",
        "prompt_len",
        "generated_len",
        "top_differing_tokens",
        "prompt_ids",
        "generated_ids",
    }
    assert required.issubset(payload.keys())
    assert payload["step"] == 7
    assert payload["prompt_len"] == 3
    assert payload["generated_len"] == 2
    assert isinstance(payload["top_differing_tokens"], list)
    assert len(payload["top_differing_tokens"]) >= 1


def test_first_divergence_only_semantics(tmp_path: Path):
    out = tmp_path / "cache_divergence_report.json"

    mismatches = [
        {"step": 3, "max_diff": 0.03},
        {"step": 5, "max_diff": 0.05},
    ]
    first = mismatches[0]

    write_cache_divergence_report(
        str(out),
        step=first["step"],
        max_logit_diff=first["max_diff"],
        prompt_ids=[1, 2, 3],
        generated_ids=[4, 5],
        extra={
            "top_differing_tokens": [{"token_id": 77, "abs_diff": 0.03}],
            "note": "first divergence",
        },
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["step"] == 3
    assert payload["max_logit_diff"] == 0.03
    assert payload.get("note") == "first divergence"

# tests/eval/test_prompt_suite_data_contract.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_lab.eval.prompt_suite import (
    bucket_prompt_cases,
    load_prompt_cases,
    validate_prompt_case,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_prompt_cases_reads_all_valid_lines(tmp_path: Path) -> None:
    rows = [
        {"case_id": "c1", "bucket": "short_prompt", "prompt": "hello"},
        {"case_id": "c2", "bucket": "long_prompt", "prompt": "world", "tags": ["a"]},
    ]
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(path, rows)

    cases = load_prompt_cases(str(path))

    assert len(cases) == 2
    assert cases[0]["case_id"] == "c1"
    assert cases[1]["case_id"] == "c2"


def test_validate_prompt_case_requires_minimal_fields() -> None:
    with pytest.raises(ValueError):
        validate_prompt_case({"bucket": "x", "prompt": "abc"})
    with pytest.raises(ValueError):
        validate_prompt_case({"case_id": "x", "prompt": "abc"})
    with pytest.raises(ValueError):
        validate_prompt_case({"case_id": "x", "bucket": "b", "prompt": ""})
    with pytest.raises(ValueError):
        validate_prompt_case({"case_id": "x", "bucket": "b", "prompt": 123})


def test_validate_prompt_case_rejects_bad_optional_types() -> None:
    with pytest.raises(ValueError):
        validate_prompt_case({"case_id": "x", "bucket": "short_prompt", "prompt": "ok", "stop_strings": "x"})
    with pytest.raises(ValueError):
        validate_prompt_case({"case_id": "x", "bucket": "short_prompt", "prompt": "ok", "max_new_tokens": 0})
    with pytest.raises(ValueError):
        validate_prompt_case({"case_id": "x", "bucket": "short_prompt", "prompt": "ok", "tags": "x"})


def test_bucket_prompt_cases_groups_correctly() -> None:
    cases = [
        {"case_id": "a", "bucket": "short_prompt", "prompt": "1"},
        {"case_id": "b", "bucket": "short_prompt", "prompt": "2"},
        {"case_id": "c", "bucket": "safety_probe", "prompt": "3"},
    ]

    grouped = bucket_prompt_cases(cases)

    assert set(grouped.keys()) == {"short_prompt", "safety_probe"}
    assert [x["case_id"] for x in grouped["short_prompt"]] == ["a", "b"]
    assert [x["case_id"] for x in grouped["safety_probe"]] == ["c"]


def test_case_ids_are_unique(tmp_path: Path) -> None:
    rows = [
        {"case_id": "dup", "bucket": "short_prompt", "prompt": "hello"},
        {"case_id": "dup", "bucket": "long_prompt", "prompt": "world"},
    ]
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match="duplicate case_id"):
        _ = load_prompt_cases(str(path))


def test_validate_prompt_case_schema_relaxed_missing_assumes_current() -> None:
    case = {"case_id": "x", "bucket": "short_prompt", "prompt": "ok", "stop_strings": [], "tags": []}
    validate_prompt_case(case)
    assert case["schema_version"] == 2


def test_validate_prompt_case_schema_rejects_non_current_version() -> None:
    with pytest.raises(ValueError, match="unsupported prompt case schema_version"):
        validate_prompt_case(
            {
                "schema_version": 3,
                "case_id": "x",
                "bucket": "short_prompt",
                "prompt": "ok",
                "stop_strings": [],
                "tags": [],
            }
        )


def test_validate_prompt_case_bucket_membership_only() -> None:
    with pytest.raises(ValueError, match="bucket"):
        validate_prompt_case(
            {
                "schema_version": 2,
                "case_id": "x",
                "bucket": "not_real",
                "prompt": "ok",
                "stop_strings": [],
                "tags": [],
            }
        )
    validate_prompt_case(
        {
            "schema_version": 2,
            "case_id": "x2",
            "bucket": "stop_trap",
            "prompt": "ok",
            "stop_strings": [],
            "tags": [],
        }
    )


def test_load_prompt_cases_normalizes_optionals_to_canonical_defaults(tmp_path: Path) -> None:
    rows = [
        {
            "case_id": "n1",
            "bucket": "short_prompt",
            "prompt": "  normalized  ",
        }
    ]
    path = tmp_path / "prompts.jsonl"
    _write_jsonl(path, rows)
    out = load_prompt_cases(str(path))
    assert len(out) == 1
    case = out[0]
    assert case["schema_version"] == 2
    assert case["stop_strings"] == []
    assert case["tags"] == []
    assert case["max_new_tokens"] is None

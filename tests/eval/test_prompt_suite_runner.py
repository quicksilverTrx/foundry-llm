# tests/eval/test_prompt_suite_runner.py
from __future__ import annotations

import json
from pathlib import Path

from llm_lab.eval.prompt_suite import (
    compare_backend_results,
    summarize_prompt_suite,
    run_prompt_case,
    run_prompt_suite,
    write_prompt_suite_outputs,
)


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 31 for ch in text]


class _FakeEngine:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def generate(
        self,
        *,
        prompt_ids: list[int],
        attention_mask: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        stop_strings,
        seed,
        return_logprobs: bool,
    ) -> dict:
        del attention_mask, temperature, top_k, top_p, stop_strings, return_logprobs
        if prompt_ids and prompt_ids[0] == 0:
            raise RuntimeError("broken case")
        n = min(max_new_tokens, 2)
        token_ids = [1] * n
        return {
            # Include seed in text so determinism-policy tests can observe seed effects.
            "completion_text": f"{'x' * n}|{seed}",
            "completion_token_ids": token_ids,
            "stop_reason": "max_new_tokens",
            "all_token_ids": prompt_ids + token_ids,
            "metrics": {
                "ttft_ms": 1.5,
                "decode_ms_per_token": 2.0,
                "tokens_per_sec": 500.0,
            },
            "seed_echo": seed,
        }


class _FakeHttpAdapter:
    def generate(self, *, prompt: str, max_new_tokens: int, stop_strings, seed) -> dict:
        del stop_strings, seed
        if "explode" in prompt:
            raise RuntimeError("http exploded")
        n = min(max_new_tokens, 2)
        return {
            "completion_text": "y" * n,
            "completion_token_ids": [2] * n,
            "stop_reason": "max_new_tokens",
            "metrics": {
                "ttft_ms": 2.0,
                "decode_ms_per_token": 3.0,
                "tokens_per_sec": 333.0,
            },
            "safety_flags": [],
            "refusal_applied": False,
        }


def _cases() -> list[dict]:
    return [
        {"case_id": "c1", "bucket": "short_prompt", "prompt": "abc", "max_new_tokens": 2},
        {"case_id": "c2", "bucket": "safety_probe", "prompt": "def", "max_new_tokens": 2},
    ]


def test_run_prompt_case_returns_required_fields() -> None:
    out = run_prompt_case(_FakeEngine(), _cases()[0], seed=123)
    required = {
        "case_id",
        "bucket",
        "prompt_hash",
        "completion_text",
        "stop_reason",
        "prompt_len_chars",
        "completion_len_chars",
        "prompt_len_tokens",
        "completion_len_tokens",
        "ttft_ms",
        "decode_ms_per_token",
        "tokens_per_sec",
        "safety_flags",
        "refusal_applied",
        "error",
    }
    assert required.issubset(out.keys())
    assert out["case_id"] == "c1"
    assert out["bucket"] == "short_prompt"


def test_run_prompt_suite_runs_all_cases() -> None:
    rows = run_prompt_suite(_FakeEngine(), _cases(), seed=7)
    assert len(rows) == 2
    assert [r["case_id"] for r in rows] == ["c1", "c2"]


def test_summarize_prompt_suite_counts_buckets_and_stop_reasons() -> None:
    rows = run_prompt_suite(_FakeEngine(), _cases(), seed=7)
    summary = summarize_prompt_suite(rows)

    assert summary["total_cases"] == 2
    assert summary["bucket_counts"]["short_prompt"] == 1
    assert summary["bucket_counts"]["safety_probe"] == 1
    assert summary["stop_reason_counts"]["max_new_tokens"] == 2


def test_write_prompt_suite_outputs_writes_jsonl_and_summary_json(tmp_path: Path) -> None:
    rows = run_prompt_suite(_FakeEngine(), _cases(), seed=1)
    summary = summarize_prompt_suite(rows)

    write_prompt_suite_outputs(rows, summary, str(tmp_path))

    outputs_path = tmp_path / "prompt_suite_outputs.jsonl"
    summary_path = tmp_path / "prompt_suite_summary.json"

    assert outputs_path.exists()
    assert summary_path.exists()
    assert outputs_path.read_text(encoding="utf-8").strip() != ""

    parsed = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(parsed) == len(rows)


def test_runner_is_deterministic_in_greedy_mode() -> None:
    engine = _FakeEngine()
    rows1 = run_prompt_suite(engine, _cases(), seed=123)
    rows2 = run_prompt_suite(engine, _cases(), seed=123)
    assert rows1 == rows2


def test_runner_captures_errors_without_crashing_entire_suite() -> None:
    engine = _FakeHttpAdapter()
    cases = [
        {"case_id": "ok", "bucket": "short_prompt", "prompt": "hello", "max_new_tokens": 2},
        {"case_id": "bad", "bucket": "short_prompt", "prompt": "explode", "max_new_tokens": 2},
        {"case_id": "ok2", "bucket": "short_prompt", "prompt": "world", "max_new_tokens": 2},
    ]
    rows = run_prompt_suite(engine, cases, seed=3)

    assert len(rows) == 3
    assert rows[1]["error"] is not None
    assert rows[0]["error"] is None
    assert rows[2]["error"] is None


def test_compare_backend_results_schema_and_match_status() -> None:
    engine = [
        {"case_id": "a", "stop_reason": "max_new_tokens", "refusal_applied": False, "ttft_ms": 1.0},
        {"case_id": "b", "stop_reason": "max_new_tokens", "refusal_applied": False, "ttft_ms": 2.0},
    ]
    http = [
        {"case_id": "a", "stop_reason": "max_new_tokens", "refusal_applied": False, "ttft_ms": 1.5},
        {"case_id": "b", "stop_reason": "safety_refusal", "refusal_applied": True, "ttft_ms": 2.5},
    ]
    rows = compare_backend_results(engine, http)
    assert len(rows) == 2
    required = {
        "case_id",
        "engine_stop_reason",
        "http_stop_reason",
        "engine_refusal",
        "http_refusal",
        "latency_delta_fields",
        "match_status",
    }
    assert required.issubset(rows[0].keys())
    by_id = {r["case_id"]: r for r in rows}
    assert by_id["a"]["match_status"] == "match"
    assert by_id["b"]["match_status"] in {"mismatch_stop_reason", "mismatch_refusal"}

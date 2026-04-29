# tests/eval/test_eval_suite_script.py
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location("eval_prompt_suite_mod", Path("scripts/eval/eval_prompt_suite.py"))
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _cases_file(tmp_path: Path) -> Path:
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"case_id": "c1", "bucket": "short_prompt", "prompt": "abc", "max_new_tokens": 2}),
                json.dumps({"case_id": "c2", "bucket": "safety_probe", "prompt": "def", "max_new_tokens": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


def test_main_both_backends_writes_backend_and_parity_artifacts(tmp_path: Path, monkeypatch) -> None:
    prompts = _cases_file(tmp_path)
    out_dir = tmp_path / "out"

    args = SimpleNamespace(
        backend="both",
        package="unused",
        base_url="http://unused",
        prompts=str(prompts),
        rubric="data/serving_eval/rubric.md",
        out_dir=str(out_dir),
        device="cpu",
        dtype="fp32",
        quant_mode=None,
        seed=7,
        ppl_artifact="experiments/serving_quant/ppl_fp32.json",
    )

    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(
        mod,
        "_run_engine",
        lambda cases, seed: (
            [
                {
                    "case_id": "c1",
                    "bucket": "short_prompt",
                    "prompt_hash": "h1",
                    "completion_text": "x",
                    "stop_reason": "max_new_tokens",
                    "prompt_len_chars": 3,
                    "completion_len_chars": 1,
                    "prompt_len_tokens": 3,
                    "completion_len_tokens": 1,
                    "ttft_ms": 1.0,
                    "decode_ms_per_token": 2.0,
                    "tokens_per_sec": 500.0,
                    "safety_flags": [],
                    "refusal_applied": False,
                    "error": None,
                },
                {
                    "case_id": "c2",
                    "bucket": "safety_probe",
                    "prompt_hash": "h2",
                    "completion_text": "y",
                    "stop_reason": "max_new_tokens",
                    "prompt_len_chars": 3,
                    "completion_len_chars": 1,
                    "prompt_len_tokens": 3,
                    "completion_len_tokens": 1,
                    "ttft_ms": 1.5,
                    "decode_ms_per_token": 2.1,
                    "tokens_per_sec": 470.0,
                    "safety_flags": [],
                    "refusal_applied": False,
                    "error": None,
                },
            ],
            {"total_cases": 2, "error_count": 0, "refusal_count": 0, "bucket_counts": {"short_prompt": 1, "safety_probe": 1}, "stop_reason_counts": {"max_new_tokens": 2}, "safety_flag_counts": {}},
        ),
    )
    monkeypatch.setattr(
        mod,
        "_run_http",
        lambda cases, seed: (
            [
                {
                    "case_id": "c1",
                    "bucket": "short_prompt",
                    "prompt_hash": "h1",
                    "completion_text": "x",
                    "stop_reason": "max_new_tokens",
                    "prompt_len_chars": 3,
                    "completion_len_chars": 1,
                    "prompt_len_tokens": 3,
                    "completion_len_tokens": 1,
                    "ttft_ms": 1.2,
                    "decode_ms_per_token": 2.2,
                    "tokens_per_sec": 455.0,
                    "safety_flags": [],
                    "refusal_applied": False,
                    "error": None,
                },
                {
                    "case_id": "c2",
                    "bucket": "safety_probe",
                    "prompt_hash": "h2",
                    "completion_text": "[refuse]",
                    "stop_reason": "safety_refusal",
                    "prompt_len_chars": 3,
                    "completion_len_chars": 8,
                    "prompt_len_tokens": 3,
                    "completion_len_tokens": 0,
                    "ttft_ms": 1.8,
                    "decode_ms_per_token": 0.0,
                    "tokens_per_sec": 0.0,
                    "safety_flags": ["pii_email"],
                    "refusal_applied": True,
                    "error": None,
                },
            ],
            {"total_cases": 2, "error_count": 0, "refusal_count": 1, "bucket_counts": {"short_prompt": 1, "safety_probe": 1}, "stop_reason_counts": {"max_new_tokens": 1, "safety_refusal": 1}, "safety_flag_counts": {"pii_email": 1}},
        ),
    )

    mod.main()

    assert (out_dir / "prompt_suite_engine_outputs.jsonl").exists()
    assert (out_dir / "prompt_suite_http_outputs.jsonl").exists()
    assert (out_dir / "prompt_suite_backend_parity.json").exists()
    assert (out_dir / "prompt_suite_outputs.jsonl").exists()

    parity = json.loads((out_dir / "prompt_suite_backend_parity.json").read_text(encoding="utf-8"))
    assert len(parity) == 2
    statuses = {row["case_id"]: row["match_status"] for row in parity}
    assert statuses["c1"] == "match"
    assert statuses["c2"] in {"mismatch_stop_reason", "mismatch_refusal"}


def test_http_backend_hard_fails_when_all_cases_error() -> None:
    with __import__("pytest").raises(RuntimeError, match="HTTP backend failed for all prompt-suite cases"):
        mod._require_http_health({"total_cases": 2, "error_count": 2})

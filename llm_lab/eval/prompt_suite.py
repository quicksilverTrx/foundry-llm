# llm_lab/eval/prompt_suite.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib import request as urllib_request


_REQUIRED_CASE_FIELDS = ("case_id", "bucket", "prompt")
_ALLOWED_BUCKETS = {
    "short_prompt",
    "long_prompt",
    "repetition_trap",
    "stop_trap",
    "code_like",
    "safety_probe",
}
_CURRENT_SCHEMA_VERSION = 2


class HttpGenerateClient:
    """Minimal HTTP adapter for `/generate` compatibility in eval runs."""

    def __init__(self, *, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        seed: int | None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "stop_strings": stop_strings,
            "stop_token_ids": None,
            "seed": seed,
            "return_logprobs": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=f"{self.base_url}/generate",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib_request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
        if "error" in data:
            code = data["error"].get("code", "http_error")
            msg = data["error"].get("message", "unknown")
            raise RuntimeError(f"http backend error: {code}: {msg}")
        return data


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _as_text_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _load_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line == "":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{idx}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"each JSONL row must be object at {path}:{idx}")
        rows.append(obj)
    return rows


def _normalize_schema_version(case: dict[str, Any]) -> None:
    raw = case.get("schema_version", _CURRENT_SCHEMA_VERSION)
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("prompt case field `schema_version` must be int-like when present") from exc
    if version != _CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported prompt case schema_version: {version}; expected {_CURRENT_SCHEMA_VERSION}"
        )
    case["schema_version"] = _CURRENT_SCHEMA_VERSION


def _normalize_optional_list_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else None
    if not isinstance(value, list):
        raise ValueError("optional list field must be list[str], str, or None")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("optional list field must contain only str")
        candidate = item.strip()
        if candidate:
            out.append(candidate)
    return out or None


def _normalize_positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if not stripped.isdigit():
            raise ValueError("max_new_tokens string must contain digits only")
        value = int(stripped)
    if not isinstance(value, int):
        raise ValueError("max_new_tokens must be int-like")
    if value <= 0:
        raise ValueError("max_new_tokens must be positive")
    return int(value)


def validate_prompt_case(case: dict) -> None:
    _normalize_schema_version(case)

    for key in _REQUIRED_CASE_FIELDS:
        if key not in case:
            raise ValueError(f"prompt case missing required field: {key}")
    if not isinstance(case["case_id"], str) or case["case_id"].strip() == "":
        raise ValueError("prompt case field `case_id` must be non-empty str")
    if not isinstance(case["bucket"], str) or case["bucket"].strip() == "":
        raise ValueError("prompt case field `bucket` must be non-empty str")
    if not isinstance(case["prompt"], str):
        raise ValueError("prompt case field `prompt` must be str")
    if case["prompt"].strip() == "":
        raise ValueError("prompt case field `prompt` must be non-empty")
    if str(case["bucket"]) not in _ALLOWED_BUCKETS:
        raise ValueError(f"prompt case field `bucket` must be one of: {sorted(_ALLOWED_BUCKETS)}")

    stop_strings = case.get("stop_strings", [])
    if not isinstance(stop_strings, list) or any(not isinstance(x, str) for x in stop_strings):
        raise ValueError("prompt case field `stop_strings` must be list[str]")

    tags = case.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
        raise ValueError("prompt case field `tags` must be list[str]")

    max_new_tokens = case.get("max_new_tokens", None)
    if max_new_tokens is not None and (not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
        raise ValueError("prompt case field `max_new_tokens` must be positive int or None")


def load_prompt_cases(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompt suite JSONL not found: {p}")

    rows = _load_jsonl_lines(p)
    out: list[dict] = []
    seen_case_ids: set[str] = set()

    for row in rows:
        row["schema_version"] = _CURRENT_SCHEMA_VERSION if "schema_version" not in row else row["schema_version"]
        stop_strings = _normalize_optional_list_str(row.get("stop_strings"))
        tags = _normalize_optional_list_str(row.get("tags"))
        row["stop_strings"] = stop_strings if stop_strings is not None else []
        row["tags"] = tags if tags is not None else []
        row["max_new_tokens"] = _normalize_positive_int_or_none(row.get("max_new_tokens"))
        # Strip legacy/mode fields from canonical loaded shape.
        row.pop("normalize_optionals_v2", None)
        row.pop("bucket_contract_mode", None)
        validate_prompt_case(row)
        case_id = str(row["case_id"])
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id in prompt suite: {case_id}")
        seen_case_ids.add(case_id)
        out.append(dict(row))

    return out


def bucket_prompt_cases(cases: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for case in cases:
        validate_prompt_case(case)
        bucket = str(case["bucket"])
        buckets.setdefault(bucket, []).append(case)
    return buckets


def _engine_generate(engine: Any, case: dict[str, Any], *, seed: int | None) -> dict[str, Any]:
    prompt = str(case["prompt"])
    max_new_tokens = int(case.get("max_new_tokens") or 32)
    stop_strings = case.get("stop_strings")

    # Engine path: tokenize in-process and call low-level engine contract.
    if hasattr(engine, "tokenizer") and hasattr(engine, "generate"):
        prompt_ids = list(engine.tokenizer.encode(prompt))
        return engine.generate(
            prompt_ids=prompt_ids,
            attention_mask=[1] * len(prompt_ids),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            stop_strings=stop_strings,
            seed=seed,
            return_logprobs=False,
        )

    # HTTP adapter path: delegated JSON request.
    if hasattr(engine, "generate"):
        return engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            seed=seed,
        )

    raise TypeError("unsupported prompt-suite backend: expected Engine-like or HTTP adapter")


def _extract_result_fields(case: dict[str, Any], payload: dict[str, Any], *, error: str | None) -> dict[str, Any]:
    completion_text = str(payload.get("completion_text", "")) if error is None else ""
    completion_token_ids = payload.get("completion_token_ids")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    prompt = str(case["prompt"])

    out: dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "bucket": str(case["bucket"]),
        "prompt_hash": _prompt_hash(prompt),
        "completion_text": completion_text,
        "stop_reason": payload.get("stop_reason") if error is None else None,
        "prompt_len_chars": len(prompt),
        "completion_len_chars": len(completion_text),
        "prompt_len_tokens": payload.get("prompt_len_tokens"),
        "completion_len_tokens": len(completion_token_ids) if isinstance(completion_token_ids, list) else None,
        "ttft_ms": metrics.get("ttft_ms"),
        "decode_ms_per_token": metrics.get("decode_ms_per_token"),
        "tokens_per_sec": metrics.get("tokens_per_sec"),
        "safety_flags": payload.get("safety_flags", []),
        "refusal_applied": bool(payload.get("refusal_applied", False)),
        "error": error,
    }

    # Engine payloads do not include prompt token count; derive when available.
    if out["prompt_len_tokens"] is None and payload.get("all_token_ids") is not None and isinstance(completion_token_ids, list):
        out["prompt_len_tokens"] = max(len(payload["all_token_ids"]) - len(completion_token_ids), 0)

    return out


def run_prompt_case(engine, case: dict, *, seed: int | None = None) -> dict:
    validate_prompt_case(case)
    try:
        payload = _engine_generate(engine, case, seed=seed)
        return _extract_result_fields(case, payload, error=None)
    except Exception as exc:
        return _extract_result_fields(case, {}, error=_as_text_error(exc))


def run_prompt_suite(engine, cases: list[dict], *, seed: int | None = None) -> list[dict]:
    results: list[dict] = []
    for idx, case in enumerate(cases):
        case_seed = None if seed is None else int(seed) + int(idx)
        results.append(run_prompt_case(engine, case, seed=case_seed))
    return results


def summarize_prompt_suite(results: list[dict]) -> dict:
    bucket_counts: dict[str, int] = {}
    stop_reason_counts: dict[str, int] = {}
    safety_flag_counts: dict[str, int] = {}

    error_count = 0
    refusal_count = 0
    ttft_values: list[float] = []
    tps_values: list[float] = []
    for row in results:
        bucket = str(row.get("bucket", "unknown"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        reason = row.get("stop_reason")
        if isinstance(reason, str) and reason != "":
            stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1

        for flag in row.get("safety_flags", []) or []:
            flag_str = str(flag)
            safety_flag_counts[flag_str] = safety_flag_counts.get(flag_str, 0) + 1

        if row.get("error"):
            error_count += 1
        if bool(row.get("refusal_applied", False)):
            refusal_count += 1

        ttft = row.get("ttft_ms")
        if isinstance(ttft, (int, float)):
            ttft_values.append(float(ttft))
        tps = row.get("tokens_per_sec")
        if isinstance(tps, (int, float)):
            tps_values.append(float(tps))

    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / float(len(values))

    return {
        "total_cases": len(results),
        "bucket_counts": bucket_counts,
        "stop_reason_counts": stop_reason_counts,
        "safety_flag_counts": safety_flag_counts,
        "error_count": error_count,
        "refusal_count": refusal_count,
        "avg_ttft_ms": _avg(ttft_values),
        "avg_tokens_per_sec": _avg(tps_values),
    }


def write_prompt_suite_outputs(
    results: list[dict],
    summary: dict,
    out_dir: str,
    *,
    stem: str = "prompt_suite",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    outputs_path = out / f"{stem}_outputs.jsonl"
    summary_path = out / f"{stem}_summary.json"

    lines = [json.dumps(row, sort_keys=True) for row in results]
    outputs_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def compare_backend_results(engine_results: list[dict], http_results: list[dict]) -> list[dict]:
    by_engine = {str(r.get("case_id")): r for r in engine_results}
    by_http = {str(r.get("case_id")): r for r in http_results}
    all_ids = sorted(set(by_engine.keys()) | set(by_http.keys()))

    rows: list[dict] = []
    for case_id in all_ids:
        e = by_engine.get(case_id, {})
        h = by_http.get(case_id, {})

        latency_delta_fields: dict[str, float | None] = {}
        for k in ("ttft_ms", "decode_ms_per_token", "tokens_per_sec"):
            ev = e.get(k)
            hv = h.get(k)
            if isinstance(ev, (int, float)) and isinstance(hv, (int, float)):
                latency_delta_fields[k] = float(hv) - float(ev)
            else:
                latency_delta_fields[k] = None

        status = "match"
        if bool(e.get("error")) != bool(h.get("error")):
            status = "mismatch_error_state"
        elif bool(e.get("refusal_applied", False)) != bool(h.get("refusal_applied", False)):
            status = "mismatch_refusal"
        elif e.get("stop_reason") != h.get("stop_reason"):
            status = "mismatch_stop_reason"

        rows.append(
            {
                "case_id": case_id,
                "engine_stop_reason": e.get("stop_reason"),
                "http_stop_reason": h.get("stop_reason"),
                "engine_refusal": bool(e.get("refusal_applied", False)),
                "http_refusal": bool(h.get("refusal_applied", False)),
                "latency_delta_fields": latency_delta_fields,
                "match_status": status,
            }
        )
    return rows

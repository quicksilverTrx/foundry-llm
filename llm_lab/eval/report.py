# llm_lab/eval/report.py
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from llm_lab.serving.engine import CacheState, build_engine_from_package

if TYPE_CHECKING:
    from torch import nn


_REQUIRED_BENCH_KEYS = {
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


def _normalize_bench_row(payload: dict, source_path: Path) -> dict | None:
    row = dict(payload)

    # Backward-compatible normalization for legacy bench payloads.
    if "batch_size" not in row and "B" in row:
        row["batch_size"] = int(row["B"])
    if "quant_mode" not in row:
        row["quant_mode"] = "none"
    if "decode_ms_total" not in row and "decode_ms_per_token" in row and "gen_len" in row:
        row["decode_ms_total"] = float(row["decode_ms_per_token"]) * float(max(int(row["gen_len"]) - 1, 0))
    if "context_len" not in row and "prompt_len" in row:
        row["context_len"] = int(row["prompt_len"])
    if "benchmark_mode" not in row:
        stem = source_path.stem.lower()
        if "cache" in stem and "recompute" not in stem:
            row["benchmark_mode"] = "cache"
        elif "recompute" in stem:
            row["benchmark_mode"] = "recompute"
        else:
            row["benchmark_mode"] = "unknown"

    if not _REQUIRED_BENCH_KEYS.issubset(row.keys()):
        return None
    row["source_file"] = str(source_path)
    return row


def _sum_unique_storage_bytes(tensors: list[torch.Tensor]) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for t in tensors:
        storage = t.untyped_storage()
        key = (int(storage.data_ptr()), int(storage.nbytes()))
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return int(total)


def _runtime_memory_policy(model: "nn.Module", cache_state: object | None) -> int:
    tensors: list[torch.Tensor] = []
    tensors.extend([p for p in model.parameters()])
    tensors.extend([b for b in model.buffers()])

    if isinstance(cache_state, CacheState) and cache_state.past_key_values is not None:
        for k, v in cache_state.past_key_values:
            tensors.append(k)
            tensors.append(v)
    elif cache_state is not None and hasattr(cache_state, "past_key_values"):
        raw = getattr(cache_state, "past_key_values")
        if raw is not None:
            for pair in raw:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    k, v = pair
                    if isinstance(k, torch.Tensor):
                        tensors.append(k)
                    if isinstance(v, torch.Tensor):
                        tensors.append(v)
    return _sum_unique_storage_bytes(tensors)


def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def estimate_runtime_memory_bytes(
    model: "nn.Module",
    cache_state: object | None = None,
) -> int:
    return int(_runtime_memory_policy(model, cache_state))


def benchmark_inference_modes(
    package_path: str,
    device: str,
    dtype: str,
    quant_mode: str | None,
    prompt_len: int,
    gen_len: int,
    batch_size: int = 1,
    *,
    loader: str = "package",
) -> dict[str, float]:
    if batch_size != 1:
        raise ValueError("benchmark_inference_modes currently supports batch_size=1 only")

    engine = build_engine_from_package(
        package_path=package_path,
        device=device,
        dtype=dtype,
        quant_mode=quant_mode,
        loader=loader,
    )

    vocab = int(getattr(getattr(engine.model, "config", None), "vocab_size", 4096))
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (prompt_len,), dtype=torch.long).tolist()

    out = engine.generate(
        prompt_ids=prompt,
        attention_mask=[1] * len(prompt),
        max_new_tokens=int(gen_len),
        temperature=0.0,
        top_k=1,
    )
    metrics = out["metrics"]

    payload: dict[str, object] = {
        "prefill_ms": float(metrics["prefill_ms"]),
        "ttft_ms": float(metrics["ttft_ms"]),
        "decode_ms_total": float(metrics["decode_ms_total"]),
        "decode_ms_per_token": float(metrics["decode_ms_per_token"]),
        "tokens_per_sec": float(metrics["tokens_per_sec"]),
        "prompt_len": float(prompt_len),
        "gen_len": float(gen_len),
        "batch_size": float(batch_size),
        "dtype": str(metrics.get("runtime_dtype", dtype)),
        "quant_mode": str(metrics.get("runtime_quant_mode", quant_mode or "none")),
    }
    payload["runtime_fallback_reason"] = metrics.get("runtime_fallback_reason")

    missing = sorted(_REQUIRED_BENCH_KEYS - set(payload.keys()))
    if missing:
        raise RuntimeError(f"benchmark payload missing keys: {missing}")
    return payload


def summarize_precision_recommendation(
    results: list[dict[str, object]],
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in results:
        device = str(row.get("device", "unknown"))
        grouped.setdefault(device, []).append(row)

    out: dict[str, str] = {}
    for device, rows in grouped.items():
        baseline_ppl = None
        for r in rows:
            if str(r.get("mode", "")) == "fp32" and isinstance(r.get("ppl"), (int, float)):
                baseline_ppl = float(r["ppl"])
                break

        eligible: list[dict[str, object]] = []
        for r in rows:
            if r.get("supported") is False:
                continue
            mode = str(r.get("mode", "")).lower()
            if mode in {"", "unknown"}:
                continue
            ppl = r.get("ppl")
            if baseline_ppl is not None and isinstance(ppl, (int, float)) and baseline_ppl > 0:
                drift = (float(ppl) - baseline_ppl) / baseline_ppl
                if drift > 0.15:
                    continue
            eligible.append(r)

        if not eligible:
            out[f"{device}_default"] = "fp32"
            out[f"{device}_reason"] = "fallback_no_quality_eligible_mode"
            continue

        tps_vals = [float(r.get("tokens_per_sec", 0.0)) for r in eligible]
        ttft_vals = [float(r.get("ttft_ms", 0.0)) for r in eligible]
        decode_vals = [float(r.get("decode_ms_per_token", 0.0)) for r in eligible]
        mem_vals = [float(r.get("memory_bytes", r.get("runtime_memory_bytes", 0.0))) for r in eligible]

        def _norm(value: float, values: list[float], invert: bool = False) -> float:
            lo, hi = min(values), max(values)
            if hi <= lo:
                return 0.5
            x = (value - lo) / (hi - lo)
            return 1.0 - x if invert else x

        ranked: list[tuple[float, dict[str, object]]] = []
        for r in eligible:
            tps = float(r.get("tokens_per_sec", 0.0))
            ttft = float(r.get("ttft_ms", 0.0))
            decode = float(r.get("decode_ms_per_token", 0.0))
            mem = float(r.get("memory_bytes", r.get("runtime_memory_bytes", 0.0)))
            score = (
                0.50 * _norm(tps, tps_vals)
                + 0.20 * _norm(ttft, ttft_vals, invert=True)
                + 0.20 * _norm(decode, decode_vals, invert=True)
                + 0.10 * _norm(mem, mem_vals, invert=True)
            )
            ranked.append((score, r))

        def _tie_key(item: tuple[float, dict[str, object]]) -> tuple[float, float, str]:
            score, row = item
            ppl = float(row.get("ppl", 1e9))
            mode = str(row.get("mode", "zzzz"))
            return (score, -ppl, mode)

        ranked.sort(key=_tie_key, reverse=True)
        best = ranked[0][1]
        out[f"{device}_default"] = str(best.get("mode", "fp32"))
        out[f"{device}_reason"] = "best_tradeoff_quality_gated"
    return out


def _mode_row(result: dict[str, object]) -> str:
    run_state = str(result.get("run_state", "executed"))
    executed_mode = str(result.get("executed_mode", ""))
    metrics_collected = bool(result.get("metrics_collected", False))
    return (
        f"| {result.get('mode','unknown')} | {result.get('supported','unknown')} "
        f"| {run_state} | {executed_mode} | {metrics_collected} | {result.get('reason','')} |"
    )


def write_quant_report(
    results: list[dict[str, object]],
    out_path: str,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Quantization Report")
    lines.append("")
    lines.append("## Environment")
    lines.append("- device/runtime captured from artifacts")
    lines.append("- package and eval corpus recorded by caller")
    lines.append("")
    lines.append("## Field Definitions")
    lines.append("- `dtype`: effective runtime compute dtype.")
    lines.append("- `quant_mode`: quantization mode metadata and honesty fields for quantized execution state.")
    lines.append("- `memory_bytes_estimate`: deduplicated unique-storage estimate of model parameters + buffers + cache (if provided).")
    lines.append("- `run_state`: `executed` | `normalized_not_run_separately` | `excluded`.")
    lines.append("- `executed_mode`: actual executed runtime mode (for normalized modes this is typically `fp32`).")
    lines.append("- `metrics_collected`: false for normalized-not-run or excluded rows.")
    lines.append("")
    lines.append("## Run-State Semantics")
    lines.append("- CPU fp16/bf16 requests are fallback-normalized to fp32 and not benchmarked separately.")
    lines.append("- Normalized rows retain `mode=requested` but have `run_state=normalized_not_run_separately` and metric fields unset.")
    lines.append("")
    lines.append("## Supported Modes")
    lines.append("| mode | supported | run_state | executed_mode | metrics_collected | reason |")
    lines.append("|---|---|---|---|---|---|")
    for row in results:
        lines.append(_mode_row(row))
    lines.append("")
    lines.append("## Memory Comparison")
    lines.append("- `memory_bytes_estimate` is an estimate, not process RSS; it includes deduplicated parameters/buffers and optional cache tensors.")
    lines.append("")
    lines.append("## Latency Comparison")
    lines.append("- prefill_ms / ttft_ms / decode_ms_per_token / tokens_per_sec")
    lines.append("")
    lines.append("## PPL Comparison")
    lines.append("- avg_nll / ppl and drift vs fp32 baseline")
    lines.append("")
    lines.append("## Drift / Caveats")
    lines.append("- Cross-mode drift and within-mode cache equivalence notes")
    lines.append("")
    lines.append("## What Failed")
    lines.append("- List unavailable modes, backend errors, or invalid runs")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("- Generated from measured results via summarize_precision_recommendation")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def save_json(path: str, payload: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_bucket_summary(results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in results:
        bucket = str(row.get("bucket", "unknown"))
        counts[bucket] = counts.get(bucket, 0) + 1
    return {"total": len(results), "by_bucket": counts}


def build_safety_summary(results: list[dict]) -> dict:
    reason_counts: dict[str, int] = {}
    refusal_count = 0
    for row in results:
        flags = row.get("safety_flags") or []
        for flag in flags:
            k = str(flag)
            reason_counts[k] = reason_counts.get(k, 0) + 1
        if bool(row.get("refusal_applied", False)):
            refusal_count += 1
    return {
        "refusal_count": refusal_count,
        "flagged_cases": sum(1 for r in results if (r.get("safety_flags") or [])),
        "reason_counts": reason_counts,
    }


def build_latency_summary(results: list[dict]) -> dict:
    ttft_vals: list[float] = []
    tps_vals: list[float] = []
    decode_vals: list[float] = []

    for row in results:
        ttft = row.get("ttft_ms")
        tps = row.get("tokens_per_sec")
        decode = row.get("decode_ms_per_token")
        if isinstance(ttft, (int, float)):
            ttft_vals.append(float(ttft))
        if isinstance(tps, (int, float)):
            tps_vals.append(float(tps))
        if isinstance(decode, (int, float)):
            decode_vals.append(float(decode))

    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / float(len(values))

    return {
        "count_with_ttft": len(ttft_vals),
        "count_with_tokens_per_sec": len(tps_vals),
        "count_with_decode_ms_per_token": len(decode_vals),
        "avg_ttft_ms": _avg(ttft_vals),
        "avg_tokens_per_sec": _avg(tps_vals),
        "avg_decode_ms_per_token": _avg(decode_vals),
    }


def build_eval_report(summary: dict, *, ppl_artifact_path: str | None = None) -> str:
    bucket = summary.get("bucket_summary", {})
    safety = summary.get("safety_summary", {})
    latency = summary.get("latency_summary", {})
    stop_reasons = summary.get("stop_reason_counts", {})
    anomalies = summary.get("notable_failures", [])

    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## scope")
    lines.append("- eval suite + safety regressions + evidence synthesis")
    lines.append("")
    lines.append("## suite size and bucket breakdown")
    lines.append(f"- total cases: {bucket.get('total', 0)}")
    for k, v in sorted((bucket.get("by_bucket") or {}).items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## stop reason breakdown")
    if stop_reasons:
        for k, v in sorted(stop_reasons.items()):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- no stop reason data")
    lines.append("")
    lines.append("## latency summary")
    lines.append(f"- avg ttft_ms: {latency.get('avg_ttft_ms')}")
    lines.append(f"- avg decode_ms_per_token: {latency.get('avg_decode_ms_per_token')}")
    lines.append(f"- avg tokens_per_sec: {latency.get('avg_tokens_per_sec')}")
    lines.append("")
    lines.append("## safety summary")
    lines.append(f"- refusal_count: {safety.get('refusal_count', 0)}")
    lines.append(f"- flagged_cases: {safety.get('flagged_cases', 0)}")
    for k, v in sorted((safety.get("reason_counts") or {}).items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## notable failures / anomalous cases")
    if anomalies:
        for item in anomalies:
            lines.append(f"- {item}")
    else:
        lines.append("- no anomalies flagged by current policy")
    lines.append("")
    lines.append("## PPL / quantization artifacts")
    if ppl_artifact_path:
        lines.append(f"- ppl artifact: {ppl_artifact_path}")
    else:
        lines.append("- ppl artifact: not provided")
    lines.append("")
    lines.append("## next-action items")
    lines.append("- Review safety refusals and false-positive rates by bucket.")
    lines.append("- Compare this run against prior baseline for regression deltas.")
    lines.append("- Update benchmark coverage and rerun final-close validation before sign-off.")
    lines.append("")
    return "\n".join(lines)


def write_eval_report(report_text: str, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text, encoding="utf-8")


def load_bench_artifacts(bench_dir: str) -> dict:
    path = Path(bench_dir)
    if not path.exists():
        raise FileNotFoundError(f"bench_dir does not exist: {path}")

    rows: list[dict] = []
    sources: dict[str, str] = {}
    for p in sorted(path.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            row = _normalize_bench_row(payload, p)
            if row is not None:
                rows.append(row)
                sources[p.stem] = str(p)
        elif isinstance(payload, list):
            accepted = 0
            for item in payload:
                if not isinstance(item, dict):
                    continue
                row = _normalize_bench_row(item, p)
                if row is None:
                    continue
                rows.append(row)
                accepted += 1
            if accepted > 0:
                sources[p.stem] = str(p)

    if not rows:
        raise FileNotFoundError(f"no benchmark JSON with required schema in: {path}")
    return {"rows": rows, "sources": sources}


def build_ttft_tps_table(bench_data: dict) -> list[dict]:
    rows = bench_data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("bench_data must contain `rows` list")

    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "context_len": int(row["prompt_len"]),
                "batch_size": int(row["batch_size"]),
                "benchmark_mode": str(row.get("benchmark_mode", "unknown")),
                "dtype": str(row.get("dtype", "unknown")),
                "quant_mode": str(row.get("quant_mode", "none")),
                "prefill_ms": float(row["prefill_ms"]),
                "ttft_ms": float(row["ttft_ms"]),
                "decode_ms_per_token": float(row["decode_ms_per_token"]),
                "tokens_per_sec": float(row["tokens_per_sec"]),
                "source_file": str(row.get("source_file", "")),
            }
        )
    return out


def estimate_kv_cache_bytes(
    *,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int,
) -> int:
    # K + V tensors => factor 2.
    return int(
        2
        * int(batch_size)
        * int(seq_len)
        * int(n_layers)
        * int(n_kv_heads)
        * int(head_dim)
        * int(bytes_per_elem)
    )


def build_kv_memory_economics_table(config_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in config_rows:
        est_bytes = estimate_kv_cache_bytes(
            batch_size=int(row["batch_size"]),
            seq_len=int(row["seq_len"]),
            n_layers=int(row["n_layers"]),
            n_kv_heads=int(row["n_kv_heads"]),
            head_dim=int(row["head_dim"]),
            bytes_per_elem=int(row["bytes_per_elem"]),
        )
        out.append(
            {
                "arch_variant": str(row["arch_variant"]),
                "batch_size": int(row["batch_size"]),
                "seq_len": int(row["seq_len"]),
                "n_layers": int(row["n_layers"]),
                "n_kv_heads": int(row["n_kv_heads"]),
                "head_dim": int(row["head_dim"]),
                "dtype": str(row.get("dtype", "unknown")),
                "estimated_kv_cache_bytes": est_bytes,
                "estimated_kv_cache_mb": float(est_bytes) / (1024.0 * 1024.0),
            }
        )
    return out


def build_evidence_pack_manifest(paths: dict[str, str]) -> dict:
    required = {
        "cache_equivalence_report_md",
        "ttft_tps_table_json",
        "ttft_tps_report_md",
        "kv_cache_memory_economics_json",
        "kv_cache_memory_economics_md",
        "eval_manifest_json",
    }
    missing = sorted(required - set(paths.keys()))
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    return dict(paths)


def write_json_artifact(obj: dict | list, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_text_artifact(text: str, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

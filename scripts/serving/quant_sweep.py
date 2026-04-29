# scripts/quant_sweep.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from llm_lab.eval.ppl import evaluate_streaming_nll
from llm_lab.eval.report import (
    benchmark_inference_modes,
    estimate_runtime_memory_bytes,
    save_json,
    summarize_precision_recommendation,
    write_quant_report,
)
from llm_lab.serving.engine import build_engine_from_package
from llm_lab.serving.precision import runtime_precision_decision
from llm_lab.serving.quant import describe_quant_runtime

_MEMORY_METRIC_DEFINITION = (
    "deduplicated_unique_storage_estimate_of_model_parameters_buffers_and_optional_cache_bytes"
)
_DTYPE_DEFINITION = "effective_runtime_compute_dtype"


def _mock_bench(*, prompt_len: int, gen_len: int, dtype: str, quant_mode: str | None) -> dict[str, object]:
    prefill_ms = 2.0 if quant_mode is None else 2.4
    decode_ms_per_token = 0.8 if quant_mode is None else 0.6
    decode_ms_total = decode_ms_per_token * float(max(gen_len - 1, 0))
    ttft_ms = prefill_ms + decode_ms_per_token
    tps = 1000.0 / max(decode_ms_per_token, 1e-9)
    return {
        "prefill_ms": float(prefill_ms),
        "ttft_ms": float(ttft_ms),
        "decode_ms_total": float(decode_ms_total),
        "decode_ms_per_token": float(decode_ms_per_token),
        "tokens_per_sec": float(tps),
        "prompt_len": int(prompt_len),
        "gen_len": int(gen_len),
        "batch_size": 1,
        "dtype": str(dtype),
        "quant_mode": str(quant_mode or "none"),
    }


def _mock_ppl(*, dtype: str, quant_mode: str | None, device: str) -> dict[str, object]:
    avg_nll = 2.3 if quant_mode is None else 2.45
    return {
        "n_tokens": 1024.0,
        "total_nll": float(avg_nll * 1024.0),
        "avg_nll": float(avg_nll),
        "ppl": float(torch.exp(torch.tensor(avg_nll)).item()),
        "device": str(device),
        "dtype": str(dtype),
        "quant_mode": str(quant_mode or "none"),
    }


def _mode_tag(dtype: str, quant_mode: str | None) -> str:
    if quant_mode == "int8":
        return "int8"
    return dtype


def _normalize_quant_mode_value(value: object) -> str | None:
    if value is None:
        return None
    norm = str(value).strip().lower()
    if norm in {"", "none", "null"}:
        return "none"
    return norm


def _run_state_for_mode(
    *,
    requested_dtype: str,
    runtime_dtype: str,
    requested_quant_mode: str | None,
    runtime_quant_mode: str | None,
) -> tuple[str, str]:
    if requested_quant_mode is not None and runtime_quant_mode != requested_quant_mode:
        return "excluded", "none"
    if requested_quant_mode is None and requested_dtype != runtime_dtype:
        return "normalized_not_run_separately", runtime_dtype
    return "executed", _mode_tag(runtime_dtype, runtime_quant_mode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quant precision sweep harness")
    p.add_argument("--package", type=str, default=None, help="Required unless --tiny-mock")
    p.add_argument("--text-path", type=str, default=None, help="Required unless --tiny-mock")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-len", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="experiments/serving_quant")
    p.add_argument("--loader", type=str, default="package", choices=["package", "nanollama"])
    p.add_argument("--tiny-mock", action="store_true")
    p.add_argument("--strict-recommendation", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.tiny_mock:
        if not args.package:
            raise ValueError("--package is required unless --tiny-mock")
        if not args.text_path:
            raise ValueError("--text-path is required unless --tiny-mock")

    mode_specs: list[tuple[str, str | None]] = [
        ("fp32", None),
        ("fp16", None),
        ("bf16", None),
        ("fp32", "int8"),
    ]

    precision_matrix: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for requested_dtype, requested_quant_mode in mode_specs:
        tag = _mode_tag(requested_dtype, requested_quant_mode)

        runtime_dtype, precision_reason = runtime_precision_decision(requested_dtype, args.device)
        quant_info = describe_quant_runtime(requested_quant_mode, args.device)
        runtime_quant_mode = quant_info.get("runtime_quant_mode")
        quant_reason = quant_info.get("quant_fallback_reason")
        run_state, executed_mode = _run_state_for_mode(
            requested_dtype=requested_dtype,
            runtime_dtype=runtime_dtype,
            requested_quant_mode=requested_quant_mode,
            runtime_quant_mode=runtime_quant_mode,
        )

        dtype_supported = runtime_dtype == requested_dtype
        quant_supported = requested_quant_mode is None or runtime_quant_mode == requested_quant_mode
        supported = bool(dtype_supported and quant_supported and run_state == "executed")
        reason = precision_reason or quant_reason

        row = {
            "mode": tag,
            "requested_dtype": requested_dtype,
            "runtime_dtype": runtime_dtype,
            "requested_quant_mode": requested_quant_mode,
            "runtime_quant_mode": runtime_quant_mode,
            "supported": supported,
            "run_state": run_state,
            "executed_mode": executed_mode,
            "metrics_collected": bool(run_state == "executed"),
            "reason": reason,
            "device": args.device,
        }
        precision_matrix.append(row)

        if run_state != "executed":
            fallback_note = ""
            if run_state == "normalized_not_run_separately":
                fallback_note = (
                    f"requested mode normalized to executed runtime '{executed_mode}' "
                    "and not benchmarked as a separate mode"
                )
            results.append(
                {
                    "mode": tag,
                    "supported": False,
                    "run_state": run_state,
                    "executed_mode": executed_mode,
                    "metrics_collected": False,
                    "reason": reason or fallback_note or "unsupported",
                    "device": args.device,
                    "dtype": None,
                    "quant_mode": None,
                    "prefill_ms": None,
                    "ttft_ms": None,
                    "decode_ms_total": None,
                    "decode_ms_per_token": None,
                    "tokens_per_sec": None,
                    "avg_nll": None,
                    "ppl": None,
                    "memory_bytes_estimate": None,
                    "memory_metric_definition": _MEMORY_METRIC_DEFINITION,
                    "dtype_definition": _DTYPE_DEFINITION,
                }
            )
            continue

        if args.tiny_mock:
            bench = _mock_bench(
                prompt_len=args.prompt_len,
                gen_len=args.gen_len,
                dtype=runtime_dtype,
                quant_mode=requested_quant_mode,
            )
            ppl = _mock_ppl(dtype=runtime_dtype, quant_mode=requested_quant_mode, device=args.device)
            memory_bytes = 0
        else:
            bench = benchmark_inference_modes(
                package_path=str(args.package),
                device=args.device,
                dtype=requested_dtype,
                quant_mode=requested_quant_mode,
                prompt_len=args.prompt_len,
                gen_len=args.gen_len,
                batch_size=1,
                loader=args.loader,
            )
            engine = build_engine_from_package(
                package_path=str(args.package),
                device=args.device,
                dtype=requested_dtype,
                quant_mode=requested_quant_mode,
                loader=args.loader,
            )
            ppl = evaluate_streaming_nll(
                model=engine.model,
                tokenizer=engine.tokenizer,
                text_path=str(args.text_path),
                device=args.device,
                max_seq_len=args.max_seq_len,
                stride=args.stride,
            )
            memory_bytes = estimate_runtime_memory_bytes(engine.model, cache_state=None)

        save_json(str(out_dir / f"bench_{tag}.json"), bench)
        save_json(str(out_dir / f"ppl_{tag}.json"), ppl)

        results.append(
            {
                "mode": tag,
                "supported": True,
                "run_state": "executed",
                "executed_mode": executed_mode,
                "metrics_collected": True,
                "reason": "",
                "device": args.device,
                "dtype": bench.get("dtype"),
                "quant_mode": _normalize_quant_mode_value(bench.get("quant_mode")),
                "prefill_ms": bench.get("prefill_ms"),
                "ttft_ms": bench.get("ttft_ms"),
                "decode_ms_total": bench.get("decode_ms_total"),
                "decode_ms_per_token": bench.get("decode_ms_per_token"),
                "tokens_per_sec": bench.get("tokens_per_sec"),
                "avg_nll": ppl.get("avg_nll"),
                "ppl": ppl.get("ppl"),
                "memory_bytes_estimate": memory_bytes,
                "memory_metric_definition": _MEMORY_METRIC_DEFINITION,
                "dtype_definition": _DTYPE_DEFINITION,
                "drift_note": "compare vs fp32 baseline in downstream analysis",
            }
        )

    save_json(str(out_dir / "precision_matrix.json"), {"modes": precision_matrix})
    save_json(
        str(out_dir / "quant_results.json"),
        {
            "dtype_definition": _DTYPE_DEFINITION,
            "memory_metric_definition": _MEMORY_METRIC_DEFINITION,
            "run_state_definition": "executed|normalized_not_run_separately|excluded",
            "results": results,
        },
    )

    write_quant_report(results=results, out_path=str(out_dir / "quant_report.md"))

    try:
        recommendation = summarize_precision_recommendation(results)
    except NotImplementedError as exc:
        if args.strict_recommendation:
            raise
        recommendation = {
            "status": "todo_user",
            "message": str(exc),
        }
    save_json(str(out_dir / "recommendation.json"), recommendation)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

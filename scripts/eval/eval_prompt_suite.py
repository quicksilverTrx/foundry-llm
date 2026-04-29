# scripts/eval_prompt_suite.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_lab.eval.prompt_suite import (
    compare_backend_results,
    HttpGenerateClient,
    load_prompt_cases,
    run_prompt_suite,
    summarize_prompt_suite,
    write_prompt_suite_outputs,
)
from llm_lab.eval.report import (
    build_bucket_summary,
    build_latency_summary,
    build_eval_report,
    build_safety_summary,
    write_eval_report,
)
from llm_lab.serving.engine import build_engine_from_package


_CLI_ARGS: argparse.Namespace | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt suite runner")
    p.add_argument("--backend", choices=["both", "engine", "http"], default="both")
    p.add_argument("--package", type=str, default="experiments/p1_pos_enc/runs/rope/package")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--prompts", type=str, default="data/serving_eval/prompts.jsonl")
    p.add_argument("--rubric", type=str, default="data/serving_eval/rubric.md")
    p.add_argument("--out-dir", type=str, default="experiments/serving_reports")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="fp32")
    p.add_argument("--quant-mode", type=str, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ppl-artifact", type=str, default="experiments/serving_quant/ppl_fp32.json")
    p.add_argument("--loader", type=str, default="package", choices=["package", "nanollama"])
    return p.parse_args()


def build_engine_from_args():
    if _CLI_ARGS is None:
        raise RuntimeError("build_engine_from_args called before CLI args initialization")
    return build_engine_from_package(
        package_path=_CLI_ARGS.package,
        device=_CLI_ARGS.device,
        dtype=_CLI_ARGS.dtype,
        quant_mode=_CLI_ARGS.quant_mode,
        loader=_CLI_ARGS.loader,
    )


def _write_prompt_suite_report(*, out_dir: Path, summary: dict, rubric_path: str) -> Path:
    p = out_dir / "prompt_suite_report.md"
    lines: list[str] = []
    lines.append("# Prompt Suite Report")
    lines.append("")
    lines.append(f"- total_cases: {summary.get('total_cases', 0)}")
    lines.append(f"- error_count: {summary.get('error_count', 0)}")
    lines.append(f"- refusal_count: {summary.get('refusal_count', 0)}")
    lines.append("")
    lines.append("## bucket_counts")
    for k, v in sorted((summary.get("bucket_counts") or {}).items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## stop_reason_counts")
    for k, v in sorted((summary.get("stop_reason_counts") or {}).items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## safety_flag_counts")
    for k, v in sorted((summary.get("safety_flag_counts") or {}).items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## rubric")
    lines.append(f"- source: {rubric_path}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_backend_parity(parity_rows: list[dict], out_dir: Path) -> Path:
    p = out_dir / "prompt_suite_backend_parity.json"
    p.write_text(json.dumps(parity_rows, indent=2, sort_keys=True), encoding="utf-8")
    return p


def _run_engine(cases: list[dict], seed: int) -> tuple[list[dict], dict]:
    engine = build_engine_from_args()
    results = run_prompt_suite(engine, cases, seed=seed)
    summary = summarize_prompt_suite(results)
    return results, summary


def _run_http(cases: list[dict], seed: int) -> tuple[list[dict], dict]:
    if _CLI_ARGS is None:
        raise RuntimeError("CLI args not initialized")
    http_backend = HttpGenerateClient(base_url=_CLI_ARGS.base_url)
    results = run_prompt_suite(http_backend, cases, seed=seed)
    summary = summarize_prompt_suite(results)
    return results, summary


def _require_http_health(http_summary: dict) -> None:
    total = int(http_summary.get("total_cases", 0))
    errors = int(http_summary.get("error_count", 0))
    if total > 0 and errors == total:
        raise RuntimeError(
            "HTTP backend failed for all prompt-suite cases; ensure serving API is running and reachable."
        )


def main() -> None:
    global _CLI_ARGS
    _CLI_ARGS = parse_args()
    out_dir = Path(_CLI_ARGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_prompt_cases(_CLI_ARGS.prompts)

    engine_results: list[dict] = []
    engine_summary: dict = {}
    http_results: list[dict] = []
    http_summary: dict = {}

    if _CLI_ARGS.backend in {"both", "engine"}:
        engine_results, engine_summary = _run_engine(cases, _CLI_ARGS.seed)
        write_prompt_suite_outputs(engine_results, engine_summary, str(out_dir), stem="prompt_suite_engine")

    if _CLI_ARGS.backend in {"both", "http"}:
        http_results, http_summary = _run_http(cases, _CLI_ARGS.seed)
        _require_http_health(http_summary)
        write_prompt_suite_outputs(http_results, http_summary, str(out_dir), stem="prompt_suite_http")

    if _CLI_ARGS.backend == "both":
        parity = compare_backend_results(engine_results, http_results)
        _write_backend_parity(parity, out_dir)

    # Canonical outputs/report source: HTTP results when available (safety-measured path).
    if http_results:
        canonical_results, canonical_summary = http_results, http_summary
    elif engine_results:
        canonical_results, canonical_summary = engine_results, engine_summary
    else:
        raise RuntimeError("No backend executed; this should not happen")

    write_prompt_suite_outputs(canonical_results, canonical_summary, str(out_dir), stem="prompt_suite")
    _write_prompt_suite_report(out_dir=out_dir, summary=canonical_summary, rubric_path=_CLI_ARGS.rubric)

    eval_summary = {
        "bucket_summary": build_bucket_summary(canonical_results),
        "safety_summary": build_safety_summary(canonical_results),
        "latency_summary": build_latency_summary(canonical_results),
        "stop_reason_counts": dict(canonical_summary.get("stop_reason_counts", {})),
        "notable_failures": [r["case_id"] for r in canonical_results if r.get("error")],
    }
    report_text = build_eval_report(eval_summary, ppl_artifact_path=_CLI_ARGS.ppl_artifact)
    write_eval_report(report_text, str(out_dir / "eval_report.md"))

    manifest = {
        "prompt_suite_outputs_jsonl": str(out_dir / "prompt_suite_outputs.jsonl"),
        "prompt_suite_summary_json": str(out_dir / "prompt_suite_summary.json"),
        "prompt_suite_report_md": str(out_dir / "prompt_suite_report.md"),
        "eval_report.md": str(out_dir / "eval_report.md"),
    }
    if engine_results:
        manifest["prompt_suite_engine_outputs_jsonl"] = str(out_dir / "prompt_suite_engine_outputs.jsonl")
        manifest["prompt_suite_engine_summary_json"] = str(out_dir / "prompt_suite_engine_summary.json")
    if http_results:
        manifest["prompt_suite_http_outputs_jsonl"] = str(out_dir / "prompt_suite_http_outputs.jsonl")
        manifest["prompt_suite_http_summary_json"] = str(out_dir / "prompt_suite_http_summary.json")
    if _CLI_ARGS.backend == "both":
        manifest["prompt_suite_backend_parity_json"] = str(out_dir / "prompt_suite_backend_parity.json")

    (out_dir / "prompt_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

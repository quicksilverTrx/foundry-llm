# tests/eval/test_evidence_pack.py
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_lab.eval.report import (
    build_evidence_pack_manifest,
    build_kv_memory_economics_table,
    build_ttft_tps_table,
    estimate_kv_cache_bytes,
    load_bench_artifacts,
    write_json_artifact,
    write_text_artifact,
)


_REQUIRED = {
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


def _bench_row(prompt_len: int, batch_size: int, *, dtype: str = "fp32") -> dict:
    return {
        "prefill_ms": 10.0,
        "ttft_ms": 11.0,
        "decode_ms_total": 100.0,
        "decode_ms_per_token": 2.5,
        "tokens_per_sec": 400.0,
        "prompt_len": prompt_len,
        "gen_len": 64,
        "batch_size": batch_size,
        "dtype": dtype,
        "quant_mode": "none",
    }


def test_load_bench_artifacts_reads_required_files(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(_bench_row(256, 1)), encoding="utf-8")

    out = load_bench_artifacts(str(tmp_path))
    assert "rows" in out
    assert len(out["rows"]) == 1
    assert _REQUIRED.issubset(out["rows"][0].keys())


def test_load_bench_artifacts_missing_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = load_bench_artifacts(str(tmp_path))


def test_build_ttft_tps_table_has_required_columns() -> None:
    bench_data = {
        "rows": [
            _bench_row(128, 1),
            _bench_row(256, 2),
            _bench_row(512, 4),
            _bench_row(512, 8),
        ]
    }
    table = build_ttft_tps_table(bench_data)
    required_cols = {
        "context_len",
        "batch_size",
        "prefill_ms",
        "ttft_ms",
        "decode_ms_per_token",
        "tokens_per_sec",
    }
    for row in table:
        assert required_cols.issubset(row.keys())

    assert {128, 256, 512}.issubset({int(r["context_len"]) for r in table})
    assert {1, 2, 4, 8}.issubset({int(r["batch_size"]) for r in table})
    assert all("synthetic" not in r for r in table)


def test_estimate_kv_cache_bytes_matches_formula() -> None:
    out = estimate_kv_cache_bytes(
        batch_size=2,
        seq_len=4,
        n_layers=3,
        n_kv_heads=5,
        head_dim=7,
        bytes_per_elem=2,
    )
    expected = 2 * 2 * 4 * 3 * 5 * 7 * 2
    assert out == expected


def test_build_kv_memory_economics_table_includes_mha_and_gqa() -> None:
    rows = [
        {
            "arch_variant": "MHA",
            "batch_size": 1,
            "seq_len": 256,
            "n_layers": 12,
            "n_kv_heads": 12,
            "head_dim": 64,
            "bytes_per_elem": 2,
            "dtype": "fp16",
        },
        {
            "arch_variant": "GQA",
            "batch_size": 1,
            "seq_len": 256,
            "n_layers": 12,
            "n_kv_heads": 4,
            "head_dim": 64,
            "bytes_per_elem": 2,
            "dtype": "fp16",
        },
    ]
    out = build_kv_memory_economics_table(rows)
    variants = {r["arch_variant"] for r in out}
    assert any(v.startswith("MHA") for v in variants)
    assert any(v.startswith("GQA") for v in variants)

    mha = [r for r in out if r["arch_variant"].startswith("MHA")][0]
    gqa = [r for r in out if r["arch_variant"].startswith("GQA")][0]
    assert gqa["estimated_kv_cache_bytes"] < mha["estimated_kv_cache_bytes"]


def test_build_evidence_pack_manifest_points_to_existing_files(tmp_path: Path) -> None:
    files = {
        "cache_equivalence_report_md": tmp_path / "cache_equivalence_report.md",
        "ttft_tps_table_json": tmp_path / "ttft_tps_table.json",
        "ttft_tps_report_md": tmp_path / "ttft_tps_report.md",
        "kv_cache_memory_economics_json": tmp_path / "kv_cache_memory_economics.json",
        "kv_cache_memory_economics_md": tmp_path / "kv_cache_memory_economics.md",
        "eval_manifest_json": tmp_path / "eval_manifest.json",
    }
    for p in files.values():
        p.write_text("x", encoding="utf-8")

    manifest = build_evidence_pack_manifest({k: str(v) for k, v in files.items()})
    for _, p in manifest.items():
        if p.endswith(".json") or p.endswith(".md"):
            assert Path(p).exists()


def test_reports_are_written_nonempty(tmp_path: Path) -> None:
    json_out = tmp_path / "table.json"
    md_out = tmp_path / "report.md"
    write_json_artifact([{"a": 1}], str(json_out))
    write_text_artifact("# ok\n", str(md_out))

    assert json_out.exists() and json_out.read_text(encoding="utf-8").strip() != ""
    assert md_out.exists() and md_out.read_text(encoding="utf-8").strip() != ""


def test_build_ttft_tps_table_ignores_policy_mode_field() -> None:
    table = build_ttft_tps_table(
        {
            "coverage_policy_mode": "advanced_v2",
            "rows": [
                {
                    "prefill_ms": 1.0,
                    "ttft_ms": 1.0,
                    "decode_ms_per_token": 1.0,
                    "tokens_per_sec": 1.0,
                    "prompt_len": 128,
                    "batch_size": 1,
                    "dtype": "fp32",
                    "quant_mode": "none",
                }
            ],
        }
    )
    assert len(table) == 1


def test_strict_grid_validator_fails_on_missing_points() -> None:
    from scripts.eval.make_evidence_pack import _validate_required_grid

    partial = [{"context_len": 128, "batch_size": 1}]
    with pytest.raises(ValueError, match="missing measured TTFT/TPS grid points"):
        _validate_required_grid(partial)


def test_strict_grid_validator_accepts_full_points() -> None:
    from scripts.eval.make_evidence_pack import _validate_required_grid

    full = [{"context_len": c, "batch_size": b} for c in (128, 256, 512) for b in (1, 2, 4, 8)]
    _validate_required_grid(full)


def test_evidence_script_grid_validation_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("evidence_mod", Path("scripts/eval/make_evidence_pack.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = SimpleNamespace(
        bench_dir=str(tmp_path / "bench"),
        reports_dir=str(tmp_path / "reports"),
        quant_dir=str(tmp_path / "quant"),
        out_dir=str(tmp_path / "out"),
        cache_divergence=str(tmp_path / "bench" / "missing.json"),
        cache_receipts=str(tmp_path / "bench" / "missing_receipts.json"),
        package_dir=str(tmp_path / "package"),
        validate_grid=False,
    )
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "load_bench_artifacts", lambda _: {"rows": [{"prompt_len": 128, "batch_size": 1}]})
    monkeypatch.setattr(
        mod,
        "build_ttft_tps_table",
        lambda _: [
            {
                "context_len": 128,
                "batch_size": 1,
                "dtype": "fp32",
                "quant_mode": "none",
                "prefill_ms": 1.0,
                "ttft_ms": 1.0,
                "decode_ms_per_token": 1.0,
                "tokens_per_sec": 1.0,
            }
        ],
    )
    monkeypatch.setattr(mod, "build_kv_memory_economics_table", lambda _: [])
    monkeypatch.setattr(mod, "build_evidence_pack_manifest", lambda d: d)
    monkeypatch.setattr(mod, "_write_cache_equivalence_report", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_validate_required_grid", lambda _: (_ for _ in ()).throw(RuntimeError("called")))
    monkeypatch.setattr(mod, "_load_package_config", lambda _: {"n_layers": 4, "n_heads": 4, "d_model": 256})

    # generator mode: must not call grid validator
    mod.main()

    args.validate_grid = True
    with pytest.raises(RuntimeError, match="called"):
        mod.main()


def test_cache_equivalence_report_surfaces_receipt_fields(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("evidence_mod", Path("scripts/eval/make_evidence_pack.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "cache_equivalence_report.md"
    receipts = tmp_path / "cache_equivalence_receipts.json"
    receipts.write_text(
        json.dumps(
            {
                "tested_prompt_lengths": [128, 256],
                "tested_generation_lengths": [16],
                "dtype": "fp32",
                "quant_mode": None,
                "greedy": True,
                "case_count": 2,
                "tolerance": 1e-4,
                "within_mode_cached_vs_recompute": True,
                "provenance": {"package_path": "x"},
            }
        ),
        encoding="utf-8",
    )

    mod._write_cache_equivalence_report(  # type: ignore[attr-defined]
        out_path=out,
        divergence_path=tmp_path / "missing_divergence.json",
        receipts_path=receipts,
        bench_dir=tmp_path,
    )

    text = out.read_text(encoding="utf-8")
    assert "tested prompt lengths" in text
    assert "tested generation lengths" in text
    assert "case count" in text
    assert "within-mode cached-vs-recompute comparison" in text

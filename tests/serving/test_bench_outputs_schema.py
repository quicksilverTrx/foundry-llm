# tests/serving/test_bench_outputs_schema.py
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _required_keys() -> set[str]:
    return {
        "prefill_ms",
        "ttft_ms",
        "decode_ms_total",
        "decode_ms_per_token",
        "tokens_per_sec",
        "context_len",
        "device",
        "dtype",
        "quant_mode",
        "batch_size",
        "B",
        "prompt_len",
        "gen_len",
        "n_iters",
        "warmup",
        "seed",
        "benchmark_mode",
        "prefill_batched",
        "decode_batched",
        "repeats_total",
        "package_path",
        "model_config_hash",
        "tokenizer_hash",
        "timestamp",
    }


def test_bench_outputs_schema_tiny_mode(tmp_path: Path):
    out_dir = tmp_path / "bench"
    script = Path("scripts/serving/bench_inference.py")

    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "both",
        "--prompt-len",
        "8",
        "--gen-len",
        "4",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--out-dir",
        str(out_dir),
        "--tiny-mock",
    ]
    subprocess.run(cmd, check=True)

    recompute_path = out_dir / "recompute.json"
    cache_path = out_dir / "cache.json"

    assert recompute_path.exists(), f"Missing benchmark artifact: {recompute_path}"
    assert cache_path.exists(), f"Missing benchmark artifact: {cache_path}"

    recompute = json.loads(recompute_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    req = _required_keys()
    assert req.issubset(recompute.keys())
    assert req.issubset(cache.keys())

    for payload in (recompute, cache):
        assert payload["ttft_ms"] > 0
        assert payload["decode_ms_per_token"] > 0
        assert math.isfinite(payload["tokens_per_sec"])


def test_bench_outputs_real_device_thresholds(tmp_path: Path):
    package_dir = Path(os.environ.get("SERVING_PACKAGE_DIR", "experiments/p1_pos_enc/runs/rope/package"))
    if not package_dir.exists():
        pytest.skip(f"SERVING_PACKAGE_DIR missing: {package_dir}")

    out_dir = tmp_path / "bench-real"
    script = Path("scripts/serving/bench_inference.py")
    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "both",
        "--prompt-len",
        "16",
        "--gen-len",
        "8",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--out-dir",
        str(out_dir),
        "--package-dir",
        str(package_dir),
    ]
    subprocess.run(cmd, check=True)

    recompute = json.loads((out_dir / "recompute.json").read_text(encoding="utf-8"))
    cache = json.loads((out_dir / "cache.json").read_text(encoding="utf-8"))

    assert recompute["decode_ms_per_token"] > 0
    assert cache["decode_ms_per_token"] > 0
    assert math.isfinite(recompute["tokens_per_sec"])
    assert math.isfinite(cache["tokens_per_sec"])

    device = str(cache["device"])
    if device.startswith("cuda"):
        # CUDA should show a clear decode-latency win with cache.
        assert cache["decode_ms_per_token"] <= recompute["decode_ms_per_token"] * 1.08
    elif device.startswith("mps"):
        # MPS can be noisier than CUDA, but cache should still avoid major regressions.
        assert cache["decode_ms_per_token"] <= recompute["decode_ms_per_token"] * 1.75
    else:
        # CPU baseline: cache may not always dominate on tiny runs, but should stay in a reasonable band.
        assert cache["decode_ms_per_token"] <= recompute["decode_ms_per_token"] * 1.75


def test_bench_grid_tiny_mode_emits_batch_aware_rows(tmp_path: Path):
    out_dir = tmp_path / "bench-grid"
    script = Path("scripts/serving/bench_inference.py")
    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "both",
        "--context-lens",
        "128,256,512",
        "--batch-sizes",
        "1,2,4,8",
        "--gen-len",
        "8",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--repeats",
        "3",
        "--out-dir",
        str(out_dir),
        "--tiny-mock",
    ]
    subprocess.run(cmd, check=True)

    cache_grid = json.loads((out_dir / "cache_grid.json").read_text(encoding="utf-8"))
    rec_grid = json.loads((out_dir / "recompute_grid.json").read_text(encoding="utf-8"))
    assert len(cache_grid) == 12
    assert len(rec_grid) == 12

    for row in cache_grid + rec_grid:
        assert row["context_len"] in {128, 256, 512}
        assert row["batch_size"] in {1, 2, 4, 8}
        assert row["benchmark_mode"] in {"cache", "recompute"}
        assert bool(row["decode_batched"]) is False
        assert bool(row["prefill_batched"]) == (int(row["batch_size"]) > 1)

    cache_raw = (out_dir / "raw" / "cache_repeats.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec_raw = (out_dir / "raw" / "recompute_repeats.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # 3 contexts x 4 batches x 3 repeats
    assert len(cache_raw) == 36
    assert len(rec_raw) == 36

# scripts/bench_inference.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llm_lab.serving.engine import Engine, build_engine_from_package


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def save_json(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_json_list(path: str, payload: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True) for r in rows]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return _sha256_bytes(path.read_bytes())


def _sha256_dir(path: Path) -> str | None:
    if not path.exists() or not path.is_dir():
        return None
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = p.relative_to(path).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _build_provenance(
    *,
    package_dir: Path,
    device: str,
    dtype: str,
    quant_mode: str | None,
) -> dict[str, Any]:
    return {
        "package_path": str(package_dir.resolve()),
        "model_config_hash": _sha256_file(package_dir / "config.json"),
        "tokenizer_hash": _sha256_dir(package_dir / "tokenizer"),
        "device": device,
        "dtype": dtype,
        "quant_mode": quant_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _median_ms(samples: list[float]) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _parse_int_csv(raw: str | None) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "":
        return []
    out: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if token == "":
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"expected positive integer in CSV list, got: {value}")
        out.append(value)
    return out


def _stat_block(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "cv": float("nan")}
    mean = float(sum(values) / float(len(values)))
    std = float(statistics.stdev(values)) if len(values) >= 2 else 0.0
    min_v = float(min(values))
    max_v = float(max(values))
    cv = float(std / mean) if abs(mean) > 1e-12 else float("inf")
    return {"mean": mean, "std": std, "min": min_v, "max": max_v, "cv": cv}


def _summarize_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty repeat rows")
    metric_keys = (
        "prefill_ms",
        "ttft_ms",
        "decode_ms_total",
        "decode_ms_per_token",
        "tokens_per_sec",
    )
    stats = {k: _stat_block([float(r[k]) for r in rows]) for k in metric_keys}
    base = dict(rows[0])
    base["n_repeats"] = int(len(rows))
    base["stats"] = stats
    for k in metric_keys:
        base[k] = float(stats[k]["mean"])
    base.pop("repeat_idx", None)
    base.pop("repeat_seed", None)
    return base


def bench_recompute(model, prompt_ids: torch.Tensor, steps: int, *, warmup: int, iters: int) -> dict[str, Any]:
    max_ctx = int(getattr(getattr(model, "config", None), "block_size", prompt_ids.shape[1]))

    @torch.no_grad()
    def _single_run() -> tuple[float, float, float]:
        cur_ids = prompt_ids.clone()

        _sync_device(cur_ids.device)
        t0 = torch.cuda.Event(enable_timing=True) if cur_ids.device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if cur_ids.device.type == "cuda" else None
        if t0 is not None:
            t0.record()
        else:
            import time

            wall0 = time.perf_counter()
        logits, _ = model(cur_ids, attention_mask=torch.ones_like(cur_ids), past_key_values=None, use_cache=False)
        _sync_device(cur_ids.device)
        if t0 is not None:
            t1.record()
            _sync_device(cur_ids.device)
            prefill_ms = float(t0.elapsed_time(t1))
        else:
            import time

            prefill_ms = (time.perf_counter() - wall0) * 1000.0

        _sync_device(cur_ids.device)
        if t0 is not None:
            t0.record()
        else:
            import time

            wall0 = time.perf_counter()
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        _sync_device(cur_ids.device)
        if t0 is not None:
            t1.record()
            _sync_device(cur_ids.device)
            select_ms = float(t0.elapsed_time(t1))
        else:
            import time

            select_ms = (time.perf_counter() - wall0) * 1000.0
        ttft_ms = prefill_ms + select_ms

        if steps <= 1:
            return prefill_ms, ttft_ms, 0.0

        cur_ids = torch.cat([cur_ids, next_id], dim=1)
        if int(cur_ids.shape[1]) > max_ctx:
            cur_ids = cur_ids[:, -max_ctx:]
        _sync_device(cur_ids.device)
        if t0 is not None:
            t0.record()
        else:
            import time

            wall0 = time.perf_counter()
        for _ in range(steps - 1):
            logits, _ = model(cur_ids, attention_mask=torch.ones_like(cur_ids), past_key_values=None, use_cache=False)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_id], dim=1)
            if int(cur_ids.shape[1]) > max_ctx:
                cur_ids = cur_ids[:, -max_ctx:]
        _sync_device(cur_ids.device)
        if t0 is not None:
            t1.record()
            _sync_device(cur_ids.device)
            decode_total_ms = float(t0.elapsed_time(t1))
        else:
            import time

            decode_total_ms = (time.perf_counter() - wall0) * 1000.0

        decode_tokens = float(max(steps - 1, 0) * max(int(prompt_ids.shape[0]), 1))
        return prefill_ms, ttft_ms, decode_total_ms / max(decode_tokens, 1.0)

    model.eval()
    for _ in range(warmup):
        _single_run()

    prefill_samples: list[float] = []
    ttft_samples: list[float] = []
    decode_samples: list[float] = []
    for _ in range(iters):
        p_ms, ttft_ms, decode_ms_per_tok = _single_run()
        prefill_samples.append(p_ms)
        ttft_samples.append(ttft_ms)
        decode_samples.append(decode_ms_per_tok)

    prefill_ms = _median_ms(prefill_samples)
    ttft_ms = _median_ms(ttft_samples)
    decode_ms_per_token = _median_ms(decode_samples)
    tps = 1000.0 / max(decode_ms_per_token, 1e-9)

    decode_token_count = float(max(steps - 1, 0) * max(int(prompt_ids.shape[0]), 1))
    decode_ms_total = decode_ms_per_token * decode_token_count
    return {
        "prefill_ms": float(prefill_ms),
        "ttft_ms": float(ttft_ms),
        "decode_ms_total": float(decode_ms_total),
        "decode_ms_per_token": float(decode_ms_per_token),
        "tokens_per_sec": float(tps),
    }


def bench_cached(engine: Engine, prompt_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int, *, warmup: int, iters: int) -> dict[str, Any]:
    @torch.no_grad()
    def _single_run() -> tuple[float, float, float]:
        batch_size = int(prompt_ids.shape[0])
        _sync_device(prompt_ids.device)
        t0 = torch.cuda.Event(enable_timing=True) if prompt_ids.device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if prompt_ids.device.type == "cuda" else None

        if t0 is not None:
            t0.record()
        else:
            import time

            wall0 = time.perf_counter()

        if batch_size == 1:
            logits, state, _ = engine.prefill(prompt_ids, attention_mask)
            logits_list = [logits[0]]
            state_list = [state]
        else:
            batch_prompt_ids = prompt_ids.detach().cpu().tolist()
            logits_list, state_list, _ = engine.prefill_batch(batch_prompt_ids)

        _sync_device(prompt_ids.device)
        if t0 is not None:
            t1.record()
            _sync_device(prompt_ids.device)
            prefill_ms = float(t0.elapsed_time(t1))
        else:
            import time

            prefill_ms = (time.perf_counter() - wall0) * 1000.0

        _sync_device(prompt_ids.device)
        if t0 is not None:
            t0.record()
        else:
            import time

            wall0 = time.perf_counter()
        next_ids: list[torch.Tensor] = []
        for logits in logits_list:
            if logits.ndim == 1:
                nxt = torch.argmax(logits, dim=-1).view(1, 1)
            else:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            next_ids.append(nxt.to(prompt_ids.device))
        _sync_device(prompt_ids.device)
        if t0 is not None:
            t1.record()
            _sync_device(prompt_ids.device)
            first_select_ms = float(t0.elapsed_time(t1))
        else:
            import time

            first_select_ms = (time.perf_counter() - wall0) * 1000.0
        ttft_ms = prefill_ms + first_select_ms

        if steps <= 1:
            return prefill_ms, ttft_ms, 0.0

        _sync_device(prompt_ids.device)
        if t0 is not None:
            t0.record()
        else:
            import time

            wall0 = time.perf_counter()

        for seq_idx in range(batch_size):
            next_id = next_ids[seq_idx]
            state = state_list[seq_idx]
            for _ in range(steps - 1):
                step_logits, state = engine.decode_step(next_id, state)
                next_id = torch.argmax(step_logits, dim=-1, keepdim=True)

        _sync_device(prompt_ids.device)
        if t0 is not None:
            t1.record()
            _sync_device(prompt_ids.device)
            decode_total_ms = float(t0.elapsed_time(t1))
        else:
            import time

            decode_total_ms = (time.perf_counter() - wall0) * 1000.0

        decode_tokens = float(max(steps - 1, 0) * max(batch_size, 1))
        return prefill_ms, ttft_ms, decode_total_ms / max(decode_tokens, 1.0)

    for _ in range(warmup):
        _single_run()

    prefill_samples: list[float] = []
    ttft_samples: list[float] = []
    decode_samples: list[float] = []
    for _ in range(iters):
        p_ms, t_ms, d_ms = _single_run()
        prefill_samples.append(p_ms)
        ttft_samples.append(t_ms)
        decode_samples.append(d_ms)

    prefill_ms = _median_ms(prefill_samples)
    ttft_ms = _median_ms(ttft_samples)
    decode_ms_per_token = _median_ms(decode_samples)
    tps = 1000.0 / max(decode_ms_per_token, 1e-9)

    decode_token_count = float(max(steps - 1, 0) * max(int(prompt_ids.shape[0]), 1))
    decode_ms_total = decode_ms_per_token * decode_token_count
    return {
        "prefill_ms": float(prefill_ms),
        "ttft_ms": float(ttft_ms),
        "decode_ms_total": float(decode_ms_total),
        "decode_ms_per_token": float(decode_ms_per_token),
        "tokens_per_sec": float(tps),
    }


def _row_common_fields(
    *,
    prompt_len: int,
    gen_len: int,
    batch_size: int,
    warmup: int,
    iters: int,
    seed: int,
    repeat_idx: int,
    repeats_total: int,
    mode: str,
) -> dict[str, Any]:
    return {
        "batch_size": int(batch_size),
        "B": int(batch_size),
        "context_len": int(prompt_len),
        "prompt_len": int(prompt_len),
        "gen_len": int(gen_len),
        "n_iters": int(iters),
        "warmup": int(warmup),
        "seed": int(seed),
        "repeat_idx": int(repeat_idx),
        "repeats_total": int(repeats_total),
        "benchmark_mode": str(mode),
        "prefill_batched": bool(batch_size > 1),
        "decode_batched": False,
    }


def _mock_metrics(
    *,
    mode: str,
    prompt_len: int,
    gen_len: int,
    device: str,
    dtype: str,
    quant_mode: str | None,
    batch_size: int,
    iters: int,
    warmup: int,
    package_dir: Path,
    benchmark_mode: str,
    seed: int,
    repeat_idx: int,
    repeats_total: int,
) -> dict[str, Any]:
    if mode == "cache":
        prefill_ms = 3.2 + (prompt_len / 256.0) + (batch_size - 1) * 0.6
        decode_ms_per_token = max(0.05, 0.65 - 0.04 * min(batch_size, 8))
    else:
        prefill_ms = 2.0 + (prompt_len / 256.0) + (batch_size - 1) * 0.8
        decode_ms_per_token = max(0.05, 2.2 - 0.06 * min(batch_size, 8))
    ttft_ms = prefill_ms + decode_ms_per_token
    tps = (1000.0 / decode_ms_per_token) if decode_ms_per_token > 0 else float("inf")
    decode_ms_total = decode_ms_per_token * float(max(gen_len - 1, 0) * max(batch_size, 1))
    payload = {
        "prefill_ms": float(prefill_ms),
        "ttft_ms": float(ttft_ms),
        "decode_ms_total": float(decode_ms_total),
        "decode_ms_per_token": float(decode_ms_per_token),
        "tokens_per_sec": float(tps),
        "device": device,
        "dtype": dtype,
        "quant_mode": quant_mode,
    }
    payload.update(
        _row_common_fields(
            prompt_len=prompt_len,
            gen_len=gen_len,
            batch_size=batch_size,
            warmup=warmup,
            iters=iters,
            seed=seed,
            repeat_idx=repeat_idx,
            repeats_total=repeats_total,
            mode=benchmark_mode,
        )
    )
    payload.update(
        _build_provenance(
            package_dir=package_dir,
            device=device,
            dtype=dtype,
            quant_mode=quant_mode,
        )
    )
    return payload


def _run_cache_equivalence(
    *,
    engine: Engine,
    prompt_len: int,
    gen_len: int,
    atol: float,
    seed: int,
) -> dict[str, Any]:
    if gen_len <= 0:
        raise ValueError("gen_len must be > 0")

    model = engine.model
    dev = next(model.parameters()).device
    max_ctx = int(getattr(getattr(model, "config", None), "block_size", prompt_len))
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 10000))
    torch.manual_seed(int(seed))
    prompt_ids = torch.randint(0, vocab_size, (1, int(prompt_len)), dtype=torch.long, device=dev)
    attention_mask = torch.ones_like(prompt_ids)

    cached_logits: list[torch.Tensor] = []
    logits, state, _ = engine.prefill(prompt_ids, attention_mask)
    cached_logits.append(logits)
    cached_tokens = [int(torch.argmax(logits, dim=-1).item())]
    for _ in range(gen_len - 1):
        nxt = torch.tensor([[cached_tokens[-1]]], dtype=torch.long, device=dev)
        logits, state = engine.decode_step(nxt, state)
        cached_logits.append(logits)
        cached_tokens.append(int(torch.argmax(logits, dim=-1).item()))

    recompute_logits: list[torch.Tensor] = []
    recompute_tokens: list[int] = []
    cur = prompt_ids.clone()
    for _ in range(gen_len):
        out, _ = model(cur, attention_mask=torch.ones_like(cur), past_key_values=None, use_cache=False)
        last = out[:, -1, :]
        recompute_logits.append(last)
        nxt = int(torch.argmax(last, dim=-1).item())
        recompute_tokens.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=dev)], dim=1)
        if int(cur.shape[1]) > max_ctx:
            cur = cur[:, -max_ctx:]

    max_abs_logit_diff = 0.0
    first_divergence_step = None
    top1_match_all_steps = True
    for step, (cached_step, recompute_step) in enumerate(zip(cached_logits, recompute_logits)):
        diff = float(torch.max(torch.abs(cached_step.float() - recompute_step.float())).item())
        if diff > max_abs_logit_diff:
            max_abs_logit_diff = diff
        if not torch.allclose(cached_step, recompute_step, atol=float(atol), rtol=0.0) and first_divergence_step is None:
            first_divergence_step = int(step)
        if int(torch.argmax(cached_step, dim=-1).item()) != int(torch.argmax(recompute_step, dim=-1).item()):
            top1_match_all_steps = False

    comparable = bool(int(prompt_len) < int(max_ctx))
    return {
        "prompt_len": int(prompt_len),
        "gen_len": int(gen_len),
        "max_abs_logit_diff": float(max_abs_logit_diff),
        "first_divergence_step": first_divergence_step,
        "exact_match_generated_tokens": bool(cached_tokens == recompute_tokens),
        "top1_match_all_steps": bool(top1_match_all_steps),
        "within_mode_cached_vs_recompute": True,
        "comparable_for_gate": comparable,
        "non_comparable_reason": (
            None
            if comparable
            else "prompt_len_at_block_size_boundary_recompute_path_cannot_replay_cached_step_semantics_exactly"
        ),
    }


def _build_cache_equivalence_receipts(
    *,
    engine: Engine,
    prompt_lens: list[int],
    gen_lens: list[int],
    atol: float,
    seed: int,
    package_dir: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    case_seed = int(seed)
    for prompt_len in prompt_lens:
        for gen_len in gen_lens:
            cases.append(
                _run_cache_equivalence(
                    engine=engine,
                    prompt_len=int(prompt_len),
                    gen_len=int(gen_len),
                    atol=float(atol),
                    seed=case_seed,
                )
            )
            case_seed += 1

    max_diff = max((float(r["max_abs_logit_diff"]) for r in cases), default=0.0)
    first_divergence = next((r["first_divergence_step"] for r in cases if r["first_divergence_step"] is not None), None)
    comparable_cases = [r for r in cases if bool(r.get("comparable_for_gate", True))]
    non_comparable_cases = [r for r in cases if not bool(r.get("comparable_for_gate", True))]
    cmp_exact_all = bool(all(bool(r["exact_match_generated_tokens"]) for r in comparable_cases)) if comparable_cases else False
    cmp_top1_all = bool(all(bool(r["top1_match_all_steps"]) for r in comparable_cases)) if comparable_cases else False
    cmp_max_diff = max((float(r["max_abs_logit_diff"]) for r in comparable_cases), default=0.0)
    cmp_first_div = next(
        (r["first_divergence_step"] for r in comparable_cases if r["first_divergence_step"] is not None),
        None,
    )

    return {
        "tested_prompt_lengths": sorted({int(x) for x in prompt_lens}),
        "tested_generation_lengths": sorted({int(x) for x in gen_lens}),
        "dtype": str(getattr(engine, "runtime_dtype", "unknown")),
        "quant_mode": getattr(engine, "quant_mode", None),
        "greedy": True,
        "case_count": int(len(cases)),
        "tolerance": float(atol),
        "within_mode_cached_vs_recompute": True,
        "max_abs_logit_diff": float(max_diff),
        "first_divergence_step": first_divergence,
        "comparable_case_count": int(len(comparable_cases)),
        "non_comparable_case_count": int(len(non_comparable_cases)),
        "exact_match_generated_tokens_all_comparable_cases": cmp_exact_all,
        "top1_match_all_steps_all_comparable_cases": cmp_top1_all,
        "max_abs_logit_diff_comparable_cases": float(cmp_max_diff),
        "first_divergence_step_comparable_cases": cmp_first_div,
        "exact_match_generated_tokens_all_cases": bool(all(bool(r["exact_match_generated_tokens"]) for r in cases)),
        "top1_match_all_steps_all_cases": bool(all(bool(r["top1_match_all_steps"]) for r in cases)),
        "source_bench_files": [
            "experiments/serving_bench/cache_grid.json",
            "experiments/serving_bench/recompute_grid.json",
        ],
        "cases": cases,
        "provenance": _build_provenance(
            package_dir=package_dir,
            device=str(next(engine.model.parameters()).device),
            dtype=str(getattr(engine, "runtime_dtype", "unknown")),
            quant_mode=getattr(engine, "quant_mode", None),
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-1 serving benchmark scaffolding")
    p.add_argument("--mode", choices=["cache", "recompute", "both"], default="both")
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--context-lens",
        type=str,
        default="",
        help="Comma-separated context lengths for grid runs; falls back to --prompt-len when empty.",
    )
    p.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes for grid runs; falls back to --batch-size when empty.",
    )
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="experiments/serving_bench")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--quant-mode", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--package-dir",
        type=str,
        default=os.environ.get("SERVING_PACKAGE_DIR", "experiments/p1_pos_enc/runs/rope/package"),
        help="Model package directory for non-mock benchmark mode.",
    )
    p.add_argument(
        "--tiny-mock",
        action="store_true",
        help="Write schema-valid mock outputs without running model loops (for test scaffolding).",
    )
    p.add_argument("--loader", type=str, default="package", choices=["package", "nanollama"],
                    help="Model loader: package (sp16k dir) or nanollama (raw .pt checkpoint).")
    p.add_argument(
        "--cache-equivalence-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for within-mode cached-vs-recompute equivalence probes.",
    )
    p.add_argument(
        "--equiv-prompt-lens",
        type=str,
        default="",
        help="Optional CSV prompt lengths for cache equivalence receipts; defaults to benchmark context set.",
    )
    p.add_argument(
        "--equiv-gen-lens",
        type=str,
        default="",
        help="Optional CSV generation lengths for cache equivalence receipts; defaults to min(gen_len,16).",
    )
    return p.parse_args()


def _selected_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["recompute", "cache"]
    return [mode]


def _stable_seed(base_seed: int, *, mode: str, prompt_len: int, batch_size: int, repeat_idx: int) -> int:
    mode_off = 13 if mode == "cache" else 29
    return int(base_seed + mode_off + prompt_len * 97 + batch_size * 997 + repeat_idx * 7919)


def _select_canonical_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows cannot be empty")
    for row in rows:
        if int(row.get("context_len", -1)) == 256 and int(row.get("batch_size", -1)) == 1:
            return row
    return rows[0]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be > 0")

    package_dir = Path(args.package_dir)
    context_lens = _parse_int_csv(args.context_lens) or [int(args.prompt_len)]
    batch_sizes = _parse_int_csv(args.batch_sizes) or [int(args.batch_size)]
    modes = _selected_modes(args.mode)

    engine: Engine | None = None
    model = None
    dtype_name = str(args.dtype)
    vocab_size = 10000

    if not args.tiny_mock:
        if not package_dir.exists():
            raise FileNotFoundError(f"package_dir does not exist: {package_dir}")
        engine = build_engine_from_package(
            package_path=str(package_dir),
            device=args.device,
            dtype=args.dtype,
            quant_mode=args.quant_mode,
            loader=args.loader,
        )
        model = engine.model
        dtype_name = str(getattr(engine, "runtime_dtype", args.dtype))
        vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 10000))
        block_size = int(engine.block_size)
        context_lens = [int(min(int(c), block_size)) for c in context_lens]

    for b in batch_sizes:
        if int(b) <= 0:
            raise ValueError("all batch sizes must be > 0")

    grid_summaries: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    raw_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}

    for mode in modes:
        for prompt_len in context_lens:
            for batch_size in batch_sizes:
                rep_rows: list[dict[str, Any]] = []
                for repeat_idx in range(int(args.repeats)):
                    repeat_seed = _stable_seed(
                        int(args.seed),
                        mode=mode,
                        prompt_len=int(prompt_len),
                        batch_size=int(batch_size),
                        repeat_idx=int(repeat_idx),
                    )
                    if args.tiny_mock:
                        row = _mock_metrics(
                            mode=mode,
                            prompt_len=int(prompt_len),
                            gen_len=int(args.gen_len),
                            device=args.device,
                            dtype=dtype_name,
                            quant_mode=args.quant_mode,
                            batch_size=int(batch_size),
                            iters=int(args.iters),
                            warmup=int(args.warmup),
                            package_dir=package_dir,
                            benchmark_mode=mode,
                            seed=repeat_seed,
                            repeat_idx=repeat_idx,
                            repeats_total=int(args.repeats),
                        )
                    else:
                        torch.manual_seed(repeat_seed)
                        prompt_ids = torch.randint(
                            0,
                            int(vocab_size),
                            (int(batch_size), int(prompt_len)),
                            dtype=torch.long,
                            device=args.device,
                        )
                        attention_mask = torch.ones_like(prompt_ids)
                        if mode == "recompute":
                            if model is None:
                                raise RuntimeError("model not initialized")
                            row = bench_recompute(
                                model,
                                prompt_ids,
                                int(args.gen_len),
                                warmup=int(args.warmup),
                                iters=int(args.iters),
                            )
                        else:
                            if engine is None:
                                raise RuntimeError("engine not initialized")
                            row = bench_cached(
                                engine,
                                prompt_ids,
                                attention_mask,
                                int(args.gen_len),
                                warmup=int(args.warmup),
                                iters=int(args.iters),
                            )
                        row.update(
                            {
                                "device": str(prompt_ids.device),
                                "dtype": dtype_name,
                                "quant_mode": getattr(engine, "quant_mode", None),
                            }
                        )
                        row.update(
                            _row_common_fields(
                                prompt_len=int(prompt_len),
                                gen_len=int(args.gen_len),
                                batch_size=int(batch_size),
                                warmup=int(args.warmup),
                                iters=int(args.iters),
                                seed=repeat_seed,
                                repeat_idx=repeat_idx,
                                repeats_total=int(args.repeats),
                                mode=mode,
                            )
                        )
                        row.update(
                            _build_provenance(
                                package_dir=package_dir,
                                device=str(prompt_ids.device),
                                dtype=dtype_name,
                                quant_mode=getattr(engine, "quant_mode", None),
                            )
                        )

                    raw_rows[mode].append(dict(row))
                    rep_rows.append(dict(row))

                summary = _summarize_repeats(rep_rows)
                summary["repeat_source_rows"] = int(len(rep_rows))
                grid_summaries[mode].append(summary)

    for mode in modes:
        save_jsonl(str(raw_dir / f"{mode}_repeats.jsonl"), raw_rows[mode])
        save_json_list(str(out_dir / f"{mode}_grid.json"), grid_summaries[mode])
        save_json(str(out_dir / f"{mode}.json"), _select_canonical_row(grid_summaries[mode]))

    if not args.tiny_mock and engine is not None:
        eq_prompt_lens = _parse_int_csv(args.equiv_prompt_lens) or list(context_lens)
        eq_gen_lens = _parse_int_csv(args.equiv_gen_lens) or [min(int(args.gen_len), 16)]
        receipts = _build_cache_equivalence_receipts(
            engine=engine,
            prompt_lens=[int(x) for x in eq_prompt_lens],
            gen_lens=[int(x) for x in eq_gen_lens],
            atol=float(args.cache_equivalence_atol),
            seed=int(args.seed),
            package_dir=package_dir,
        )
        save_json(str(out_dir / "cache_equivalence_receipts.json"), receipts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

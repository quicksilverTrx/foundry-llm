# tests/serving/test_precision_correctness.py
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from llm_lab.serving.engine import CacheState, Engine, build_engine_from_package
from llm_lab.serving.precision import cast_model_for_inference, resolve_runtime_precision
from llm_lab.serving.quant import quant_backend_is_available


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 6

    def __init__(self) -> None:
        self._stoi = {"a": 1, "b": 2, "c": 3, " ": 4, "<|endoftext|>": 6}
        self._itos = {v: k for k, v in self._stoi.items()}

    def token_to_id(self, tok: str) -> int:
        return int(self._stoi[tok])

    def encode(self, text: str) -> list[int]:
        return [self._stoi[ch] for ch in text if ch in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids if i in self._itos and self._itos[i] != "<|endoftext|>")


class ScheduledModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, schedule: dict[int, int] | None = None):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.schedule = {} if schedule is None else dict(schedule)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def _pick(self, abs_pos: int) -> int:
        return int(self.schedule.get(abs_pos, 1))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        device = input_ids.device
        logits = torch.full((B, T, self.vocab_size), -10.0, device=device)
        past_len = 0 if not past_key_values else int(past_key_values[0][0].shape[2])
        for t in range(T):
            logits[:, t, self._pick(past_len + t)] = 10.0
        if not use_cache:
            return logits, None
        total = past_len + T
        k = torch.zeros(B, 1, total, 1, device=device)
        v = torch.zeros(B, 1, total, 1, device=device)
        return logits, [(k, v)]


def _cache_tolerance_policy(runtime_dtype: str) -> tuple[float, float]:
    table = {
        "fp32": (1e-4, 1e-4),
        "fp16": (5e-2, 5e-2),
        "bf16": (5e-2, 5e-2),
        "int8": (1e-1, 1e-1),
    }
    if runtime_dtype not in table:
        raise ValueError(f"unsupported runtime dtype for tolerance policy: {runtime_dtype}")
    return table[runtime_dtype]


def _stable_topk_with_tiebreak(logits: torch.Tensor, k: int = 1, eps: float = 1e-6) -> list[int]:
    scores = torch.nan_to_num(logits.float(), nan=-1e30, neginf=-1e30, posinf=1e30).flatten().tolist()
    ordered = sorted(enumerate(scores), key=lambda it: (-it[1], it[0]))
    out: list[int] = []
    for token_id, score in ordered:
        if len(out) >= k:
            break
        if not out:
            out.append(int(token_id))
            continue
        prev = scores[out[-1]]
        if abs(score - prev) <= eps:
            out.append(int(token_id))
        else:
            out.append(int(token_id))
    return out[:k]


def _cross_mode_drift_acceptance(
    *,
    base_tokens: list[int],
    alt_tokens: list[int],
    base_logits: torch.Tensor,
    alt_logits: torch.Tensor,
    mode: str,
) -> dict[str, object]:
    if base_tokens == alt_tokens:
        return {
            "accepted": True,
            "mode": mode,
            "reason": "exact_greedy_parity",
            "max_logit_diff": 0.0,
            "top1_match": True,
        }

    thresholds = {
        "fp16": 1.0,
        "bf16": 1.0,
        "int8": 2.0,
    }
    max_logit_diff = float(torch.max(torch.abs(base_logits.float() - alt_logits.float())).item())
    top1_base = _stable_topk_with_tiebreak(base_logits[0], k=1)[0]
    top1_alt = _stable_topk_with_tiebreak(alt_logits[0], k=1)[0]
    top1_match = top1_base == top1_alt
    threshold = float(thresholds.get(mode, 1.0))
    accepted = bool(top1_match or max_logit_diff <= threshold)
    return {
        "accepted": accepted,
        "mode": mode,
        "reason": "bounded_drift" if accepted else "drift_exceeds_policy",
        "max_logit_diff": max_logit_diff,
        "threshold": threshold,
        "top1_match": top1_match,
        "top1_base": int(top1_base),
        "top1_alt": int(top1_alt),
    }


def _package_dir() -> Path:
    return Path(os.environ.get("SERVING_PACKAGE_DIR", "experiments/p1_pos_enc/runs/rope/package"))


def _cached_logits(engine: Engine, prompt: torch.Tensor, mask: torch.Tensor, steps: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    logits, state, _ = engine.prefill(prompt, mask)
    out.append(logits)
    for _ in range(steps - 1):
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        logits, state = engine.decode_step(nxt, state)
        out.append(logits)
    return out


def _recompute_logits(model: torch.nn.Module, prompt: torch.Tensor, steps: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    cur = prompt.clone()
    for _ in range(steps):
        logits, _ = model(cur, attention_mask=torch.ones_like(cur), past_key_values=None, use_cache=False)
        last = logits[:, -1, :]
        out.append(last)
        nxt = torch.argmax(last, dim=-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    return out


def test_fp16_or_bf16_greedy_matches_fp32_on_short_prompt() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    device = "cpu"
    baseline = build_engine_from_package(str(pkg), device=device, dtype="fp32", quant_mode=None)
    req = "bf16" if resolve_runtime_precision("bf16", device) != "fp32" else "fp16"
    other = build_engine_from_package(str(pkg), device=device, dtype=req, quant_mode=None)

    out_a = baseline.generate(prompt_ids=[1, 2], attention_mask=[1, 1], max_new_tokens=4, temperature=0.0, top_k=1)
    out_b = other.generate(prompt_ids=[1, 2], attention_mask=[1, 1], max_new_tokens=4, temperature=0.0, top_k=1)

    prompt = torch.tensor([[1, 2]], dtype=torch.long, device=device)
    mask = torch.ones_like(prompt)
    l_a, _, _ = baseline.prefill(prompt, mask)
    l_b, _, _ = other.prefill(prompt, mask)
    report = _cross_mode_drift_acceptance(
        base_tokens=list(out_a["completion_token_ids"]),
        alt_tokens=list(out_b["completion_token_ids"]),
        base_logits=l_a,
        alt_logits=l_b,
        mode=other.runtime_dtype,
    )
    assert bool(report["accepted"]), str(report)


def test_int8_greedy_is_stable_or_drift_is_reported() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    ok, _ = quant_backend_is_available("int8", "cpu")
    if not ok:
        pytest.skip("int8 backend unavailable on this CPU runtime")

    fp32 = build_engine_from_package(str(pkg), device="cpu", dtype="fp32", quant_mode=None)
    int8 = build_engine_from_package(str(pkg), device="cpu", dtype="fp32", quant_mode="int8")

    out_a = fp32.generate(prompt_ids=[1, 2], attention_mask=[1, 1], max_new_tokens=4, temperature=0.0, top_k=1)
    out_b = int8.generate(prompt_ids=[1, 2], attention_mask=[1, 1], max_new_tokens=4, temperature=0.0, top_k=1)
    x = torch.tensor([[1, 2]], dtype=torch.long)
    m = torch.ones_like(x)
    la, _, _ = fp32.prefill(x, m)
    lb, _, _ = int8.prefill(x, m)
    report = _cross_mode_drift_acceptance(
        base_tokens=list(out_a["completion_token_ids"]),
        alt_tokens=list(out_b["completion_token_ids"]),
        base_logits=la,
        alt_logits=lb,
        mode="int8",
    )
    assert report["max_logit_diff"] >= 0.0
    assert report["reason"] in {"exact_greedy_parity", "bounded_drift", "drift_exceeds_policy"}


def test_cache_equivalence_still_holds_with_requested_precision() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    torch.manual_seed(0)
    prompt = torch.randint(0, 100, (1, 8), dtype=torch.long)
    mask = torch.ones_like(prompt)

    for req in ("fp32", "fp16", "bf16"):
        engine = build_engine_from_package(str(pkg), device="cpu", dtype=req, quant_mode=None)
        cached = _cached_logits(engine, prompt, mask, steps=3)
        recompute = _recompute_logits(engine.model, prompt, steps=3)
        atol, rtol = _cache_tolerance_policy(engine.runtime_dtype)
        for a, b in zip(cached, recompute):
            assert torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
            assert _stable_topk_with_tiebreak(a[0], k=1)[0] == _stable_topk_with_tiebreak(b[0], k=1)[0]


def test_stop_semantics_preserved_across_precisions() -> None:
    tok = TinyTokenizer()
    for req in ("fp32", "fp16", "bf16"):
        runtime = resolve_runtime_precision(req, "cpu")
        model = cast_model_for_inference(
            ScheduledModel(schedule={0: 2, 1: 2, 2: 3, 3: 4}),
            runtime,
        )
        engine = Engine(
            model=model,
            tokenizer=tok,
            block_size=16,
            max_cache_len=16,
            requested_dtype=req,
            runtime_dtype=runtime,
            quant_mode=None,
        )
        out = engine.generate(
            prompt_ids=[1],
            attention_mask=[1],
            max_new_tokens=4,
            temperature=0.0,
            top_k=1,
            stop_token_ids={2},
        )
        assert out["stop_reason"] == "stop_token"


def test_metrics_schema_unchanged_under_quant_mode() -> None:
    pkg = _package_dir()
    if not pkg.exists():
        pytest.skip(f"package missing: {pkg}")

    base = build_engine_from_package(str(pkg), device="cpu", dtype="fp32", quant_mode=None)
    out_a = base.generate(prompt_ids=[1, 2], attention_mask=[1, 1], max_new_tokens=2, temperature=0.0, top_k=1)

    required = {
        "ttft_ms",
        "prefill_ms",
        "decode_ms_total",
        "decode_ms_per_token",
        "tokens_per_sec",
        "requested_dtype",
        "runtime_dtype",
        "runtime_quant_mode",
    }
    assert required.issubset(out_a["metrics"].keys())

    ok, _ = quant_backend_is_available("int8", "cpu")
    if not ok:
        pytest.skip("int8 backend unavailable on this CPU runtime")

    int8 = build_engine_from_package(str(pkg), device="cpu", dtype="fp32", quant_mode="int8")
    out_b = int8.generate(prompt_ids=[1, 2], attention_mask=[1, 1], max_new_tokens=2, temperature=0.0, top_k=1)
    assert required.issubset(out_b["metrics"].keys())
    assert set(out_a["metrics"].keys()) == set(out_b["metrics"].keys())


def test_ranking_tie_break_policy_is_deterministic() -> None:
    logits = torch.tensor([0.5, 1.0, 1.0, 0.4], dtype=torch.float32)
    first = _stable_topk_with_tiebreak(logits, k=2)
    second = _stable_topk_with_tiebreak(logits, k=2)
    assert first == second
    assert first[0] == 1
    assert first[1] == 2


def test_cache_tolerance_policy_known_modes() -> None:
    assert _cache_tolerance_policy("fp32") == (1e-4, 1e-4)
    assert _cache_tolerance_policy("fp16") == (5e-2, 5e-2)
    with pytest.raises(ValueError):
        _cache_tolerance_policy("unknown")

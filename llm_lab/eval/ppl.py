# llm_lab/eval/ppl.py
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn


def _streaming_alignment_reference_loop(
    model: "nn.Module",
    token_ids: list[int],
    *,
    max_seq_len: int,
    stride: int,
    device: str,
) -> tuple[float, int]:
    if max_seq_len <= 1:
        raise ValueError("max_seq_len must be > 1")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if stride > max_seq_len:
        raise ValueError("stride must be <= max_seq_len to ensure contiguous target coverage")
    if len(token_ids) < max_seq_len + 1:
        raise ValueError(f"Need at least {max_seq_len + 1} tokens; got {len(token_ids)}")

    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    last_target = ids.numel() - 1
    next_target = 1
    total_nll = 0.0
    n_tokens = 0

    for start in range(0, last_target, stride):
        end = min(start + max_seq_len, last_target)
        x = ids[start:end].unsqueeze(0)
        if x.shape[1] == 0:
            continue

        logits, _ = model(x, attention_mask=torch.ones_like(x), past_key_values=None, use_cache=False)
        score_from = max(start + 1, next_target)
        if score_from > end:
            continue

        offset = int(score_from - (start + 1))
        y = ids[score_from : end + 1]
        pred = logits[:, offset : offset + y.numel(), :]
        ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y, reduction="sum")
        total_nll += float(ce.item())
        n_tokens += int(y.numel())
        next_target = end + 1
        if next_target > last_target:
            break

    if next_target <= last_target:
        # Tail-rescue window: ensures no final target positions are skipped.
        end = last_target
        start = max(0, end - max_seq_len)
        x = ids[start:end].unsqueeze(0)
        logits, _ = model(x, attention_mask=torch.ones_like(x), past_key_values=None, use_cache=False)
        score_from = max(start + 1, next_target)
        offset = int(score_from - (start + 1))
        y = ids[score_from : end + 1]
        pred = logits[:, offset : offset + y.numel(), :]
        ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y, reduction="sum")
        total_nll += float(ce.item())
        n_tokens += int(y.numel())

    return float(total_nll), int(n_tokens)


def _read_eval_text(text_path: str) -> str:
    p = Path(text_path)
    if not p.exists():
        raise FileNotFoundError(f"text_path does not exist: {p}")
    return p.read_text(encoding="utf-8")


def _encode_eval_text(tokenizer: object, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text))
    except Exception:
        # Fallback path for corpora containing symbols outside tokenizer training coverage.
        sanitized = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
        try:
            return list(tokenizer.encode(sanitized))
        except Exception:
            token_ids: list[int] = []
            for chunk in sanitized.split():
                try:
                    token_ids.extend(list(tokenizer.encode(chunk + " ")))
                except Exception:
                    continue
            return token_ids


def _infer_runtime_dtype(model: "nn.Module") -> str:
    try:
        param = next(model.parameters())
    except StopIteration:
        return "fp32"
    if param.dtype == torch.float16:
        return "fp16"
    if param.dtype == torch.bfloat16:
        return "bf16"
    return "fp32"


@torch.no_grad()
def evaluate_streaming_nll(
    model: "nn.Module",
    tokenizer: object,
    text_path: str,
    device: str,
    max_seq_len: int,
    stride: int,
) -> dict[str, float]:
    if max_seq_len <= 1:
        raise ValueError("max_seq_len must be > 1")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    text = _read_eval_text(text_path)
    token_ids = _encode_eval_text(tokenizer, text)
    if len(token_ids) < max_seq_len + 1:
        raise ValueError(
            f"Need at least {max_seq_len + 1} tokens for streaming eval; got {len(token_ids)}"
        )

    model = model.to(device)
    model.eval()

    total_nll, n_tokens = _streaming_alignment_reference_loop(
        model,
        token_ids,
        max_seq_len=max_seq_len,
        stride=stride,
        device=device,
    )
    avg_nll = total_nll / max(n_tokens, 1)

    result = {
        "n_tokens": float(n_tokens),
        "total_nll": float(total_nll),
        "avg_nll": float(avg_nll),
        "ppl": float(math.exp(avg_nll)),
        "device": str(device),
        "dtype": str(getattr(model, "_runtime_dtype", _infer_runtime_dtype(model))),
        "quant_mode": str(getattr(model, "_runtime_quant_mode", "none")),
    }
    return result

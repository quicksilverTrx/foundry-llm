# llm_lab/serving/diagnostics.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_cache_divergence_report(
    path: str,
    *,
    step: int,
    max_logit_diff: float,
    prompt_ids: list[int],
    generated_ids: list[int],
    extra: dict[str, Any],
) -> None:
    """Write a compact first-divergence report JSON artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_differing_tokens = extra.get("top_differing_tokens", [])
    payload: dict[str, Any] = {
        "step": int(step),
        "max_logit_diff": float(max_logit_diff),
        "prompt_len": int(len(prompt_ids)),
        "generated_len": int(len(generated_ids)),
        "prompt_ids": [int(x) for x in prompt_ids],
        "generated_ids": [int(x) for x in generated_ids],
        "top_differing_tokens": top_differing_tokens,
    }

    # Preserve optional debug context without requiring a fixed schema.
    for k, v in extra.items():
        if k not in payload:
            payload[k] = v

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

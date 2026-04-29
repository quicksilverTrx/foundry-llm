# scripts/eval_perplexity.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_lab.core.package.io import load_model_package
from llm_lab.eval.ppl import evaluate_streaming_nll
from llm_lab.serving.precision import cast_model_for_inference, runtime_precision_decision
from llm_lab.serving.quant import describe_quant_runtime, maybe_quantize_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3 streaming-NLL evaluation")
    p.add_argument("--package", type=str, required=True)
    p.add_argument("--text-path", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="fp32")
    p.add_argument("--quant-mode", type=str, default=None)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _, tokenizer, model = load_model_package(args.package, device=args.device)

    runtime_dtype, _ = runtime_precision_decision(args.dtype, args.device)
    model = cast_model_for_inference(model.eval(), runtime_dtype)

    qinfo = describe_quant_runtime(args.quant_mode, args.device)
    if qinfo.get("runtime_quant_mode") is not None:
        model = maybe_quantize_model(model, qinfo["runtime_quant_mode"], args.device)

    setattr(model, "_runtime_dtype", runtime_dtype)
    setattr(model, "_runtime_quant_mode", qinfo.get("runtime_quant_mode") or "none")

    result = evaluate_streaming_nll(
        model=model,
        tokenizer=tokenizer,
        text_path=args.text_path,
        device=args.device,
        max_seq_len=args.max_seq_len,
        stride=args.stride,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

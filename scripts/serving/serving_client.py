# scripts/serving_client.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import torch

from llm_lab.serving.engine import build_engine_from_package


def pick_device(dev: str | None) -> str:
    if dev is not None:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serving client: local engine or HTTP API")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stop_strings", type=str, default=None, help="Pipe-separated stop strings")
    p.add_argument("--stop_token_ids", type=str, default=None, help="Comma-separated token IDs")
    p.add_argument("--return_logprobs", action="store_true", default=False)

    p.add_argument("--server_url", type=str, default=None, help="If set, call HTTP API at this base URL")
    p.add_argument("--stream", action="store_true", default=False, help="Use /stream SSE endpoint")

    p.add_argument("--package", type=str, default=None, help="Model package dir for local mode")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="fp32")
    p.add_argument("--quant-mode", type=str, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--frequency_penalty", type=float, default=None)
    p.add_argument("--eos_token_id", type=int, default=None)
    return p.parse_args()


def _parse_stop_token_ids(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


def _parse_stop_strings(raw: str | None) -> list[str] | None:
    if raw is None or raw == "":
        return None
    out = [s for s in raw.split("|") if s != ""]
    return out or None


def _request_payload(args: argparse.Namespace) -> dict:
    return {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "stop_strings": _parse_stop_strings(args.stop_strings),
        "stop_token_ids": _parse_stop_token_ids(args.stop_token_ids),
        "seed": args.seed,
        "return_logprobs": args.return_logprobs,
    }


def _run_http(args: argparse.Namespace) -> None:
    assert args.server_url is not None
    payload = _request_payload(args)
    base = args.server_url.rstrip("/")

    with httpx.Client(timeout=60.0) as client:
        if not args.stream:
            r = client.post(f"{base}/generate", json=payload)
            r.raise_for_status()
            out = r.json()
            print(out["completion_text"])
            print(f"stop_reason={out['stop_reason']}")
            print(f"completion_tokens={len(out['completion_token_ids'])}")
            return

        with client.stream("POST", f"{base}/stream", json=payload) as r:
            r.raise_for_status()
            current_event = None
            for line in r.iter_lines():
                if line is None or line == "":
                    continue
                if line.startswith("event: "):
                    current_event = line[len("event: ") :]
                    continue
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[len("data: ") :])
                if current_event == "token":
                    print(data["token_text"], end="", flush=True)
                elif current_event == "final":
                    print()
                    print(f"stop_reason={data['stop_reason']}")
                    print(f"completion_tokens={len(data['completion_token_ids'])}")
                    print("metrics=" + json.dumps(data["metrics"], sort_keys=True))


def _run_local(args: argparse.Namespace) -> None:
    if args.package is None:
        raise ValueError("--package is required for local mode")

    device = pick_device(args.device)
    pkg_dir = Path(args.package)
    engine = build_engine_from_package(
        package_path=str(pkg_dir),
        device=device,
        dtype=args.dtype,
        quant_mode=args.quant_mode,
    )
    tokenizer = engine.tokenizer

    prompt_ids = tokenizer.encode(args.prompt)
    stop_token_ids = _parse_stop_token_ids(args.stop_token_ids)
    stop_strings = _parse_stop_strings(args.stop_strings)

    out = engine.generate(
        prompt_ids=prompt_ids,
        attention_mask=[1] * len(prompt_ids),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        eos_token_id=args.eos_token_id,
        stop_token_ids=set(stop_token_ids) if stop_token_ids else None,
        stop_strings=stop_strings,
        seed=args.seed,
        return_logprobs=args.return_logprobs,
    )
    full_ids = prompt_ids + out["completion_token_ids"]
    text = tokenizer.decode(full_ids)
    print(text)
    print(f"stop_reason={out['stop_reason']}")
    print(f"completion_tokens={len(out['completion_token_ids'])}")


def main() -> None:
    args = parse_args()
    if args.server_url is not None:
        _run_http(args)
        return
    _run_local(args)


if __name__ == "__main__":
    main()

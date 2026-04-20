# scripts/serve.py
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from llm_lab.serving.api import create_app
from llm_lab.serving.config import ServingConfig
from llm_lab.serving.engine import build_engine_from_package


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serving API")
    p.add_argument("--package", type=str, required=True, help="Model package directory")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="fp32")
    p.add_argument("--quant-mode", type=str, default=None)
    p.add_argument("--log-raw-prompts", action="store_true", default=False)
    p.add_argument("--disable-rate-limit", action="store_true", default=False)
    p.add_argument("--rate-limit-max-requests", type=int, default=60)
    p.add_argument("--rate-limit-window-s", type=float, default=60.0)
    p.add_argument("--loader", type=str, default="package", choices=["package", "nanollama"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pkg_dir = Path(args.package)
    engine = build_engine_from_package(
        package_path=str(pkg_dir),
        device=args.device,
        dtype=args.dtype,
        quant_mode=args.quant_mode,
        loader=args.loader,
    )
    serving_config = ServingConfig(
        host=args.host,
        port=args.port,
        device=args.device,
        dtype=args.dtype,
        quant_mode=args.quant_mode,
        log_raw_prompts=args.log_raw_prompts,
        rate_limit_enabled=not args.disable_rate_limit,
        rate_limit_max_requests=args.rate_limit_max_requests,
        rate_limit_window_s=args.rate_limit_window_s,
    )
    app = create_app(engine, config=serving_config)

    uvicorn.run(app, host=serving_config.host, port=serving_config.port)


if __name__ == "__main__":
    main()

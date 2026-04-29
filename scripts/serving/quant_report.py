# scripts/quant_report.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_lab.eval.report import summarize_precision_recommendation, write_quant_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantization report writer")
    p.add_argument("--results-json", type=str, required=True, help="JSON list of runtime result objects")
    p.add_argument("--out", type=str, default="experiments/serving_quant/quant_report.md")
    p.add_argument("--recommendation-out", type=str, default="experiments/serving_quant/recommendation.json")
    p.add_argument(
        "--strict-recommendation",
        action="store_true",
        help="Fail if the recommendation scorer is not available.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        results = payload["results"]
    elif isinstance(payload, list):
        results = payload
    else:
        raise ValueError("--results-json must contain either a JSON list or an object with a 'results' list")

    write_quant_report(results=results, out_path=args.out)

    try:
        recommendation = summarize_precision_recommendation(results)
    except NotImplementedError as exc:
        if args.strict_recommendation:
            raise
        recommendation = {
            "status": "todo_user",
            "message": str(exc),
        }
    out = Path(args.recommendation_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recommendation, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

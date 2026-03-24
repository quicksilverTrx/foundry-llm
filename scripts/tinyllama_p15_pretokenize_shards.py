from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# TinyLlama bridge START: tinyllama_p15 pretokenize shards CLI
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.data.pretok_shards import (
    build_tinyllama_p15_pretokenized_shards,
    load_tokenizer_from_artifact_dir,
    spot_decode_shard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build tinyllama_p15 pretokenized uint16 shards from raw docs and split manifest, "
            "or run spot-decode on a shard."
        )
    )

    # Build mode inputs.
    parser.add_argument("--input-file", type=Path, help="Raw docs source file; split manifest never provides doc text.")
    parser.add_argument("--tokenizer-artifact-dir", type=Path, required=True, help="Tokenizer artifact directory.")
    parser.add_argument("--split-manifest", type=Path, help="Split membership/provenance manifest (not doc text).")
    parser.add_argument("--output-dir", type=Path, help="Output directory for train/val shards and root manifest.")
    parser.add_argument("--max-tokens-per-shard", type=int, default=1_000_000, help="Deterministic token cap per shard.")

    # Spot-decode mode inputs.
    parser.add_argument("--spot-decode-shard", type=Path, help="Optional shard .bin path for spot-decode utility.")
    parser.add_argument("--spot-decode-max-docs", type=int, default=5, help="Maximum docs to decode during spot-decode.")
    return parser


def _validate_build_args(args: argparse.Namespace) -> None:
    required = {
        "--input-file": args.input_file,
        "--split-manifest": args.split_manifest,
        "--output-dir": args.output_dir,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise SystemExit(
            "Error: build mode requires " + ", ".join(missing)
        )



def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.spot_decode_shard is not None:
            tokenizer, _ = load_tokenizer_from_artifact_dir(args.tokenizer_artifact_dir)
            eot_id = tokenizer.token_to_id("<|endoftext|>")
            out = spot_decode_shard(
                args.spot_decode_shard,
                tokenizer=tokenizer,
                eot_token_id=eot_id,
                max_docs=args.spot_decode_max_docs,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0

        _validate_build_args(args)
        root_manifest_path = build_tinyllama_p15_pretokenized_shards(
            input_file=args.input_file,
            tokenizer_artifact_dir=args.tokenizer_artifact_dir,
            split_manifest_path=args.split_manifest,
            output_dir=args.output_dir,
            max_tokens_per_shard=args.max_tokens_per_shard,
        )
        print(f"tinyllama_p15 shard manifest written: {root_manifest_path}")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
# TinyLlama bridge END: tinyllama_p15 pretokenize shards CLI

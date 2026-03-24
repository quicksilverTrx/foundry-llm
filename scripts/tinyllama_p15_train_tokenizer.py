from __future__ import annotations

import argparse
from pathlib import Path
import sys

# TinyLlama bridge START: tinyllama_p15 tokenizer training CLI
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import (
    TOKENIZER_ARTIFACT_FILENAMES,
    build_tokenizer_artifact_from_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train tinyllama_p15 tokenizer artifact from raw docs (one non-empty line per doc)."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Text corpus path (one non-empty line per doc).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output artifact directory.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Requested tokenizer vocab size.")
    parser.add_argument("--train-ratio", type=float, default=0.98, help="Train split ratio in (0, 1).")
    parser.add_argument("--split-seed", type=int, default=0, help="Deterministic split seed.")
    parser.add_argument(
        "--backend-family",
        type=str,
        default="sentencepiece",
        choices=("sentencepiece", "legacy_bpe", "bpe"),
        help="Tokenizer backend family to train.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        # End-to-end build: read docs -> split -> train tokenizer -> emit artifact + sidecars.
        paths = build_tokenizer_artifact_from_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.backend_family,
            train_ratio=args.train_ratio,
            split_seed=args.split_seed,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Tokenizer artifact created at: {args.output_dir}")
    # Print canonical artifact filenames for easy shell piping and verification.
    for key in (
        "vocab",
        "merges",
        "sentencepiece_model",
        "external_id_map",
        "sentencepiece_meta",
        "config",
        "stats",
        "reserved_tokens",
        "split_manifest",
        "tokenizer_hash",
    ):
        if key in paths:
            print(f"{TOKENIZER_ARTIFACT_FILENAMES[key]}: {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# TinyLlama bridge END: tinyllama_p15 tokenizer training CLI

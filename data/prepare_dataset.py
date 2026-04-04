#!/usr/bin/env python3
"""
Download and tokenise FineWeb-Edu for NanoLlama pretraining.

Streams the FineWeb-Edu 10B-token sample from HuggingFace, tokenises each
document with GPT-2 BPE (tiktoken), and writes uint16 .npy shard files that
the ShardLoader / ShardTrainer pipeline can consume directly.

Shard layout:

    <out_dir>/
        edufineweb_val_000000.npy        ← first ~100 M tokens (validation)
        edufineweb_train_000001.npy      ← next  ~100 M tokens
        edufineweb_train_000002.npy
        …
The tokenisation approach and shard naming convention are adapted from
Andrej Karpathy's build-nanogpt project:
    https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
    (MIT Licence — see the URL above for the original source)
Usage
-----
    # Install extra deps once:
    pip install datasets tiktoken numpy tqdm

    # Default: writes to ./data/edu_fineweb10B  (~20 GB, ~45 min)
    python data/prepare_dataset.py

    # Custom output path:
    python data/prepare_dataset.py --out_dir /fast_disk/edu_fineweb10B

    # Smaller test run (first 5 shards only):
    python data/prepare_dataset.py --max_shards 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"   # ≈10 B GPT-2 tokens
SHARD_SIZE = 10 ** 8           # 100 M tokens per shard
DEFAULT_OUT = "data/edu_fineweb10B"


# ── helpers ───────────────────────────────────────────────────────────────────

def _shard_path(out_dir: Path, idx: int) -> Path:
    """Shard 0 is the held-out validation set; shards 1-N are training."""
    if idx == 0:
        return out_dir / "edufineweb_val_000000.npy"
    return out_dir / f"edufineweb_train_{idx:06d}.npy"


def _flush_shard(tokens: list[int], path: Path) -> None:
    arr = np.array(tokens, dtype=np.uint16)
    np.save(path, arr)
    mb = arr.nbytes / 1e6
    kind = "val " if "val" in path.name else "train"
    print(f"  [{kind}] {path.name}  {len(tokens):>12,} tokens  {mb:>6.0f} MB")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu shards for NanoLlama pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT,
        help="Output directory for .npy shard files",
    )
    parser.add_argument(
        "--shard_size", type=int, default=SHARD_SIZE,
        help="Maximum tokens per shard",
    )
    parser.add_argument(
        "--max_shards", type=int, default=None,
        help="Stop after writing this many shards (useful for quick tests)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Late imports so users get a clear error if deps are missing
    try:
        import tiktoken
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}\n"
            "Install with: pip install datasets tiktoken tqdm"
        ) from exc

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token   # <|endoftext|> = 50256

    print(f"Streaming {DATASET_ID} / {DATASET_NAME} from HuggingFace …")
    ds = load_dataset(DATASET_ID, name=DATASET_NAME, split="train", streaming=True)

    shard_tokens: list[int] = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    print(f"Writing shards to: {out_dir.resolve()}")
    print(f"Shard size: {args.shard_size:,} tokens  |  val = shard 0, train = shards 1+\n")

    for doc in tqdm(ds, desc="tokenising", unit=" docs"):
        # Prepend EOT so each document boundary is marked in the token stream
        doc_tokens = [eot] + enc.encode_ordinary(doc["text"])
        shard_tokens.extend(doc_tokens)
        total_tokens += len(doc_tokens)
        total_docs += 1

        # Flush complete shards
        while len(shard_tokens) >= args.shard_size:
            _flush_shard(shard_tokens[: args.shard_size], _shard_path(out_dir, shard_idx))
            shard_tokens = shard_tokens[args.shard_size :]
            shard_idx += 1
            if args.max_shards is not None and shard_idx >= args.max_shards:
                print(f"\nStopped after {args.max_shards} shards (--max_shards).")
                _summary(out_dir, total_docs, total_tokens, shard_idx)
                return

    # Flush final partial shard (may be <shard_size tokens)
    if shard_tokens:
        _flush_shard(shard_tokens, _shard_path(out_dir, shard_idx))

    _summary(out_dir, total_docs, total_tokens, shard_idx + 1)


def _summary(out_dir: Path, total_docs: int, total_tokens: int, n_shards: int) -> None:
    print(f"\n{'─'*60}")
    print(f"  Documents : {total_docs:,}")
    print(f"  Tokens    : {total_tokens/1e9:.3f} B")
    print(f"  Shards    : {n_shards}  (val=1, train={max(0, n_shards-1)})")
    print(f"  Output    : {out_dir.resolve()}")
    print(f"{'─'*60}")
    print("\nNext step:")
    print(f"  python scripts/pretrain_nanollama.py --data_dir {out_dir}")


if __name__ == "__main__":
    main()

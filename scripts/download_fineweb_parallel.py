#!/usr/bin/env python3
"""
FineWebEDU-10BT download + tokenize — exact Karpathy fineweb.py spec.

Uses HuggingFace `datasets` library (handles download, caching, retries)
+ multiprocessing.Pool.imap for per-document parallel tokenization.

Memory profile per worker: O(one document) — no giant Python int lists.
Pre-allocated 100M-token numpy shard buffer in main process.

Matches Karpathy's build-nanogpt/fineweb.py spec exactly:
  - Tokenizer:    tiktoken GPT-2 BPE
  - EOT:          prepended to each document ([EOT, tok1, tok2, ...])
  - Shard size:   100,000,000 tokens
  - Val shard:    first shard (shard 000000) → edufineweb_val_000000.npy
  - Train shards: subsequent shards → edufineweb_train_000001.npy …

Expected wall time: ~40-70 min on H100 pod (HF download is the bottleneck).

Usage:
    export HF_TOKEN=hf_...
    python3 scripts/download_fineweb_parallel.py \\
        --out-dir ./data/edu_fineweb10B \\
        --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"
TIKTOKEN_ENC = "gpt2"
EOT_TOKEN    = 50256        # <|endoftext|> in GPT-2 BPE
SHARD_SIZE   = int(1e8)     # 100,000,000 tokens per shard


# ── worker (runs in pool process — one doc at a time, tiny memory) ─────────────

def tokenize(doc: dict) -> np.ndarray:
    """
    Tokenize a single FineWeb document.

    Returns uint16 array: [EOT, tok1, tok2, ...]
    Each worker process initializes tiktoken once (LRU-cached by tiktoken).
    """
    import tiktoken
    enc = tiktoken.get_encoding(TIKTOKEN_ENC)
    tokens = enc.encode_ordinary(doc["text"] or "")
    out = np.empty(1 + len(tokens), dtype=np.uint16)
    out[0] = EOT_TOKEN
    out[1:] = tokens
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="FineWebEDU-10BT download + tokenize (Karpathy spec)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path, default=Path("./data/edu_fineweb10B"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="Parallel tokenizer worker processes")
    ap.add_argument("--hf-token", type=str, default=None)
    args = ap.parse_args()

    hf_token: Optional[str] = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if hf_token:
        os.environ["HF_TOKEN"]                 = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"]   = hf_token

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir : {out_dir}")
    print(f"Workers    : {args.workers}")
    print(f"Shard size : {SHARD_SIZE:,} tokens")
    print(f"Val split  : shard 000000 → edufineweb_val_000000.npy")
    print(f"HF token   : {'set' if hf_token else 'NOT SET — will likely fail'}")
    print()

    # ── download + cache dataset ───────────────────────────────────────────────
    print("Loading dataset (this downloads to HF_DATASETS_CACHE on first run)...",
          flush=True)
    from datasets import load_dataset

    fw = load_dataset(
        DATASET_REPO,
        name=DATASET_NAME,
        split="train",
        token=hf_token,
    )
    print(f"Dataset ready: {len(fw):,} documents", flush=True)
    print()

    # ── tokenize + shard ───────────────────────────────────────────────────────
    t0          = time.time()
    shard_index = 0
    buf         = np.empty((SHARD_SIZE,), dtype=np.uint16)   # 200 MB, reused
    buf_pos     = 0
    tokens_total = 0
    shards_done  = 0
    docs_done    = 0
    n_docs       = len(fw)

    print(f"Tokenizing {n_docs:,} documents with {args.workers} workers...",
          flush=True)

    with mp.Pool(args.workers) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            docs_done += 1

            # Fill current shard, spilling remainder into next
            pos = 0
            while pos < len(tokens):
                space = SHARD_SIZE - buf_pos
                take  = min(space, len(tokens) - pos)
                buf[buf_pos : buf_pos + take] = tokens[pos : pos + take]
                buf_pos      += take
                pos          += take
                tokens_total += take

                if buf_pos == SHARD_SIZE:
                    split   = "val" if shard_index == 0 else "train"
                    fname   = out_dir / f"edufineweb_{split}_{shard_index:06d}.npy"
                    np.save(str(fname), buf)
                    shards_done += 1
                    elapsed = time.time() - t0
                    rate    = tokens_total / elapsed if elapsed > 0 else 0
                    print(
                        f"[{elapsed/60:.1f}m] shard {shard_index:06d} ({split})  "
                        f"docs={docs_done:,}/{n_docs:,}  "
                        f"total={tokens_total/1e9:.3f}B tok  "
                        f"rate={rate/1e6:.1f}M tok/s",
                        flush=True,
                    )
                    shard_index += 1
                    buf_pos = 0

        # Flush partial last shard
        if buf_pos > 0:
            split = "val" if shard_index == 0 else "train"
            fname = out_dir / f"edufineweb_{split}_{shard_index:06d}.npy"
            np.save(str(fname), buf[:buf_pos])
            shards_done += 1
            elapsed = time.time() - t0
            print(
                f"[{elapsed/60:.1f}m] shard {shard_index:06d} ({split}, partial)  "
                f"tokens={buf_pos:,}",
                flush=True,
            )

    # ── summary + sanity check ────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"DONE  shards={shards_done}  "
          f"tokens={tokens_total/1e9:.3f}B  "
          f"elapsed={elapsed/60:.1f}min")

    train_shards = sorted(out_dir.glob("edufineweb_train_*.npy"))
    val_shards   = sorted(out_dir.glob("edufineweb_val_*.npy"))
    print(f"Train shards : {len(train_shards)}")
    print(f"Val   shards : {len(val_shards)}")

    if val_shards:
        v = np.load(str(val_shards[0]))
        print(f"Val[0]   : {len(v):,} tokens  dtype={v.dtype}  max={v.max()}")

    if train_shards:
        s = np.load(str(train_shards[0]))
        print(f"Train[0] : {len(s):,} tokens  dtype={s.dtype}  max={s.max()}")
        assert s.max() < 50304, f"Token out of range: {s.max()}"

    print("Token range check: PASSED")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

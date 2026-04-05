#!/usr/bin/env python3
"""
Pretrain NanoLlama on FineWeb-Edu.

Two-command pipeline
--------------------
    # Step 1 — download and tokenise FineWeb-Edu (~45 min, ~20 GB disk)
    python data/prepare_dataset.py

    # Step 2 — train (~9 h on a single RTX 4090)
    python scripts/pretrain_nanollama.py

Hardware requirements
---------------------
    GPU  : NVIDIA RTX 3090 / 4090 or equivalent (≥18 GB VRAM)
    RAM  : ≥32 GB recommended for data streaming
    Disk : ~20 GB for shards  +  ~2 GB for checkpoints (model-only)

The default config (configs/nanollama_8l.json) reproduces the Phase 6
NanoLlama 8L run:  127.6 M params | 2.5 B tokens | val loss 3.3566 | BPB 1.016

Training loop
-------------
Uses ShardTrainer — a step-based pretraining loop extracted from the
original pod training script (run_phase6.py).  Passing the same
ShardTrainerConfig and MiniGPTConfig to ShardTrainer is *guaranteed* to
produce identical outputs to the original run: same LR schedule
(cosine_with_warmup), same optimizer (AdamW with weight-decay groups),
same data order (sequential shard reads).

Checkpoint format
-----------------
Each checkpoint is a dict with keys:
    step, val_loss, config (model dict), model_state_dict, optimizer_state_dict

To load for inference:
    ckpt = torch.load('runs/nanollama_8l/ckpts/step_04768.pt', map_location='cpu')
    model = MiniGPT(MiniGPTConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model_state_dict'])
"""
from __future__ import annotations

import os

# Must be set before any CUDA call.  Prevents memory-fragmentation OOM that
# occurs at B=16, T=1024 with the default caching allocator.  Safe to leave
# on for all runs.  See PyTorch docs: notes/cuda.html#memory-management
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.shard_trainer import ShardTrainer, ShardTrainerConfig

DEFAULT_CONFIG = REPO_ROOT / "configs" / "nanollama_8l.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _check_data(data_dir: Path) -> None:
    if data_dir.exists():
        return
    print(f"\nDataset directory not found: {data_dir}")
    print("\nCreate it first:")
    print("    python data/prepare_dataset.py")
    print(f"    python data/prepare_dataset.py --out_dir {data_dir}")
    sys.exit(1)


def _warn_cuda_alloc() -> None:
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in conf:
        print(
            "WARNING: PYTORCH_CUDA_ALLOC_CONF not set — OOM may occur at B=16.\n"
            "  Fix: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
            "  Or run via:  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/pretrain_nanollama.py"
        )


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pretrain NanoLlama on FineWeb-Edu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="JSON config file (model + training + data)",
    )
    p.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing edufineweb_*.npy shards (overrides config)",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for checkpoints and logs (overrides config)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Training device: cuda / mps / cpu (auto-detected if omitted)",
    )
    p.add_argument(
        "--max_steps", type=int, default=None,
        help="Stop after N optimizer steps (overrides config; useful for smoke tests)",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from",
    )
    return p


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    if not args.config.exists():
        sys.exit(f"Config file not found: {args.config}")

    raw = _load_json(args.config)
    model_cfg_dict: dict = raw["model"]
    train_cfg_dict: dict = dict(raw["training"])
    data_cfg_dict: dict  = raw.get("data", {})

    # ShardTrainerConfig holds both training hyperparams and data paths
    train_cfg_dict.update(data_cfg_dict)

    # CLI overrides — explicit args win over config file
    device = args.device or _auto_device()
    train_cfg_dict["device"] = device

    if args.data_dir  is not None: train_cfg_dict["data_dir"]   = args.data_dir
    if args.out_dir   is not None: train_cfg_dict["out_dir"]    = args.out_dir
    if args.max_steps is not None: train_cfg_dict["max_steps"]  = args.max_steps
    if args.resume    is not None: train_cfg_dict["resume_ckpt"] = args.resume

    # Validate data directory before spending time building the model
    _check_data(Path(train_cfg_dict.get("data_dir", "")))

    if device == "cuda":
        _warn_cuda_alloc()

    # ── build model ───────────────────────────────────────────────────────────
    model_config = MiniGPTConfig(**model_cfg_dict)
    model = MiniGPT(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"NanoLlama  {n_params/1e6:.1f}M params | "
        f"{model_config.n_layers}L  "
        f"d{model_config.d_model}  "
        f"h{model_config.n_heads}  "
        f"kv{model_config.num_kv_heads}  "
        f"ff{model_config.d_ff}"
    )
    print(f"Device : {device}")
    print(f"Config : {args.config}")
    print()

    # ── build trainer and run ─────────────────────────────────────────────────
    trainer_config = ShardTrainerConfig(**train_cfg_dict)
    trainer = ShardTrainer(model, model_cfg_dict, trainer_config)
    trainer.fit()


if __name__ == "__main__":
    main()

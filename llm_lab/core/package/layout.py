# llm_lab/core/package/layout.py
from __future__ import annotations

from pathlib import Path

MODEL_CONFIG_FILENAME = "config.json"
TOKENIZER_DIRNAME = "tokenizer"
CHECKPOINTS_DIRNAME = "checkpoints"
BEST_CHECKPOINT_NAME = "best_val.pt"

def tokenizer_dir(package_dir: Path) -> Path:
    return package_dir / TOKENIZER_DIRNAME


def checkpoints_dir(package_dir: Path) -> Path:
    return package_dir / CHECKPOINTS_DIRNAME


def best_checkpoint_path(package_dir: Path) -> Path:
    return checkpoints_dir(package_dir) / BEST_CHECKPOINT_NAME


def step_checkpoint_path(package_dir: Path, step: int) -> Path:
    return checkpoints_dir(package_dir) / f"model_step_{step:06d}.pt"
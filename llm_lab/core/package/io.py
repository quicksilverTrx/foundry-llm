# llm_lab/core/package/io.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Union, Optional

import torch
from torch import nn

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer
from llm_lab.core.package.layout import (
    MODEL_CONFIG_FILENAME,
    tokenizer_dir,
    checkpoints_dir,
    best_checkpoint_path,
)


def save_model_package(
    package_dir: Union[str, Path],
    config: MiniGPTConfig,
    tokenizer: SubwordTokenizer,
    model: nn.Module,
    *,
    is_best: bool = True,
    step: Optional[int] = None,
) -> Path:
    """
    Save config, tokenizer artifacts, and model weights into a package folder.

    Returns the path to the checkpoint file just written.
    """
    pkg = Path(package_dir)
    pkg.mkdir(parents=True, exist_ok=True)


    config_path = pkg / MODEL_CONFIG_FILENAME
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)

    # 2) save tokenizer under tokenizer/
    tok_dir = tokenizer_dir(pkg)
    tok_dir.mkdir(exist_ok=True)
    vocab_path = tok_dir / "vocab.json"
    merges_path = tok_dir / "merges.txt"
    tokenizer.save(vocab_path, merges_path)

    # 3) save model weights under checkpoints/
    ckpt_dir = checkpoints_dir(pkg)
    ckpt_dir.mkdir(exist_ok=True)

    if is_best:
        ckpt_path = best_checkpoint_path(pkg)
    else:
        assert step is not None, "step must be provided when is_best=False"
        ckpt_path = ckpt_dir / f"model_step_{step:06d}.pt"

    state = {
        "model_state": model.state_dict()
    }
    torch.save(state, ckpt_path)

    return ckpt_path


def load_model_package(
    package_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[MiniGPTConfig, SubwordTokenizer, MiniGPT]:
    """
    Load config, tokenizer, and model from a package folder.
    For now, always loads the 'best_val.pt' checkpoint.
    """
    pkg = Path(package_dir)


    config_path = pkg / MODEL_CONFIG_FILENAME
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    config = MiniGPTConfig(**raw_cfg)


    tok_dir = tokenizer_dir(pkg)
    tokenizer = SubwordTokenizer.load_from_files(
        vocab_path=tok_dir / "vocab.json",
        merges_path=tok_dir / "merges.txt",
    )


    model = MiniGPT(config)
    ckpt = torch.load(best_checkpoint_path(pkg), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    return config, tokenizer, model

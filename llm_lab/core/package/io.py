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
NANOLLAMA_DEFAULTS = {
    "norm_type": "rmsnorm",
    "mlp_type": "swiglu",
    "pos_encoding_type": "rope",
    "attention_type": "gqa",
}

def _apply_arch_family_defaults(raw: dict) -> dict:
    arch = raw.get("arch_family", "miniGPT")
    if arch == "nanollama":
        for k, v in NANOLLAMA_DEFAULTS.items():
            raw.setdefault(k, v)
    return raw

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

    # 2) save tokenizer under tokenizer/ (backend-aware artifact payload)
    tok_dir = tokenizer_dir(pkg)
    tok_dir.mkdir(exist_ok=True)
    tokenizer.save(artifact_dir=tok_dir)

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
    raw_cfg = _apply_arch_family_defaults(raw_cfg)
    config = MiniGPTConfig(**raw_cfg)


    tok_dir = tokenizer_dir(pkg)
    tokenizer = SubwordTokenizer.load(tok_dir)
    tokenizer_vocab_size = len(tokenizer.stoi)
    if tokenizer_vocab_size != int(config.vocab_size):
        raise ValueError(
            "tokenizer artifact vocab size does not match model config "
            f"(tokenizer={tokenizer_vocab_size}, config={config.vocab_size})"
        )


    model = MiniGPT(config)
    ckpt = torch.load(best_checkpoint_path(pkg), map_location=device)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(ckpt["model_state"].keys())
    extra = ckpt_keys - model_keys
    if extra:
        filtered = {k: v for k, v in ckpt["model_state"].items() if k in model_keys}
        model.load_state_dict(filtered, strict=True)
    else:
        model.load_state_dict(ckpt["model_state"])
    model.to(device)

    return config, tokenizer, model

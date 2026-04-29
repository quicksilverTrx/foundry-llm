# llm_lab/core/package/nanollama_loader.py
"""Load a raw NanoLlama .pt checkpoint into (config, tokenizer, model).

NanoLlama checkpoints use different keys from the sp16k package format:
  - "config" (dict)            vs  model_config.json
  - "model_state_dict"         vs  "model_state"
  - "step" / "val_loss"        vs  "global_step" / "best_val_loss"

Returns the same (MiniGPTConfig, tokenizer, MiniGPT) triple as
load_model_package() so it plugs directly into Engine().
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.package.io import _apply_arch_family_defaults
from llm_lab.core.tokenization.tiktoken_wrapper import TiktokenWrapper


def load_nanollama_checkpoint(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    encoding_name: str = "gpt2",
) -> Tuple[MiniGPTConfig, TiktokenWrapper, MiniGPT]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    raw_cfg = ckpt["config"]
    raw_cfg = _apply_arch_family_defaults(raw_cfg)
    cfg = MiniGPTConfig(**raw_cfg)

    model = MiniGPT(cfg)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
    model.load_state_dict(ckpt[state_key])
    model.to(device).eval()

    tokenizer = TiktokenWrapper(encoding_name=encoding_name)
    return cfg, tokenizer, model

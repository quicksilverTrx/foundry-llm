# tests/core/test_nanollama_config_identity.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_lab.core.package.io import save_model_package, load_model_package
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig
from llm_lab.core.package.layout import MODEL_CONFIG_FILENAME


def _read_config(pkg_dir: Path) -> dict:
    cfg_path = pkg_dir / MODEL_CONFIG_FILENAME
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _write_config(pkg_dir: Path, raw: dict) -> None:
    cfg_path = pkg_dir / MODEL_CONFIG_FILENAME
    cfg_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")


def _build_valid_nanollama_pkg(tmp_path: Path, *, mlp_type: str = "swiglu", num_kv_heads: int = 2) -> Path:
    """
    Create a fully loadable package directory (config.json + tokenizer + checkpoint).
    We intentionally build a config that is already self-consistent so load_state_dict succeeds.
    """
    torch.manual_seed(0)

    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    texts = ["hello world", "hello there", "world hello", "hello <|user|> world"]
    tok_cfg = SubwordTokenizerConfig(vocab_size=80, model_type="bpe")
    tok = SubwordTokenizer.train_from_iterator(texts, tok_cfg)

    cfg = MiniGPTConfig(
        arch_family="nanollama",
        vocab_size=len(tok.stoi),
        d_model=64,
        n_layers=2,
        n_heads=8,
        d_ff=256,
        block_size=64,
        dropout=0.0,
        # Build-time config must be explicit so model weights match checkpoint
        norm_type="rmsnorm",
        mlp_type=mlp_type,
        pos_encoding_type="rope",
        attention_type="gqa",
        num_kv_heads=num_kv_heads,
    )

    model = MiniGPT(cfg).eval()
    save_model_package(pkg_dir, cfg, tok, model, is_best=True)
    return pkg_dir


def test_nanollama_defaults_are_auto_filled(tmp_path: Path):
    """
    Remove nanollama-default fields from config.json.
    load_model_package() should inject them via _apply_arch_family_defaults(raw_cfg)
    and then MiniGPTConfig(**raw_cfg) should succeed.
    """
    pkg_dir = _build_valid_nanollama_pkg(tmp_path, mlp_type="swiglu", num_kv_heads=2)

    raw = _read_config(pkg_dir)

    # Intentionally omit fields that NANOLLAMA_DEFAULTS should set.
    for k in ["norm_type", "mlp_type", "pos_encoding_type", "attention_type"]:
        raw.pop(k, None)

    # Keep num_kv_heads: required for gqa once defaults apply
    assert raw.get("num_kv_heads") == 2

    _write_config(pkg_dir, raw)

    cfg2, tok2, model2 = load_model_package(pkg_dir, device="cpu")

    assert cfg2.arch_family == "nanollama"
    assert cfg2.norm_type == "rmsnorm"
    assert cfg2.mlp_type == "swiglu"
    assert cfg2.pos_encoding_type == "rope"
    assert cfg2.attention_type == "gqa"
    assert cfg2.num_kv_heads == 2


def test_gqa_requires_divisible_heads(tmp_path: Path):
    """
    Set num_kv_heads to an invalid value (n_heads % num_kv_heads != 0).
    This should fail *during MiniGPTConfig(**raw_cfg)* with ValueError/AssertionError
    (i.e., before any state dict loading).
    """
    pkg_dir = _build_valid_nanollama_pkg(tmp_path, mlp_type="swiglu", num_kv_heads=2)

    raw = _read_config(pkg_dir)
    raw["num_kv_heads"] = 3  # invalid: 8 % 3 != 0
    _write_config(pkg_dir, raw)

    with pytest.raises((ValueError, AssertionError)):
        _ = load_model_package(pkg_dir, device="cpu")


def test_nanollama_does_not_override_explicit_user_choices(tmp_path: Path):
    """
    Verify _apply_arch_family_defaults(raw) uses setdefault, so explicit config keys
    are preserved (defaults should NOT overwrite user-provided values).
    We make a package whose *checkpoint matches* mlp_type='gelu',
    then delete other defaulted fields and ensure they are injected while mlp_type stays gelu.
    """
    pkg_dir = _build_valid_nanollama_pkg(tmp_path, mlp_type="gelu", num_kv_heads=2)

    raw = _read_config(pkg_dir)

    # Keep explicit user override
    assert raw["mlp_type"] == "gelu"

    # Omit other nanollama-default fields so loader fills them
    for k in ["norm_type", "pos_encoding_type", "attention_type"]:
        raw.pop(k, None)

    _write_config(pkg_dir, raw)

    cfg2, tok2, model2 = load_model_package(pkg_dir, device="cpu")

    # must preserve explicit choice
    assert cfg2.mlp_type == "gelu"

    # must inject missing defaults
    assert cfg2.norm_type == "rmsnorm"
    assert cfg2.pos_encoding_type == "rope"
    assert cfg2.attention_type == "gqa"
    assert cfg2.num_kv_heads == 2

# tests/core/test_package_io.py
import json

import pytest
import torch

from llm_lab.core.package.io import save_model_package, load_model_package
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig
from llm_lab.core.package.layout import MODEL_CONFIG_FILENAME


def test_save_load_model_package_roundtrip(tmp_path):
    torch.manual_seed(0)

    texts = ["hello world", "hello there", "world hello"]
    tok_cfg = SubwordTokenizerConfig(vocab_size=80, model_type="bpe")
    tok = SubwordTokenizer.train_from_iterator(texts, tok_cfg)

    cfg = MiniGPTConfig(
        vocab_size=len(tok.stoi),
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        block_size=16,
        dropout=0.0,
        pos_encoding_type="learned",
    )

    model = MiniGPT(cfg).eval()

    # fixed input
    ids = tok.encode("hello world")
    x = torch.tensor([ids[: cfg.block_size]], dtype=torch.long)  # [1, T]

    with torch.no_grad():
        logits1,_ = model(x)

    save_model_package(tmp_path, cfg, tok, model, is_best=True)

    cfg2, tok2, model2 = load_model_package(tmp_path, device="cpu")
    model2.eval()

    with torch.no_grad():
        logits2,_ = model2(x)

    assert cfg2.vocab_size == cfg.vocab_size
    assert tok2.encode("hello world") == tok.encode("hello world")
    assert logits1.shape == logits2.shape
    assert torch.allclose(logits1, logits2, atol=0, rtol=0)  # exact match in eval, dropout=0


def test_save_load_model_package_roundtrip_sentencepiece(tmp_path):
    torch.manual_seed(0)

    texts = ["alpha beta gamma", "beta gamma alpha", "gamma alpha beta"]
    tok_cfg = SubwordTokenizerConfig(vocab_size=96, model_type="sentencepiece")
    tok = SubwordTokenizer.train_from_iterator(texts, tok_cfg)

    cfg = MiniGPTConfig(
        vocab_size=len(tok.stoi),
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_ff=64,
        block_size=16,
        dropout=0.0,
        pos_encoding_type="learned",
    )
    model = MiniGPT(cfg).eval()

    save_model_package(tmp_path, cfg, tok, model, is_best=True)
    cfg2, tok2, _ = load_model_package(tmp_path, device="cpu")

    assert cfg2.vocab_size == cfg.vocab_size
    assert tok2.backend_family == "sentencepiece"
    assert tok2.encode("alpha beta") == tok.encode("alpha beta")


def test_load_model_package_fails_when_tokenizer_vocab_mismatches_config(tmp_path):
    texts = ["hello world", "world hello"]
    tok = SubwordTokenizer.train_from_iterator(texts, SubwordTokenizerConfig(vocab_size=64, model_type="bpe"))
    cfg = MiniGPTConfig(
        vocab_size=len(tok.stoi),
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_ff=32,
        block_size=8,
        dropout=0.0,
    )
    model = MiniGPT(cfg).eval()
    save_model_package(tmp_path, cfg, tok, model, is_best=True)

    cfg_path = tmp_path / MODEL_CONFIG_FILENAME
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw["vocab_size"] = int(raw["vocab_size"]) + 1
    cfg_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer artifact vocab size does not match model config"):
        _ = load_model_package(tmp_path, device="cpu")

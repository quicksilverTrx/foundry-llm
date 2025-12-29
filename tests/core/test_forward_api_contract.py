import torch
import pytest
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig

def _cfg():
    return MiniGPTConfig(
        vocab_size=30, d_model=16, n_layers=2, n_heads=2, d_ff=64,
        block_size=8, dropout=0.0, pos_encoding_type="learned",
        norm_type="layernorm", mlp_type="gelu"
    )

def test_forward_returns_tuple_and_logits_shape():
    m = MiniGPT(_cfg())
    x = torch.randint(0, m.config.vocab_size, (2, 5))
    logits, cache = m(x, attention_mask=None, past_key_values=None, use_cache=False)
    assert logits.shape == (2, 5, m.config.vocab_size)
    assert cache is None

def test_use_cache_true_behavior_is_explicit():
    m = MiniGPT(_cfg())
    x = torch.randint(0, m.config.vocab_size, (2, 5))




    with pytest.raises(NotImplementedError):
        _ = m(x, use_cache=True)


def test_forward_returns_tuple_and_logits_shape_1():
    cfg = MiniGPTConfig(vocab_size=30, d_model=16, n_layers=2, n_heads=2, d_ff=32, block_size=8, dropout=0.0)
    m = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 5))
    out = m(x)
    assert isinstance(out, tuple) and len(out) == 2
    logits, past = out
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert past is None  # Phase 1 stub
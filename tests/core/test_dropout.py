# tests/core/test_dropout.py
import torch
import pytest
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig

def _tiny_cfg(vocab=20, block =8,dropout =0):
    return MiniGPTConfig(vocab_size = vocab,d_model=16,n_layers =2,n_heads=2, d_ff=64, block_size=block,dropout = dropout)

def test_deterministic_when_dropout_zero():
    torch.manual_seed(0)
    m = MiniGPT(_tiny_cfg(dropout =0))
    x = torch.randint(0,m.config.vocab_size,(2,5))
    m.train()
    y1,_ = m(x)
    y2,_ = m(x)
    assert torch.allclose(y1,y2)

def test_nondeterministic_when_dropout_nonzero():
    torch.manual_seed(0)
    m = MiniGPT(_tiny_cfg(dropout =0.2))
    x = torch.randint(0,m.config.vocab_size,(2,5))
    m.train()
    y1,_ = m(x)
    y2,_ = m(x)
    assert not torch.allclose(y1,y2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()

def test_dropout_disabled_in_eval_mode():
    torch.manual_seed(0)
    m = MiniGPT(_tiny_cfg(dropout=0.2))
    x = torch.randint(0, m.config.vocab_size, (2, 5))
    m.eval()
    y1, _ = m(x)
    y2, _ = m(x)
    assert torch.allclose(y1, y2)

    
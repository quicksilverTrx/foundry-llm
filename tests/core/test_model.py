# tests/core/test_model.py
import torch
import pytest
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
"""(foundry-llm) ron@MacBookPro-7 foundry-llm % pytest tests/core/test_model.py 
=================================================================================== test session starts ===================================================================================
platform darwin -- Python 3.11.11, pytest-9.0.1, pluggy-1.6.0
rootdir: /Users/ron/Desktop/github_projects/foundry-llm
configfile: pyproject.toml
collected 3 items                                                                                                                                                                         

tests/core/test_model.py ...                                                                                                                                                        [100%]

==================================================================================== 3 passed in 0.94s ===================================================================================="""
def _tiny_cfg(vocab=20, block =8):
    return MiniGPTConfig(vocab_size = vocab,d_model=16,n_layers =2,n_heads=2, d_ff=64, block_size=block,dropout = 0.0)

def test_forward_shape():
    m = MiniGPT(_tiny_cfg())
    x = torch.randint(0,m.config.vocab_size,(2,5))
    y,_ = m(x)
    assert y.shape == (2,5,m.config.vocab_size)

def test_block_size_guard():
    m = MiniGPT(_tiny_cfg(block=4))
    x = torch.randint(0, m.config.vocab_size, (1, 5))
    with pytest.raises(ValueError):
        _ = m(x)

def test_causality_end_to_end():
    m = MiniGPT(_tiny_cfg(vocab=50, block=16))
    m.eval()
    prefix = torch.tensor([[1,2,3,4,5]])
    a = torch.cat([prefix,torch.tensor([[6,6,6]])],dim = 1)
    b = torch.cat([prefix,torch.tensor([[7,7,7]])],dim = 1)

    y_a,_ = m(a) # [1, T, V]
    y_b,_ = m(b)  # [1, T, V]

    prefix_size = prefix.shape[1] # length shared prefix (number of tokens)
    y_a_prefix_logits = y_a[:,:prefix_size,:] # Slice logits for prefix positions only
    y_b_prefix_logits = y_b[:,:prefix_size,:]

    diff_prefix_logits = y_a_prefix_logits - y_b_prefix_logits #  Compute elementwise difference

    abs_diff = diff_prefix_logits.abs()
    max_abs_diff = abs_diff.max()

    assert max_abs_diff <= 1e-3

    
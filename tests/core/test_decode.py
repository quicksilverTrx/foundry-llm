# tests/core/test_decode.py

"""
(foundry-llm) ron@MacBookPro-7 foundry-llm % pytest tests/core/test_decode.py
=================================================================================== test session starts ===================================================================================
platform darwin -- Python 3.11.11, pytest-9.0.1, pluggy-1.6.0
rootdir: /Users/ron/Desktop/github_projects/foundry-llm
configfile: pyproject.toml
collected 1 item                                                                                                                                                                          

tests/core/test_decode.py .                                                                                                                                                         [100%]

==================================================================================== 1 passed in 0.86s ====================================================================================

"""
import torch
from llm_lab.core.decode.sampling import greedy_decode
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig

def test_greedy_decode_length_and_range():
     # Tiny, deterministic small vocab/model to keep test fast.
    cfg = MiniGPTConfig(vocab_size=30, d_model=16, n_layers=1, n_heads=1, d_ff=32, block_size=8, dropout=0.0)
    m = MiniGPT(cfg).eval() 

    x = torch.randint(0,cfg.vocab_size,(2,4)) #[B,T]

    y = greedy_decode(m, x, max_new_tokens=5, block_size=cfg.block_size)

    x_len = x.shape[1] # T
    assert(torch.equal(y[:,:x_len],x)) # same prefix

    assert y.dtype == torch.long
    assert y.device == x.device

    assert y.shape == (2, 9)  # T_new = T + max_new_tokens

    assert (y >= 0).all()
    assert (y < cfg.vocab_size).all() #within vocab limits

    with torch.no_grad():
        logits = m(x)   # [B, T, V]
        last_logits = logits[:,-1,:] # [B, V]
        last_tokens = last_logits.argmax(dim=-1) # [B]
    assert torch.equal(y[:,x_len],last_tokens)

    y0 = greedy_decode(m, x, max_new_tokens=0, block_size=cfg.block_size)
    assert torch.equal(y0, x)

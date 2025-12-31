import torch
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig

def _run(pos_type: str):
    cfg = MiniGPTConfig(
        vocab_size=101,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        block_size=16,
        dropout=0.0,
        pos_encoding_type=pos_type,
    )
    m = MiniGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    logits, past = m(x, use_cache=False)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert past is None  # use_cache=False

def test_pos_encoding_learned_forward():
    _run("learned")

def test_pos_encoding_sinusoidal_forward():
    _run("sinusoidal")

def test_pos_encoding_rope_forward():
    _run("rope")

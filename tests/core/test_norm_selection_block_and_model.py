import torch
from llm_lab.core.model.blocks import TransformerBlock, TransformerBlockConfig
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.model.norms import RMSNorm

def test_block_uses_rmsnorm_when_configured():
    cfg = TransformerBlockConfig(
        d_model=16, n_heads=2, d_ff=64, dropout=0.0,
        norm_type="rmsnorm", mlp_type="gelu", use_rope=False
    )
    block = TransformerBlock(cfg)
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.norm2, RMSNorm)

def test_model_final_norm_respects_norm_type():
    cfg = MiniGPTConfig(
        vocab_size=50, d_model=16, n_layers=2, n_heads=2, d_ff=64,
        block_size=8, dropout=0.0, norm_type="rmsnorm", mlp_type="gelu",
        pos_encoding_type="learned"
    )
    m = MiniGPT(cfg)
    # This should be RMSNorm after you fix MiniGPT to use make_norm
    assert isinstance(m.ln_f, RMSNorm)



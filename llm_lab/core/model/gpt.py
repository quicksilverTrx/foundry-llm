from __future__ import annotations
from dataclasses import dataclass

from torch import nn
import torch

from llm_lab.core.model.blocks import TransformerBlock, TransformerBlockConfig

@dataclass
class MiniGPTConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    block_size: int
    dropout: float = 0.0

class MiniGPT(nn.Module):
    """
    Char-level GPT-style decoder.
    """
    def __init__(self,config: MiniGPTConfig):
        super().__init__()
        self.config = config

        #embedding layer
        self.token_embed = nn.Embedding(config.vocab_size,config.d_model)
        self.pos_embed = nn.Embedding(config.block_size,config.d_model)

        # Blocks
        block_cfg = TransformerBlockConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList([TransformerBlock(block_cfg) for _ in range(config.n_layers)])

        # Final norm + LM head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model,config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        return: logits [B, T, vocab_size]
        """
        B,T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError("sequence length exceeds block size")
        token_embeddings = self.token_embed(input_ids)
        position_indices = torch.arange(T,device=input_ids.device)
        position_embeddings = self.pos_embed(position_indices)
        x = token_embeddings + position_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x) # [B, T, vocab_size]
        return x
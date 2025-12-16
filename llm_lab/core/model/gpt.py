#llm_lab/core/model/gpt
from __future__ import annotations
from dataclasses import dataclass

from torch import nn
import torch
from typing import Literal

from llm_lab.core.model.blocks import TransformerBlock, TransformerBlockConfig
from llm_lab.core.model.pos_encodings import SinusoidalPositionalEncoding

@dataclass
class MiniGPTConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    block_size: int
    dropout: float = 0.0

    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    mlp_type: Literal["gelu", "swiglu"] = "gelu"
    attention_type: Literal["mha", "mqa"] = "mha"

    pos_encoding_type: Literal["learned", "sinusoidal", "rope"] = "learned"

class MiniGPT(nn.Module):
    """
    Char-level GPT-style decoder.
    """
    def __init__(self,config: MiniGPTConfig):
        super().__init__()
        self.config = config

        self.sin_pos = None
        if config.pos_encoding_type == "sinusoidal":
            self.sin_pos = SinusoidalPositionalEncoding(config.block_size, config.d_model)
        #embedding layer
        self.token_embed = nn.Embedding(config.vocab_size,config.d_model)
        self.pos_embed = nn.Embedding(config.block_size,config.d_model)

        # Blocks
        block_cfg = TransformerBlockConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            use_rope= (config.pos_encoding_type=="rope")

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
        token_embeddings = self.token_embed(input_ids)  # [B,T,D]
        x = token_embeddings  # always defined
        position_indices = torch.arange(T,device=input_ids.device) # [T]
        if self.config.pos_encoding_type=="learned":
            position_embeddings = self.pos_embed(position_indices)
            x = x + position_embeddings
        elif self.config.pos_encoding_type =="sinusoidal":
            position_embeddings = self.sin_pos(position_indices)
            x = x + position_embeddings
        elif self.config.pos_encoding_type =="rope":
            pass
        for block in self.blocks:
            x = block(x, position_ids=position_indices if self.config.pos_encoding_type == "rope" else None)
        x = self.ln_f(x)
        x = self.lm_head(x) # [B, T, vocab_size]
        return x
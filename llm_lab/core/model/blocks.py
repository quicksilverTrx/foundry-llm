from __future__ import annotations

from dataclasses import dataclass
from torch import nn
import torch

from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
from llm_lab.core.model.mlp import FeedForward, FeedForwardConfig

@dataclass
class TransformerBlockConfig:
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

class TransformerBlock(nn.Module):
    def __init__(self,config: TransformerBlockConfig):
        super().__init__()
        self.config = config

        #Normalization layer
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        #Attention Block
        attention_config = MultiHeadAttentionConfig(d_model=config.d_model,
                                                    n_heads=config.n_heads,
                                                    dropout=config.dropout)
        self.blocks = MultiHeadAttention(attention_config)

        #FF Layer
        ff_config = FeedForwardConfig(d_model=config.d_model,
                                      d_ff=config.d_ff,
                                      dropout=config.dropout)
        self.mlp = FeedForward(ff_config)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        h = x + self.blocks(self.ln1(x))
        y = h + self.mlp(self.ln2(h))
        return y


from __future__ import annotations 
from dataclasses import dataclass
from torch import nn
import torch

@dataclass
class FeedForwardConfig:
    d_model: int
    d_ff: int
    dropout: int 

class FeedForward(nn.Module):
    def __init__(self,config :FeedForwardConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.d_model,config.d_ff)
        self.fc2 = nn.Linear(config.d_ff,config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        return: [B, T, d_model]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
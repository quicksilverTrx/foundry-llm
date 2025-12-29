from __future__ import annotations 
from dataclasses import dataclass
from torch import nn
import torch
from typing import Literal

@dataclass
class FeedForwardConfig:
    d_model: int
    d_ff: int
    dropout: float 
    mlp_type : Literal["swiglu","gelu"] = "gelu"

class FeedForward(nn.Module):
    def __init__(self,config :FeedForwardConfig):
        super().__init__()
        self.config = config
        if config.mlp_type=="gelu":
            self.fc1 = nn.Linear(config.d_model,config.d_ff)
            self.fc2 = nn.Linear(config.d_ff,config.d_model)
        elif config.mlp_type == "swiglu":
            self.w_up = nn.Linear(config.d_model,config.d_ff, bias=False)
            self.w_gate = nn.Linear(config.d_model,config.d_ff, bias=False)
            self.w_down = nn.Linear(config.d_ff,config.d_model, bias=False)
        self.act = nn.GELU() if config.mlp_type=="gelu" else nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        return: [B, T, d_model]
        """
        if self.config.mlp_type == "gelu":
            x = self.fc1(x)   # [B, T, d_model] -> [B, T, d_ff]   (EXPAND)
            x = self.act(x)
            x = self.fc2(x) # [B, T, d_ff]    -> [B, T, d_model] (CONTRACT)
            x = self.dropout(x)
        elif self.config.mlp_type == "swiglu":
            a = self.w_up(x)
            b = self.w_gate(x)
            h = self.act(b) * a
            x = self.w_down(h)
            x = self.dropout(x)
        return x
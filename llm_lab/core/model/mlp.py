# llm_lab/core/model/mlp.py
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
    mlp_type: Literal["swiglu", "gelu", "relu_squared"] = "gelu"

class FeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config
        if config.mlp_type == "gelu":
            self.fc1 = nn.Linear(config.d_model, config.d_ff)
            self.fc2 = nn.Linear(config.d_ff, config.d_model)
        elif config.mlp_type == "swiglu":
            self.w_up   = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        elif config.mlp_type == "relu_squared":
            # Two weight matrices (no gate branch): up then down.
            # ReLU²(x) = max(0, x)²  — reference: modded-nanogpt train_gpt.py
            self.w_up   = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        else:
            raise ValueError(f"Unknown mlp_type: {config.mlp_type!r}")
        self.act     = nn.GELU() if config.mlp_type == "gelu" else nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        return: [B, T, d_model]
        """
        if self.config.mlp_type == "gelu":
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            x = self.dropout(x)
        elif self.config.mlp_type == "swiglu":
            a = self.w_up(x)
            b = self.w_gate(x)
            h = self.act(b) * a
            x = self.w_down(h)
            x = self.dropout(x)
        elif self.config.mlp_type == "relu_squared":
            h = self.w_up(x)
            h = torch.relu(h).pow(2)  # ReLU²
            x = self.w_down(h)
            x = self.dropout(x)
        return x
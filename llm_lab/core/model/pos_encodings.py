# llm_lab/core/model/pos_encodings.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import math

class LearnedPositionEmbedding(nn.Module):
    def __init__(self,block_size: int,d_model : int):
        super().__init__()
        self.emb = nn.Embedding(block_size,d_model)

    def forward(self, position_ids : torch.Tensor) -> torch.Tensor:
        return self.emb(position_ids) # position_ids: [T] or [B, T]  ->  [T, D] or [B, T, D]
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"even d_model is required")
        positional_encoding = torch.zeros(block_size,d_model, dtype=torch.float32) # [T,D_model]

        positions = torch.arange(0,block_size,dtype=torch.float) # [T]
        positions = positions.unsqueeze(1) # [T,1]

        #[d_model/2]
        div_term = torch.exp(
            torch.arange(0,d_model,2).float() #[d_model/2]
              * (-math.log(10000)/d_model)
            ) 
        
        angles = positions * div_term  # [T, d_model/2] 
        positional_encoding[:,0::2] = torch.sin(angles) # [T,d_model/2]
        positional_encoding[:,1::2] = torch.cos(angles) # [T,d_model/2]

        self.register_buffer("positional_encoding",positional_encoding)

    def forward(self,position_ids : torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding[position_ids] # [T] -> [T,D]
        return x

@dataclass
class RoPEConfig:
    theta: float = 10000.0

def apply_rope(q: torch.Tensor,  # [B, T, head_dim]
    k: torch.Tensor,  # [B, T, head_dim]
    position_ids: torch.Tensor,  # [T] 
    *,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B,T, head_dim = q.shape
    assert head_dim % 2 ==0
    device = q.device
    if q.shape != k.shape:
        raise ValueError(f"q and k must have same shape, got {q.shape} vs {k.shape}")
    # [H/2]
    inv_freq = torch.exp(
                torch.arange(0,head_dim,2,device=device, dtype=torch.float32)#[head_dim/2]
                * (-math.log(theta)/head_dim)
                ) 
    position_ids_unsqueezed=position_ids.to(device=device, dtype=torch.float32).unsqueeze(1) #[T,1]

    angles = position_ids_unsqueezed * inv_freq # [T,head_dim/2]

    cos = torch.cos(angles).to(dtype=q.dtype) # [T,head_dim/2]
    sin = torch.sin(angles).to(dtype=q.dtype) # [T,head_dim/2]

    cos = cos.unsqueeze(0)  # [1, T, head_dim/2]
    sin = sin.unsqueeze(0)  # [1, T, head_dim/2]
    
    q_even, q_odd = q[..., 0::2], q[..., 1::2]  # [..., T, H/2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]

    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd  = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd  = k_even * sin + k_odd * cos

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_out[..., 0::2] = q_rot_even
    q_out[..., 1::2] = q_rot_odd
    k_out[..., 0::2] = k_rot_even
    k_out[..., 1::2] = k_rot_odd

    return q_out, k_out
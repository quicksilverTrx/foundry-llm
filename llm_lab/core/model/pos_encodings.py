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
    rope_scaling_type: str = "none",      # "none" | "linear"
    rope_scaling_factor: float = 1.0,     # >1 extends context
    position_offset: int = 0,             # KV-cache ready
    rope_fraction: float = 1.0,           # fraction of head dims to rotate (0,1]; 1.0=full RoPE
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rope_scaling_factor <= 0:
        raise ValueError("rope_scaling_factor must be > 0")
    if rope_scaling_type not in {"none", "linear"}:
        raise ValueError(f"rope_scaling_type must be 'none' or 'linear', got {rope_scaling_type!r}")
    if rope_scaling_type == "linear" and rope_scaling_factor < 1.0:
        raise ValueError("rope_scaling_factor must be >= 1.0 for linear scaling")
    if not (0.0 < rope_fraction <= 1.0):
        raise ValueError(f"rope_fraction must be in (0, 1], got {rope_fraction}")

    if q.shape != k.shape:
        raise ValueError(f"q and k must have same shape, got {q.shape} vs {k.shape}")
    if position_ids.ndim != 1:
        raise ValueError(f"position_ids must be 1D [T], got shape {tuple(position_ids.shape)}")

    B, T, head_dim = q.shape
    assert head_dim % 2 == 0
    device = q.device
    if position_ids.shape[0] != T:
        raise ValueError(f"position_ids length must equal T={T}, got {position_ids.shape[0]}")

    # Partial RoPE: only rotate the first rotary_dim dimensions.
    # rotary_dim must be even; remaining dims pass through unchanged.
    rotary_dim = int(head_dim * rope_fraction)
    if rotary_dim % 2 != 0:
        rotary_dim -= 1  # round down to nearest even
    rotary_dim = max(2, rotary_dim)  # at least one pair

    # [rotary_dim/2] in float32 for numeric stability
    inv_freq = torch.exp(
        torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(theta) / rotary_dim)
    )

    pos = position_ids.to(device=device, dtype=torch.float32)
    pos = pos + position_offset
    if rope_scaling_type == "linear":
        pos = pos / float(rope_scaling_factor)

    position_ids_unsqueezed = pos.unsqueeze(1)  # [T, 1]
    angles = position_ids_unsqueezed * inv_freq  # [T, rotary_dim/2]

    cos = torch.cos(angles).to(dtype=q.dtype)  # [T, rotary_dim/2]
    sin = torch.sin(angles).to(dtype=q.dtype)  # [T, rotary_dim/2]

    cos = cos.unsqueeze(0)  # [1, T, rotary_dim/2]
    sin = sin.unsqueeze(0)  # [1, T, rotary_dim/2]

    # Split q/k into the rotary portion and the pass-through portion.
    q_rot_in = q[..., :rotary_dim]   # [B, T, rotary_dim]
    q_pass   = q[..., rotary_dim:]   # [B, T, head_dim - rotary_dim]
    k_rot_in = k[..., :rotary_dim]
    k_pass   = k[..., rotary_dim:]

    q_even, q_odd = q_rot_in[..., 0::2], q_rot_in[..., 1::2]  # [B, T, rotary_dim/2]
    k_even, k_odd = k_rot_in[..., 0::2], k_rot_in[..., 1::2]

    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd  = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd  = k_even * sin + k_odd * cos

    q_rotated = torch.empty_like(q_rot_in)
    k_rotated = torch.empty_like(k_rot_in)
    q_rotated[..., 0::2] = q_rot_even
    q_rotated[..., 1::2] = q_rot_odd
    k_rotated[..., 0::2] = k_rot_even
    k_rotated[..., 1::2] = k_rot_odd

    q_out = torch.cat([q_rotated, q_pass], dim=-1)
    k_out = torch.cat([k_rotated, k_pass], dim=-1)

    return q_out, k_out
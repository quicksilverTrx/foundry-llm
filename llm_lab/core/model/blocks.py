# llm_lab/core/model/blocks.py
#llm_lab/core/model/blocks
from __future__ import annotations

from dataclasses import dataclass
from torch import nn
import torch
from typing import Optional, Literal,List,Tuple
from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig,PastKeyValue
from llm_lab.core.model.mlp import FeedForward, FeedForwardConfig
from llm_lab.core.model.norms import RMSNorm,make_norm



@dataclass
class TransformerBlockConfig:
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    use_rope: bool = False
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    mlp_type: Literal ["swiglu","gelu"] = "gelu"
    attention_type :  Literal["mha", "gqa"] = "mha"
    num_kv_heads: Optional[int] = None
    rope_scaling_type: Literal["none", "linear"] = "none"
    rope_scaling_factor: float = 1.0
    use_sdpa: bool = False   # passed through to MultiHeadAttentionConfig
    qk_norm: bool = False    # passed through to MultiHeadAttentionConfig


class TransformerBlock(nn.Module):
    def __init__(self,config: TransformerBlockConfig):
        super().__init__()
        self.config = config

        #Normalization layer
        self.norm1 = make_norm(config.norm_type,config.d_model)
        self.norm2 = make_norm(config.norm_type,config.d_model)


        #Attention Block
        attention_config = MultiHeadAttentionConfig(d_model=config.d_model,
                                                    n_heads=config.n_heads,
                                                    dropout=config.dropout,
                                                    use_rope=config.use_rope,
                                                    attention_type = config.attention_type,
                                                    num_kv_heads = config.num_kv_heads,
                                                    rope_scaling_type=config.rope_scaling_type,
                                                    rope_scaling_factor=config.rope_scaling_factor,
                                                    use_sdpa=config.use_sdpa,
                                                    qk_norm=config.qk_norm)
        self.attn = MultiHeadAttention(attention_config)

        #FF Layer
        ff_config = FeedForwardConfig(d_model=config.d_model,
                                      d_ff=config.d_ff,
                                      dropout=config.dropout,
                                      mlp_type=config.mlp_type)
        self.mlp = FeedForward(ff_config)
    
    def forward(self,x:torch.Tensor,position_ids : Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, past_key_value : Optional[PastKeyValue] = None,
                use_cache = False) -> Tuple[torch.Tensor,Optional[PastKeyValue]]:
        """
        x: [B, T, D]
        position_ids: [T] or None
        attention_mask: [B, T] (current chunk)
        past_key_value: (k, v) each [B, H_kv, T_past, D_head] or None
        returns:
          y: [B, T, D]
          present_key_value: (k, v) each [B, H_kv, T_total, D_head] or None
        """
        assert x.ndim == 3, f"TransformerBlock expects x [B,T,D], got {tuple(x.shape)}"
        xn = self.norm1(x)
        assert xn.ndim == 3, f"norm1 must preserve [B,T,D], got {tuple(xn.shape)}"
        # Residual branch: self-attention over normalized input.
        attention_output,present_key_value = self.attn(xn, position_ids=position_ids,
                                 attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        # Standard pre-norm decoder block: x + attn, then + mlp(norm(.)).
        h = attention_output + x
        y = h + self.mlp(self.norm2(h))
        return y, present_key_value

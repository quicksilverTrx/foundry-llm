# llm_lab/core/model/attention.py

from __future__ import annotations

from dataclasses import dataclass 
from typing import Literal,Optional,List,Tuple
import torch
from torch import nn
from llm_lab.core.model.pos_encodings import apply_rope
PastKeyValue = Tuple[torch.Tensor, torch.Tensor]
PastKeyValues = List[PastKeyValue] # len == n_layers (model-level)


@dataclass
class SingleHeadAttentionConfig:
    d_model : int
    head_dim : int
    dropout : float = 0.0
    use_rope: bool = False 


class SingleHeadAttention(nn.Module):
    """
    Single attention head operating over the time dimension.

    Input:  x [B, T, d_model]
    Output: y [B, T, head_dim]  
    """

    def __init__(self,config : SingleHeadAttentionConfig):
        super().__init__()
        assert config.head_dim <= config.d_model

        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout
        self.config = config

        # Linear projections: d_model -> head_dim
        self.q_proj = nn.Linear(self.d_model,self.head_dim)
        self.k_proj = nn.Linear(self.d_model,self.head_dim)
        self.v_proj = nn.Linear(self.d_model,self.head_dim)

        # Dropout on attention probabilities
        self.dropout_p_layer = nn.Dropout(self.dropout_p)

    def _causal_mask (self, T:int, device : torch.device, dtype = torch.float32) -> torch.Tensor :
        """
        Returns a [T, T] mask with -inf on positions j > i (future),
        0 on allowed positions.
        Causal mask [T, T]:
          0      for j <= i (allowed: past + self)
          -inf   for j > i  (future positions)
        """
        mask = torch.full((T,T),float(-1e9),device=device,dtype=dtype)
        mask = torch.triu(mask,diagonal = 1)  # upper triangle (j > i) stays -inf; rest becomes 0
        return mask
    
    def forward(self, x:torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, past_key_value : Optional[PastKeyValue] = None, use_cache = False) -> Tuple[torch.Tensor,Optional[PastKeyValue]]:  
        """
        x: [B, T, d_model]
        returns: [B, T, head_dim]
        """
        B, T, C = x.shape
        assert C == self.d_model

        # Project to q, k, v in head space
        q_val = self.q_proj(x) # [B, T, head_dim]
        assert q_val.shape == (B, T, self.head_dim)
        k_val = self.k_proj(x) # [B, T, head_dim]
        v_val = self.v_proj(x) # [B, T, head_dim]

        if self.config.use_rope:
            if position_ids is None:
                raise ValueError("use_rope=True but position_ids=None")

            q_val, k_val = apply_rope(q_val, k_val, position_ids)  # theta default is fine for base RoPE

        k_t = k_val.transpose(-2,-1) # [B,  head_dim, T]


        attention_weights = q_val @ k_t  # [B, T, T]
        attention_weights = attention_weights/(self.head_dim ** 0.5) # [B, T, T]
        assert attention_weights.shape == (B, T, T)

         # Apply causal mask so each position sees only <= its index
        mask = self._causal_mask(T,x.device,dtype=attention_weights.dtype) # [T, T]
        attention_weights_masked = attention_weights + mask # [B, T, T] + [T, T] (broadcast) -> [B, T, T]
        
        # Turn scores into probabilities over positions j
        attention_weights_probability = nn.Softmax(dim=-1)(attention_weights_masked) # dim=-1 to normalize over the “which position to attend to” dimension.
        attention_weights_probability = self.dropout_p_layer(attention_weights_probability) 
        
        # Weighted sum of value vectors → new representation
        y_t = attention_weights_probability@ v_val # [B, T, T] * [B, T, head_dim] =  [B, T, head_dim]
        return y_t,None

@dataclass
class MultiHeadAttentionConfig:
    d_model: int
    n_heads: int
    dropout: float = 0.0
    use_rope: bool = False 

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention composed from multiple SingleHeadAttention heads.

    Input:  x [B, T, d_model]
    Output: y [B, T, d_model]
    """
    def __init__(self,config : MultiHeadAttentionConfig):
        super().__init__()
        assert config.d_model %config.n_heads ==0
        self.d_model = config.d_model
        self.n_head = config.n_heads
        self.head_dim = self.d_model//self.n_head

        # Each head has its own q/k/v projections
        single_head_config = SingleHeadAttentionConfig(self.d_model,self.head_dim,config.dropout,use_rope=config.use_rope)

        self.heads = nn.ModuleList([SingleHeadAttention(single_head_config) for _ in range(self.n_head)])
        
        # Final projection back to d_model
        self.out_proj = nn.Linear(self.d_model,self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x:torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                attention_mask : Optional[torch.Tensor]= None, past_key_value : Optional[PastKeyValue] = None, use_cache = False) -> Tuple[torch.Tensor,Optional[PastKeyValue]]:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        B, T, C = x.shape
        assert C == self.d_model

        # Run each head independently, then concat along feature dim
        head_outputs = []
        for head in self.heads:
            attention_outputs,_ = head(x, position_ids=position_ids,attention_mask = attention_mask ,past_key_value = past_key_value ,use_cache = use_cache)  # each [B, T, head_dim]
            head_outputs.append(attention_outputs) 
        head_outputs = torch.cat(head_outputs, dim=-1)  # # [B, T, n_heads * head_dim] = [B, T, d_model]

        projected_head = self.out_proj(head_outputs) # [B, T, d_model]
        y = self.dropout(projected_head)
        return y,None

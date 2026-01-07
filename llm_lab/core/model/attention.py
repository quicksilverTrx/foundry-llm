# llm_lab/core/model/attention.py

from __future__ import annotations

from dataclasses import dataclass 
from typing import Literal,Optional,List,Tuple
import torch
from torch import nn
from llm_lab.core.model.pos_encodings import apply_rope
PastKeyValue = Tuple[torch.Tensor, torch.Tensor]
PastKeyValues = List[PastKeyValue] # len == n_layers (model-level)

_CAUSAL_MASK_CACHE = {} # # key: (T, str(device), dtype) -> Tensor[T,T]
USE_CACHE = True

def _causal_mask ( T:int, device : torch.device, dtype = torch.float32) -> torch.Tensor :
    """
    Returns a [T, T] mask with -inf on positions j > i (future),
    0 on allowed positions.
    Causal mask [T, T]:
        0      for j <= i (allowed: past + self)
        -inf   for j > i  (future positions)
    """
    if USE_CACHE:
        key = (T, str(device), dtype)

        m = _CAUSAL_MASK_CACHE.get(key)
        if m is None:
            neg = torch.finfo(dtype).min
            mask = torch.full((T, T), float(neg), device=device, dtype=dtype)
            mask = torch.triu(mask, diagonal=1)
            _CAUSAL_MASK_CACHE[key] = mask
            m = mask
        return m
    else:
        mask = torch.full((T, T), float(-1e9), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1) 
        return mask

def _apply_masks(scores: torch.Tensor, *, attention_mask: Optional[torch.Tensor], T: int, device, dtype):
    # scores: [B, ..., T, T]
    scores = scores + _causal_mask(T, device, dtype=dtype)

    if attention_mask is None:
        return scores

    # Accept [B, T] with 1=keep, 0=mask
    if attention_mask.ndim != 2 or attention_mask.shape[-1] != T:
        raise ValueError(f"attention_mask must be [B,T], got {tuple(attention_mask.shape)}")

    # Convert to boolean keep mask, then mask out keys (last dim)
    keep = attention_mask.to(torch.bool)  # [B,T]
    neg = torch.finfo(dtype).min
    # Broadcast: [B,1,1,T] over heads/query positions
    scores = scores.masked_fill(~keep[:, None, None, :], neg)
    return scores

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
        self.scale = self.head_dim ** -0.5


    
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
        attention_weights = attention_weights*(self.scale) # [B, T, T]
        assert attention_weights.shape == (B, T, T)

         # Apply causal mask so each position sees only <= its index
        # mask = _causal_mask(T,x.device,dtype=attention_weights.dtype) # [T, T]
        #attention_weights_masked = attention_weights + mask # [B, T, T] + [T, T] (broadcast) -> [B, T, T]
        attention_weights_masked = _apply_masks(
                attention_weights, attention_mask=attention_mask, T=T, device=x.device, dtype=attention_weights.dtype
            )
        # Turn scores into probabilities over positions j
        #attention_weights_probability = nn.Softmax(dim=-1)(attention_weights_masked) # dim=-1 to normalize over the “which position to attend to” dimension.
        attention_weights_probability = torch.softmax(attention_weights_masked,dim=-1)
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
    attention_type: Literal["mha", "gqa"] = "mha"
    num_kv_heads: Optional[int] = None

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
        self.attention_type = config.attention_type
        self.config = config
        # Each head has its own q/k/v projections
        if self.attention_type == "mha":
            single_head_config = SingleHeadAttentionConfig(self.d_model,self.head_dim,config.dropout,use_rope=config.use_rope)

            self.heads = nn.ModuleList([SingleHeadAttention(single_head_config) for _ in range(self.n_head)])
            
            # Final projection back to d_model
            self.out_proj = nn.Linear(self.d_model,self.d_model)
            self.dropout = nn.Dropout(config.dropout)

        elif self.attention_type == "gqa":
            self.H_kv = self.config.num_kv_heads
            if self.config.num_kv_heads is  None :
                raise ValueError("num kv head is None")
            assert 1<= self.H_kv <= self.n_head
            assert self.n_head % self.config.num_kv_heads == 0
            # Linear projections: d_model -> head_dim * n_heads
            self.q_proj = nn.Linear(self.d_model,self.head_dim * self.n_head)
            # Linear projections: d_model -> head_dim * n_heads_kv
            self.k_proj = nn.Linear(self.d_model,self.head_dim * self.H_kv)
            self.v_proj = nn.Linear(self.d_model,self.head_dim * self.H_kv)

            
            self.attention_dropout_layer = nn.Dropout(config.dropout)
            self.scale = self.head_dim ** -0.5
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


        if self.config.attention_type == "mha":
            # Run each head independently, then concat along feature dim
            head_outputs = []
            for head in self.heads:
                attention_outputs,_ = head(x, position_ids=position_ids,attention_mask = attention_mask ,past_key_value = past_key_value ,use_cache = use_cache)  # each [B, T, head_dim]
                head_outputs.append(attention_outputs) 
            head_outputs = torch.cat(head_outputs, dim=-1)  # # [B, T, n_heads * head_dim] = [B, T, d_model]

            projected_head = self.out_proj(head_outputs) # [B, T, d_model]
            y = self.dropout(projected_head)



            return y,None
        elif self.config.attention_type == "gqa":
            H = self.n_head
            H_kv = self.H_kv
            D = self.head_dim
            repeat = H//H_kv
            # [B, T, d_model] -> [B, T, H * D] -> project to -> [B,H,T,D]
            q = self.q_proj(x).view(B,T,H,D).transpose(1,2)
            # [B, T, d_model] -> [B, T, H_kv * D] -> project to -> [B,H_kv,T,D]
            k = self.k_proj(x).view(B,T,H_kv,D).transpose(1,2)
            v = self.v_proj(x).view(B,T,H_kv,D).transpose(1,2)

            #Expand KV along the head axis to get [B,H,T,D]
            k = k.repeat_interleave(repeat,dim=1) 
            v = v.repeat_interleave(repeat,dim=1)
            
            if self.config.use_rope:
                if position_ids is None:
                    raise ValueError("use_rope=True but position_ids=None")
                if position_ids is not None:
                   assert position_ids.ndim == 1 and position_ids.shape[0] == T

                q_batched = q.reshape(B*H,T,D)
                k_batched = k.reshape(B*H,T,D)
                q, k = apply_rope(q_batched, k_batched, position_ids)  # theta default is fine for base RoPE
                q = q.reshape(B,H,T,D)
                k = k.reshape(B,H,T,D)

            k_t = k.transpose(-2,-1) # [B,  n_head,D, T]

            
            attention_weights = q @ k_t  # [B, n_head,T, T]
            attention_weights = attention_weights*(self.scale) # [B,n_head,  T, T]
            assert attention_weights.shape == (B, H,T, T)

            # # Apply causal mask so each position sees only <= its index
            # mask = _causal_mask(T,x.device,dtype=attention_weights.dtype) # [T, T]
            # attention_weights_masked = attention_weights + mask # [B,n_head, T, T] + [T, T] (broadcast) -> [B, n_head, T, T]

            attention_weights_masked = _apply_masks(
                attention_weights, attention_mask=attention_mask, T=T, device=x.device, dtype=attention_weights.dtype
            ) # [B, n_head, T, T]

            attention_weights_probability = torch.softmax(attention_weights_masked,dim=-1)
            attention_weights_probability = self.attention_dropout_layer(attention_weights_probability) 
            
            # Weighted sum of value vectors → new representation
            s = attention_weights_probability@ v # [B, n_head, T, T] * [B,n_head,T,D] = [B,n_head,T,D] 
            output = s.transpose(1,2).contiguous().view(B, T, H * D) # n_head * D = D_model
            projection = self.out_proj(output)
            y_t = self.dropout(projection)
            
            return y_t,None 




# llm_lab/core/model/attention.py
from __future__ import annotations

from dataclasses import dataclass 
from typing import Literal,Optional,List,Tuple
import torch
import torch.nn.functional as F
from torch import nn
from llm_lab.core.model.pos_encodings import apply_rope
PastKeyValue = Tuple[torch.Tensor, torch.Tensor]
PastKeyValues = List[PastKeyValue] # len == n_layers (model-level)

_CAUSAL_MASK_CACHE = {} # key: (T, str(device), dtype) -> Tensor[T, T]
USE_CACHE = True

def _validate_kv_cache(k: torch.Tensor, v: torch.Tensor) -> None:
    """Helper validator for canonical KV cache layout."""
    if k.shape != v.shape:
        raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
    if k.ndim != 4:
        raise ValueError(f"expected KV as 4D [B,n_kv_heads,T,head_dim], got {tuple(k.shape)}")


def _append_past(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past: Optional[PastKeyValue],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Concatenates cache along canonical time axis=2."""
    _validate_kv_cache(k_new, v_new)
    if past is None:
        return k_new,v_new,0
    k_past,v_past = past
    _validate_kv_cache(k_past, v_past)
    time_past = k_past.shape[2]
    k = torch.cat([k_past,k_new],dim=2)
    v = torch.cat([v_past,v_new],dim=2)
    _validate_kv_cache(k, v)
    return k,v,time_past


def _causal_mask(
    query_len: int,
    key_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Returns a [T_q, T_k] mask with -inf on future positions.
    """
    # Backward-compatible path: _causal_mask(T, device, dtype)
    if isinstance(key_len, torch.device):
        if isinstance(device, torch.dtype):
            dtype = device
        device = key_len
        key_len = None
    # Backward-compatible path: _causal_mask(T, device=<...>, dtype=<...>)
    if key_len is None:
        key_len = query_len
    if device is None:
        raise ValueError("device is required")

    if USE_CACHE and query_len == key_len:
        key = (query_len, str(device), dtype)

        m = _CAUSAL_MASK_CACHE.get(key)
        if m is None:
            neg = torch.finfo(dtype).min
            mask = torch.full((query_len, key_len), (neg), device=device, dtype=dtype)
            mask = torch.triu(mask, diagonal=1)
            _CAUSAL_MASK_CACHE[key] = mask
            m = mask
        return m

    neg = torch.finfo(dtype).min
    query_idx = torch.arange(query_len, device=device)[:, None]
    key_idx = torch.arange(key_len, device=device)[None, :]
    max_allowed = (key_len - query_len) + query_idx
    mask = torch.zeros((query_len, key_len), device=device, dtype=dtype)
    mask = mask.masked_fill(key_idx > max_allowed, neg)
    return mask

def _apply_masks(scores: torch.Tensor, *, attention_mask: Optional[torch.Tensor], device, dtype):
    # scores: [B, ..., T_q, T_k]
    query_len = scores.shape[-2]
    key_len = scores.shape[-1]
    scores = scores + _causal_mask(query_len, key_len, device, dtype=dtype)

    if attention_mask is None:
        return scores

    # Accept [B, L] with 1=keep, 0=mask, where L is either T_q or T_k.
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B,T_k], got {tuple(attention_mask.shape)}")
    if attention_mask.shape[-1] == query_len and key_len > query_len:
        # Decode path commonly passes mask for new tokens only; treat cached keys as valid.
        pad = torch.ones((attention_mask.shape[0], key_len - query_len), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([pad, attention_mask], dim=-1)
    elif attention_mask.shape[-1] != key_len:
        raise ValueError(f"attention_mask must have key length {key_len}, got {tuple(attention_mask.shape)}")

    # Convert to boolean keep mask, then mask out keys (last dim)
    keep = attention_mask.to(torch.bool)  # [B, T_k]
    neg = torch.finfo(dtype).min
    if scores.ndim == 3:
        # scores is [B,T_q,T_k], so broadcast over query axis only.
        key_mask = ~keep[:, None, :]  # [B,1,T_k]
    elif scores.ndim == 4:
        # scores is [B,H,T_q,T_k], so broadcast over heads and query axes.
        key_mask = ~keep[:, None, None, :]  # [B,1,1,T_k]
    else:
        raise ValueError(f"scores must be [B,T_q,T_k] or [B,H,T_q,T_k], got {tuple(scores.shape)}")
    scores = scores.masked_fill(key_mask, neg)
    return scores

@dataclass
class SingleHeadAttentionConfig:
    d_model : int
    head_dim : int
    dropout : float = 0.0
    use_rope: bool = False
    rope_scaling_type: Literal["none", "linear"] = "none"
    rope_scaling_factor: float = 1.0
    use_sdpa: bool = False   # if True: F.scaled_dot_product_attention; if False: explicit softmax


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
        self.use_sdpa = config.use_sdpa


    
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

        if past_key_value is not None:
            past_len = past_key_value[0].shape[2]
        else :
            past_len = 0
        if self.config.use_rope:
            if position_ids is None:
                raise ValueError("use_rope=True but position_ids=None")

            q_val, k_val = apply_rope(q_val, k_val, position_ids,
                                      rope_scaling_type=self.config.rope_scaling_type,
                                      rope_scaling_factor=self.config.rope_scaling_factor,
                                      position_offset=past_len,)  # theta default is fine for base RoPE
        
        k_present, v_present, _ = _append_past(
            k_val.unsqueeze(1),
            v_val.unsqueeze(1),
            past_key_value,
        )
        k_val = k_present.squeeze(1)
        v_val = v_present.squeeze(1)
        if not self.use_sdpa:                                                   # ── raw attention path ──
            k_t = k_val.transpose(-2,-1) # [B, D_head, T_k]
            Tq = q_val.shape[1]
            Tk = k_val.shape[1]
            attention_weights = q_val @ k_t  # [B, T_q, T_k]
            attention_weights = attention_weights*(self.scale) # [B, T_q, T_k]
            assert attention_weights.shape == (B, Tq, Tk)
            attention_weights_masked = _apply_masks(
                    attention_weights, attention_mask=attention_mask, device=x.device, dtype=attention_weights.dtype
                )
            # Turn scores into probabilities over positions j
            #attention_weights_probability = nn.Softmax(dim=-1)(attention_weights_masked) # dim=-1 to normalize over the "which position to attend to" dimension.
            attention_weights_probability = torch.softmax(attention_weights_masked,dim=-1)
            attention_weights_probability = self.dropout_p_layer(attention_weights_probability)
            # Weighted sum of value vectors → new representation
            y_t = attention_weights_probability @ v_val  # [B,T_q,T_k] x [B,T_k,D] -> [B,T_q,D]
        else:                                                                   # ── SDPA path ──
            # k_t: SDPA takes k_val as [B,1,T_k,head_dim] via unsqueeze — no explicit transpose needed
            Tq = q_val.shape[1]
            Tk = k_val.shape[1]
            # attention_weights_masked: build attn_mask_sdpa that mirrors _apply_masks(causal + padding).
            # None  → is_causal flag encodes the causal mask internally (fused flash-attention kernel).
            # else  → additive float bias: causal_bias [1,1,Tq,Tk] + padding_bias [B,1,1,Tk].
            if attention_mask is None:
                attn_mask_sdpa = None
                is_causal      = (Tq == Tk)  # full-sequence: causal; decode step (Tq<Tk): attend all cached keys
            else:
                if attention_mask.ndim != 2:
                    raise ValueError(f"attention_mask must be [B,T_k], got {tuple(attention_mask.shape)}")
                neg            = torch.finfo(q_val.dtype).min
                causal_bias    = _causal_mask(Tq, Tk, x.device, dtype=q_val.dtype)[None, None]  # [1,1,Tq,Tk]
                keep           = attention_mask.to(torch.bool)
                if keep.shape[-1] == Tq and Tk > Tq:
                    pad  = torch.ones(keep.shape[0], Tk - Tq, device=keep.device, dtype=keep.dtype)
                    keep = torch.cat([pad, keep], dim=-1)
                padding_bias   = torch.zeros(B, 1, 1, Tk, device=x.device, dtype=q_val.dtype).masked_fill(
                    ~keep[:, None, None, :], neg
                )
                attn_mask_sdpa = causal_bias + padding_bias                          # [B,1,Tq,Tk] additive bias
                is_causal      = False  # causality already encoded in causal_bias
            # attention_weights = q_val @ k_t * scale               → fused inside SDPA
            # Turn scores into probabilities over positions j        → fused (attn_mask_sdpa + softmax)
            # attention_weights_probability = dropout_p_layer(...)   → dropout_p passed directly
            dropout_p = self.dropout_p if self.training else 0.0
            # Weighted sum of value vectors → new representation
            y_4d = F.scaled_dot_product_attention(
                q_val.unsqueeze(1), k_val.unsqueeze(1), v_val.unsqueeze(1),
                attn_mask=attn_mask_sdpa, dropout_p=dropout_p, is_causal=is_causal,
            )                                                                        # [B,1,T_q,head_dim]
            y_t = y_4d.squeeze(1)  # [B, T_q, head_dim]
        assert y_t.shape == (B, Tq, self.head_dim)
        if use_cache:
            return y_t,(k_present,v_present)
        
        return y_t,None

@dataclass
class MultiHeadAttentionConfig:
    d_model: int
    n_heads: int
    dropout: float = 0.0
    use_rope: bool = False
    attention_type: Literal["mha", "gqa"] = "mha"
    num_kv_heads: Optional[int] = None
    rope_scaling_type: Literal["none", "linear"] = "none"
    rope_scaling_factor: float = 1.0
    use_sdpa: bool = False   # propagated to each head (MHA) or used directly (GQA)

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
            single_head_config = SingleHeadAttentionConfig(self.d_model,self.head_dim,config.dropout,use_rope=config.use_rope,
                                                           rope_scaling_type=config.rope_scaling_type,
                                                        rope_scaling_factor=config.rope_scaling_factor,
                                                        use_sdpa=config.use_sdpa)

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
            self.use_sdpa = config.use_sdpa



    def forward(self,x:torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                attention_mask : Optional[torch.Tensor]= None, past_key_value : Optional[PastKeyValue] = None, use_cache = False) -> Tuple[torch.Tensor,Optional[PastKeyValue]]:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        present_k_list = []
        present_v_list = []
        if x.ndim == 4 and x.shape[1] == 1:
            # Defensive: some upstream path accidentally introduces a singleton dim.
            # Squeeze to recover [B,T,D] expected by attention.
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"MultiHeadAttention expected x [B,T,D] (or [B,1,T,D]), got {tuple(x.shape)}")

        B, T, C = x.shape
        assert C == self.d_model


        if self.config.attention_type == "mha":
            # Run each head independently, then concat along feature dim
            head_outputs = []
            for head_index, head in enumerate(self.heads):
                head_past = None
                if past_key_value is not None:
                    k_past, v_past = past_key_value
                    head_past = (k_past[:, head_index : head_index + 1, :, :], v_past[:, head_index : head_index + 1, :, :])

                attention_outputs,present_kv = head(
                    x,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_value=head_past,
                    use_cache=use_cache,
                )  # each [B, T, head_dim]
                head_outputs.append(attention_outputs) 
                if use_cache:
                    if present_kv is None:
                        raise RuntimeError(f"Head {head_index} returned no present KV while use_cache=True")
                    k_head, v_head = present_kv
                    present_k_list.append(k_head)
                    present_v_list.append(v_head)
            head_outputs = torch.cat(head_outputs, dim=-1)  # # [B, T, n_heads * head_dim] = [B, T, d_model]

            projected_head = self.out_proj(head_outputs) # [B, T, d_model]
            y = self.dropout(projected_head)

            if use_cache:
                k_present = torch.cat(present_k_list, dim=1)  # [B, n_heads, T_total, head_dim]
                v_present = torch.cat(present_v_list, dim=1)
                return y,(k_present,v_present)
            return y,None
        elif self.config.attention_type == "gqa":
            H = self.n_head
            H_kv = self.H_kv
            D = self.head_dim
            repeat = H//H_kv
            # [B, T, d_model] -> [B, T, H * D] -> project to -> [B,H,T,D]
            q = self.q_proj(x).view(B,T,H,D).transpose(1,2)  # [B,H,T_q,D]
            # [B, T, d_model] -> [B, T, H_kv * D] -> project to -> [B,H_kv,T,D]
            k = self.k_proj(x).view(B,T,H_kv,D).transpose(1,2)  # [B,H_kv,T_q,D]
            v = self.v_proj(x).view(B,T,H_kv,D).transpose(1,2)  # [B,H_kv,T_q,D]
            # Expand KV to H heads for compute. Cache stays in base H_kv space.
            k = k.repeat_interleave(repeat,dim=1)
            v = v.repeat_interleave(repeat,dim=1)
            
            if self.config.use_rope:
                if position_ids is None:
                    raise ValueError("use_rope=True but position_ids=None")
                if position_ids is not None:
                   assert position_ids.ndim == 1 and position_ids.shape[0] == T

                q_batched = q.reshape(B*H,T,D)
                k_batched = k.reshape(B*H,T,D)
                if past_key_value is not None:
                    past_len = past_key_value[0].shape[2]
                else :
                    past_len=0
                q, k = apply_rope(q_batched, k_batched, position_ids,
                                  rope_scaling_type=self.config.rope_scaling_type,
                                    rope_scaling_factor=self.config.rope_scaling_factor,
                                    position_offset=past_len)  # theta default is fine for base RoPE
                q = q.reshape(B,H,T,D)
                k = k.reshape(B,H,T,D)

            # Collapse to base heads so cache ABI remains [B,H_kv,T,D].
            k_base = k[:, ::repeat, :, :]
            v_base = v[:, ::repeat, :, :]
            k_present, v_present, _ = _append_past(k_base, v_base, past_key_value)
            # Expand cached totals back to H heads for attention math.
            k_total = k_present.repeat_interleave(repeat, dim=1)
            v_total = v_present.repeat_interleave(repeat, dim=1)
            if not self.use_sdpa:                                               # ── raw attention path ──
                k_t = k_total.transpose(-2,-1) # [B, n_head, D, T_k]
                Tq = q.shape[2]
                Tk = k_total.shape[2]
                attention_weights = q @ k_t  # [B, n_head, T_q, T_k]
                attention_weights = attention_weights*(self.scale) # [B, n_head, T_q, T_k]
                assert attention_weights.shape == (B, H, Tq, Tk)
                attention_weights_masked = _apply_masks(
                    attention_weights, attention_mask=attention_mask, device=x.device, dtype=attention_weights.dtype
                ) # [B, n_head, T_q, T_k]
                attention_weights_probability = torch.softmax(attention_weights_masked,dim=-1)
                attention_weights_probability = self.attention_dropout_layer(attention_weights_probability)
                # Weighted sum of value vectors → new representation
                s = attention_weights_probability @ v_total # [B,n_head,T_q,T_k] * [B,n_head,T_k,D] = [B,n_head,T_q,D]
            else:                                                               # ── SDPA path ──
                # k_t: SDPA takes k_total as [B,n_head,T_k,D] directly — no explicit transpose needed
                Tq = q.shape[2]
                Tk = k_total.shape[2]
                # attention_weights_masked: build attn_mask_sdpa that mirrors _apply_masks(causal + padding).
                # None  → is_causal flag encodes the causal mask internally (fused flash-attention kernel).
                # else  → additive float bias: causal_bias [1,1,Tq,Tk] + padding_bias [B,1,1,Tk].
                if attention_mask is None:
                    attn_mask_sdpa = None
                    is_causal      = (Tq == Tk)  # full-sequence: causal; decode step (Tq<Tk): attend all cached keys
                else:
                    if attention_mask.ndim != 2:
                        raise ValueError(f"attention_mask must be [B,T_k], got {tuple(attention_mask.shape)}")
                    neg            = torch.finfo(q.dtype).min
                    causal_bias    = _causal_mask(Tq, Tk, q.device, dtype=q.dtype)[None, None]  # [1,1,Tq,Tk]
                    keep           = attention_mask.to(torch.bool)
                    if keep.shape[-1] == Tq and Tk > Tq:
                        pad  = torch.ones(keep.shape[0], Tk - Tq, device=keep.device, dtype=keep.dtype)
                        keep = torch.cat([pad, keep], dim=-1)
                    padding_bias   = torch.zeros(B, 1, 1, Tk, device=q.device, dtype=q.dtype).masked_fill(
                        ~keep[:, None, None, :], neg
                    )
                    attn_mask_sdpa = causal_bias + padding_bias                      # [B,n_head,Tq,Tk] via broadcast
                    is_causal      = False  # causality already encoded in causal_bias
                # attention_weights = q @ k_t * scale               → fused inside SDPA
                # attention_weights_masked                           → driven by attn_mask_sdpa / is_causal
                # attention_weights_probability = dropout(softmax()) → dropout_p passed directly
                dropout_p = self.config.dropout if self.training else 0.0
                # Weighted sum of value vectors → new representation
                s = F.scaled_dot_product_attention(
                    q, k_total, v_total,
                    attn_mask=attn_mask_sdpa, dropout_p=dropout_p, is_causal=is_causal,
                )                                                                    # [B,n_head,T_q,D]
            # s: [B, n_head, T_q, D]
            output = s.transpose(1,2).contiguous().view(B, Tq, H * D) # n_head * D = D_model
            projection = self.out_proj(output)
            y_t = self.dropout(projection)
            
            if use_cache:
                return y_t,(k_present,v_present)
            return y_t,None 

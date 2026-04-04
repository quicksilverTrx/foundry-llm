# llm_lab/core/model/gpt.py
from __future__ import annotations
from dataclasses import dataclass
import math

from torch import nn
import torch
from typing import Literal,Optional,List,Tuple
from llm_lab.core.model.attention import PastKeyValues
from llm_lab.core.model.blocks import TransformerBlock, TransformerBlockConfig
from llm_lab.core.model.pos_encodings import SinusoidalPositionalEncoding
from llm_lab.core.model.norms import make_norm
from llm_lab.core.model.masks import _normalize_attention_mask

@dataclass
class MiniGPTConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    block_size: int
    dropout: float = 0.0

    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    mlp_type: Literal["gelu", "swiglu"] = "gelu"
    attention_type: Literal["mha", "gqa"] = "mha"
    num_kv_heads: Optional[int] = None

    pos_encoding_type: Literal["learned", "sinusoidal", "rope"] = "learned"
    rope_scaling_type: Literal["none", "linear"] = "none"
    rope_scaling_factor: float = 1.0
    arch_family :Literal["miniGPT","nanollama"] = "miniGPT"
    tie_weights: bool = False   # share token_embed ↔ lm_head (saves params; correct for small LLaMA-style models)
    use_sdpa: bool = False      # enable F.scaled_dot_product_attention in every attention block

    logit_softcap: Optional[float] = None
    qk_norm: bool = False

    def __post_init__(self):
        if self.attention_type == "gqa":
            if self.num_kv_heads is None:
                raise ValueError("GQA requires num_kv_heads")
            if not (1 <= self.num_kv_heads <= self.n_heads):
                raise ValueError("num_kv_heads out of range")
            if self.n_heads % self.num_kv_heads != 0:
                raise ValueError("n_heads must be divisible by num_kv_heads")

        if self.arch_family == "nanollama":
            if self.attention_type != "gqa":
                raise ValueError("nanollama requires attention_type='gqa'")
            if self.pos_encoding_type != "rope":
                raise ValueError("nanollama requires pos_encoding_type='rope'")
            if self.norm_type != "rmsnorm":
                raise ValueError("nanollama requires norm_type='rmsnorm'")


        if self.pos_encoding_type == "rope":
            if self.rope_scaling_factor <= 0:
                raise ValueError("rope_scaling_factor must be > 0")
            if self.rope_scaling_type == "linear" and self.rope_scaling_factor < 1.0:
                raise ValueError("rope_scaling_factor must be >= 1 for linear scaling")


def _validate_past_key_values(
    past_key_values: Optional[PastKeyValues],
    *,
    n_layers: int,
    use_cache: bool,
) -> None:
    """Validates model-level KV cache ABI: list[(k,v)] with k/v [B,H_kv,T,D]."""
    if past_key_values is None:
        return
    if not use_cache:
        raise ValueError("past_key_values requires use_cache=True")
    if not isinstance(past_key_values, list):
        raise TypeError("past_key_values must be a list")
    if len(past_key_values) != n_layers:
        raise ValueError(f"past_key_values length mismatch: got {len(past_key_values)} expected {n_layers}")
    for layer_index, layer_kv in enumerate(past_key_values):
        if not isinstance(layer_kv, tuple) or len(layer_kv) != 2:
            raise TypeError(f"past_key_values[{layer_index}] must be a (k, v) tuple")
        k, v = layer_kv
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError(f"past_key_values[{layer_index}] must contain tensors")
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"past_key_values[{layer_index}] must be 4D tensors")


def _infer_past_len(past_key_values: Optional[PastKeyValues]) -> int:
    """Returns cached time length from canonical cache layout."""
    if past_key_values is None:
        return 0
    first_k, _ = past_key_values[0]
    return int(first_k.shape[2])




class MiniGPT(nn.Module):
    """
    subword tokenizer, reserved tokens, RoPE, GQA powered GPT-style decoder.
    """
    def __init__(self,config: MiniGPTConfig):
        super().__init__()
        self.config = config

        self.sin_pos = None
        if config.pos_encoding_type == "sinusoidal":
            self.sin_pos = SinusoidalPositionalEncoding(config.block_size, config.d_model)
        #embedding layer
        self.token_embed = nn.Embedding(config.vocab_size,config.d_model)
        # pos_embed is only used for learned/sinusoidal — RoPE injects position via attention
        self.pos_embed = (
            nn.Embedding(config.block_size, config.d_model)
            if config.pos_encoding_type != "rope"
            else None
        )

        # Blocks
        block_cfg = TransformerBlockConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            norm_type=config.norm_type,
            mlp_type=config.mlp_type,
            use_rope= (config.pos_encoding_type=="rope"),
            attention_type = config.attention_type,
            num_kv_heads=config.num_kv_heads,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_scaling_type = config.rope_scaling_type,
            use_sdpa=config.use_sdpa,
            qk_norm=config.qk_norm,
        )

        self.blocks = nn.ModuleList([TransformerBlock(block_cfg) for _ in range(config.n_layers)])

        # Final norm + LM head
        self.ln_f = make_norm(config.norm_type,config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()
        if config.tie_weights:
            self.lm_head.weight = self.token_embed.weight

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.config.d_model)
        residual_std = std / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if p.ndim < 2:
                continue  # biases: leave as zero
            if any(x in name for x in ('out_proj', 'w_down', 'fc2')):
                nn.init.normal_(p, mean=0.0, std=residual_std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)
        # std=0.02 for embedding and lm_head weight (overrides general init)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask : Optional[torch.Tensor] = None,
                past_key_values : Optional[PastKeyValues] = None,
                use_cache : bool = False) -> Tuple[torch.Tensor,Optional[PastKeyValues]]:
        """
        input_ids: [B, T]
        return: logits [B, T, vocab_size]
        """
        B,T = input_ids.shape
        attention_mask_bool = None
        new_past: Optional[PastKeyValues] = [] if use_cache else None
        if attention_mask is not None:
            # Model mask always tracks current chunk length T; attention handles cache expansion.
            attention_mask_bool = _normalize_attention_mask(attention_mask,B=B,T=T,device=input_ids.device)
        _validate_past_key_values(
            past_key_values,
            n_layers=self.config.n_layers,
            use_cache=use_cache,
        )

        if T > self.config.block_size:
            raise ValueError("sequence length exceeds block size")
        token_embeddings = self.token_embed(input_ids)  # [B, T, D]
        assert token_embeddings.ndim == 3, f"token_embed must return [B,T,D], got {tuple(token_embeddings.shape)}"
        x = token_embeddings  # always defined
        assert x.ndim == 3, f"MiniGPT expects x [B,T,D] before blocks, got {tuple(x.shape)}"
        past_len = _infer_past_len(past_key_values)
        if self.config.pos_encoding_type =="rope":
            # RoPE uses local [0..T-1] positions plus per-layer position_offset=past_len.
            position_indices = torch.arange(0, T,device=input_ids.device)
        else:
            # Learned/sinusoidal embeddings use absolute indices in the sequence.
            position_indices = torch.arange(past_len, past_len + T,device=input_ids.device) # [T]
        if self.config.pos_encoding_type == "learned":
            assert self.pos_embed is not None
            position_embeddings = self.pos_embed(position_indices)
            x = x + position_embeddings
            assert x.ndim == 3, f"pos-embed add must keep x [B,T,D]; got {tuple(x.shape)}"
        elif self.config.pos_encoding_type == "sinusoidal":
            position_embeddings = self.sin_pos(position_indices)
            x = x + position_embeddings
            assert x.ndim == 3, f"pos-embed add must keep x [B,T,D]; got {tuple(x.shape)}"
        # rope: no additive position encoding — handled per-layer in attention

        # Guard against accidental broadcast bugs that introduce an extra batch axis.
        # Expected: x is [B, T, D].
        if x.ndim == 4 and x.shape[0] == B and x.shape[1] == B and x.shape[2] == T:
            idx = torch.arange(B, device=x.device)
            x = x[idx, idx]  # take the diagonal -> [B, T, D]
        assert x.ndim == 3 and x.shape[0] == B and x.shape[1] == T, \
            f"x must be [B,T,D] before blocks; got {tuple(x.shape)}"

        for block_index,block in enumerate(self.blocks):
            layer_past = None
            if past_key_values is not None : 
                 assert (len(past_key_values)) == self.config.n_layers
                 # Per-layer cache shape is [B, H_kv, T_past, D].
                 layer_past = past_key_values[block_index]

            x,present_key_value = block(x, position_ids=position_indices if self.config.pos_encoding_type == "rope" else None, 
                      attention_mask = attention_mask_bool ,past_key_value = layer_past ,use_cache = use_cache)
            if use_cache:

                if present_key_value is None:
                    raise RuntimeError(f"Block {block_index} returned no present key/value while use_cache=True")
                
                new_past.append(present_key_value)
        x = self.ln_f(x)
        x = self.lm_head(x) # [B, T, vocab_size]
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            x = cap * torch.tanh(x / cap)
        return x, new_past

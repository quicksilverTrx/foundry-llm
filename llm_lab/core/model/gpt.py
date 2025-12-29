#llm_lab/core/model/gpt
from __future__ import annotations
from dataclasses import dataclass

from torch import nn
import torch
from typing import Literal,Optional,List,Tuple
from llm_lab.core.model.attention import PastKeyValues
from llm_lab.core.model.blocks import TransformerBlock, TransformerBlockConfig
from llm_lab.core.model.pos_encodings import SinusoidalPositionalEncoding
from llm_lab.core.model.norms import make_norm


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
    attention_type: Literal["mha", "mqa"] = "mha"

    pos_encoding_type: Literal["learned", "sinusoidal", "rope"] = "learned"


class MiniGPT(nn.Module):
    """
    Char-level GPT-style decoder.
    """
    def __init__(self,config: MiniGPTConfig):
        super().__init__()
        self.config = config

        self.sin_pos = None
        if config.pos_encoding_type == "sinusoidal":
            self.sin_pos = SinusoidalPositionalEncoding(config.block_size, config.d_model)
        #embedding layer
        self.token_embed = nn.Embedding(config.vocab_size,config.d_model)
        self.pos_embed = nn.Embedding(config.block_size,config.d_model)

        # Blocks
        block_cfg = TransformerBlockConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            norm_type=config.norm_type,
            mlp_type=config.mlp_type,
            use_rope= (config.pos_encoding_type=="rope")

        )

        self.blocks = nn.ModuleList([TransformerBlock(block_cfg) for _ in range(config.n_layers)])

        # Final norm + LM head
        self.ln_f = make_norm(config.norm_type,config.d_model)
        self.lm_head = nn.Linear(config.d_model,config.vocab_size)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask : Optional[torch.Tensor] = None,
                past_key_values : Optional[PastKeyValues] = None,
                use_cache : bool = False) -> Tuple[torch.Tensor,Optional[PastKeyValues]]:
        """
        input_ids: [B, T]
        return: logits [B, T, vocab_size]
        """
        B,T = input_ids.shape
        new_past: Optional[PastKeyValues] = [] if use_cache else None
        if attention_mask is not None:
            raise NotImplementedError("attention_mask support not implemented yet .")
        if past_key_values is not None :
            raise NotImplementedError("past_key_values input not supported until KV-cache is implemented.")
        if use_cache: 
            raise NotImplementedError("use_cache=True not supported yet.")
        if T > self.config.block_size:
            raise ValueError("sequence length exceeds block size")
        token_embeddings = self.token_embed(input_ids)  # [B,T,D]
        x = token_embeddings  # always defined
        position_indices = torch.arange(T,device=input_ids.device) # [T]
        if self.config.pos_encoding_type=="learned":
            position_embeddings = self.pos_embed(position_indices)
            x = x + position_embeddings
        elif self.config.pos_encoding_type =="sinusoidal":
            position_embeddings = self.sin_pos(position_indices)
            x = x + position_embeddings
        elif self.config.pos_encoding_type =="rope":
            pass
        for block_index,block in enumerate(self.blocks):
            layer_past = None
            if past_key_values is not None : 
                 assert (len(past_key_values)) == self.config.n_layers
                 layer_past = past_key_values[block_index]

            x,present_key_value = block(x, position_ids=position_indices if self.config.pos_encoding_type == "rope" else None, 
                      attention_mask = attention_mask ,past_key_value = layer_past ,use_cache = use_cache)
            if use_cache:
                if present_key_value is None:
                    raise NotImplementedError(
                    "use_cache=True but KV-cache isn't implemented yet. "
                    "Either implement PastKeyValues returns in attention, or call with use_cache=False."
                )
                new_past.append(present_key_value)
        x = self.ln_f(x)
        x = self.lm_head(x) # [B, T, vocab_size]
        return x, new_past
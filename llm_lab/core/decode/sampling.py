# llm_lab/core/decode/sampling.py
from __future__ import annotations
from typing import Optional
from torch import nn
import torch

@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int
) -> torch.Tensor:
    """
    Iteratively generate tokens by taking argmax at each step.
    input_ids: [B, T]
    returns: [B, T + max_new_tokens].
    """
    model.eval() 

    output_ids = input_ids
    for _ in range(max_new_tokens):
        B,T = output_ids.shape
        if T>block_size:
            cur_ids = output_ids[:,-block_size:] # B,block_size
        else:
            cur_ids = output_ids # [B, T_cur]
        logits, _ = model(cur_ids, attention_mask = None, past_key_values = None, use_cache = False) #[B,T_cur,V]
        logits_last = logits[:,-1,:] #[B,V]
        next_token_greedy = torch.argmax(logits_last,dim=-1,keepdim=True) # [B, 1]
        output_ids = torch.cat([output_ids,next_token_greedy],dim=1) # [B, T_cur+1]
    return output_ids


@torch.no_grad()
def sample_with_temperature(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
) -> torch.Tensor:
    """
    Sample from softmax(logits / temperature).
    input_ids: [B, T]
    returns: [B, T + max_new_tokens]
    """
    model.eval() 
    if temperature <= 0:
        raise ValueError ("temperature must be > 0")

    output_ids = input_ids
    for _ in range(max_new_tokens):
        B,T = output_ids.shape
        if T>block_size:
            cur_ids = output_ids[:,-block_size:] # B,block_size
        else:
            cur_ids = output_ids # [B, T_cur]
        logits, _ = model(cur_ids, attention_mask = None, past_key_values = None, use_cache = False) #[B,T_cur,V]
        logits_last = logits[:,-1,:] #[B,V]
        logits_scaled  = logits_last/temperature
        probs = torch.softmax(logits_scaled,dim=-1)  #[B,V]
        # Categorical per batch row → [B] → [B, 1]
        next_token = torch.distributions.Categorical(probs).sample().unsqueeze(-1)  # [B, 1]
        output_ids = torch.cat([output_ids,next_token],dim=1)  # [B, T_cur+1]
        
    return output_ids

@torch.no_grad()
def sample_top_k(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    k: int,
) -> torch.Tensor:
    """
    Top-k sampling: keep only k highest logits before sampling.
    """
    model.eval() 
    if temperature <=0:
        raise ValueError ("temperature must be > 0")

    output_ids = input_ids
    for _ in range(max_new_tokens):
        B,T = output_ids.shape
        if T>block_size:
            cur_ids = output_ids[:,-block_size:] # B,block_size
        else:
            cur_ids = output_ids # [B, T_cur]
        logits, _ = model(cur_ids, attention_mask = None, past_key_values = None, use_cache = False) #[B,block_size,V]
        logits_last = logits[:,-1,:] #[B,V]
        logits_scaled  = logits_last/temperature

        # topk over vocab dimension
        B, V = logits_scaled.shape
        k_eff = min(k,V)
        (values, indices) = torch.topk(logits_scaled,k=k_eff,dim = -1) # each row: top-k logits + indices

        values = torch.softmax(values,dim=-1)  # [B, k_eff]
        dist = torch.distributions.Categorical(values)

        # which of the k indices to pick per batch
        topk_choice = dist.sample()            # [B] in [0, k_eff-1]


         # map back to vocab ids
        next_token = indices[
            torch.arange(B, device=indices.device), topk_choice
        ].unsqueeze(-1)                        # [B, 1]

        output_ids = torch.cat([output_ids, next_token], dim=1)
    return output_ids

@torch.no_grad()
def sample_top_p(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """
    Top-p (nucleus) sampling:
    Keep the smallest set of tokens whose cumulative probability >= top_p,
    then sample from that truncated distribution.

    input_ids: [B, T]
    returns: [B, T + max_new_tokens]
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    model.eval()
    output_ids = input_ids  # [B, T]

    for _ in range(max_new_tokens):
        B, T = output_ids.shape

        if T > block_size:
            cur_ids = output_ids[:, -block_size:]   # [B, block_size]
        else:
            cur_ids = output_ids                    # [B, T_cur]

        logits , _ = model(cur_ids, attention_mask = None, past_key_values = None, use_cache = False)  # [B, T_cur, V]
        last_logits = logits[:, -1, :]              # [B, V]

        # Temperature scaling
        scaled = last_logits / temperature          # [B, V]
        probs = torch.softmax(scaled, dim=-1)       # [B, V]

        # Sort probabilities descending
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)  # [B, V], [B, V]
        cumprobs = torch.cumsum(sorted_probs, dim=-1)                               # [B, V]

        # Mask out tokens beyond top_p mass
        # keep tokens where cumulative prob <= top_p, but always keep at least 1
        cutoff_mask = cumprobs > top_p
        # shift mask right so we always keep the first token where cumprob crosses top_p
        cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
        cutoff_mask[..., 0] = False

        sorted_probs_trunc = sorted_probs.clone()
        sorted_probs_trunc[cutoff_mask] = 0.0

        # Renormalize
        sorted_probs_trunc = sorted_probs_trunc / sorted_probs_trunc.sum(dim=-1, keepdim=True)

        dist = torch.distributions.Categorical(sorted_probs_trunc)
        sorted_choice = dist.sample()  # [B] index in sorted space

        next_token = sorted_indices[
            torch.arange(B, device=sorted_indices.device), sorted_choice
        ].unsqueeze(-1)                # [B, 1]

        output_ids = torch.cat([output_ids, next_token], dim=1)

    return output_ids

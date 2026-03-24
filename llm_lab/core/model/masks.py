# llm_lab/core/model/masks.py
# boilerplate: safe attention_mask normalization
import torch
def _normalize_attention_mask(attention_mask: torch.Tensor, *, B: int, T: int, device: torch.device) -> torch.Tensor:
    """
    Returns mask as torch.bool on correct device, shape [B,T].
    Convention: True = keep token, False = pad.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B,T], got {tuple(attention_mask.shape)}")
    if attention_mask.shape != (B, T):
        raise ValueError(f"attention_mask shape mismatch: expected {(B,T)}, got {tuple(attention_mask.shape)}")
    if attention_mask.device != device:
        attention_mask = attention_mask.to(device)
    # accept {0,1} int masks or bool masks
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask != 0
    return attention_mask
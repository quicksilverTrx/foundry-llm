#llm_lab/core/model/norms.py
from torch import nn
import torch

def make_norm(norm_type: str, d_model: int, eps: float = 1e-6) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    if norm_type == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    raise ValueError(f"Unknown norm_type={norm_type!r}. Expected 'layernorm' or 'rmsnorm'.")


class RMSNorm(nn.Module) :
    def __init__(self,d_model : int, eps : float = 1e-6 ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        D = x.shape[-1]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight  # broadcast over leading dims

        


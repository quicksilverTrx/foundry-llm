# llm_lab/core/train/optim.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List,Dict,Tuple
import inspect
import torch

@dataclass(frozen=True)
class OptimConfig:
    lr : float
    weight_decay: float 
    betas : Tuple[float,float] = (0.9,0.95)
    eps : float = 1e-8

def build_adamw_with_decay_groups(model: torch.nn.Module, cfg: OptimConfig)  -> Tuple[torch.optim.Optimizer,List[Dict]]:
    decay: List[torch.nn.Parameter] = []
    no_decay : List[torch.nn.Parameter] = []

    decay_names: List[str] = []
    no_decay_names : List[str] = []

    for name,p in model.named_parameters():
        if p.requires_grad == False:
            continue
        if 'embed' in name or 'lm_head' in name:
            no_decay.append(p)
            no_decay_names.append(name)
        elif p.ndim == 1:
            no_decay.append(p)
            no_decay_names.append(name)
        else:
            decay.append(p)
            decay_names.append(name)

    assert len(decay) > 0
    assert len(no_decay) > 0
    
    param_groups: List[Dict] = [
        {"params": decay, "weight_decay": cfg.weight_decay, "names": decay_names},
        {"params": no_decay, "weight_decay": 0.0, "names": no_decay_names},
    ]

    # fused AdamW is only supported on CUDA; fall back to standard on cpu/mps.
    fused_available = (
        'fused' in inspect.signature(torch.optim.AdamW).parameters
        and all(p.device.type == 'cuda' for p in decay + no_decay)
    )
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                            fused=fused_available)
    return opt, {"decay": decay_names, "no_decay": no_decay_names}

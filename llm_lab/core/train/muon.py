# llm_lab/core/train/muon.py
"""
Muon optimizer — Newton-Schulz orthogonalization of gradients before momentum.

Reference: modded-nanogpt (KellerJordan), train_gpt.py
  - Newton-Schulz coefficients: 3.4445, -4.7750, 2.0315  (copied exactly)
  - Newton-Schulz iterations: 5
  - Applied ONLY to 2D weight matrices (ndim >= 2)
  - Embeddings, norms, biases, scalars use standard AdamW

Usage::

    from llm_lab.core.train.muon import build_muon_optimizer

    opt = build_muon_optimizer(
        model,
        muon_lr=0.02,
        adam_lr=6e-4,
        adam_betas=(0.9, 0.95),
        weight_decay=0.1,
    )
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# ── Newton-Schulz iteration ──────────────────────────────────────────────────

# Coefficients copied EXACTLY from modded-nanogpt train_gpt.py.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_NS_ITERS  = 5


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = _NS_ITERS) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute G / ||G||₂ ≈ orthogonal matrix.

    Input:  G [m, n]  (m >= n preferred for numerical stability)
    Output: X [m, n]  approximately orthonormal columns

    Uses the degree-5 minimax polynomial approximation with the exact
    coefficients from modded-nanogpt.
    """
    assert G.ndim == 2
    a, b, c = _NS_COEFFS

    # Work in float32 for stability; input may be bf16.
    X = G.float()
    # Normalise so the largest singular value is ~1.
    X = X / (X.norm() + 1e-7)

    if X.shape[0] > X.shape[1]:
        # Tall matrix: work on X X^T (smaller side = n columns).
        transpose = False
    else:
        # Wide matrix: transpose so we always have m >= n.
        X = X.T
        transpose = True

    for _ in range(steps):
        A = X @ X.T
        # Degree-5 polynomial: X ← a*X + b*A*X + c*A²*X
        X = a * X + b * (A @ X) + c * (A @ A @ X)

    if transpose:
        X = X.T

    return X.to(dtype=G.dtype)


# ── Muon optimizer class ─────────────────────────────────────────────────────

class Muon(torch.optim.Optimizer):
    """
    Muon — momentum optimizer with Newton-Schulz gradient orthogonalization.

    Applied ONLY to 2D (or higher) weight matrices.  Embeddings, norms,
    biases, and scalar parameters must be in a separate AdamW group.

    Args:
        params:       iterable of 2D parameters (from build_muon_optimizer)
        lr:           learning rate (default 0.02)
        momentum:     momentum factor (default 0.95)
        nesterov:     use Nesterov momentum (default True)
        ns_steps:     Newton-Schulz iterations (default 5)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = _NS_ITERS,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if g.ndim < 2:
                    # Scalar / bias: fall back to plain SGD with momentum.
                    # (Shouldn't reach here if parameter split is correct.)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    p.add_(buf, alpha=-lr)
                    continue

                # Flatten to 2D: [rows, cols]
                orig_shape = g.shape
                g2d = g.view(g.shape[0], -1)

                # Newton-Schulz orthogonalization.
                g_orth = _zeropower_via_newtonschulz5(g2d, steps=ns_steps)

                # Scale by sqrt(max(rows, cols)) as in modded-nanogpt.
                scale = max(g2d.shape[0], g2d.shape[1]) ** 0.5
                g_orth = g_orth * scale

                g_orth = g_orth.view(orig_shape)

                # Momentum buffer.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g_orth)

                if nesterov:
                    update = g_orth + momentum * buf
                else:
                    update = buf

                p.add_(update, alpha=-lr)

                # Decoupled weight decay (applied after momentum update).
                wd = group.get("weight_decay", 0.0)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)

        return loss


# ── Combined optimizer factory ───────────────────────────────────────────────

def build_muon_optimizer(
    model: nn.Module,
    *,
    muon_lr: float = 0.02,
    adam_lr: float = 6e-4,
    adam_betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
) -> Tuple[List[torch.optim.Optimizer], Dict[str, List[str]]]:
    """
    Split model parameters into two optimizer groups:

    - **Muon group** (2D transformer weight matrices): Q/K/V/out projections,
      MLP weight matrices.  No weight decay applied here — Muon's orthogonal
      update implicitly regularises.
    - **Adam group** (everything else): token embeddings, lm_head, positional
      embeddings, RMSNorm/LayerNorm scales, biases, value embed params,
      x0-mixin scalars.  Standard AdamW with weight_decay on ≥2D params.

    Returns
    -------
    (muon_opt, adam_opt), name_groups
        Two optimizers ready for use; caller is responsible for stepping both
        and applying the LR schedule to each.
    name_groups: dict with keys "muon" and "adam" containing parameter names.
    """
    muon_params:      List[torch.nn.Parameter] = []
    adam_decay_params: List[torch.nn.Parameter] = []
    adam_nodecay_params: List[torch.nn.Parameter] = []

    muon_names:       List[str] = []
    adam_decay_names: List[str] = []
    adam_nodecay_names: List[str] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Embeddings and lm_head always go to Adam (no decay).
        is_embed_or_head = any(x in name for x in ("token_embed", "pos_embed", "lm_head", "value_embed", "x0_lambda"))
        # 1D params (biases, norm scales) → Adam no-decay.
        is_1d = p.ndim == 1
        # Norm weight tensors are 1D for RMSNorm/LayerNorm — caught above.

        if is_embed_or_head or is_1d:
            adam_nodecay_params.append(p)
            adam_nodecay_names.append(name)
        elif p.ndim >= 2:
            # 2D+ weight matrices: Muon for transformer weights.
            muon_params.append(p)
            muon_names.append(name)
        else:
            adam_nodecay_params.append(p)
            adam_nodecay_names.append(name)

    muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum, nesterov=muon_nesterov, weight_decay=weight_decay)

    import inspect
    fused_ok = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
        and all(p.device.type == "cuda" for p in adam_decay_params + adam_nodecay_params)
    )
    adam_opt = torch.optim.AdamW(
        [
            {"params": adam_decay_params,   "weight_decay": weight_decay},
            {"params": adam_nodecay_params, "weight_decay": 0.0},
        ],
        lr=adam_lr,
        betas=adam_betas,
        fused=fused_ok,
    )

    name_groups = {
        "muon":        muon_names,
        "adam_decay":  adam_decay_names,
        "adam_nodecay": adam_nodecay_names,
    }
    return [muon_opt, adam_opt], name_groups


def build_muon_optimizer_nanochat(
    model: nn.Module,
    *,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    weight_decay: float = 0.2,      # Muon group only; all Adam groups use WD=0
    embed_lr: float = 0.3,          # token_embed — fast adaptation of lookup table
    unembed_lr: float = 0.004,      # lm_head — conservative; tied to output distribution
    scalar_lr: float = 0.5,         # x0_lambda / x0_lambda0 scalars
    adam_lr: float = 6e-4,          # default Adam group (norms, biases, value_embed)
    embed_betas: Tuple[float, float] = (0.8, 0.95),
    scalar_betas: Tuple[float, float] = (0.96, 0.95),   # higher β₁ for slow-moving scalars
    default_betas: Tuple[float, float] = (0.8, 0.95),
    eps: float = 1e-10,
) -> Tuple[List[torch.optim.Optimizer], Dict[str, List[str]]]:
    """
    Nanochat-recipe combined optimizer: Muon for 2D weights + AdamW with four
    separate LR groups matching the nanochat gpt.py setup_optimizer pattern.

    Parameter groups
    ----------------
    Muon  : 2D transformer weight matrices (Q/K/V/O projections, MLP gate/up/down).
            weight_decay=0.2 (LR-coupled: p *= 1 − lr × wd each step).
            WD effect is cautious — decays proportionally with LR during warmdown.

    Adam embed   : token_embed.weight — lr=0.3   betas=(0.8, 0.95)  WD=0
    Adam unembed : lm_head.weight     — lr=0.004  betas=(0.8, 0.95)  WD=0
    Adam scalar  : x0_lambda*, x0_lambda0* (1-D scalars) — lr=0.5  betas=(0.96, 0.95)  WD=0
    Adam default : everything else (RMSNorm scales, biases, value_embed, pos_embed if any)
                   — lr=6e-4  betas=(0.8, 0.95)  WD=0

    Design notes
    ------------
    - embed_lr=0.3 is 50× higher than the default Adam group.  Validated by
      nanochat across 320 experiments at depth=12.  The embeddings are 1-D
      lookup tables that need fast early adaptation; Muon already handles the
      2-D transformer weights at lr=0.02 with orthogonalised updates.
    - scalar_betas=(0.96, 0.95): higher β₁ stabilises the slow-moving x0
      residual scalars.
    - Non-fused AdamW used to support per-group betas (fused AdamW ignores
      per-group betas in PyTorch ≤ 2.4).

    Returns
    -------
    ([muon_opt, adam_opt], name_groups)
        name_groups keys: "muon", "embed", "unembed", "scalar", "default"
    """
    muon_params:    List[torch.nn.Parameter] = []
    embed_params:   List[torch.nn.Parameter] = []
    unembed_params: List[torch.nn.Parameter] = []
    scalar_params:  List[torch.nn.Parameter] = []
    default_params: List[torch.nn.Parameter] = []

    muon_names:    List[str] = []
    embed_names:   List[str] = []
    unembed_names: List[str] = []
    scalar_names:  List[str] = []
    default_names: List[str] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "token_embed" in name:
            embed_params.append(p)
            embed_names.append(name)
        elif "lm_head" in name:
            unembed_params.append(p)
            unembed_names.append(name)
        elif "x0_lambda" in name:
            # Catches both x0_lambda.* and x0_lambda0.* (1-D scalars, shape [1])
            scalar_params.append(p)
            scalar_names.append(name)
        elif p.ndim >= 2 and not any(x in name for x in ("pos_embed", "value_embed")):
            # 2-D+ transformer weight matrix → Muon
            muon_params.append(p)
            muon_names.append(name)
        else:
            # 1-D params (RMSNorm, biases), pos_embed (unused for RoPE models),
            # value_embed parameters — standard Adam with low LR.
            default_params.append(p)
            default_names.append(name)

    muon_opt = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        weight_decay=weight_decay,
    )

    # Non-fused AdamW — required for per-group betas support.
    adam_opt = torch.optim.AdamW(
        [
            {"params": embed_params,   "lr": embed_lr,   "betas": embed_betas,   "weight_decay": 0.0},
            {"params": unembed_params, "lr": unembed_lr, "betas": embed_betas,   "weight_decay": 0.0},
            {"params": scalar_params,  "lr": scalar_lr,  "betas": scalar_betas,  "weight_decay": 0.0},
            {"params": default_params, "lr": adam_lr,    "betas": default_betas, "weight_decay": 0.0},
        ],
        lr=adam_lr,   # default; overridden per group above
        eps=eps,
    )

    name_groups = {
        "muon":    muon_names,
        "embed":   embed_names,
        "unembed": unembed_names,
        "scalar":  scalar_names,
        "default": default_names,
    }
    return [muon_opt, adam_opt], name_groups

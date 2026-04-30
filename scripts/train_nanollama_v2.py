#!/usr/bin/env python3
"""
Train NanoLlama-127M v2 on FineWebEDU-10BT (pretokenized .npy shards).

Default config: 127M params, 5B tokens (9537 steps), H100.

Optimizer modes
---------------
  --optimizer muon_adam   nanochat recipe: Muon for 2D weights +
                          4 separate Adam groups (embed/unembed/scalar/default)
                          LR schedule: constant + warmdown (last 40% of max_steps)

  --optimizer adamw       standard AdamW, decay/no-decay groups
                          LR schedule: cosine with linear warmup

Probe vs production mode
------------------------
  --max_steps 9537   Full production LR schedule length (5B tokens at default batch).
                     Used for LR schedule computation in ALL runs.
  --run_steps 1500   Actual steps to execute (probes only).
                     Omit or set equal to max_steps for production.

  CRITICAL: probe probes set run_steps=1500, max_steps=9537.
  At step 1500, the constant_warmdown schedule is still at 100% peak LR
  (warmdown doesn't start until step 5722 = 9537 * 0.6).
  All three probes (N1/N2/N3) are therefore tested at comparable LR regimes.

Quick start (H100 pod)
---------------------
    # Probe N1 (nanochat Muon, v2 arch, 1500 steps)
    python3 scripts/train_nanollama_v2.py \\
        --data_dir ./data/edu_fineweb10B \\
        --run_dir  ./runs/nanollama_v2/N1 \\
        --optimizer muon_adam --max_steps 9537 --run_steps 1500 \\
        --batch_size 16 --grad_accum 32 --compile

    # Production run (9537 steps, uses winning probe config)
    python3 scripts/train_nanollama_v2.py \\
        --data_dir ./data/edu_fineweb10B \\
        --run_dir  ./runs/nanollama_v2/production \\
        --optimizer muon_adam --max_steps 9537 \\
        --batch_size 16 --grad_accum 32 --compile

Kill rules (auto-applied)
--------------------------
  - val rising 3 consecutive evals after step 1000
  - train loss < 0.5  (diverged or overfit)
  - train−val gap > 3.0 after step 2000

Checkpoint format
-----------------
    {step, val_loss, config, model_state_dict,
     muon_state_dict, adam_state_dict,
     shard_idx, position}

Load for inference::

    ckpt = torch.load('best_val.pt', map_location='cpu')
    model = MiniGPT(MiniGPTConfig(**ckpt['config']))
    model.load_state_dict(ckpt['model_state_dict'])
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import csv
import datetime
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.data.shard_loader import ShardLoader
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.lr_schedule import constant_warmdown, cosine_with_warmup
from llm_lab.core.train.muon import Muon, _build_muon_param_groups


# ── NanoLlama-127M v2 architecture config ─────────────────────────────────────
#
# Changes vs v1 (nanollama_8l.json / pretrain_nanollama.py):
#   rope_fraction  : 1.0 → 0.5  (Partial RoPE, nanochat/modded-nanogpt standard)
#   n_value_embeds : 0   → 2    (per-layer learnable value-embedding biases)
#   use_x0_mixin   : False → True  (per-layer residual: x = λ·x + λ₀·x₀)
#
# Probe N3 overrides these back to v1 values via CLI flags.

NANOLLAMA_V2_CONFIG = MiniGPTConfig(
    arch_family="nanollama",
    vocab_size=50304,
    d_model=768,
    n_layers=8,
    n_heads=12,
    num_kv_heads=4,
    d_ff=2048,
    block_size=1024,
    dropout=0.0,
    norm_type="rmsnorm",
    mlp_type="swiglu",
    attention_type="gqa",
    pos_encoding_type="rope",
    tie_weights=False,
    logit_softcap=30.0,
    qk_norm=True,
    rope_fraction=0.5,
    n_value_embeds=2,
    use_x0_mixin=True,
    use_sdpa=True,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, log_path: Optional[Path] = None) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(line + "\n")


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast(device_type: str):
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class _NoOp:
    """Duck-typed no-op optimizer for the Muon slot when running AdamW-only.

    Does NOT inherit from torch.optim.Optimizer — modern PyTorch rejects
    empty parameter lists in the base Optimizer.__init__.
    """
    param_groups: list = []

    def step(self, closure=None) -> None:
        pass

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass


@torch.no_grad()
def _evaluate(
    model: MiniGPT,
    loader: ShardLoader,
    val_steps: int,
    device_type: str,
) -> float:
    model.eval()
    loader.reset()
    total = 0.0
    for _ in range(val_steps):
        x, y = loader.next_batch()
        with _autocast(device_type):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        total += loss.item()
    model.train()
    return total / val_steps


def _save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizers: list,
    step: int,
    val_loss: Optional[float],
    config_dict: dict,
    train_loader: Optional[ShardLoader] = None,
) -> None:
    muon_opt, adam_opt = optimizers
    ckpt: dict = {
        "step":              step,
        "val_loss":          val_loss,
        "config":            config_dict,
        "model_state_dict":  model.state_dict(),
        "muon_state_dict":   muon_opt.state_dict(),
        "adam_state_dict":   adam_opt.state_dict(),
    }
    if train_loader is not None:
        ckpt["shard_idx"] = train_loader.shard_idx
        ckpt["position"]  = train_loader.pos
    torch.save(ckpt, path)


def _load_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizers: list,
    device: torch.device,
    train_loader: Optional[ShardLoader] = None,
) -> Tuple[int, Optional[float]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    muon_opt, adam_opt = optimizers
    if "muon_state_dict" in ckpt:
        muon_opt.load_state_dict(ckpt["muon_state_dict"])
    if "adam_state_dict" in ckpt:
        adam_opt.load_state_dict(ckpt["adam_state_dict"])
    if train_loader is not None and "shard_idx" in ckpt:
        train_loader.fast_forward(ckpt["shard_idx"], ckpt["position"])
    return int(ckpt.get("step", 0)), ckpt.get("val_loss")


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train NanoLlama-127M v2 on FineWebEDU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--data_dir",  type=str, required=True,
                   help="Directory with edufineweb_{train,val}_*.npy shards")
    p.add_argument("--run_dir",   type=str, default="./runs/nanollama_v2",
                   help="Output directory for checkpoints, logs, CSV")
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--file_pattern", type=str,
                   default="edufineweb_{split}_*.npy",
                   help="Shard filename glob; {split} replaced by train/val")

    # Training steps — CRITICAL distinction
    p.add_argument("--max_steps", type=int, default=9537,
                   help="Full production LR schedule length. "
                        "9537 = 5B tokens at B=16 GA=32 T=1024. "
                        "Used for LR computation in ALL runs (probes + production).")
    p.add_argument("--run_steps", type=int, default=None,
                   help="Actual steps to execute. For probes: 1500. "
                        "Omit for production (defaults to max_steps).")

    # Batch / sequence
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Micro-batch size (sequences per micro-step)")
    p.add_argument("--grad_accum",  type=int, default=32,
                   help="Gradient accumulation steps; effective batch = B×GA×T")
    p.add_argument("--block_size",  type=int, default=1024)

    # LR schedule params
    p.add_argument("--warmup_steps",   type=int,   default=100,
                   help="Linear warmup steps (for both schedules)")
    p.add_argument("--warmdown_ratio", type=float, default=0.4,
                   help="Fraction of max_steps for LR warmdown (muon_adam only). "
                        "Warmdown starts at step floor(max_steps × (1−ratio)).")
    p.add_argument("--grad_clip",      type=float, default=1.0)

    # Eval / logging / checkpoints
    p.add_argument("--val_every",  type=int, default=500)
    p.add_argument("--val_steps",  type=int, default=300,
                   help="Val batches per eval (300 × B=16 × T=1024 ≈ 4.9M tokens, SE≈0.001 nats)")
    p.add_argument("--ckpt_every", type=int, default=2000,
                   help="Periodic checkpoint interval (skipped if run_steps < ckpt_every)")
    p.add_argument("--log_every",  type=int, default=50)

    # Device / compile
    p.add_argument("--device",  type=str, default=None,
                   help="cuda / mps / cpu (auto-detected)")
    p.add_argument("--compile", action="store_true",
                   help="Wrap model with torch.compile")
    p.add_argument("--seed",    type=int, default=42)

    # Probe metadata
    p.add_argument("--probe_name", type=str, default=None,
                   help="Short label written to run_summary.json")

    # Optimizer selection
    p.add_argument("--optimizer", type=str, default="muon_adam",
                   choices=["muon_adam", "adamw"],
                   help="muon_adam = nanochat recipe (Muon + 4 Adam groups); "
                        "adamw = standard single AdamW")

    # Muon+Adam nanochat LRs (muon_adam mode)
    p.add_argument("--muon_lr",    type=float, default=0.02,
                   help="Muon LR for 2D weight matrices")
    p.add_argument("--embed_lr",   type=float, default=0.3,
                   help="Adam LR for token embeddings (50× higher than default)")
    p.add_argument("--unembed_lr", type=float, default=0.004,
                   help="Adam LR for lm_head")
    p.add_argument("--scalar_lr",  type=float, default=0.5,
                   help="Adam LR for x0_lambda scalars")
    p.add_argument("--adam_lr",    type=float, default=6e-4,
                   help="Default Adam group LR (norms, biases, value_embed). "
                        "Also used as the global LR in adamw mode.")

    # Weight decay
    p.add_argument("--weight_decay", type=float, default=0.2,
                   help="WD for Muon 2D weights (muon_adam mode) or AdamW 2D params. "
                        "All Adam groups in muon_adam mode use WD=0.0.")

    # Architecture overrides (probe N3: v1 arch)
    p.add_argument("--rope_fraction",   type=float, default=None,
                   help="Override rope_fraction (N3: 1.0 for full RoPE)")
    p.add_argument("--n_value_embeds",  type=int,   default=None,
                   help="Override n_value_embeds (N3: 0)")
    p.add_argument("--use_x0_mixin",
                   type=lambda x: x.lower() == "true",
                   default=None,
                   help="Override use_x0_mixin (N3: false)")
    p.add_argument("--n_layers",       type=int, default=None)
    p.add_argument("--d_ff",           type=int, default=None)
    p.add_argument("--attention_type", type=str, default=None,
                   choices=["mha", "gqa"])
    p.add_argument("--num_kv_heads",   type=int, default=None)
    p.add_argument("--mlp_type",       type=str, default=None,
                   choices=["swiglu", "gelu", "relu_squared"])
    return p


# ── main training loop ────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    # Actual steps to execute (probes: run_steps < max_steps)
    run_steps: int = args.run_steps if args.run_steps is not None else args.max_steps

    torch.manual_seed(args.seed)

    device_str = args.device or _auto_device()
    device = torch.device(device_str)

    run_dir  = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    csv_path = run_dir / "trajectory.csv"

    # ── build model ───────────────────────────────────────────────────────────
    cfg = NANOLLAMA_V2_CONFIG
    overrides: dict = {}
    if args.block_size != cfg.block_size:
        overrides["block_size"] = args.block_size
    if args.rope_fraction is not None:
        overrides["rope_fraction"] = args.rope_fraction
    if args.n_value_embeds is not None:
        overrides["n_value_embeds"] = args.n_value_embeds
    if args.use_x0_mixin is not None:
        overrides["use_x0_mixin"] = args.use_x0_mixin
    if args.n_layers is not None:
        overrides["n_layers"] = args.n_layers
    if args.d_ff is not None:
        overrides["d_ff"] = args.d_ff
    if args.attention_type is not None:
        overrides["attention_type"] = args.attention_type
        if args.attention_type == "mha":
            overrides["num_kv_heads"] = args.num_kv_heads or cfg.n_heads
    if args.num_kv_heads is not None:
        overrides["num_kv_heads"] = args.num_kv_heads
    if args.mlp_type is not None:
        overrides["mlp_type"] = args.mlp_type
    if overrides:
        cfg = replace(cfg, **overrides)

    model = MiniGPT(cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    _log(f"NanoLlama-v2  {n_params/1e6:.2f}M params", log_path)
    _log(
        f"  {cfg.n_layers}L  d{cfg.d_model}  h{cfg.n_heads}  "
        f"kv{cfg.num_kv_heads}  ff{cfg.d_ff}  block={cfg.block_size}",
        log_path,
    )
    _log(
        f"  rope_frac={cfg.rope_fraction}  n_value_embeds={cfg.n_value_embeds}  "
        f"x0_mixin={cfg.use_x0_mixin}  qk_norm={cfg.qk_norm}  "
        f"softcap={cfg.logit_softcap}",
        log_path,
    )

    total_batch = args.batch_size * args.grad_accum * args.block_size
    _log(
        f"Batch: B={args.batch_size}  GA={args.grad_accum}  T={args.block_size}"
        f"  → {total_batch:,} tok/step",
        log_path,
    )
    _log(
        f"Schedule steps: {args.max_steps}  Run steps: {run_steps}"
        f"  ({total_batch * run_steps / 1e9:.3f}B tokens executed)",
        log_path,
    )
    _log(f"Device: {device_str}  seed={args.seed}", log_path)

    # ── torch.compile ─────────────────────────────────────────────────────────
    if args.compile:
        _log("Compiling with torch.compile (first ~50 steps slow)...", log_path)
        model = torch.compile(model)

    # ── optimizers ────────────────────────────────────────────────────────────
    if args.optimizer == "adamw":
        import inspect
        fused_ok = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device.type == "cuda"
        )
        _adamw = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.parameters()
                               if p.requires_grad and p.ndim >= 2],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for p in model.parameters()
                               if p.requires_grad and p.ndim < 2],
                    "weight_decay": 0.0,
                },
            ],
            lr=args.adam_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=fused_ok,
        )
        optimizers = [_NoOp(), _adamw]
        _log(
            f"Optimizer: AdamW  lr={args.adam_lr}  wd={args.weight_decay}"
            f"  betas=(0.9,0.95)  schedule=cosine_with_warmup",
            log_path,
        )
    else:
        _groups = _build_muon_param_groups(model)
        muon_opt = Muon(_groups["muon"], lr=args.muon_lr, momentum=0.95,
                        nesterov=True, weight_decay=args.weight_decay)
        adam_opt = torch.optim.AdamW(
            [
                {"params": _groups["embed"],   "lr": args.embed_lr,   "betas": (0.8, 0.95)},
                {"params": _groups["unembed"], "lr": args.unembed_lr, "betas": (0.8, 0.95)},
                {"params": _groups["scalar"],  "lr": args.scalar_lr,  "betas": (0.96, 0.95)},
                {"params": _groups["default"], "lr": args.adam_lr,    "betas": (0.8, 0.95)},
            ],
            lr=args.adam_lr, eps=1e-10, weight_decay=0.0,
        )
        optimizers = [muon_opt, adam_opt]
        n_muon = len(_groups["muon"])
        n_adam = sum(len(v) for k, v in _groups.items() if k != "muon")
        _log(
            f"Optimizer: Muon+Adam (nanochat)  "
            f"muon_lr={args.muon_lr}  embed_lr={args.embed_lr}  "
            f"unembed_lr={args.unembed_lr}  scalar_lr={args.scalar_lr}  "
            f"adam_lr={args.adam_lr}  wd(Muon)={args.weight_decay}",
            log_path,
        )
        _log(
            f"  Muon params: {n_muon}  Adam params: {n_adam}"
            f"  schedule=constant_warmdown  warmdown_ratio={args.warmdown_ratio}"
            f"  embed_lr={args.embed_lr}  unembed_lr={args.unembed_lr}"
            f"  scalar_lr={args.scalar_lr}",
            log_path,
        )

    muon_opt, adam_opt = optimizers

    # ── data loaders ──────────────────────────────────────────────────────────
    train_loader = ShardLoader(
        args.data_dir, "train", args.batch_size, args.block_size,
        file_pattern=args.file_pattern, device=device_str,
    )
    val_loader = ShardLoader(
        args.data_dir, "val", args.batch_size, args.block_size,
        file_pattern=args.file_pattern, device=device_str,
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_step = 0
    last_val_loss: Optional[float] = None
    if args.resume is not None:
        start_step, last_val_loss = _load_checkpoint(
            Path(args.resume), model, optimizers, device, train_loader
        )
        _log(f"Resumed from {args.resume}  step={start_step}  val={last_val_loss}", log_path)

    # ── config JSON ───────────────────────────────────────────────────────────
    run_config: dict = {
        "probe_name":            args.probe_name,
        "optimizer":             args.optimizer,
        "n_params":              n_params,
        "B":  args.batch_size,  "T": args.block_size, "GA": args.grad_accum,
        "total_batch_tokens":    total_batch,
        "max_steps":             args.max_steps,
        "run_steps":             run_steps,
        "tokens_executed_B":     (run_steps * total_batch) / 1e9,
        "muon_lr":               args.muon_lr,
        "embed_lr":              args.embed_lr,
        "unembed_lr":            args.unembed_lr,
        "scalar_lr":             args.scalar_lr,
        "adam_lr":               args.adam_lr,
        "warmup_steps":          args.warmup_steps,
        "warmdown_ratio":        args.warmdown_ratio,
        "weight_decay":          args.weight_decay,
        "grad_clip":             args.grad_clip,
        "device":                device_str,
        **{k: getattr(cfg, k) for k in [
            "arch_family", "d_model", "n_layers", "n_heads", "num_kv_heads",
            "d_ff", "block_size", "vocab_size", "norm_type", "mlp_type",
            "attention_type", "pos_encoding_type", "rope_fraction",
            "n_value_embeds", "use_x0_mixin", "qk_norm", "logit_softcap",
        ]},
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # ── CSV header ────────────────────────────────────────────────────────────
    if start_step == 0:
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "step", "train_loss", "muon_lr", "adam_lr",
                "grad_norm", "tok_per_sec", "gpu_mem_gb", "val_loss",
            ])

    # ── initial val (step 0 only) ─────────────────────────────────────────────
    if start_step == 0:
        init_val = _evaluate(model, val_loader, args.val_steps, device.type)
        last_val_loss = init_val
        _log(f"step=   0  init_val={init_val:.4f}", log_path)

    # ── kill-rule state ───────────────────────────────────────────────────────
    val_history: List[float] = []
    best_val: float = float("inf")
    best_val_path = run_dir / "best_val.pt"

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    completed = start_step  # will be updated each step; used in summary

    for step in range(start_step, run_steps):
        completed = step + 1  # 1-indexed for logging and gate checks

        # ── LR schedule ───────────────────────────────────────────────────────
        if args.optimizer == "adamw":
            current_adam_lr = cosine_with_warmup(
                step,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
                max_lr=args.adam_lr,
                min_lr=args.adam_lr * 0.1,
            )
            for pg in adam_opt.param_groups:
                pg["lr"] = current_adam_lr
            current_muon_lr = 0.0

        else:
            # constant_warmdown: compute normalised factor in [0, 1], then
            # scale each group by its own peak LR.
            lr_factor = constant_warmdown(
                step,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
                max_lr=1.0,
                min_lr=0.0,
                warmdown_ratio=args.warmdown_ratio,
            )
            current_muon_lr = args.muon_lr * lr_factor
            current_adam_lr = args.adam_lr * lr_factor

            for pg in muon_opt.param_groups:
                pg["lr"] = current_muon_lr

            # Adam groups are ordered: [embed, unembed, scalar, default]
            # group order: muon_opt (index 0), adam_opt (index 1)
            adam_peak_lrs = [
                args.embed_lr,
                args.unembed_lr,
                args.scalar_lr,
                args.adam_lr,
            ]
            for pg, peak_lr in zip(adam_opt.param_groups, adam_peak_lrs):
                pg["lr"] = peak_lr * lr_factor

        # ── forward + backward (gradient accumulation) ────────────────────────
        muon_opt.zero_grad()
        adam_opt.zero_grad()
        train_loss_accum = 0.0

        for _ in range(args.grad_accum):
            x, y = train_loader.next_batch()
            with _autocast(device.type):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
            (loss / args.grad_accum).backward()
            train_loss_accum += loss.item() / args.grad_accum

        # ── gradient clip + optimizer step ───────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        ).item()
        muon_opt.step()
        adam_opt.step()

        # ── timing + VRAM ─────────────────────────────────────────────────────
        t1 = time.time()
        tok_per_sec = total_batch / (t1 - t0)
        t0 = t1
        gpu_mem_gb = 0.0
        if device.type == "cuda":
            gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3

        # ── val eval ──────────────────────────────────────────────────────────
        is_val_step = (
            (completed % args.val_every == 0)
            or (completed == run_steps)
        )
        val_loss_now: Optional[float] = None

        if is_val_step:
            val_loss_now = _evaluate(model, val_loader, args.val_steps, device.type)
            last_val_loss = val_loss_now
            val_history.append(val_loss_now)
            _log(
                f"step={completed:5d}  train={train_loss_accum:.4f}  "
                f"val={val_loss_now:.4f}  tok/s={tok_per_sec:.0f}  "
                f"vram={gpu_mem_gb:.1f}GB",
                log_path,
            )

            # Save best-val checkpoint.
            if val_loss_now < best_val:
                best_val = val_loss_now
                _save_checkpoint(
                    best_val_path, model, optimizers, completed,
                    val_loss_now, cfg_dict, train_loader,
                )
                _log(f"  → new best val {best_val:.4f}", log_path)

            # Kill rules.
            if completed > 1000 and len(val_history) >= 3:
                if all(val_history[-i] > val_history[-i - 1] for i in range(1, 3)):
                    _log("KILL: val rising 3 consecutive evals — stopping.", log_path)
                    break
            if train_loss_accum < 0.5:
                _log(
                    f"KILL: train loss {train_loss_accum:.4f} < 0.5 — diverged or overfit.",
                    log_path,
                )
                break
            if completed > 2000 and (train_loss_accum - val_loss_now) > 3.0:
                _log(
                    f"KILL: train−val gap {train_loss_accum - val_loss_now:.4f} > 3.0.",
                    log_path,
                )
                break

            model.train()

        # ── CSV row (every step) ──────────────────────────────────────────────
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                completed,
                f"{train_loss_accum:.6f}",
                f"{current_muon_lr:.6e}",
                f"{current_adam_lr:.6e}",
                f"{grad_norm:.4f}",
                f"{tok_per_sec:.1f}",
                f"{gpu_mem_gb:.2f}",
                f"{val_loss_now:.4f}" if val_loss_now is not None else "",
            ])

        # ── console log (non-val steps) ───────────────────────────────────────
        if completed % args.log_every == 0 and not is_val_step:
            _log(
                f"step={completed:5d}  loss={train_loss_accum:.4f}  "
                f"muon_lr={current_muon_lr:.3e}  adam_lr={current_adam_lr:.3e}  "
                f"norm={grad_norm:.3f}  tok/s={tok_per_sec:.0f}  "
                f"vram={gpu_mem_gb:.1f}GB",
                log_path,
            )

        # ── periodic checkpoint ───────────────────────────────────────────────
        if completed % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"step_{completed:05d}.pt"
            _save_checkpoint(
                ckpt_path, model, optimizers, completed,
                last_val_loss, cfg_dict, train_loader,
            )
            _log(f"Checkpoint saved: {ckpt_path}", log_path)

    # ── run summary JSON ─────────────────────────────────────────────────────
    val_at: dict = {}
    for i, vl in enumerate(val_history):
        val_at[f"val@{(i + 1) * args.val_every}"] = vl

    def _descent(k1: str, k2: str) -> Optional[float]:
        if k1 in val_at and k2 in val_at:
            return (val_at[k1] - val_at[k2]) / args.val_every
        return None

    ve = args.val_every
    summary = {
        "final_step":       completed,
        "run_steps":        run_steps,
        "best_val":         best_val,
        "last_val":         last_val_loss,
        "total_tokens_B":   (completed * total_batch) / 1e9,
        "val_history":      val_history,
        "val_at":           val_at,
        "descent_rate_1":   _descent(f"val@{ve}",    f"val@{2*ve}"),
        "descent_rate_2":   _descent(f"val@{2*ve}",  f"val@{3*ve}"),
        **run_config,
    }
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    _log(f"Summary written: {summary_path}", log_path)
    _log(f"Done.  best_val={best_val:.4f}  steps={completed}", log_path)


if __name__ == "__main__":
    main()

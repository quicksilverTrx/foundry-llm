#!/usr/bin/env python3
"""
Train SwiftLlama-350M on FineWebEDU (pretokenized .npy shards).

Default config: 345M params, 15B tokens (~28,610 steps), RTX 4090.

Quick-start (pod)
-----------------
    python3 scripts/train_swiftllama_350m.py \\
        --data_dir ./data/edu_fineweb10B \\
        --run_dir  ./runs/swiftllama_350m

OOM fallback (fits in <22 GB)
------------------------------
    python3 scripts/train_swiftllama_350m.py \\
        --data_dir ./data/edu_fineweb10B \\
        --run_dir  ./runs/swiftllama_350m \\
        --block_size 2048 --batch_size 4 --grad_accum 64

Kill rules (auto-applied)
--------------------------
  - val rising 5 consecutive evals after step 1000
  - train loss < 0.5  (diverged or overfit)
  - train loss - val loss gap > 4.0 after step 2000

Checkpoint format
-----------------
    {step, val_loss, config, model_state_dict,
     muon_state_dict, adam_state_dict}

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
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.data.shard_loader import ShardLoader
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.lr_schedule import cosine_with_warmup
from llm_lab.core.train.muon import build_muon_optimizer


# ── SwiftLlama-350M architecture config ──────────────────────────────────────

SWIFTLLAMA_350M_CONFIG = MiniGPTConfig(
    arch_family="swiftllama",
    d_model=1024,
    n_layers=22,
    n_heads=16,
    num_kv_heads=4,
    d_ff=2728,  # must be multiple of 8 for H100 tensor-core alignment
    block_size=4096,
    vocab_size=50304,
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


# ── helpers ──────────────────────────────────────────────────────────────────

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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
    model.train()
    return total / val_steps


def _save_checkpoint(
    path: Path,
    model,
    optimizers: list,
    step: int,
    val_loss: Optional[float],
    config_dict: dict,
    train_loader=None,
) -> None:
    muon_opt, adam_opt = optimizers
    ckpt = {
        "step": step,
        "val_loss": val_loss,
        "config": config_dict,
        "model_state_dict": model.state_dict(),
        "muon_state_dict":  muon_opt.state_dict(),
        "adam_state_dict":  adam_opt.state_dict(),
    }
    if train_loader is not None:
        ckpt["shard_idx"] = train_loader.shard_idx
        ckpt["position"]  = train_loader.pos
    torch.save(ckpt, path)


def _load_checkpoint(
    path: Path,
    model,
    optimizers: list,
    device: torch.device,
    train_loader=None,
) -> tuple[int, Optional[float]]:
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
        description="Train SwiftLlama-350M on FineWebEDU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",  type=str, required=True,
                   help="Directory with edufineweb_{train,val}_*.npy shards")
    p.add_argument("--run_dir",   type=str, default="./runs/swiftllama_350m",
                   help="Output: checkpoints, logs, CSV")
    p.add_argument("--max_steps", type=int, default=28610,
                   help="Total optimizer steps (28610 = 15B tokens at default batch)")
    p.add_argument("--batch_size",  type=int, default=4,
                   help="Micro-batch size (sequences per micro-step)")
    p.add_argument("--grad_accum",  type=int, default=32,
                   help="Gradient accumulation steps; effective batch = B×GA×T tokens")
    p.add_argument("--block_size",  type=int, default=4096,
                   help="Sequence length (tokens). OOM fallback: 2048")
    p.add_argument("--muon_lr",     type=float, default=0.02,
                   help="Muon LR for 2D weight matrices")
    p.add_argument("--adam_lr",     type=float, default=6e-4,
                   help="Adam LR for embeddings, norms, scalars")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--val_every",    type=int, default=500,
                   help="Run val eval every N optimizer steps")
    p.add_argument("--val_steps",    type=int, default=300,
                   help="Number of shard batches for val estimate (300 = ~2.46M tokens, SE ~0.0019 nats)")
    p.add_argument("--ckpt_every",   type=int, default=2000,
                   help="Save periodic checkpoint every N steps")
    p.add_argument("--log_every",    type=int, default=50,
                   help="Print + CSV-log every N steps")
    p.add_argument("--device",  type=str, default=None,
                   help="cuda / mps / cpu (auto-detected)")
    p.add_argument("--resume",  type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--file_pattern", type=str,
                   default="edufineweb_{split}_*.npy",
                   help="Shard filename pattern; {split} is replaced by train/val")
    p.add_argument("--compile", action="store_true",
                   help="Wrap model with torch.compile (first ~50 steps slow for compilation)")
    p.add_argument("--optimizer", type=str, default="muon_adam",
                   choices=["muon_adam", "adamw"],
                   help="Optimizer type: muon_adam (default) or pure adamw")
    p.add_argument("--probe_name", type=str, default=None,
                   help="Short label written to run_summary.json; used by ablation runner")
    # Architecture overrides (for ablation probes)
    p.add_argument("--d_ff",          type=int,   default=None, help="Override d_ff")
    p.add_argument("--n_layers",      type=int,   default=None, help="Override n_layers")
    p.add_argument("--rope_fraction", type=float, default=None, help="Override rope_fraction")
    p.add_argument("--n_value_embeds",type=int,   default=None, help="Override n_value_embeds")
    p.add_argument("--use_x0_mixin",  type=lambda x: x.lower()=="true", default=None,
                   help="Override use_x0_mixin (true/false)")
    p.add_argument("--attention_type",type=str,   default=None, choices=["mha","gqa"],
                   help="Override attention_type")
    p.add_argument("--num_kv_heads",  type=int,   default=None, help="Override num_kv_heads")
    p.add_argument("--mlp_type",      type=str,   default=None, choices=["swiglu","gelu","relu_squared"],
                   help="Override mlp_type")
    p.add_argument("--seed",          type=int,   default=42,   help="Random seed (same seed = same init)")
    return p


# ── main training loop ────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    device_str = args.device or _auto_device()
    device = torch.device(device_str)

    # ── seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    run_dir  = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    csv_path = run_dir / "trajectory.csv"

    # ── build model ───────────────────────────────────────────────────────────
    from dataclasses import replace
    cfg = SWIFTLLAMA_350M_CONFIG
    # Apply CLI architecture overrides (used by ablation probes).
    overrides = {}
    if args.block_size != cfg.block_size:
        overrides["block_size"] = args.block_size
    if args.d_ff is not None:
        overrides["d_ff"] = args.d_ff
    if args.n_layers is not None:
        overrides["n_layers"] = args.n_layers
    if args.rope_fraction is not None:
        overrides["rope_fraction"] = args.rope_fraction
    if args.n_value_embeds is not None:
        overrides["n_value_embeds"] = args.n_value_embeds
    if args.use_x0_mixin is not None:
        overrides["use_x0_mixin"] = args.use_x0_mixin
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

    _log(f"SwiftLlama-350M  {n_params/1e6:.2f}M params", log_path)
    _log(f"  {cfg.n_layers}L  d{cfg.d_model}  h{cfg.n_heads}  kv{cfg.num_kv_heads}  ff{cfg.d_ff}", log_path)
    _log(f"  rope_fraction={cfg.rope_fraction}  n_value_embeds={cfg.n_value_embeds}  x0_mixin={cfg.use_x0_mixin}", log_path)

    total_batch = args.batch_size * args.grad_accum * args.block_size
    total_tokens = total_batch * args.max_steps
    _log(f"Batch: B={args.batch_size} GA={args.grad_accum} T={args.block_size} → {total_batch:,} tok/step", log_path)
    _log(f"Steps: {args.max_steps}  ≈ {total_tokens/1e9:.1f}B tokens", log_path)
    _log(f"Device: {device_str}", log_path)

    # ── torch.compile ─────────────────────────────────────────────────────────
    if args.compile:
        _log("Compiling model with torch.compile (first ~50 steps will be slow)...", log_path)
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
                {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
                 "weight_decay": args.weight_decay},
                {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
                 "weight_decay": 0.0},
            ],
            lr=args.adam_lr,
            betas=(0.9, 0.95),
            fused=fused_ok,
        )
        # Wrap as a list so the rest of the script (which expects [muon, adam]) works.
        # We use a dummy no-op optimizer in the muon slot.
        class _NoOp(torch.optim.Optimizer):
            def __init__(self): super().__init__([], {})
            def step(self, closure=None): pass
        optimizers = [_NoOp(), _adamw]
        muon_opt, adam_opt = optimizers
        _log(f"Optimizer: pure AdamW  lr={args.adam_lr}  wd={args.weight_decay}", log_path)
    else:
        optimizers, name_groups = build_muon_optimizer(
            model,
            muon_lr=args.muon_lr,
            adam_lr=args.adam_lr,
            weight_decay=args.weight_decay,
        )
        muon_opt, adam_opt = optimizers
        _log(f"Muon params: {len(name_groups['muon'])}  Adam params: {len(name_groups['adam_decay'])+len(name_groups['adam_nodecay'])}", log_path)

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

    # ── run config JSON ───────────────────────────────────────────────────────
    run_config = {
        "probe_name": args.probe_name,
        "optimizer": args.optimizer,
        "n_params": n_params,
        "B": args.batch_size, "T": args.block_size, "GA": args.grad_accum,
        "total_batch_tokens": total_batch,
        "max_steps": args.max_steps, "total_tokens_B": total_tokens / 1e9,
        "muon_lr": args.muon_lr, "adam_lr": args.adam_lr,
        "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "device": device_str,
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
            csv.writer(f).writerow(
                ["step", "train_loss", "muon_lr", "adam_lr",
                 "grad_norm", "tok_per_sec", "gpu_mem_gb", "val_loss"]
            )

    # ── initial val ───────────────────────────────────────────────────────────
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

    for step in range(start_step, args.max_steps):
        completed = step + 1  # 1-indexed step number used for logging/eval gates

        # ── LR schedule (same cosine for both optimizers, scaled by peak lr) ─
        muon_lr = cosine_with_warmup(
            step, warmup_steps=args.warmup_steps, max_steps=args.max_steps,
            max_lr=args.muon_lr, min_lr=args.muon_lr * 0.1,
        )
        adam_lr = cosine_with_warmup(
            step, warmup_steps=args.warmup_steps, max_steps=args.max_steps,
            max_lr=args.adam_lr, min_lr=args.adam_lr * 0.1,
        )
        for pg in muon_opt.param_groups:
            pg["lr"] = muon_lr
        for pg in adam_opt.param_groups:
            pg["lr"] = adam_lr  # for adamw-only mode this drives the single optimizer

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
        is_val_step = (completed % args.val_every == 0) or (completed == args.max_steps)
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
            # Save best val checkpoint.
            if val_loss_now < best_val:
                best_val = val_loss_now
                _save_checkpoint(best_val_path, model, optimizers, completed, val_loss_now, cfg_dict, train_loader)
                _log(f"  → new best val {best_val:.4f}", log_path)

            # Kill rules.
            if completed > 1000 and len(val_history) >= 5:
                if all(val_history[-i] > val_history[-i-1] for i in range(1, 5)):
                    _log("KILL: val rising 5 consecutive evals — stopping.", log_path)
                    break
            if train_loss_accum < 0.5:
                _log(f"KILL: train loss {train_loss_accum:.4f} < 0.5 — stopping.", log_path)
                break
            if completed > 2000 and (train_loss_accum - val_loss_now) > 4.0:
                _log(f"KILL: train-val gap {train_loss_accum - val_loss_now:.4f} > 4.0 — stopping.", log_path)
                break

            model.train()

        # ── CSV row ───────────────────────────────────────────────────────────
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                completed,
                f"{train_loss_accum:.6f}",
                f"{muon_lr:.6e}",
                f"{adam_lr:.6e}",
                f"{grad_norm:.4f}",
                f"{tok_per_sec:.1f}",
                f"{gpu_mem_gb:.2f}",
                f"{val_loss_now:.4f}" if val_loss_now is not None else "",
            ])

        # ── console log ───────────────────────────────────────────────────────
        if completed % args.log_every == 0:
            _log(
                f"step={completed:5d}  loss={train_loss_accum:.4f}  "
                f"muon_lr={muon_lr:.3e}  adam_lr={adam_lr:.3e}  "
                f"norm={grad_norm:.3f}  tok/s={tok_per_sec:.0f}  "
                f"vram={gpu_mem_gb:.1f}GB",
                log_path,
            )

        # ── periodic checkpoint ───────────────────────────────────────────────
        is_ckpt_step = (completed % args.ckpt_every == 0) or (completed == args.max_steps)
        if is_ckpt_step:
            ckpt_path = ckpt_dir / f"step_{completed:05d}.pt"
            _save_checkpoint(ckpt_path, model, optimizers, completed, last_val_loss, cfg_dict, train_loader)
            _log(f"Checkpoint: {ckpt_path}", log_path)

    # ── final checkpoint (last) ───────────────────────────────────────────────
    last_path = run_dir / "last.pt"
    _save_checkpoint(last_path, model, optimizers, completed, last_val_loss, cfg_dict, train_loader)
    _log(f"Last checkpoint: {last_path}", log_path)

    # ── run summary JSON ─────────────────────────────────────────────────────
    # Build val-at-step dict for probe analysis.
    val_every = args.val_every
    val_at = {}
    for i, vl in enumerate(val_history):
        step_of_eval = (i + 1) * val_every
        val_at[f"val@{step_of_eval}"] = vl
    descent_rate_1 = (
        (val_at.get(f"val@{val_every}", None) - val_at.get(f"val@{2*val_every}", None)) / val_every
        if f"val@{val_every}" in val_at and f"val@{2*val_every}" in val_at
        else None
    )
    descent_rate_2 = (
        (val_at.get(f"val@{2*val_every}", None) - val_at.get(f"val@{3*val_every}", None)) / val_every
        if f"val@{2*val_every}" in val_at and f"val@{3*val_every}" in val_at
        else None
    )
    summary = {
        "final_step": completed,
        "best_val": best_val,
        "last_val": last_val_loss,
        "total_tokens_B": (completed * total_batch) / 1e9,
        "val_history": val_history,
        "val_at": val_at,
        "descent_rate_1": descent_rate_1,
        "descent_rate_2": descent_rate_2,
        **run_config,
    }
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    _log(f"Summary written to {summary_path}", log_path)
    _log(f"Done. best_val={best_val:.4f}  steps={completed}", log_path)


if __name__ == "__main__":
    main()

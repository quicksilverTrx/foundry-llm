# llm_lab/core/train/shard_trainer.py
"""
ShardTrainer — step-based pretraining loop for raw .npy shard data.

Extracted and generalized from the Phase 6 run_phase6.py pod script.
Uses:
  - llm_lab.core.data.shard_loader.ShardLoader   (data)
  - llm_lab.core.train.lr_schedule.cosine_with_warmup  (LR)
  - llm_lab.core.train.optim.build_adamw_with_decay_groups  (optimizer)
  - torch.autocast (bf16 on CUDA; fp32 elsewhere)

Key design decisions vs the existing Trainer:
  - Step-based (not epoch-based); more natural for pretraining
  - Works with ShardLoader directly (no DataLoader wrapping needed)
  - Device-aware autocast: bf16 on cuda, fp32 on cpu/mps
  - Checkpoint format: {step, val_loss, config, model_state_dict,
    optimizer_state_dict} — matches Phase 6 checkpoint format
  - Resume from checkpoint supported (Trainer lacks this)
  - Trajectory CSV matches Phase 6 column layout
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_lab.core.data.shard_loader import ShardLoader
from llm_lab.core.train.lr_schedule import cosine_with_warmup
from llm_lab.core.train.optim import OptimConfig, build_adamw_with_decay_groups


@dataclass
class ShardTrainerConfig:
    """
    Full configuration for a ShardTrainer pretraining run.

    Defaults match the Phase 6 NanoLlama 8L run exactly.
    """
    # ── output ─────────────────────────────────────────────────────────────
    out_dir: str = "./out"

    # ── compute ────────────────────────────────────────────────────────────
    B: int = 16               # micro-batch size (sequences per micro-step)
    T: int = 1024             # sequence length (tokens per sequence)
    grad_accum: int = 32      # gradient accumulation steps
    device: str = "cuda"      # "cuda", "cpu", "mps"

    # ── learning rate ──────────────────────────────────────────────────────
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    max_steps: int = 4768

    # ── regularisation ─────────────────────────────────────────────────────
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    # ── logging / evaluation ───────────────────────────────────────────────
    val_every: int = 250      # run val eval every N optimizer steps
    val_steps: int = 20       # number of shard batches used for val
    ckpt_every: int = 500     # save checkpoint every N optimizer steps
    log_every: int = 50       # print train stats every N steps

    # ── data ───────────────────────────────────────────────────────────────
    data_dir: str = ""        # path to shard directory (required)
    file_pattern: str = "edufineweb_{split}_*.npy"

    # ── resume ─────────────────────────────────────────────────────────────
    resume_ckpt: Optional[str] = None  # path to checkpoint to resume from

    @property
    def total_batch(self) -> int:
        return self.B * self.T * self.grad_accum

    @property
    def total_tokens(self) -> float:
        return self.total_batch * self.max_steps


class ShardTrainer:
    """
    Step-based pretraining trainer for raw .npy shard data.

    Typical usage::

        from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
        from llm_lab.core.train.shard_trainer import ShardTrainer, ShardTrainerConfig

        cfg = ShardTrainerConfig(
            out_dir="/workspace/runs/phase6",
            data_dir="./data/edu_fineweb10B",
            B=16, T=1024, grad_accum=32,
            max_lr=6e-4, min_lr=6e-5,
            warmup_steps=200, max_steps=4768,
        )
        model_cfg = MiniGPTConfig(...)
        model = MiniGPT(model_cfg)
        trainer = ShardTrainer(model, model_cfg.__dict__, cfg)
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        model_config_dict: Dict,
        config: ShardTrainerConfig,
    ) -> None:
        self.model = model
        self.model_config_dict = model_config_dict
        self.config = config

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # ── autocast context ───────────────────────────────────────────────
        # bf16 only on CUDA; MPS and CPU fall back to fp32
        self._use_autocast = (self.device.type == "cuda")

        # ── optimizer ─────────────────────────────────────────────────────
        optim_cfg = OptimConfig(lr=config.max_lr, weight_decay=config.weight_decay)
        self.opt, self.param_groups = build_adamw_with_decay_groups(model, optim_cfg)

        # ── output dirs ───────────────────────────────────────────────────
        self.out_dir = Path(config.out_dir)
        self.ckpt_dir = self.out_dir / "ckpts"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── state ─────────────────────────────────────────────────────────
        self.start_step = 0
        self.last_val_loss: Optional[float] = None

        # ── data loaders ──────────────────────────────────────────────────
        self.train_loader = ShardLoader(
            config.data_dir, "train", config.B, config.T,
            file_pattern=config.file_pattern, device=config.device,
        )
        self.val_loader = ShardLoader(
            config.data_dir, "val", config.B, config.T,
            file_pattern=config.file_pattern, device=config.device,
        )

        # ── resume ────────────────────────────────────────────────────────
        if config.resume_ckpt is not None:
            self.start_step = self.load_checkpoint(config.resume_ckpt)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """Run the full training loop from start_step to max_steps."""
        cfg = self.config
        model = self.model
        opt = self.opt

        n_params = sum(p.numel() for p in model.parameters())
        self._log(
            f"ShardTrainer starting: {n_params/1e6:.2f}M params | "
            f"B={cfg.B} T={cfg.T} GA={cfg.grad_accum} "
            f"total_batch={cfg.total_batch:,} | "
            f"steps={self.start_step}→{cfg.max_steps} | "
            f"tokens≈{cfg.total_tokens/1e9:.2f}B | "
            f"device={cfg.device}"
        )
        self._log(
            f"decay_params={len(self.param_groups['decay'])}  "
            f"nodecay_params={len(self.param_groups['no_decay'])}"
        )

        # Save run config
        self._save_run_config(n_params)

        # CSV header (create fresh file; append if resuming)
        csv_path = self.out_dir / "trajectory.csv"
        if self.start_step == 0:
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["step", "train_loss", "lr", "grad_norm", "tok_per_sec", "val_loss"]
                )

        # Initial val (step 0 only, not on resume)
        if self.start_step == 0:
            val_loss = self._evaluate()
            self.last_val_loss = val_loss
            self._log(f"step=   0  VAL={val_loss:.4f}")

        model.train()
        t0 = time.time()

        for step in range(self.start_step, cfg.max_steps):
            # ── LR update ─────────────────────────────────────────────────
            lr = cosine_with_warmup(
                step,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
                max_lr=cfg.max_lr,
                min_lr=cfg.min_lr,
            )
            for pg in opt.param_groups:
                pg["lr"] = lr

            # ── forward + backward (grad accumulation) ────────────────────
            opt.zero_grad()
            train_loss_accum = 0.0

            for _ in range(cfg.grad_accum):
                x, y = self.train_loader.next_batch()
                with self._autocast():
                    logits, _ = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
                (loss / cfg.grad_accum).backward()
                train_loss_accum += loss.item() / cfg.grad_accum

            # ── optimizer step ────────────────────────────────────────────
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            ).item()
            opt.step()

            # ── timing ────────────────────────────────────────────────────
            t1 = time.time()
            tok_per_sec = cfg.total_batch / (t1 - t0)
            t0 = t1
            completed = step + 1  # 1-indexed step number

            # ── val eval ──────────────────────────────────────────────────
            val_loss_str = ""
            is_val_step = (completed % cfg.val_every == 0) or (step == cfg.max_steps - 1)
            if is_val_step:
                val_loss = self._evaluate()
                self.last_val_loss = val_loss
                val_loss_str = f"  VAL={val_loss:.4f}"
                self._log(f"step={completed:4d}{val_loss_str}")
                model.train()

            # ── CSV row ───────────────────────────────────────────────────
            with csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    completed,
                    f"{train_loss_accum:.6f}",
                    f"{lr:.6e}",
                    f"{grad_norm:.4f}",
                    f"{tok_per_sec:.2f}",
                    f"{self.last_val_loss:.4f}" if is_val_step else "",
                ])

            # ── periodic console log ───────────────────────────────────────
            if completed % cfg.log_every == 0:
                self._log(
                    f"step={completed:4d}  loss={train_loss_accum:.4f}  "
                    f"lr={lr:.2e}  norm={grad_norm:.4f}  "
                    f"tok/s={tok_per_sec:.0f}{val_loss_str}"
                )

            # ── checkpoint ────────────────────────────────────────────────
            is_ckpt_step = (completed % cfg.ckpt_every == 0) or (step == cfg.max_steps - 1)
            if is_ckpt_step:
                ckpt_path = self.ckpt_dir / f"step_{completed:05d}.pt"
                self.save_checkpoint(str(ckpt_path), step=completed)
                self._log(f"Checkpoint: {ckpt_path}")

        self._log("ShardTrainer fit() complete.")

    def save_checkpoint(self, path: str, step: Optional[int] = None) -> None:
        """
        Save full checkpoint (model + optimizer state).

        Checkpoint format is compatible with Phase 6 pod checkpoints:
            {step, val_loss, config, model_state_dict, optimizer_state_dict}
        """
        torch.save(
            {
                "step": step,
                "val_loss": self.last_val_loss,
                "config": self.model_config_dict,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            },
            path,
        )

    def save_model_only(self, path: str, step: Optional[int] = None) -> None:
        """
        Save a slimmed checkpoint (model weights only, no optimizer state).
        Suitable for distribution / inference; ~487MB vs ~1.5GB for full.
        """
        torch.save(
            {
                "step": step,
                "val_loss": self.last_val_loss,
                "config": self.model_config_dict,
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        """
        Load model + optimizer state from a checkpoint.

        Returns the step number stored in the checkpoint so training can resume.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        resume_step = int(ckpt.get("step", 0))
        self.last_val_loss = ckpt.get("val_loss")
        self._log(f"Resumed from {path}  step={resume_step}  val_loss={self.last_val_loss}")
        return resume_step

    # ── internals ─────────────────────────────────────────────────────────────

    def _evaluate(self) -> float:
        """Run val eval for val_steps batches; returns mean cross-entropy."""
        model = self.model
        model.eval()
        self.val_loader.reset()
        total = 0.0
        with torch.no_grad():
            for _ in range(self.config.val_steps):
                x, y = self.val_loader.next_batch()
                with self._autocast():
                    logits, _ = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
                total += loss.item()
        return total / self.config.val_steps

    def _autocast(self):
        """Return bf16 autocast context on CUDA; no-op on cpu/mps."""
        import contextlib
        if self._use_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_path = self.out_dir / "run.log"
        with log_path.open("a") as f:
            f.write(line + "\n")

    def _save_run_config(self, n_params: float) -> None:
        cfg = self.config
        payload = {
            "B": cfg.B, "T": cfg.T, "grad_accum": cfg.grad_accum,
            "total_batch": cfg.total_batch,
            "max_lr": cfg.max_lr, "min_lr": cfg.min_lr,
            "warmup_steps": cfg.warmup_steps, "max_steps": cfg.max_steps,
            "grad_clip": cfg.grad_clip, "weight_decay": cfg.weight_decay,
            "val_every": cfg.val_every, "val_steps": cfg.val_steps,
            "total_tokens_B": cfg.total_tokens / 1e9,
            "total_params_M": n_params / 1e6,
            **self.model_config_dict,
        }
        config_path = self.out_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)

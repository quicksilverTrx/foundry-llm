# llm_lab/core/train/trainer.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Callable, Literal
import contextlib

from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from llm_lab.core.train.optim import build_adamw_with_decay_groups, OptimConfig
from llm_lab.core.train.lr_schedule import cosine_with_warmup
import torch
import csv
from pathlib import Path
import math
import time

@dataclass
class TrainerConfig:
    device: str = "cpu"
    dtype: Literal["fp32", "bf16"] = "fp32"
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    log_dir: Optional[str] = None     
    log_every_n_steps: int = 100       
    num_epochs: int = 1                
    sample_every_n_steps_multiple: Optional[int] = None

    max_steps: Optional[int] = None          # stop after N optimizer steps
    eval_every_n_steps: Optional[int] = None # if set, run val eval periodically
    eval_max_batches: Optional[int] = None   # if set, cap val batches per eval pass
    optimizer: Literal["Adam", "AdamW"] = "AdamW"
    weight_decay: float = 0.0
    grad_accum_steps: int = 1
    lr_schedule_type: Literal["constant", "cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_min: float = 0.0
    val_steps: Optional[int] = None  # alias for eval_max_batches; required when val_loader is IterableDataset
    progress_enabled: bool = True
    progress_update_every_n_steps: Optional[int] = None


class Trainer:
    def __init__(self,
                 model : nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 config: TrainerConfig):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optim_group_names = None
        self.start_time = time.monotonic()

        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        print("Model device (after .to):", next(model.parameters()).device)
        if self.config.dtype not in ("fp32", "bf16"):
            raise ValueError(f"unsupported dtype={self.config.dtype!r}; expected 'fp32' or 'bf16'")
        if self.config.dtype == "bf16" and self.device.type != "cuda":
            raise ValueError("dtype='bf16' is only supported on CUDA devices")
    
        
        self.lr = self.config.lr
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "AdamW":
            optim_config = OptimConfig(lr=self.config.lr, weight_decay=self.config.weight_decay)
            AdamW_optimizer, named_parameters = build_adamw_with_decay_groups(self.model, optim_config)
            self.optimizer = AdamW_optimizer
            self.optim_group_names = named_parameters
        else:
            raise ValueError ("not supported optimiser ")
        if self.config.lr_schedule_type not in ("constant", "cosine"):
            raise ValueError(
                f"unsupported lr_schedule_type={self.config.lr_schedule_type!r}; expected 'constant' or 'cosine'"
            )
        if int(self.config.lr_warmup_steps) < 0:
            raise ValueError(f"lr_warmup_steps must be >= 0, got {self.config.lr_warmup_steps}")
        if float(self.config.lr_min) < 0.0:
            raise ValueError(f"lr_min must be >= 0.0, got {self.config.lr_min}")
        if float(self.config.lr_min) > float(self.config.lr):
            raise ValueError(
                f"lr_min must be <= base lr (lr={self.config.lr}, lr_min={self.config.lr_min})"
            )
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.global_step = 0
        self.last_val_step = -1
        self.last_sample_step = -1
        self.last_train_logged_loss: Optional[float] = None
        self.last_val_logged_loss: Optional[float] = None
        self.sample_callback: Optional[Callable[[int, int], None]] = None
        self.best_val_loss = math.inf
        self.best_ckpt_path: Optional[Path] = None
        self.ckpt_dir: Optional[Path] = None
        if config.log_dir is not None:
            self.log_dir = Path(config.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path = self.log_dir / "loss_curve.csv"
            self.status_metrics_path = self.log_dir / "status_curve.csv"
            self.ckpt_dir = self.log_dir / "checkpoints"
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = None
            self.metrics_path = None
            self.status_metrics_path = None

    def _autocast_context(self):
        if self.config.dtype != "bf16":
            return contextlib.nullcontext()
        # Use actual device type so the context works on any CUDA device.
        # (bf16 is still gated to CUDA-only via the __init__ guard above.)
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def _set_optimizer_lr(self, lr_value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr_value)

    def _lr_for_step(self, step: int) -> float:
        """
        Return the learning rate for optimizer step `step` (0-indexed).

        For lr_schedule_type="cosine" this delegates to cosine_with_warmup so
        the schedule is bit-for-bit identical to ShardTrainer:
            step 0  → max_lr / warmup_steps   (first warm-up step, non-zero)
            step warmup_steps → max_lr         (peak)
            step max_steps    → min_lr         (floor, stays there)
        """
        if self.config.lr_schedule_type == "constant":
            return float(self.config.lr)

        # cosine schedule — requires max_steps to be set
        max_steps = self.config.max_steps
        if max_steps is None or max_steps <= 0:
            return float(self.config.lr)

        return cosine_with_warmup(
            step,
            warmup_steps=int(self.config.lr_warmup_steps),
            max_steps=int(max_steps),
            max_lr=float(self.config.lr),
            min_lr=float(self.config.lr_min),
        )

    @staticmethod
    def progress_stats(*, start_time: float, step: int, max_steps: Optional[int], now: Optional[float] = None) -> dict[str, float | int | None]:
        if now is None:
            now = time.monotonic()
        elapsed = max(0.0, float(now) - float(start_time))
        steps_per_sec = (float(step) / elapsed) if step > 0 and elapsed > 0 else 0.0
        eta_sec: Optional[float] = None
        if max_steps is not None and step > 0 and steps_per_sec > 0.0:
            remaining = max(0, int(max_steps) - int(step))
            eta_sec = float(remaining) / steps_per_sec
        return {
            "elapsed_sec": elapsed,
            "steps_per_sec": steps_per_sec,
            "eta_sec": eta_sec,
        }

    def _progress_interval(self) -> int:
        interval = self.config.progress_update_every_n_steps
        if interval is None:
            interval = self.config.log_every_n_steps
        return max(int(interval), 1)

    def _emit_progress(self, *, epoch: int, split: str, loss: Optional[float]) -> None:
        if not self.config.progress_enabled:
            return
        if self.global_step <= 0:
            return
        if (self.global_step % self._progress_interval()) != 0:
            return
        stats = self.progress_stats(
            start_time=self.start_time,
            step=self.global_step,
            max_steps=self.config.max_steps,
        )
        elapsed_sec = float(stats["elapsed_sec"])
        steps_per_sec = float(stats["steps_per_sec"])
        eta_sec = stats["eta_sec"]
        eta_str = "?" if eta_sec is None else f"{float(eta_sec):.1f}s"
        max_step_str = "?" if self.config.max_steps is None else str(self.config.max_steps)
        loss_str = "n/a" if loss is None else f"{float(loss):.6f}"
        val_str = "n/a" if self.last_val_logged_loss is None else f"{float(self.last_val_logged_loss):.6f}"
        train_str = "n/a" if self.last_train_logged_loss is None else f"{float(self.last_train_logged_loss):.6f}"

        print(
            "[progress] "
            f"step={self.global_step}/{max_step_str} "
            f"elapsed={elapsed_sec:.1f}s "
            f"speed={steps_per_sec:.3f}it/s "
            f"eta={eta_str} "
            f"device={self.config.device} "
            f"train_last={train_str} "
            f"val_last={val_str} "
            f"split={split} "
            f"loss={loss_str}",
            flush=True,
        )
        self._log_status_metrics(
            split=split,
            epoch=epoch,
            step=self.global_step,
            loss=loss,
            elapsed_sec=elapsed_sec,
            steps_per_sec=steps_per_sec,
            eta_sec=eta_sec,
        )

    def train_epoch(self,epoch_index: int) -> float:
        self.model.train()
        total_loss = 0.0
        micro_batches = 0
        # IterableDataset has no __len__; guard against calling len() on it.
        _loader_has_len = not isinstance(self.train_loader.dataset, IterableDataset)
        if _loader_has_len and len(self.train_loader) <= 0:
            raise ValueError("train_loader is empty")
        if not _loader_has_len and self.config.max_steps is None:
            raise ValueError(
                "max_steps must be set in TrainerConfig when training with an IterableDataset"
            )
        grad_accum_steps = int(self.config.grad_accum_steps)
        if grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be > 0, got {grad_accum_steps}")

        self.optimizer.zero_grad()
        accum_loss = 0.0
        accum_count = 0

        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with self._autocast_context():
                outputs,_ = self.model(inputs,attention_mask = None, past_key_values = None, use_cache = False) # [B, T, vocab_size]
                B, T, V = outputs.shape
                outputs_flat = outputs.reshape(B*T,V)
                labels_flat = labels.reshape(B*T)
                loss = self.loss_fn(outputs_flat,labels_flat)
            loss_item = float(loss.item())
            (loss / grad_accum_steps).backward()
            total_loss += loss_item
            micro_batches += 1
            accum_loss += loss_item
            accum_count += 1

            # IterableDataset has no __len__, so is_last_batch is always False;
            # max_steps (checked above) ensures the loop terminates.
            is_last_batch = _loader_has_len and (i + 1) >= len(self.train_loader)
            should_step = (accum_count >= grad_accum_steps) or is_last_batch
            if not should_step:
                continue

            # Pass current global_step (0-indexed) so the schedule matches
            # ShardTrainer's cosine_with_warmup(step, ...) call exactly.
            scheduled_lr = self._lr_for_step(self.global_step)
            self._set_optimizer_lr(scheduled_lr)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            step_loss = accum_loss / max(accum_count, 1)
            accum_loss = 0.0
            accum_count = 0

            if (self.global_step % self.config.log_every_n_steps) == 0:
                self.last_train_logged_loss = float(step_loss)
                self._log_metrics(
                    split="train",
                    epoch=epoch_index,
                    step=self.global_step,
                    loss=step_loss,
                )
                print(f"[train] step={self.global_step} loss={step_loss:.6f}", flush=True)
            self._emit_progress(epoch=epoch_index, split="train", loss=step_loss)
            val_loss = self._maybe_run_val(epoch_index=epoch_index)
            if val_loss is not None:
                self.model.train()
            self._maybe_run_sample(epoch_index=epoch_index)

            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break

        return total_loss/max(micro_batches,1)
    
    def evaluate(self, epoch_index: int) -> float:
        if self.val_loader is None: return 0.0
        if len(self.val_loader)>0:
            self.model.eval()
            eval_loss = 0.0
            eval_batches = 0
            max_eval_batches = self.config.eval_max_batches
            if max_eval_batches is not None and max_eval_batches <= 0:
                raise ValueError(f"eval_max_batches must be > 0 when set, got {max_eval_batches}")
            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    if max_eval_batches is not None and i >= max_eval_batches:
                        break
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    with self._autocast_context():
                        outputs ,_= self.model(inputs,attention_mask = None, past_key_values = None, use_cache = False)
                        B,T,V = outputs.shape
                        outputs_flat = outputs.reshape(B*T,V)
                        labels_flat = labels.reshape(B*T)
                        loss = self.loss_fn(outputs_flat,labels_flat)
                    eval_loss += loss.item()
                    eval_batches += 1
            if eval_batches <= 0:
                return 0.0
            avg_loss = eval_loss / eval_batches
            self.last_val_logged_loss = float(avg_loss)
            self._log_metrics(
                split="val",
                epoch=epoch_index,
                step=self.global_step,
                loss=avg_loss,
                )
            print(
                f"[val] step={self.global_step} loss={avg_loss:.6f} batches={eval_batches}",
                flush=True,
            )
            self._emit_progress(epoch=epoch_index, split="val", loss=avg_loss)
            if self.ckpt_dir is not None and avg_loss < self.best_val_loss:
                self.best_val_loss = float(avg_loss)
                self.best_ckpt_path = self.ckpt_dir / "best_val.pt"
                self.save_checkpoint(str(self.best_ckpt_path))
            return avg_loss
        else :
            return 0
    
    def _maybe_run_val(self, epoch_index: int) -> Optional[float]:
        if self.val_loader is None:
            return None
        if self.config.eval_every_n_steps is not None:
            interval = int(self.config.eval_every_n_steps)
        else:
            interval = self.config.log_every_n_steps * 10
        if interval <= 0:
            return None
        if (self.global_step % interval) != 0:
            return None
        if self.global_step == self.last_val_step:
            return None
        self.last_val_step = self.global_step
        return self.evaluate(epoch_index=epoch_index)

    def set_sample_callback(self, callback: Callable[[int, int], None]) -> None:
        self.sample_callback = callback

    def _maybe_run_sample(self, epoch_index: int) -> None:
        if self.sample_callback is None:
            return
        multiple = self.config.sample_every_n_steps_multiple
        if multiple is None or multiple <= 0:
            return
        interval = self.config.log_every_n_steps * multiple
        if interval <= 0:
            return
        if (self.global_step % interval) != 0:
            return
        if self.global_step == self.last_sample_step:
            return
        self.last_sample_step = self.global_step
        was_training = self.model.training
        self.model.eval()
        self.sample_callback(self.global_step, epoch_index)
        if was_training:
            self.model.train()

    def fit(self,num_epochs:Optional[int] = None) -> None : 
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        for i in range(num_epochs):
            training_loss_per_batch=self.train_epoch(epoch_index=i)
            # If we ran out of steps during training, still do one final eval then stop.
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                eval_loss_per_batch = self.evaluate(epoch_index=i)
                print(
                    f"[epoch {i}] train_epoch_avg_loss={training_loss_per_batch:.4f} "
                    f"val_epoch_avg_loss={eval_loss_per_batch:.4f} "
                    f"last_train_logged={self.last_train_logged_loss} "
                    f"last_val_logged={self.last_val_logged_loss}"
                )
                return
            eval_loss_per_batch=self.evaluate(epoch_index=i)
            print(f"[epoch {i}] train_epoch_avg_loss={training_loss_per_batch:.4f} "
                  f"val_epoch_avg_loss={eval_loss_per_batch:.4f} "
                  f"last_train_logged={self.last_train_logged_loss} "
                  f"last_val_logged={self.last_val_logged_loss}")
    def save_checkpoint(self, path: str) -> None:
        check_point = {"model_state":self.model.state_dict(),
                       "optimizer_state": self.optimizer.state_dict(),
                       "trainer_config": asdict(self.config)}
        torch.save(check_point,path)

    def load_checkpoint(self,path) -> None :
        checkpoint = torch.load(path,map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def _log_metrics(
        self,
        *,
        split: str,
        epoch: int,
        step: int,
        loss: float,
    ) -> None:
        """
        Minimal helper to append a row to loss_curve.csv.
        """
        if self.metrics_path is None:
            return
        
        is_new_file = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(["split", "epoch", "step", "loss"])
            writer.writerow([split, epoch, step, loss])

    def _log_status_metrics(
        self,
        *,
        split: str,
        epoch: int,
        step: int,
        loss: Optional[float],
        elapsed_sec: float,
        steps_per_sec: float,
        eta_sec: Optional[float],
    ) -> None:
        if self.status_metrics_path is None:
            return
        is_new_file = not self.status_metrics_path.exists()
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.status_metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(
                    [
                        "timestamp_utc",
                        "split",
                        "epoch",
                        "step",
                        "loss",
                        "elapsed_sec",
                        "steps_per_sec",
                        "eta_sec",
                        "max_steps",
                        "device",
                        "last_train_loss",
                        "last_val_loss",
                    ]
                )
            writer.writerow(
                [
                    timestamp,
                    split,
                    epoch,
                    step,
                    "" if loss is None else float(loss),
                    float(elapsed_sec),
                    float(steps_per_sec),
                    "" if eta_sec is None else float(eta_sec),
                    "" if self.config.max_steps is None else int(self.config.max_steps),
                    self.config.device,
                    "" if self.last_train_logged_loss is None else float(self.last_train_logged_loss),
                    "" if self.last_val_logged_loss is None else float(self.last_val_logged_loss),
                ]
            )

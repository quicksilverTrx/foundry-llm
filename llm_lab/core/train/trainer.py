# llm_lab/core/train/trainer.py

from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Optional,Dict,Any

from torch import nn
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import csv
from pathlib import Path

@dataclass
class TrainerConfig:
    device: str = "cpu"
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    log_dir: Optional[str] = None     
    log_every_n_steps: int = 100       
    num_epochs: int = 1                

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

        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        self.lr = self.config.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.lr)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.global_step = 0
        if config.log_dir is not None:
            self.log_dir = Path(config.log_dir)
            self.log_dir.mkdir (parents = True, exist_ok = True)
            self.metrics_path = self.log_dir/"loss_curve.csv"
        else:
            self.log_dir = None
            self.metrics_path = None

    def train_epoch(self,epoch_index: int) -> float:
        self.model.train()
        total_loss = 0
        if len(self.train_loader) <=0 :
            raise ValueError ("train_loader is empty")
        for i, data in enumerate(self.train_loader):
            self.global_step += 1
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs) # [B, T, vocab_size]
            B, T, V = outputs.shape
            outputs_flat = outputs.view(B*T,V)
            labels_flat = labels.view(B*T)
            loss = self.loss_fn(outputs_flat,labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.max_grad_norm)
            self.optimizer.step()
            total_loss +=loss.item()
            if (self.global_step % self.config.log_every_n_steps) == 0:
                self._log_metrics(
                    split="train",
                    epoch=epoch_index,
                    step=self.global_step,
                    loss=loss.item(),
                )

        return total_loss/len(self.train_loader)
    
    def evaluate(self, epoch_index: int) -> float:
        if self.val_loader is None: return 0.0
        if len(self.val_loader)>0:
            self.model.eval()
            eval_loss = 0
            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    B,T,V = outputs.shape
                    outputs_flat = outputs.view(B*T,V)
                    labels_flat = labels.view(B*T)
                    loss = self.loss_fn(outputs_flat,labels_flat)
                    eval_loss += loss.item()
            avg_loss = eval_loss / len(self.val_loader)   
            self._log_metrics(
                split="val",
                epoch=epoch_index,
                step=self.global_step,
                loss=avg_loss,
                )
            return avg_loss
        else :
            return 0
    
    def fit(self,num_epochs:Optional[int] = None) -> None : 
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        for i in range(num_epochs):
            training_loss_per_batch=self.train_epoch(epoch_index=i)
            eval_loss_per_batch=self.evaluate(epoch_index=i)
            if i % 10 == 0 :
                print(f"training loss per batch at epoch {i} is {training_loss_per_batch}")
                print(f"Eval loss per batch at epoch {i} is {eval_loss_per_batch}")
    
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
        with self.metrics_path.open("a", newline ="") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(["split", "epoch", "step", "loss"])
            writer.writerow([split, epoch, step, loss])

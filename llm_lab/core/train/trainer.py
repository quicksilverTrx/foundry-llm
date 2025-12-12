# llm_lab/core/train/trainer.py

from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Optional,Dict,Any

from torch import nn
from torch.utils.data import DataLoader
import torch

@dataclass
class TrainerConfig:
    device: str = "cpu"
    lr: float = 3e-4
    max_grad_norm: float = 1.0

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

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        if len(self.train_loader) <=0 :
            raise ValueError ("train_loader is empty")
        for i, data in enumerate(self.train_loader):
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

        return total_loss/len(self.train_loader)
    
    def evaluate(self) -> float:
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
            return eval_loss/len(self.val_loader)
        else :
            return 0
    
    def fit(self,num_epochs: int) -> None : 
        for i in range(num_epochs):
            training_loss_per_batch=self.train_epoch()
            eval_loss_per_batch=self.evaluate()
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
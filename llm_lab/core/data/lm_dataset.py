# llm_lab/core/data/lm_dataset.py
from __future__ import annotations

from typing import Sequence,Tuple,Union
from torch.utils.data import Dataset
import torch
from llm_lab.core.tokenization import SubwordTokenizer

class LanguageModelingDataset(Dataset):
    """
    Generic LM dataset based on a subword tokenizer.

    Produces (input_ids, labels) where:
      input_ids: tokens[t : t+block_size]
      labels:    tokens[t+1 : t+block_size+1]
    """
    def __init__(
            self,
            text: Union[str, Sequence[str]],
            tokenizer: SubwordTokenizer,
            block_size: int) -> None:
        assert block_size > 0 
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        if(isinstance(text,str)):
            ids = self.tokenizer.encode(text)
        else:
            all_ids = []
            for t in text:
                all_ids.extend(self.tokenizer.encode(t))
            ids = all_ids
        self.data = torch.tensor(ids,dtype = torch.long)

    def __len__(self):
        return max(0,len(self.data.size(0))-self.block_size)
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x,y

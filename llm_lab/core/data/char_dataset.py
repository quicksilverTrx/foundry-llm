# llm_lab/core/data/char_dataset.py
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset

from llm_lab.core.tokenization import CharTokenizer

class CharDataset(Dataset):
    def __init__(self,text : str, tokenizer : CharTokenizer, block_size : int):
        """
        Args:
            text: full corpus as a single string.
            tokenizer: CharTokenizer instance.
            block_size: context length (number of tokens).
        """
        assert block_size > 0
        self.tokenizer = tokenizer
        self.block_size = block_size

        encoded = tokenizer.encode(text)
        self.data = torch.tensor(encoded, dtype=torch.long)

    def __len__(self) -> int :
        return len(self.data) - self.block_size
    
    def __getitem__(self,idx : int) -> Tuple[torch.Tensor,torch.Tensor]:
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1 : idx+self.block_size+1]
        return x,y
# scripts/p1_train_char.py

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader,random_split

# Allow running this script directly without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_lab.core.tokenization import CharTokenizer
from llm_lab.core.data import CharDataset
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.trainer import Trainer, TrainerConfig

def main():
    text = "hello world"
    tokenizer = CharTokenizer.from_text(text)
    dataset = CharDataset(text,tokenizer=tokenizer,block_size=8)
    generator = torch.Generator().manual_seed(2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset,val_dataset = random_split(dataset,[train_size,val_size],generator=generator)
    train_dataloader = DataLoader(train_dataset,batch_size=8,shuffle=True)
    eval_dataloader = DataLoader(val_dataset,batch_size=8,shuffle=False)
    config = MiniGPTConfig(
        vocab_size=len(tokenizer.stoi),
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_ff=32,
        block_size=8,
        dropout=0.0,
    )
    gpt_small = MiniGPT(config)
    trainer_config = TrainerConfig()
    trainer = Trainer(gpt_small,train_dataloader,eval_dataloader,trainer_config)
    trainer.fit(10)

if __name__ == "__main__":
    main()

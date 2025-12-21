import torch
import os
from torch.utils.data import DataLoader
from llm_lab.core.tokenization import CharTokenizer
from llm_lab.core.data import CharDataset
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.trainer import Trainer, TrainerConfig
"""(foundry-llm) ron@MacBookPro-7 foundry-llm % pytest tests/core/test_trainer.py 
=================================================================================== test session starts ===================================================================================
platform darwin -- Python 3.11.11, pytest-9.0.1, pluggy-1.6.0
rootdir: /Users/ron/Desktop/github_projects/foundry-llm
configfile: pyproject.toml
collected 1 item                                                                                                                                                                          

tests/core/test_trainer.py .                                                                                                                                                        [100%]

=================================================================================== 1 passed in 41.85s ====================================================================================
"""
def test_loss_decreases_tiny():
    data_path = "data/tiny_shakespeare.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected dataset at {data_path}. "
            "Please place a text file there (e.g., tiny_shakespeare)."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text[:10000]
    tokenizer = CharTokenizer.from_text(text)
    block_size = 16
    dataset = CharDataset(text,tokenizer,16)
    dataloader = DataLoader(dataset,batch_size = 16,shuffle=True)

    vocab_size = len(tokenizer.stoi)
    config = MiniGPTConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        block_size=block_size,
        dropout=0.1,
    )
    model = MiniGPT(config=config)

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    training_config = TrainerConfig(
        device=device,
        lr=3e-4,
        max_grad_norm=1.0,
        num_epochs=2)
    
    trainer = Trainer(model,dataloader,None,training_config)
    loss1 = trainer.train_epoch(epoch_index=0)
    loss2 = trainer.train_epoch(epoch_index=1)

    assert loss2<loss1
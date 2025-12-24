# scripts/p1_train_char_real.py

import os
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

# Allow running this script directly without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_lab.core.tokenization import CharTokenizer
from llm_lab.core.data import CharDataset
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.trainer import Trainer, TrainerConfig


def main():
    data_path = "data/tiny_shakespeare.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected dataset at {data_path}. "
            "Please place a text file there (e.g., tiny_shakespeare)."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text[:100000]
    print(f"Loaded dataset with {len(text)} characters.")

    # Tokenizer
    tokenizer = CharTokenizer.from_text(text)

    block_size = 128
    dataset = CharDataset(text=text, tokenizer=tokenizer, block_size=block_size)

    # Train/val split
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,drop_last=True)

    # Model config: small-ish but non-trivial
    vocab_size = len(tokenizer.stoi)
    print(f"Vocab size: {vocab_size}")

    config = MiniGPTConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        block_size=block_size,
        dropout=0.1,
        pos_encoding_type="learned"
    )
    model = MiniGPT(config)

    # Device: M1 if available
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    run_dir = ROOT/"experiments/p1_char_runs/run_001"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = TrainerConfig(
        device=device,
        lr=3e-4,
        max_grad_norm=1.0,
        log_dir= str(run_dir)   ,
        log_every_n_steps= 100     ,  
        num_epochs = 1
        )
    trainer = Trainer(model, train_dataloader, eval_dataloader, trainer_config)

    # trainer.fit() -> Working for the entire run 

    for epoch in range(trainer_config.num_epochs):
        train_loss = trainer.train_epoch(epoch_index=epoch)
        val_loss = trainer.evaluate(epoch_index=epoch)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        checkpoint_path = run_dir/f"chkpt_epoch_{epoch}.pt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    (run_dir/"vocab.txt").write_text("".join(tokenizer.itos[i] for i in range(vocab_size)))
    print(f"Saved vocab: {run_dir/'vocab.txt'}")


if __name__ == "__main__":
    main()

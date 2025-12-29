# scripts/p2_train_bpe.py
from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig
from llm_lab.core.data.lm_dataset import LanguageModelingDataset
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.trainer import Trainer, TrainerConfig
from llm_lab.core.package.io import save_model_package

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def main():
    # ---- 0) device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    # ---- 1) data
    data_path = Path("data/tiny_shakespeare.txt")  # adjust if your repo uses a different location
    text = read_text(data_path)
    print(f"Loaded dataset with {len(text)} characters.")
    text = text[:(int)(len(text)*0.5)]
    print(f"Final  dataset with {len(text)} characters.")
    # ---- 2) train tokenizer (small + fast)
    tok_cfg = SubwordTokenizerConfig(vocab_size=2000, model_type="bpe")
    tokenizer = SubwordTokenizer.train_from_iterator([text], config=tok_cfg)
    vocab_size = len(tokenizer.stoi)
    print(f"Vocab size: {vocab_size}")
    # ---- 3) dataset
    block_size = 128
    all_ids = tokenizer.encode(text)
    split_idx = int(0.98 * len(all_ids))
    train_ids = all_ids [ :split_idx]
    val_ids = all_ids[split_idx:]
    train_ds = LanguageModelingDataset(text="", tokenizer=tokenizer, block_size=block_size, token_ids=train_ids)
    val_ds   = LanguageModelingDataset(text="", tokenizer=tokenizer, block_size=block_size, token_ids=val_ids)

    print("train_windows:", len(train_ds), "val_windows:", len(val_ds), "split_idx:", split_idx)


    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)

    # ---- 4) model
    cfg = MiniGPTConfig(
        vocab_size=len(tokenizer.stoi),
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=4*256,
        block_size=block_size,
        dropout=0.1,
        # keep pos_encoding_type="learned" for Phase2 baseline
    )
    model = MiniGPT(cfg)
    ROOT = Path(__file__).resolve().parents[1]
    run_dir = ROOT/"experiments/p2/run_003"
    run_dir.mkdir(parents=True, exist_ok=True)
    # ---- 5) trainer
    tcfg = TrainerConfig(
        device=device,
        lr=3e-4,
        max_grad_norm=1.0,
        log_dir= str(run_dir)   ,
        log_every_n_steps= 10     ,  
        num_epochs = 1,
        sample_every_n_steps_multiple=10,
        )
    trainer = Trainer(model, train_loader, val_loader, tcfg)

    # ---- 6) run (keep small first)
    trainer.fit(num_epochs=1)

    # ---- 7) save package
    pkg_dir = Path("artifacts/p2_bpe_smoke/3")
    save_model_package(
        package_dir=pkg_dir,
        config=cfg,
        tokenizer=tokenizer,
        model=model,
        is_best=True,
    )
    print(f"Saved package to: {pkg_dir}")

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig
from llm_lab.core.data.lm_dataset import LanguageModelingDataset
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.trainer import Trainer, TrainerConfig
from llm_lab.core.package.io import save_model_package

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def pick_device(dev: Optional[str]) -> str:
    if dev is not None:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _load_tokenizer(tokenizer_dir: Path, vocab_size: int, text: str) -> SubwordTokenizer:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if vocab_path.exists() and merges_path.exists():
        # support both signatures (with/without config)
        try:
            tok = SubwordTokenizer.load_from_files(vocab_path, merges_path)
        except TypeError:
            tok_cfg = SubwordTokenizerConfig(vocab_size=vocab_size, model_type="bpe")
            tok = SubwordTokenizer.load_from_files(vocab_path, merges_path, config=tok_cfg)
        return tok

    tok_cfg = SubwordTokenizerConfig(vocab_size=vocab_size, model_type="bpe")
    tok = SubwordTokenizer.train_from_iterator([text], config=tok_cfg)

    # support both signatures
    try:
        tok.save(vocab_path, merges_path)
    except TypeError:
        tok.save(vocab_path=vocab_path, merges_path=merges_path)

    return tok

def _load_or_make_split(split_dir: Path, tokenizer: SubwordTokenizer, text: str) -> tuple[list[int], list[int]]:
    train_path = split_dir / "train_ids.pt"
    val_path = split_dir / "val_ids.pt"

    if train_path.exists() and val_path.exists():
        train_ids = torch.load(train_path).tolist()
        val_ids = torch.load(val_path).tolist()
        return train_ids, val_ids

    all_ids = tokenizer.encode(text)
    split_idx = int(0.98 * len(all_ids))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    torch.save(torch.tensor(train_ids, dtype=torch.long), train_path)
    torch.save(torch.tensor(val_ids, dtype=torch.long), val_path)
    return train_ids, val_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data", type=str, default="data/tiny_shakespeare.txt")
    p.add_argument("--run_dir", type=str, required=True)

    p.add_argument("--tokenizer_dir", type=str, required=True)
    p.add_argument("--split_dir", type=str, required=True)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=200)  # val logging cadence

    args = p.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = load_json(Path(args.config))
    cfg = MiniGPTConfig(**cfg_dict)

    # snapshot config
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    text = read_text(Path(args.data))
    print(f"Loaded dataset with {len(text)} characters.")

    tokenizer = _load_tokenizer(tokenizer_dir, vocab_size=cfg.vocab_size, text=text)
    assert len(tokenizer.stoi) == cfg.vocab_size, f"tokenizer vocab={len(tokenizer.stoi)} != cfg.vocab_size={cfg.vocab_size}"

    train_ids, val_ids = _load_or_make_split(split_dir, tokenizer=tokenizer, text=text)
    print(f"Split token stream: train_ids={len(train_ids)} val_ids={len(val_ids)}")

    # datasets must match cfg.block_size for training
    train_ds = LanguageModelingDataset(text="", tokenizer=tokenizer, block_size=cfg.block_size, token_ids=train_ids)
    val_ds = LanguageModelingDataset(text="", tokenizer=tokenizer, block_size=cfg.block_size, token_ids=val_ids)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = MiniGPT(cfg)

    tcfg = TrainerConfig(
        device=device,
        lr=args.lr,
        max_grad_norm=1.0,
        log_dir=str(run_dir),
        log_every_n_steps=10,
        num_epochs=1,
        sample_every_n_steps_multiple=None,
        max_steps=args.max_steps,
        eval_every_n_steps=args.eval_every,
    )

    trainer = Trainer(model, train_loader, val_loader, tcfg)
    trainer.fit(num_epochs=1)

    # Save package for this run
    pkg_dir = run_dir / "package"
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

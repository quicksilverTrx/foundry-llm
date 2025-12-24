# scripts/p1_train_char_real.py

import os
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
from typing import Callable, Any

# Allow running this script directly without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_lab.core.tokenization import CharTokenizer
from llm_lab.core.data import CharDataset
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.trainer import Trainer, TrainerConfig
from llm_lab.core.decode.sampling import (
    greedy_decode,
    sample_top_k,
    sample_with_temperature,
    sample_top_p,
)



def main():
    data_path = "data/tiny_shakespeare.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Expected dataset at {data_path}. "
            "Please place a text file there (e.g., tiny_shakespeare)."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()


    print(f"Loaded dataset with {len(text)} characters.")
    text = text[:(int)(len(text)*0.5)]
    print(f"Final  dataset with {len(text)} characters.")
    # Tokenizer
    tokenizer = CharTokenizer.from_text(text)

    block_size = 256

    # Train/val split
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(text))
    val_size = len(text) - train_size

    train_text = text[:train_size]
    val_text = text[train_size:]
    ## Not using random split as it would not cause the correct holdout val set -> results in overlap between in train and val
    # train_dataset, val_dataset = random_split(
    #     dataset, [train_size, val_size], generator=generator
    # )
    train_dataset = CharDataset(text=train_text, tokenizer=tokenizer, block_size=block_size)
    val_dataset = CharDataset(text=val_text, tokenizer=tokenizer, block_size=block_size)
    print("text_len:", len(text))
    print("block_size:", block_size)

    print("train_size:", len(train_dataset), "val_size:", len(val_dataset))


    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False,drop_last=False)

    print("len(train_dataset)=", len(train_dataset), "len(val_dataset)=", len(val_dataset))
    print("len(train_dataloader)=", len(train_dataloader), "len(eval_dataloader)=", len(eval_dataloader))
    print("train_batches_per_epoch:", len(train_dataloader))
    print("val_batches_per_eval:", len(eval_dataloader))

    # Model config: small-ish but non-trivial
    vocab_size = len(tokenizer.stoi)
    print(f"Vocab size: {vocab_size}")

    config = MiniGPTConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        block_size=block_size,
        dropout=0.1,
        pos_encoding_type="learned"
    )
    model = MiniGPT(config)


    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

    print("Model device:", next(model.parameters()).device)

    # Device: M1 if available
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    run_dir = ROOT/"experiments/p1_actual/run_003"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = TrainerConfig(
        device=device,
        lr=3e-4,
        max_grad_norm=1.0,
        log_dir= str(run_dir)   ,
        log_every_n_steps= 200     ,  
        num_epochs = 3,
        sample_every_n_steps_multiple=10,
        )
    trainer = Trainer(model, train_dataloader, eval_dataloader, trainer_config)

    # trainer.fit() -> Working for the entire run 
    import json
    (run_dir/"run_config.json").write_text(json.dumps({
        "model": config.__dict__,
        "trainer": trainer_config.__dict__,
        "data": {"path": data_path, "block_size": block_size, "text_len": len(text)},
        "seed": 42,
    }, indent=2))

    def sample_text(
        tag: str,
        sampler: Callable[..., torch.Tensor],
        *sampler_args: Any,
        **sampler_kwargs: Any,
    ) -> None:
        prompt = "First Citizen:\n"
        x = torch.tensor([tokenizer.encode(prompt)], device=device)  # [1, T_prompt]
        with torch.no_grad():
            y = sampler(
                model,
                x,
                *sampler_args,
                **sampler_kwargs,
            )
        sample = tokenizer.decode(y[0].tolist())
        print(f"{tag} {sample[:500]}")

    def sample_callback(step: int, epoch: int) -> None:
        prompt_info = f"[step {step} | epoch {epoch}]"
        sample_text(
            f"{prompt_info} greedy_decode:",
            greedy_decode,
            max_new_tokens=200,
            block_size=block_size,
        )
        sample_text(
            f"{prompt_info} sample_with_temperature(0.8):",
            sample_with_temperature,
            max_new_tokens=200,
            block_size=block_size,
            temperature=0.8,
        )
        sample_text(
            f"{prompt_info} sample_top_k(k=50):",
            sample_top_k,
            max_new_tokens=200,
            block_size=block_size,
            temperature=0.9,
            k=50,
        )
        sample_text(
            f"{prompt_info} sample_top_p(p=0.92):",
            sample_top_p,
            max_new_tokens=200,
            block_size=block_size,
            temperature=0.8,
            top_p=0.92,
        )

    trainer.set_sample_callback(sample_callback)

    for epoch in range(trainer_config.num_epochs):
        train_loss = trainer.train_epoch(epoch_index=epoch)
        val_loss = trainer.evaluate(epoch_index=epoch)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        checkpoint_path = run_dir/f"chkpt_epoch_{epoch}.pt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        prompt = "First Citizen:\n"
        x = torch.tensor([tokenizer.encode(prompt)], device=device)  # [1, T_prompt]
        with torch.no_grad():
            y = greedy_decode(model.eval(), x, max_new_tokens=200, block_size=block_size)
        sample = tokenizer.decode(y[0].tolist())
        print("greedy decoding: ",sample[:500])

    
    (run_dir/"vocab.txt").write_text("".join(tokenizer.itos[i] for i in range(vocab_size)))
    print(f"Saved vocab: {run_dir/'vocab.txt'}")
    (run_dir/"vocab_debug.tsv").write_text(
    "\n".join(f"{i}\t{repr(tokenizer.itos[i])}" for i in range(vocab_size)),
    encoding="utf-8"
)

if __name__ == "__main__":
    main()

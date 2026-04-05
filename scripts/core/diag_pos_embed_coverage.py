# scripts/diag_pos_embed_coverage.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from llm_lab.core.package.io import load_model_package


def make_batch_from_stream(token_ids: list[int], T: int, B: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build B windows of length T+1 from a single token stream, then shift to (x,y).
    x,y: [B, T]
    """
    needed = B * (T + 1)
    if len(token_ids) < needed + 10:
        raise ValueError(f"Not enough tokens in stream: need ~{needed}, have {len(token_ids)}")

    xs = []
    ys = []
    for b in range(B):
        s = b * (T + 1)
        chunk = token_ids[s : s + (T + 1)]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


@torch.no_grad()
def summarize_weight_norms(pos_w: torch.Tensor, train_seq_len: int) -> None:
    # pos_w: [block_size, d_model]
    norms = pos_w.norm(dim=1)
    a = norms[:train_seq_len]
    b = norms[train_seq_len:]
    print(f"pos_embed weight norms:")
    print(f"  trained range [0:{train_seq_len})  mean={a.mean().item():.4f} std={a.std().item():.4f}")
    print(f"  untrained range[{train_seq_len}:] mean={b.mean().item():.4f} std={b.std().item():.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package_dir", type=str, required=True)
    p.add_argument("--split_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--train_seq_len", type=int, default=256, help="the seq len you trained with (e.g., 256)")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    pkg_dir = Path(args.package_dir)
    split_dir = Path(args.split_dir)

    val_ids = torch.load(split_dir / "val_ids.pt").tolist()

    config, tokenizer, model = load_model_package(pkg_dir, device=args.device)
    model.train()

    if not hasattr(model, "pos_embed"):
        raise SystemExit("This model has no pos_embed (pos_encoding_type is not 'learned').")

    # 1) Show weight stats
    pos_w = model.pos_embed.weight.detach().float().cpu()
    summarize_weight_norms(pos_w, train_seq_len=args.train_seq_len)

    # 2) Prove gradient coverage: with seq_len=train_seq_len, only positions < train_seq_len get grads
    T = args.train_seq_len
    B = args.batch_size
    x, y = make_batch_from_stream(val_ids, T=T, B=B)
    x = x.to(args.device)
    y = y.to(args.device)

    model.zero_grad(set_to_none=True)
    logits, _ = model(x)  # [B, T, V]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    grad = model.pos_embed.weight.grad.detach().float().cpu()  # [block_size, d_model]
    grad_norms = grad.norm(dim=1)

    trained = grad_norms[:T]
    untrained = grad_norms[T:]

    print("\npos_embed gradient coverage test:")
    print(f"  loss={loss.item():.4f}")
    print(f"  max grad norm in trained [0:{T})   = {trained.max().item():.6e}")
    print(f"  max grad norm in untrained[{T}:]  = {untrained.max().item():.6e}")
    print(f"  nonzero grad positions (trained)  = {(trained > 0).sum().item()} / {trained.numel()}")
    print(f"  nonzero grad positions (untrained)= {(untrained > 0).sum().item()} / {untrained.numel()}")

    # Expectation:
    # - untrained max grad ~ 0 for T=256 batch, proving positions >=256 were never updated in training runs at seq_len=256.


if __name__ == "__main__":
    main()

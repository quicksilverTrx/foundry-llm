# scripts/eval_val_by_context.py
"""python scripts/eval_val_by_context.py \\
  --package_dir experiments/p1_pos_enc/runs/learned/package \\
  --split_dir /Users/ron/Desktop/github_projects/foundry-llm/artifacts/p2_bpe_smoke/4/token_ids \\
  --out_csv experiments/serving_eval/val_by_ctx.csv \\
  --batch_size 32 \\
  --device mps

Optional knobs:
  --max_ctx 512
  --eval_stride 8
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from llm_lab.core.package.io import load_model_package


def pick_device(dev: str | None) -> str:
    if dev is not None:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def eval_stream_nll_fixed_targets(
    model,
    token_ids: list[int],
    *,
    ctx_len: int,
    max_ctx: int,
    batch_size: int,
    device: str,
    stride: int = 1,
) -> float:
    assert 1 <= ctx_len <= max_ctx
    model.eval()
    model.to(device)

    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    N = ids.numel()
    win = max_ctx + 1
    if N < win + 1:
        raise ValueError(f"Need at least {win+1} tokens, got N={N} (max_ctx={max_ctx}).")

    windows = ids.unfold(0, win, 1)
    if stride > 1:
        windows = windows[::stride]

    x_max = windows[:, :-1]
    y = windows[:, -1]

    x = x_max[:, max_ctx - ctx_len :]

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits, _ = model(xb)
        last = logits[:, -1, :]
        loss = F.cross_entropy(last, yb, reduction="sum")
        total_nll += float(loss.item())
        total_tokens += int(yb.numel())

    return total_nll / max(total_tokens, 1)


@torch.no_grad()
def eval_stream_nll_last_token(
    model,
    token_ids: list[int],
    ctx_len: int,
    batch_size: int,
    device: str,
) -> float:
    model.eval()
    model.to(device)

    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    total_loss = 0.0
    n_tokens = 0

    start = ctx_len
    end = ids.numel() - 1

    for i in range(start, end, batch_size):
        j = min(i + batch_size, end)
        idx = torch.arange(i, j, device=device)
        x = torch.stack([ids[t - ctx_len : t] for t in idx], dim=0)
        y = ids[idx]

        logits, _ = model(x)
        last_logits = logits[:, -1, :]
        loss = F.cross_entropy(last_logits, y)
        total_loss += float(loss.item()) * (j - i)
        n_tokens += (j - i)

    return total_loss / max(n_tokens, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package_dir", type=str, required=True)
    p.add_argument("--split_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_ctx", type=int, default=None)
    p.add_argument("--eval_stride", type=int, default=1)

    args = p.parse_args()

    device = pick_device(args.device)
    pkg_dir = Path(args.package_dir)
    split_dir = Path(args.split_dir)

    val_ids = torch.load(split_dir / "val_ids.pt").tolist()

    config, tokenizer, model = load_model_package(pkg_dir, device=device)
    max_ctx = args.max_ctx if args.max_ctx is not None else int(config.block_size)

    ctx_lens = [128, 256, 384, 512]
    ctx_lens = [T for T in ctx_lens if T <= max_ctx]

    rows = []
    print(f"package_dir={pkg_dir}")
    print(f"device={device} max_ctx={max_ctx} eval_stride={args.eval_stride} val_tokens={len(val_ids)}")

    for T in ctx_lens:
        print(f"\n=== ctx_len={T} ===")

        loss_last = eval_stream_nll_last_token(
            model, val_ids, ctx_len=T, batch_size=args.batch_size, device=device
        )
        ppl_last = float(torch.exp(torch.tensor(loss_last)))

        loss_fixed = eval_stream_nll_fixed_targets(
            model,
            val_ids,
            ctx_len=T,
            max_ctx=max_ctx,
            batch_size=args.batch_size,
            device=device,
            stride=args.eval_stride,
        )
        ppl_fixed = float(torch.exp(torch.tensor(loss_fixed)))

        print(f"last_token (confounded)  loss={loss_last:.4f} ppl={ppl_last:.2f}")
        print(f"fixed_targets (correct)  loss={loss_fixed:.4f} ppl={ppl_fixed:.2f}")

        rows.append((T, loss_fixed, ppl_fixed, loss_last, ppl_last))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["context_len", "loss_fixed_targets", "ppl_fixed_targets", "loss_last_token", "ppl_last_token"])
        w.writerows(rows)

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

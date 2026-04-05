"""
HellaSwag evaluation for NanoLlama 8L (or any MiniGPT checkpoint).

Downloads hellaswag_val.jsonl (~10K items × 4 endings) from GitHub and scores
each item using normalized completion loss.

Normalized accuracy:
  For each item, compute cross-entropy loss over the ENDING tokens only
  (tokens after the context boundary). Pick the ending with the lowest loss.
  This avoids length bias that occurs when scoring the full context + ending.

Usage:
  python scripts/eval_hellaswag.py --ckpt <path/to/checkpoint.pt>
  python scripts/eval_hellaswag.py --ckpt <path> --limit 1000   # quick test
  python scripts/eval_hellaswag.py --ckpt <path> --device cpu
"""

import sys, math, json, argparse, urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import tiktoken

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig

DATA_URL  = ("https://raw.githubusercontent.com/rowanz/hellaswag/"
             "master/data/hellaswag_val.jsonl")
DATA_PATH = REPO_ROOT / "data" / "hellaswag_val.jsonl"

# published reference numbers
REF_RANDOM   = 0.2500
REF_BASELINE = 0.2381   # GPT-2 124M at 1.05B tokens (normalized loss method)


def download_hellaswag():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"  Using cached {DATA_PATH}")
        return
    print("  Downloading HellaSwag val split from GitHub...", end=" ", flush=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    n = sum(1 for _ in DATA_PATH.open())
    print(f"done ({n} items)")


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = MiniGPTConfig(**ckpt["config"])
    mdl  = MiniGPT(cfg)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device).eval()
    step = ckpt.get("step", "?")
    vloss = ckpt.get("val_loss", float("nan"))
    print(f"  Loaded step={step}  val_loss={vloss:.4f}  "
          f"params={sum(p.numel() for p in mdl.parameters())/1e6:.1f}M")
    return mdl


@torch.no_grad()
def ending_loss(model, ctx_ids, ending_ids, device, block_size):
    """Cross-entropy averaged over the ending tokens only."""
    full = ctx_ids + ending_ids
    if len(full) > block_size + 1:
        full = full[-(block_size + 1):]
        ctx_len = max(0, len(full) - len(ending_ids))
    else:
        ctx_len = len(ctx_ids)

    x = torch.tensor(full[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(full[1:],  dtype=torch.long, device=device)

    logits, _ = model(x)
    end_start = ctx_len
    if end_start >= y.shape[0]:
        return float("nan")

    return F.cross_entropy(logits[0, end_start:, :], y[end_start:]).item()


def evaluate(model, enc, device, limit=None):
    block_size = model.config.block_size
    total = correct_norm = n_nan = 0

    with DATA_PATH.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item    = json.loads(line)
            ctx_ids = enc.encode(item["ctx"], allowed_special="all")
            label   = int(item["label"])

            losses = [
                ending_loss(model, ctx_ids,
                            enc.encode(" " + e, allowed_special="all"),
                            device, block_size)
                for e in item["endings"]
            ]

            if any(math.isnan(l) for l in losses):
                n_nan += 1
                continue

            if int(min(range(4), key=lambda j: losses[j])) == label:
                correct_norm += 1
            total += 1

            if (i + 1) % 500 == 0:
                acc = correct_norm / total if total else 0
                print(f"  [{i+1:5d}]  acc={acc:.4f}  ({correct_norm}/{total})",
                      flush=True)

    acc = correct_norm / total if total else 0
    print(f"\n  Final: {correct_norm}/{total}  accuracy={acc:.4f}  "
          f"({n_nan} skipped)")
    return acc, correct_norm, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Evaluate on first N items (default: all ~10K)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / mps / cpu  (auto-detected if omitted)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("  HellaSwag Evaluation")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print(f"  Items      : {'all' if args.limit is None else args.limit}")
    print("=" * 60)

    download_hellaswag()
    enc   = tiktoken.get_encoding("gpt2")
    model = load_model(ckpt_path, device)

    print("\nRunning evaluation (progress every 500 items)...\n")
    acc, correct, total = evaluate(model, enc, device, limit=args.limit)

    print("\n" + "=" * 60)
    print(f"  HellaSwag val  acc = {acc:.4f}  ({correct}/{total})")
    print(f"  Random baseline   = {REF_RANDOM:.4f}")
    print(f"  GPT-2 124M ref    ≈ {REF_BASELINE:.4f}  (1.05B tokens)")
    print("=" * 60)


if __name__ == "__main__":
    main()

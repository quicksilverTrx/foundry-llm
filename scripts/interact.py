"""
interact.py — Interactive sampling session for a MiniGPT checkpoint.

Usage (script mode — runs demo then exits):
    python scripts/interact.py --ckpt <path/to/checkpoint.pt>

Usage (REPL mode — load model then drop into interactive shell):
    python -i scripts/interact.py --ckpt <path/to/checkpoint.pt>
    >>> out = nucleus("Photosynthesis is")
    >>> out = greedy("The capital of France is")
    >>> print(ppl("The mitochondria is the powerhouse of the cell."))

Requirements:
    pip install torch tiktoken
    pip install -e .   (installs llm_lab)
"""

import sys, math, argparse, torch, torch.nn.functional as F, tiktoken
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.decode.sampling import (
    greedy_decode,
    sample_with_temperature,
    sample_top_k,
    sample_top_p,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",   required=True, help="Path to checkpoint .pt file")
parser.add_argument("--device", default=None,  help="cuda / mps / cpu (auto)")
args, _ = parser.parse_known_args()

# ── device ────────────────────────────────────────────────────────────────────
if args.device:
    DEVICE = args.device
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

# ── load model ────────────────────────────────────────────────────────────────
CKPT_PATH = Path(args.ckpt)
if not CKPT_PATH.exists():
    sys.exit(f"Checkpoint not found: {CKPT_PATH}")

print(f"Loading checkpoint: {CKPT_PATH.name} ...")
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
step  = ckpt.get("step", "?")
vloss = ckpt.get("val_loss", float("nan"))
print(f"  step={step}  val_loss={vloss:.4f}  (ppl={math.exp(vloss):.2f})")

cfg   = MiniGPTConfig(**ckpt["config"])
model = MiniGPT(cfg).to(DEVICE).eval()
model.load_state_dict(ckpt["model_state_dict"])

n_params = sum(p.numel() for p in model.parameters())
print(f"  arch: {cfg.n_layers}L d{cfg.d_model} h{cfg.n_heads} "
      f"kv{cfg.num_kv_heads} ff{cfg.d_ff}")
print(f"  params: {n_params/1e6:.2f}M  vocab: {cfg.vocab_size}")

# ── tokenizer ─────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")

def tok(text: str) -> list[int]:
    return enc.encode(text, allowed_special="all")

def detok(ids) -> str:
    return enc.decode(list(ids))

# ── generation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def greedy(prompt: str, max_new: int = 100) -> str:
    """Greedy decoding. Fast but prone to repetition. Good for fact probing."""
    ids = tok(prompt)
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = greedy_decode(model, inp, max_new_tokens=max_new,
                        block_size=cfg.block_size)
    return prompt + detok(out[0, len(ids):].tolist())


@torch.no_grad()
def sample(prompt: str, max_new: int = 150, temp: float = 0.8) -> str:
    """Temperature sampling. Try temp=0.7–1.0."""
    ids = tok(prompt)
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = sample_with_temperature(model, inp, max_new_tokens=max_new,
                                  block_size=cfg.block_size, temperature=temp)
    return prompt + detok(out[0, len(ids):].tolist())


@torch.no_grad()
def nucleus(prompt: str, max_new: int = 150,
            temp: float = 0.8, top_p: float = 0.9) -> str:
    """Nucleus (top-p) sampling. Best quality for prose. Recommended defaults."""
    ids = tok(prompt)
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = sample_top_p(model, inp, max_new_tokens=max_new,
                       block_size=cfg.block_size, temperature=temp, top_p=top_p)
    return prompt + detok(out[0, len(ids):].tolist())


@torch.no_grad()
def topk(prompt: str, max_new: int = 150,
         temp: float = 0.8, k: int = 50) -> str:
    """Top-k sampling. Alternative to nucleus."""
    ids = tok(prompt)
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = sample_top_k(model, inp, max_new_tokens=max_new,
                       block_size=cfg.block_size, temperature=temp, k=k)
    return prompt + detok(out[0, len(ids):].tolist())


@torch.no_grad()
def ppl(text: str) -> float:
    """Perplexity of a string. Lower = model finds it more probable."""
    ids = tok(text)
    if len(ids) < 2:
        raise ValueError("Need at least 2 tokens")
    x = torch.tensor(ids[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
    y = torch.tensor(ids[1:],  dtype=torch.long, device=DEVICE)
    logits, _ = model(x)
    return math.exp(F.cross_entropy(logits[0], y).item())


# ── demo run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DEMO — running samples then dropping into REPL if -i flag used.")
    print("=" * 60)

    prompts = [
        "Photosynthesis is the process by which",
        "The theory of evolution states that",
        "Once upon a time, a scientist discovered",
    ]
    for p in prompts:
        print(f"\nPROMPT: {p!r}")
        print("greedy :", greedy(p, max_new=40))
        print("nucleus:", nucleus(p, max_new=80, temp=0.8, top_p=0.9))

    print("\n" + "=" * 60)
    print("Perplexity ladder:")
    for t in [
        "The mitochondria is the powerhouse of the cell.",
        "The mitochondria is the banana of the cell.",
        "xkqz mfrp wlvb qjth nzxk.",
    ]:
        print(f"  ppl={ppl(t):.1f}  {t!r}")

    print("\n" + "=" * 60)
    print("Available functions (use with python -i):")
    print("  greedy(prompt, max_new=100)")
    print("  sample(prompt, max_new=150, temp=0.8)")
    print("  nucleus(prompt, max_new=150, temp=0.8, top_p=0.9)  ← recommended")
    print("  topk(prompt, max_new=150, temp=0.8, k=50)")
    print("  ppl(text)")

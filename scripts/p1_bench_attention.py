from __future__ import annotations
import argparse
import torch

from llm_lab.utils.bench import median_ms
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.model.attention import MultiHeadAttentionConfig,MultiHeadAttention

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["cpu","mps","cuda"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(0)

    # TODO: replace with your attention config/dataclass if you have one
    config = MultiHeadAttentionConfig(d_model=args.d_model, n_heads=args.n_heads, dropout=0.0)


    attention = MultiHeadAttention(config).to(device)

    attention.eval()

    x = torch.randn(args.B, args.T, args.d_model, device=device)

    @torch.no_grad()
    def step():
        output,_ = attention(x)

    ms = median_ms(step, iters=args.iters, warmup=args.warmup, device=args.device)
    print(f"[bench_attention] device={args.device} B={args.B} T={args.T} median_ms={ms:.2f}")

if __name__ == "__main__":
    main()
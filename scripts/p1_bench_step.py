from __future__ import annotations
import argparse
import torch

from llm_lab.utils.bench import median_ms
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.package.io import  load_model_package
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default ="mps", choices = ["mps","cuda","cpu"])
    ap.add_argument("--B", type = int, default=4)
    ap.add_argument("--T", type = int, default=128)
    ap.add_argument("--iters", type = int, default = 30)
    ap.add_argument("--warmup", type = int, default =10)
    args = ap.parse_args()

    torch.manual_seed(0)

    cfg = MiniGPTConfig(
        vocab_size=8192, d_model=256, n_layers=6, n_heads=8, d_ff=1024,
        block_size=args.T, dropout=0.0,
        pos_encoding_type="learned", norm_type="layernorm", mlp_type="gelu",
        attention_type="mha",
        # TODO (optional): attn_backend="manual" | "sdpa"
    )

    model = MiniGPT(cfg).to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Synthetic batch = stable perf signal; avoids tokenizer/dataloader noise.
    x = torch.randint(0, cfg.vocab_size, (args.B, args.T), device=args.device)
    y = torch.randint(0, cfg.vocab_size, (args.B, args.T), device=args.device)

    def step():
        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    ms = median_ms(step, iters=args.iters, warmup=args.warmup, device=args.device)
    print(f"[bench_step] device={args.device} B={args.B} T={args.T} median_ms={ms:.2f}")

if __name__ == "__main__":
    main()
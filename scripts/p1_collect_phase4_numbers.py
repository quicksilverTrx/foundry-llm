from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch

from llm_lab.utils.bench import median_ms
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig

def _sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

def bench_attn_forward(device: str, B: int, T: int, d_model: int, n_heads: int, iters: int, warmup: int) -> float:
    torch.manual_seed(0)
    cfg = MultiHeadAttentionConfig(d_model=d_model, n_heads=n_heads, dropout=0.0, use_rope=False)
    attn = MultiHeadAttention(cfg).to(device).eval()
    x = torch.randn(B, T, d_model, device=device)

    @torch.no_grad()
    def fn():
        _ = attn(x)

    _sync(device)
    return median_ms(fn, iters=iters, warmup=warmup, device=device)

def bench_train_step(device: str, B: int, T: int, vocab_size: int, iters: int, warmup: int) -> float:
    torch.manual_seed(0)
    cfg = MiniGPTConfig(
        vocab_size=vocab_size,
        d_model=256, n_layers=6, n_heads=8, d_ff=1024,
        block_size=T, dropout=0.0,
        pos_encoding_type="learned",
        norm_type="layernorm",
        mlp_type="gelu",
        attention_type="mha",
    )
    model = MiniGPT(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randint(0, vocab_size, (B, T), device=device)
    y = torch.randint(0, vocab_size, (B, T), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    _sync(device)
    return median_ms(step, iters=iters, warmup=warmup, device=device)

def md_table(baseline: dict, optimized: dict) -> str:
    rows = []
    rows.append("| Benchmark | Shape | Baseline (ms) | Optimized (ms) | Speedup |")
    rows.append("|---|---:|---:|---:|---:|")

    for key in ["attn_fwd_ms", "train_step_ms"]:
        b = baseline.get(key)
        o = optimized.get(key)
        if b is None or o is None:
            continue
        speedup = b / o if o > 0 else float("inf")
        shape = optimized.get("shape", "")
        rows.append(f"| {key} | {shape} | {b:.2f} | {o:.2f} | {speedup:.2f}× |")
    return "\n".join(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--tag", required=True, choices=["baseline", "optimized"])
    ap.add_argument("--out-dir", default="experiments/p1_attention_perf")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "tag": args.tag,
        "device": args.device,
        "shape": f"B={args.B},T={args.T}",
        "timestamp_unix": time.time(),
        "attn_fwd_ms": bench_attn_forward(args.device, args.B, args.T, args.d_model, args.n_heads, args.iters, args.warmup),
        "train_step_ms": bench_train_step(args.device, args.B, args.T, args.vocab_size, args.iters, args.warmup),
    }

    path = out_dir / f"phase4_numbers_{args.tag}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote: {path}")

    # If both exist, print comparison table
    bpath = out_dir / "phase4_numbers_baseline.json"
    opath = out_dir / "phase4_numbers_optimized.json"
    if bpath.exists() and opath.exists():
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        optimized = json.loads(opath.read_text(encoding="utf-8"))
        print("\n" + md_table(baseline, optimized) + "\n")

if __name__ == "__main__":
    main()

# llm_lab/utils/bench.py
from __future__ import annotations
import time
import torch

def sync(device : str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda" : 
        torch.cuda.synchronize()

def median_ms(fn, iters: int = 30, warmup: int = 10, device: str = "mps") -> float:
    for _ in range(warmup):
        fn(); sync(device)
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(); sync(device)
        xs.append((time.perf_counter() - t0) * 1000.0)
    xs.sort()
    return xs[len(xs)//2]
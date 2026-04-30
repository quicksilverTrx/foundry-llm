# Serving Results — NanoLlama 8L

All measurements on **NanoLlama 8L v1** (127.6M params, 8 layers, GQA, SwiGLU, RoPE, RMSNorm, val loss 3.357) running on **CPU M1 Pro, fp32** unless noted. Source: `experiments/nanollama_serving_bench/` and `experiments/nanollama_serving_quant/`.

The serving layer separates inference into `prefill()` (one forward pass, seeds KV cache) and `decode_step()` (one token per call, extends cache). All numbers below measure this two-phase execution.

---

## KV-cache correctness gate

Before any performance measurement, cached decode was verified to match full recompute logit-for-logit at the `atol=1e-4` threshold:

| Prompt length | Gen length | Max |logit diff| | Greedy token match |
|-------------|-----------|------------------|-------------------|
| 256 | 16 | **2.10e-05** | True |
| 512 | 16 | **1.91e-05** | True |

The third case (prompt=1024) diverges due to KV-cache sliding window truncation — expected behaviour when the context exceeds `block_size=1024` and is documented as a known limitation. The first two cases confirm correctness for the primary operating range.

---

## KV-cache speedup — single-point benchmark

**Config:** prompt=256, gen=128, B=1, fp32, warmup=5, iters=20.

| Mode | Prefill (ms) | Decode (ms/tok) | Tokens/sec |
|------|-------------|----------------|-----------|
| KV-cache (cached decode) | 99.9 | **17.8** | **56.0** |
| Full recompute | 162.9 | 196.2 | 5.1 |
| **Speedup** | — | **11.0×** | **11.0×** |

Cached decode is O(T) per token (attend new token against cached keys). Full recompute is O(T²) per token (reattend the growing context at every step). The gap widens as gen_len increases.

---

## TTFT / TPS grid — KV-cache mode

**Config:** gen=64, fp32, B=1/2/4, context lengths 256/512/1024.

| Context | B=1 TTFT | B=1 TPS | B=2 TTFT | B=2 TPS | B=4 TTFT | B=4 TPS |
|---------|----------|---------|----------|---------|----------|---------|
| 256 | 100ms | 56 | 170ms | 56 | 323ms | 56 |
| 512 | 189ms | 49 | 375ms | 50 | 675ms | 51 |
| 1024 | 463ms | 42 | 872ms | 40 | 2015ms | 43 |

**TTFT** scales roughly quadratically with context length (prefill is O(T²)) and linearly with batch size — consistent with compute-bound prefill on CPU. **Decode TPS** is stable across batch sizes at a given context length, indicating the bottleneck is memory bandwidth (loading model weights per step) rather than the attention computation itself.

The same grid in recompute mode for comparison:

| Context | B=1 TPS | B=2 TPS | B=4 TPS |
|---------|---------|---------|---------|
| 256 | 5 | 6 | 6 |
| 512 | 3 | 3 | 3 |
| 1024 | 1 | 1 | 2 |

---

## Quantization — fp32 vs int8

**Config:** prompt=256, gen=64, B=1, CPU M1 Pro, qnnpack backend. Dynamic int8 quantization via `torch.quantization.quantize_dynamic` (weights only, Linear layers only, no calibration dataset required).

| Mode | Decode (ms/tok) | Prefill (ms) | Memory | PPL |
|------|----------------|-------------|--------|-----|
| fp32 | 17.6 | 118ms | 486 MB | 268.2 |
| int8 | 13.9 | 341ms | **154 MB** | 284.3 |
| **Delta** | **−1.27× (faster)** | **+2.9× (slower)** | **−68%** | **+6.0%** |

int8 decode is 1.27× faster because smaller quantized weights reduce memory bandwidth pressure — the bottleneck on CPU. int8 prefill is 2.9× slower because dynamic quantization adds per-batch quantization overhead on the compute-bound prefill path.

**PPL drift** of +6.0% (268.2 → 284.3) on held-out text is within acceptable range for most serving use cases. Greedy token match between fp32 and int8 on short prompts: True (exact).

**When to use int8:** memory-constrained deployment (edge, shared GPU, CPU-only). 154 MB vs 486 MB is the difference between fitting in cache and not. For latency-sensitive prefill-heavy workloads, fp32 is preferred.

---

## Prompt suite — functional correctness

6 test cases across buckets: short prompt, long prompt, repetition trap, stop trap, code-like, safety probe. Backend: Engine (direct) and HTTP (FastAPI). Results:

| Metric | Value |
|--------|-------|
| Cases run | 6 |
| Errors | 0 |
| Safety refusals triggered | 0 (no PII in test prompts) |
| Backend parity (engine vs HTTP) | Exact match |
| Avg TTFT | 44.8ms |
| Avg TPS | 49.0 |

All 6 cases pass. Backend parity confirms the HTTP layer adds no generation-path divergence.

---

## Memory economics — KV-cache formula

KV-cache memory per token for NanoLlama 8L:

```
2 × n_layers × n_kv_heads × head_dim × bytes_per_element
= 2 × 8 × 4 × 64 × 4 bytes (fp32)
= 16,384 bytes per token
= 16 KB/tok
```

At `block_size=1024`: max KV cache = 1024 × 16 KB = **16 MB** — negligible relative to the 486 MB model weight footprint. GQA (4 KV heads vs 12 query heads) reduces this by 3× vs MHA: MHA would be 48 KB/tok, or 48 MB at block_size=1024.

---

## Test coverage

152 tests covering all serving and eval paths, all passing. Breakdown:

| Category | Tests |
|----------|-------|
| KV-cache ABI, correctness, equivalence | 11 |
| Engine structure, decode controls, sampling | 14 |
| HTTP API (/generate, /stream, /health, /metrics) | 22 |
| Prefill batching, rate limiting, SSE streaming | 15 |
| Safety filters, privacy logging | 13 |
| Precision modes (fp32/fp16/bf16) | 10 |
| Quantization (int8, backend, audit) | 15 |
| Eval: PPL, prompt suite, evidence pack | 52 |
| **Total** | **152** |

---

## Known limitations

| Limitation | Impact |
|-----------|--------|
| Decode is B=1 only | Single-request decode; no multi-sequence continuous batching |
| fp16/bf16 fall back to fp32 on CPU | Real benefit requires CUDA |
| int8 prefill 2.9× slower than fp32 | Use fp32 for prefill-heavy workloads |
| KV-cache sliding window at block_size boundary | Logit drift at ctx=1024+ (documented in equivalence gate) |
| Safety is regex-based heuristics | Not a learned classifier; suitable for demonstration |

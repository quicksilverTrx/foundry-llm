# NanoLlama 8L Serving Experiment Provenance

All results in `nanollama_serving_bench/`, `nanollama_serving_quant/`, and `nanollama_serving_reports/` were produced from the NanoLlama 8L model with model-appropriate benchmark configs.

## Model

| Field | Value |
|-------|-------|
| Checkpoint | `experiments/tinyllama_pretrain_2026-03-31/phase6/ckpts/step_04768_model_only.pt` |
| Architecture | GQA (Grouped-Query Attention) |
| Positional encoding | RoPE |
| Layers | 8 |
| Heads (Q) | 12 |
| KV heads | 4 |
| d_model | 768 |
| d_ff | 2048 |
| vocab_size | 50,304 (tiktoken GPT-2 padded to 64) |
| block_size | 1024 |
| Parameters | 127.6M |
| Norm | RMSNorm |
| MLP | SwiGLU |
| Softcap | 30.0 |
| QK-Norm | Yes |
| arch_family | nanollama |
| Tokenizer | tiktoken GPT-2 via TiktokenWrapper (eos=50256, pad=50256) |
| Training data | FineWeb-Edu, 2.5B tokens |
| Training step | 4768 |
| Val loss | 3.3566 nats |

## Environment

| Field | Value |
|-------|-------|
| Device | CPU |
| Hardware | Apple M1 Pro |
| OS | macOS 26.3.1 arm64 |
| Python | 3.11.11 |
| PyTorch | 2.2.2 |
| Quant engine | qnnpack |

## Benchmark configs

| Experiment | prompt_len | gen_len | warmup | iters | ctx_lens | batch_sizes | repeats |
|-----------|-----------|---------|--------|-------|----------|-------------|---------|
| Single-point | 256 | 128 | 5 | 20 | — | — | — |
| Grid sweep | — | 64 | 2 | 5 | 256, 512, 1024 | 1, 2, 4 | 2 |
| Quant sweep | 256 | 64 | — | — | — | — | — |

Config rationale: block_size=1024 so grid tests up to 1024; gen_len=128 for single-point to see steady-state decode; batch sizes 1/2/4 (B=8 impractical at ctx=1024 on CPU).

## Key results

### Inference benchmark (prompt=256, gen=128, fp32)

| Metric | Cache | Recompute | Speedup |
|--------|-------|-----------|---------|
| prefill_ms | 136.1 | 151.1 | — |
| TTFT_ms | 136.2 | — | — |
| decode_ms/token | 24.2 | 182.1 | **7.53x** |
| tokens/sec | 41.3 | 5.5 | 7.53x |

### TTFT / TPS grid (cache mode, fp32)

| ctx_len | B=1 | B=2 | B=4 |
|---------|-----|-----|-----|
| 256 | 100ms / 56.0 TPS | 170ms / 55.7 TPS | 323ms / 56.4 TPS |
| 512 | 189ms / 49.4 TPS | 375ms / 50.2 TPS | 675ms / 51.0 TPS |
| 1024 | 463ms / 41.8 TPS | 872ms / 40.3 TPS | 2015ms / 42.6 TPS |

Observations:
- TTFT scales ~quadratically with context (prefill is O(T²)): 100ms → 189ms → 463ms at B=1
- TTFT scales ~linearly with batch size at fixed context (batched prefill)
- Decode TPS is stable across batch sizes at each context length — memory bandwidth bound, not compute bound
- TPS degrades ~25% from ctx=256 to ctx=1024 due to growing KV cache memory pressure

### Quantization (prompt=256, gen=64)

| Mode | prefill_ms | decode_ms/tok | TPS | PPL | Memory |
|------|-----------|---------------|-----|-----|--------|
| fp32 | 118.1 | 17.6 | 56.7 | 268.2 | 511 MB |
| int8 | 341.1 | 13.9 | 72.1 | 284.3 | 155 MB |

Analysis:
- int8 decode is **1.27x faster** (13.9 vs 17.6 ms/tok) — smaller weights reduce memory bandwidth
- int8 prefill is **2.9x slower** (341 vs 118 ms) — dynamic quantization overhead on compute-bound path
- int8 memory is **70% smaller** (155 vs 511 MB)
- int8 PPL drift is **+6.0%** (268.2 → 284.3) — acceptable for most use cases
- Recommendation: int8 default on CPU for memory-constrained; fp32 for prefill-heavy workloads

### Prompt suite (6 cases)

| Metric | Value |
|--------|-------|
| Total cases | 6 |
| Errors | 0 |
| Refusals | 0 |
| Safety flags | 0 |
| Buckets covered | short_prompt, long_prompt, repetition_trap, stop_trap, code_like, safety_probe |

## Comparison with toy model

The toy RoPE MHA model (8.4M params) was used during development. NanoLlama numbers are the production benchmarks.

| Metric | Toy MHA (8.4M) | NanoLlama 8L (127.6M) | Ratio |
|--------|---------------|----------------------|-------|
| Parameters | 8.4M | 127.6M | 15.2x |
| KV cache decode speedup | ~5x | 7.5x | Larger model benefits more from cache |
| TTFT @ ctx=256 B=1 | ~17ms | 100ms | 5.9x (scales with param count) |
| Decode TPS (fp32) | ~254 | 56 | 4.5x slower (expected from 15x params) |
| fp32 memory | 34 MB | 511 MB | 15.1x (linear in params) |
| int8 memory | 11 MB | 155 MB | 14.3x |
| int8 decode speedup vs fp32 | 0.65x (slower) | 1.27x (faster) | int8 helps larger models more |
| int8 PPL drift | +0.3% | +6.0% | Larger model more sensitive to quantization |

Key insight: int8 dynamic quantization **hurts** the tiny model (decode gets slower) but **helps** NanoLlama (1.27x faster). At 8.4M params the compute overhead of quantized ops dominates; at 127.6M params the memory bandwidth savings from smaller weights dominate.

## Experiment directories

- `experiments/nanollama_serving_bench/` — cache.json, recompute.json, cache_grid.json (9 rows), recompute_grid.json (9 rows), cache_equivalence_receipts.json, raw/
- `experiments/nanollama_serving_quant/` — bench_fp32.json, bench_int8.json, ppl_fp32.json, ppl_int8.json, precision_matrix.json, quant_results.json, quant_report.md, recommendation.json
- `experiments/nanollama_serving_reports/` — eval_report.md, prompt_suite_*.json/jsonl, prompt_manifest.json

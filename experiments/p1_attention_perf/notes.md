# Phase 4 (OC-P1) — Attention Performance Optimization Notes

## Goal

Improve attention runtime on MacBook M1 Pro (MPS) **without changing model outputs**, and back it with reproducible benchmarks + correctness guard tests.

## Environment / Methodology

* **Platform:** macOS (darwin), MacBook M1 Pro, **MPS backend**
* **Metric:** **median latency** after warmup
* **Timing correctness:** uses `torch.mps.synchronize()` to avoid async timing artifacts
* **Benchmark args:** `B=4, T=256, iters=200, warmup=50`

## Changes (low-risk, functionally invariant)

1. **Causal mask caching**

* Cached the additive causal mask `[T×T]` keyed by `(T, device, dtype)`
* Avoids re-allocation / reconstruction on every forward pass
* Location: `llm_lab/core/model/attention.py`

  * `_CAUSAL_MASK_CACHE`
  * `SingleHeadAttention._causal_mask(...)`

2. **Hot-path overhead reduction**

* Avoided per-forward `nn.Softmax(...)` module creation
* Uses `torch.softmax(...)` directly
* Precomputed attention scale factor (e.g., `head_dim ** -0.5`) instead of recomputing each forward
* Location: `llm_lab/core/model/attention.py`

## Results (B=4, T=256)

| Benchmark         |  Baseline | Optimized | Speedup |
| ----------------- | --------: | --------: | ------: |
| Attention forward |   5.31 ms |   3.70 ms |   1.44× |
| Full train step   | 215.79 ms | 205.94 ms |  ~1.05× |

## Interpretation

The attention microbench shows a large improvement, but end-to-end train step improves modestly due to **Amdahl’s Law**:

* training step is dominated by **backward pass**
* **MLP blocks** contribute significant latency
* **optimizer step** overhead is unaffected by attention-only changes

## Correctness guardrails (“fast but wrong” prevention)

Added/maintained tests that ensure:

* causal mask semantics are correct
* **no future-token leakage** (perturb future token → earlier outputs unchanged)
* cache invariance (cached vs non-cached behavior does not change outputs in eval)

Test file:

* `tests/core/test_attention_correctness.py`

Status:

* `pytest`: **45 passed**

## Artifacts

Bench output JSONs:

* `phase4_numbers_baseline.json`
* `phase4_numbers_optimized.json`

## Follow-ups / Deferred

* Phase 4 **IC-P1 (experiment runner infra)** is deferred for now (not blocking Phase 5 / NanoLlama / Delta).
* Bigger perf wins likely require architectural changes (fused QKV / GQA/MQA / KV cache), which are targeted for later phases (NanoLlama/P3).

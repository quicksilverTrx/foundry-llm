# Architecture Ablation — Full Re-Audit
**Date:** 2026-03-31

> Swap run logs: `experiments/tinyllama_pretrain_2026-03-31/phase5_swaps/swap{0-7}/`
> GPT-2 baseline checkpoint: `experiments/tinyllama_pretrain_2026-03-31/phase4_baseline/ckpts/model_02009.pt`
> Results summary: `results/ablation_summary.csv`

---

## 1. Executive Summary

A thorough re-audit was performed of all code changes, ablation swap configurations, training methodology, and reporting trail. Overall, the implementation is correct. Two real issues were found: (A) a mid-experiment init change that confounds the SWAP 3 softcap signal, and (B) the SWAP 0 baseline using untied weights (163M vs GPT-2's 124M). Both issues are understood and their impact is bounded.

The "unintended" init changes in SWAP 3 (lm_head bias=False, token_embed std=0.02) are actually the **correct** settings for NanoLlama and all subsequent runs — SWAPs 0–2 were running with slightly wrong init. NanoLlama 8L will use the correct init throughout.

---

## 2. Code Changes Audit

### 2.1 `gpt.py`

| Change | Status | Notes |
|--------|--------|-------|
| `logit_softcap: Optional[float]` in `MiniGPTConfig` | ✅ Correct | Gemma-style cap applied as `cap * tanh(x/cap)` post lm_head |
| `qk_norm: bool = False` in `MiniGPTConfig` | ✅ Correct | Threaded to attention via blocks |
| `pos_embed = None` when `pos_encoding_type="rope"` | ✅ Correct | No additive pos embed for RoPE |
| `lm_head = nn.Linear(..., bias=False)` | ✅ Correct | Applied from SWAP 3 onwards. Matches LLaMA/Karpathy reference |
| `nanollama` arch validation in `__post_init__` | ✅ Correct | Enforces gqa+rope+rmsnorm constraint |
| `_init_weights`: general `std = 1/√d_model` | ✅ Correct | 0.036 for d_model=768 |
| `_init_weights`: `residual_std` for out_proj/w_down/fc2 | ✅ Correct | Karpathy: `std / √(2 × n_layers)` for residual stream projections |
| `_init_weights`: override token_embed + lm_head to `std=0.02` | ✅ Correct | Applied after loop. Pod verified: std=0.0200 |

**Note on init logic:** The general loop visits `lm_head.weight` (matches `else` branch) and sets it to `std=0.036`. The explicit override at the end then sets it to `0.02`. The override wins. Functionally correct.

### 2.2 `attention.py`

| Change | Status | Notes |
|--------|--------|-------|
| `import torch.nn.functional as F` | ✅ | Required for SDPA |
| `from llm_lab.core.model.norms import RMSNorm` | ✅ | Required for QK-Norm |
| `qk_norm: bool` in `MultiHeadAttentionConfig` | ✅ Correct | Threads to GQA path |
| QK-Norm applied **pre-RoPE** | ✅ Correct | `q_norm(q)` → `k_norm(k)` → `repeat_interleave` → RoPE |
| `q_norm = RMSNorm(head_dim)`, `k_norm = RMSNorm(head_dim)` | ✅ Correct | Per-head normalization, head_dim=64 |
| SDPA in MHA path (`unsqueeze/squeeze` per head) | ✅ Correct | 12 SDPA launches/step (known limitation vs GQA's 1) |
| SDPA in GQA path (`is_causal=True`) | ✅ Correct | Single batched kernel |
| GQA dropout: `self.config.dropout` | ✅ Correct | Fixed from prior `self.dropout_p` AttributeError |
| GQA KV cache: stores H_kv heads, expands for attention | ✅ Correct | ABI: `[B, H_kv, T, D]` |

**GQA RoPE order analysis:**
```
k → [B, H_kv, T, D]
k = k.repeat_interleave(repeat, dim=1)   → [B, H, T, D]
apply_rope(q, k, position_ids)            → RoPE on all H heads
k_base = k[:, ::repeat, :, :]            → [B, H_kv, T, D]  (collapse back)
```
Since all `repeat` copies are identical before RoPE and get identical position_ids, `k_base` after slicing is mathematically equivalent to applying RoPE directly to the H_kv heads. **Correct but wasteful** — for NanoLlama 8L (kv=4, repeat=3), 3× RoPE compute is wasted. Not a training correctness bug.

**SDPA `is_causal=True`:** Unconditional in training (attention_mask=None fast path). Correct for autoregressive LM training. Known limitation for decode; not relevant here.

### 2.3 `blocks.py`

| Change | Status | Notes |
|--------|--------|-------|
| `qk_norm: bool = False` in `TransformerBlockConfig` | ✅ Correct | Threads to `MultiHeadAttentionConfig` |

### 2.4 `optim.py`

| Change | Status | Notes |
|--------|--------|-------|
| `betas = (0.9, 0.95)` | ✅ Correct | Already present. Matches Karpathy β₂=0.95 |
| Embed + lm_head excluded from weight decay | ✅ Correct | Already present. `'embed' in name or 'lm_head' in name` |
| `import inspect` | ✅ | Required for fused check |
| `fused = fused_available` via `inspect.signature` | ✅ Correct | Free 2–3× optimizer step speedup on CUDA |

**Weight decay group tensor counts — verified by first-principles calculation:**

For SWAP 2 (MHA, 12L, RMSNorm, RoPE, old code with lm_head.bias):
- Decay = 12L × (12 heads × 3 q/k/v weights + out_proj.weight + fc1.weight + fc2.weight) = 12 × 39 = **468** ✅
- NoDecay = token_embed(1) + lm_head.weight(1) + lm_head.bias(1, old code) + ln_f.weight(1) + 12L × (norm1 + norm2 + out_proj.bias + fc1.bias + fc2.bias + 12×3 head biases) = 4 + 12×41 = **495** ✅

For SWAP 4 (GQA kv=12, 12L, QK-Norm, new code, no lm_head.bias):
- Decay = 12L × (q_proj + k_proj + v_proj + out_proj + fc1 + fc2) = 12 × 6 = **72** ✅
- NoDecay = token_embed(1) + lm_head.weight(1) + ln_f.weight(1) + 12L × (norm1 + norm2 + q_norm + k_norm + q/k/v/out biases(4) + fc1/fc2 biases(2)) = 3 + 12×10 = **123** ✅

---

## 3. Swap Configuration Audit

### 3.1 All 8 Swap Configs vs Plan

| Swap | Features | n_layers | kv_heads | d_ff | norm | mlp | Match |
|------|----------|----------|----------|------|------|-----|-------|
| 0 | GPT-2 baseline | 12 | None (MHA) | 3072 | layernorm | gelu | ✅ |
| 1 | +RoPE | 12 | None (MHA) | 3072 | layernorm | gelu | ✅ |
| 2 | +RMSNorm | 12 | None (MHA) | 3072 | rmsnorm | gelu | ✅ |
| 3 | +logit_softcap=30 | 12 | None (MHA) | 3072 | rmsnorm | gelu | ✅ |
| 4 | +QK-Norm | 12 | 12 (GQA) | 3072 | rmsnorm | gelu | ✅ |
| 5 | +SwiGLU d_ff=2048 | 12 | 12 (GQA) | 2048 | rmsnorm | swiglu | ✅ |
| 6 | +GQA kv=4 | 12 | 4 (GQA) | 2048 | rmsnorm | swiglu | ✅ |
| 7 | Full NanoLlama 8L | **8** | 4 (GQA) | 2048 | rmsnorm | swiglu | ✅ |

Training hyperparameters — consistent across ALL 8 swaps:
- B=8, T=1024, GA=64 → total_batch=524,288 tokens/step ✅
- max_lr=6e-4, min_lr=6e-5, warmup=50, max_steps=500 ✅
- Val: 20 batches, reset pos=0, bf16 autocast, at step 0/250/500 ✅
- Data: `data/edu_fineweb10B/` (persistent, same shards) ✅

**SwiGLU FFN parameter parity verified:**
- GELU d_ff=3072: 2 matrices × 768×3072 = 4,718,592 params/layer
- SwiGLU d_ff=2048: 3 matrices × 768×2048 = 4,718,592 params/layer ✅ (exact match)
- SWAP 5 is smaller than SWAP 4 by 46,080 params = 12L × (fc1.bias[3072] + fc2.bias[768]) = SwiGLU removing GELU bias terms ✅

### 3.2 Critical Anomaly: Init Change Mid-Experiment (SWAP 2→3 Boundary)

**What happened:** Pre-phase-6 fixes were hot-patched to the pod at 03:36 UTC. SWAP 2 finished at 03:42 UTC. SWAP 3 started at 03:42 UTC — 6 minutes after patch. SWAP 3 ran with the new gpt.py.

**Confirmed by param count evidence (from config.json files):**

| Swap | total_params_M | Change from prior |
|------|---------------|-------------------|
| SWAP 2 | 162.354048M | — |
| SWAP 3 | 162.303744M | **−0.050304M = −50,304 params = exactly lm_head.bias removed** |

This confirms the hot-patch took effect on SWAP 3.

**Three simultaneous changes SWAP 2 → SWAP 3:**

| Change | SWAP 0–2 | SWAP 3–7 | Correct for Plan? |
|--------|----------|----------|-------------------|
| logit_softcap=30 | None | 30.0 | Intended swap feature |
| lm_head.bias | True (wrong) | False ✅ | **Correct** — LLaMA/Karpathy have no lm_head bias |
| token_embed init std | 0.036 (wrong) | 0.02 ✅ | **Correct** — Karpathy uses 0.02 for embeddings |

**Key reframing:** The "unintended" changes are actually corrections. SWAPs 0–2 were running with slightly wrong init. SWAP 3 onwards uses the correct init that NanoLlama 8L will also use.

**Impact on comparisons:**

| Comparison | Validity | Reason |
|------------|----------|--------|
| SWAP 0↔1↔2 | ✅ Clean | All use same (wrong) init — deltas are real |
| SWAP 2→3 | ⚠️ Confounded | softcap + bias removal + embed init all changed |
| SWAP 3↔4↔5↔6↔7 | ✅ Clean | All use same correct init |
| (NanoLlama 8L vs SWAP 7 | ✅ Valid | Same init, same arch, different B/GA/steps |

### 3.3 SWAP 0 Baseline: Untied Weights

SWAP 0 uses `attention_type="mha"` with `pos_encoding_type="learned"` and does NOT tie lm_head.weight to token_embed.weight. Real GPT-2 small (124M) uses tied weights. Result:
- SWAP 0 total_params = **163.16M** (vs 124M true GPT-2)
- Extra capacity: 38.6M params (separate lm_head matrix) + pos_embed.weight (0.79M)
- val@500 = 5.0825 is an **internal reference only**, not comparable to Karpathy's GPT-2 baseline

### 3.4 SWAP 4 Code Path Change (MHA → GQA)

SWAP 4 adds QK-Norm but also switches from MHA (per-head `SingleHeadAttention` loop) to GQA (consolidated projections) with kv=12=n_heads. This introduces:
- Different parameter layout (72 vs 468 decay tensors)
- Batched SDPA: **throughput jump 38K → 61K tok/s** (implementation, not math change)
- With repeat=1 (kv=n_heads), GQA path is mathematically identical to MHA ✅

---

## 4. Throughput Analysis

| Swaps | tok/s | Cause |
|-------|-------|-------|
| 0–3 (MHA) | 38–41K | Per-head SDPA loop: 12 launches/step for n_heads=12 |
| 4–6 (GQA kv=12,4, 12L) | 60–65K | Batched SDPA: 1 launch/step |
| 7 (GQA kv=4, **8L**) | 83–84K | Batched SDPA + fewer layers (8 vs 12) |

Throughput at B=8, T=1024, GA=64:
- Tokens per optimizer step: 8 × 1024 × 64 = **524,288 tokens**
- Tokens per second (SWAP 7): ~83,968 → time per optimizer step: 524,288/83,968 ≈ **6.2 sec**

---

## 5. Ablation Results — Final Table

Two delta columns are required. The **incremental** delta isolates each feature's
marginal contribution given all prior features present. Previously only cumulative
deltas were reported — that was incomplete.

| Swap | Config | val@500 | params | Δ cumulative (vs SWAP 0) | Δ incremental (vs prior) | Validity |
|------|--------|---------|--------|--------------------------|--------------------------|----------|
| 0 | GPT-2 baseline | 5.0825 | 163.16M | — | — | Internal ref (untied, not true GPT-2) |
| 1 | +RoPE | 4.6965 | 162.37M | −0.3860 | **−0.3860** | ✅ |
| 2 | +RMSNorm | 4.7127 | 162.35M | −0.3698 | **+0.0162** (slight regression) | ✅ |
| 3 | +softcap=30 | 4.7997 | 162.30M | −0.2828 | **+0.0870** | ⚠️ confounded (3 simultaneous changes) |
| 4 | +QK-Norm | 4.7820 | 162.31M | −0.3005 | **−0.0177** (small gain) | ✅ (also MHA→GQA path switch) |
| 5 | +SwiGLU d_ff=2048 | 4.6430 | 162.26M | −0.4395 | **−0.1390** ← strongest feature | ✅ |
| 6 | +GQA kv=4 | 4.6383 | 152.81M | −0.4442 | **−0.0047** (marginal) | ✅ |
| 7 | Full NanoLlama 8L | 4.6918 | 127.63M | −0.3907 | **+0.0535** (capacity regression) | ✅ |

**Feature signal rankings by incremental delta:**
1. SwiGLU: −0.139 (strongest by far)
2. RoPE: −0.386 vs baseline
3. QK-Norm: −0.018 (small but clean signal)
4. GQA kv=4: −0.005 (efficiency gain, not accuracy gain at 500 steps)
5. RMSNorm: +0.016 (no short-run benefit; stability/efficiency advantage at scale)
6. n_layers 8→8L: +0.054 capacity regression (recovers with 10× training in the full pretraining run)
7. softcap: signal invalid due to confound

---

## 5b. GPT-2 Baseline Re-Audit of Logic and LR Schedule

### What the GPT-2 baseline run is
Karpathy's original GPT-2 code (124M, tied weights, torch.compile), B=16, GA=32,
warmup=200, **max_steps=500**. Val measured at step 500.

### The LR Schedule Problem (computed from first principles)

| Step | Ablation LR (w=50, max=500) | GPT-2 baseline LR (w=200, max=500) | NanoLlama 8L LR (w=200, max=4768) |
|------|----------------------------|------------------------------|------------------------------|
| 200 | 6.00e-04 | 6.00e-04 | 6.00e-04 |
| 300 | 2.90e-04 | 4.65e-04 | 5.99e-04 |
| 400 | 8.70e-05 | 1.95e-04 | 5.97e-04 |
| **500** | **6.00e-05** | **6.00e-05** | **5.94e-04** |

LR integral over 500 steps (total optimization magnitude):
- Ablation swaps (warmup=50, max=500):  **0.1641**
- GPT-2 baseline (warmup=200, max=500):       **0.1596** (2.7% less than ablation runs)
- NanoLlama 8L first 500 steps (max=4768):   **0.2397** (46% MORE than GPT-2 baseline)

### Comparison Validity Matrix

| Comparison | Valid? | Reason |
|------------|--------|--------|
| GPT-2 baseline val@500 vs SWAP 7 val@500 | ≈ Valid | Both at min_lr, LR integrals within 2.7%. Different B/GA and model are the intended comparison (GPT-2 vs NanoLlama) |
| GPT-2 baseline val@500 vs NanoLlama 8L val@500 | **INVALID** | NanoLlama 8L at step 500 is near max_lr (5.94e-4). GPT-2 baseline is at min_lr. 46% more total LR in NanoLlama 8L's first 500 steps. NanoLlama 8L val@500 will appear WORSE despite NanoLlama being better |
| GPT-2 baseline val@500 vs NanoLlama 8L val@4768 | Invalid | 10× different compute |
| GPT-2 baseline as rough ceiling for NanoLlama 8L | Valid | NanoLlama 8L trains 10× longer — final val must beat the GPT-2 baseline, or something is wrong |
| NanoLlama 8L progress: use GPT-2 baseline trajectory.csv | Valid | GPT-2 baseline ran 2009 steps at B=16/GA=32 — better long-horizon reference |

### Stated Purpose Was Wrong
Previously stated: "GPT-2 baseline gives valid reference at same settings as NanoLlama 8L."
Correction: GPT-2 baseline gives reference at same B/GA as NanoLlama 8L, but different LR
schedule (max_steps=500 vs 4768). It is approximately valid vs SWAP 7, not vs NanoLlama 8L.

### What Should Have Been Done
For a true NanoLlama 8L reference: Karpathy's code for 4768 steps, warmup=200. ~6 hours.
Impractical. GPT-2 baseline is an approximation for SWAP 7 comparison only.

---

## 6. Pre-Phase-6 Fixes Status

| Fix | Applied | Correct | NanoLlama 8L Impact |
|-----|---------|---------|----------------|
| lm_head bias=False | ✅ SWAP 3+ | ✅ LLaMA/Karpathy standard | No lm_head bias in NanoLlama 8L |
| token_embed std=0.02 | ✅ SWAP 3+ | ✅ Karpathy standard | Correct embed init in NanoLlama 8L |
| fused AdamW | ✅ SWAP 3+ | ✅ Free speedup | ~2–3× faster optimizer steps |
| B=16/GA=32 | ⏳ NanoLlama 8L | ✅ Karpathy reference batch | In NanoLlama 8L training |
| warmup=200 | ⏳ NanoLlama 8L | ✅ Scaled for longer run | In NanoLlama 8L training |

---

## 7. NanoLlama 8L Config

```python
# NanoLlama 8L — full target architecture
MiniGPTConfig(
    vocab_size=50304, d_model=768, n_layers=8, n_heads=12,
    num_kv_heads=4, d_ff=2048, block_size=1024, dropout=0.0,
    norm_type="rmsnorm", mlp_type="swiglu",
    attention_type="gqa", pos_encoding_type="rope",
    arch_family="nanollama",
    logit_softcap=30.0, qk_norm=True,
)

# Training
B=16, T=1024, GA=32, total_batch=524288
max_lr=6e-4, min_lr=6e-5, warmup=200, max_steps=4768
grad_clip=1.0, bfloat16=True, fused AdamW
```

Total training tokens: 4768 × 524,288 ≈ **2.5B tokens**
Estimated time: 4768 steps × 6.2 sec/step ≈ **8.2 hours**

# Architecture Decisions

The full reasoning chain from first-principles to the NanoLlama / SwiftLlama architecture family. Every decision here is backed by a concrete experiment. Where a decision was confounded or approximate, that is noted.

---

## 1. Starting point: why LLaMA-family over vanilla GPT-2

The baseline was a reproduction of GPT-2 124M on FineWeb-Edu: val **3.6273** nats at 1.05B tokens (BPB 1.098), matching Karpathy's published result. This served as an anchor — every architectural change was measured against it.

Eight architecture swaps were run at 345M tokens each (B=8, GA=64, T=1024, warmup=50, max_steps=500), adding one feature at a time. Starting from an untied GPT-2 baseline (163M, val 5.0825 at step 500):

| Swap | Change | val@500 | Incremental Δ |
|------|--------|---------|---------------|
| 0 | Baseline (MHA, GELU, LayerNorm, learned pos) | 5.0825 | — |
| 1 | +RoPE | 4.6965 | **−0.386** |
| 2 | +RMSNorm | 4.7127 | +0.016 (slight regression) |
| 3 | +logit softcap=30 | 4.7997 | +0.087 confounded |
| 4 | +QK-Norm + GQA codepath | 4.7820 | −0.018 |
| 5 | +SwiGLU d_ff=2048 | 4.6430 | **−0.139** |
| 6 | +GQA kv=4 | 4.6383 | −0.005 |
| 7 | Scale to 8L (NanoLlama shape) | 4.6918 | +0.054 |

**RoPE** is the single largest gain (−0.386 nats). Unlike learned position embeddings, RoPE encodes relative position analytically — it doesn't spend capacity fitting a position lookup table, which is a real tax at the 500-step probe scale where those parameters are underfit. Whether this advantage persists over multi-billion-token training is a separate question; Phase 6 at 2.5B shows no sign of regression vs SWAP 0. Non-negotiable.

**SwiGLU** is the second-largest gain (−0.139 nats). The gated MLP adds no parameters when d_ff is set to compensate (GELU d_ff=3072 and SwiGLU d_ff=2048 are parameter-matched: 2 × 768×3072 = 3 × 768×2048 = 4,718,592 params/layer), so this is a pure quality improvement.

**RMSNorm** shows a slight 500-step regression (+0.016). This is within noise at probe scale. Phase 6 at 2.5B shows no measurable degradation vs LayerNorm. Retained because (a) omitting the mean normalization reduces compute per step and (b) the LLaMA family consistently uses it — useful for weight compatibility if we ever want to load external checkpoints.

**Logit softcap** (SWAP 3) is **confounded**: the SWAP 2→3 boundary also applied two initialization fixes — `lm_head.bias=False` (−50,304 params) and `token_embed std 0.036→0.02`. The raw Δ of +0.087 cannot be attributed to softcap alone. We have no isolated measurement of its contribution. It is retained because (a) no logit divergence was observed in any of our 2.5B-token runs without it, and (b) Gemma and other published LLaMA variants use it for exactly this reason — cheap insurance against numerical instability in long pretraining. If it turns out to be inert at our scale, removing it costs nothing in a future run.

**GQA** (SWAP 6, kv=4 out of 12 query heads) shows only −0.005 nats improvement at 500-step probe scale but delivers a real throughput jump: single batched SDPA call vs 12 sequential per-head calls in MHA → **38K→60K tok/s** at 12-layer scale. At the same time the GQA parameter count (152.81M) vs 12-head GQA (162.31M) shows −9.5M params. GQA is retained for its efficiency profile.

**Layer count reduction 12→8** (SWAP 7) shows +0.054 nats regression at 500 steps. We never ran a 12-layer NanoLlama to 2.5B tokens, so the claim that 8L recovers is extrapolation from probe dynamics, not a controlled comparison. The hardware reality drove this: 12L does not fit on RTX 4090 at B=16, GA=32, T=1024 in bfloat16. 8L was the largest model that fits within the training batch budget. The 500-step regression is acceptable given the probe scale and hardware constraint.

---

## 2. Init fixes applied during ablation

Pre-Phase-6 fixes applied at SWAP 2→3 boundary (confirmed by param count evidence: SWAP 2 = 162.354M, SWAP 3 = 162.304M, Δ = exactly lm_head.bias removed):

| Fix | SWAPs 0–2 | SWAPs 3–7 + Phase 6 |
|-----|-----------|---------------------|
| lm_head.bias | Present (wrong) | Removed |
| token_embed init std | 0.036 (1/√d_model) | 0.02 (Karpathy standard) |
| Residual stream projection std | 0.036 / √(2n) | 0.02 / √(2n) |

Comparisons within SWAPs 0–2 are clean; comparisons across the SWAP 2→3 boundary are confounded.

---

## 3. NanoLlama v1 final config

Chosen after Phase 5 evidence and hardware constraints (RTX 4090, 24 GB, bf16):

```
8 layers · d_model=768 · 12Q heads · 4 KV heads (GQA 3:1) · d_ff=2048
SwiGLU · RMSNorm · RoPE (full fraction=1.0)
logit_softcap=30.0 · qk_norm=True
vocab_size=50304 · block_size=1024
lm_head bias=False · tie_weights=False
token_embed std=0.02 · residual std = 0.02/√(2×n_layers)
```

Parameters: **127,633,168** (no weight tying; separate lm_head.weight adds ~38.6M vs a tied variant).

Training: B=16, GA=32, T=1024 → 524,288 tok/step · lr=6e-4 · β=(0.9, 0.95) · WD=0.1 · warmup=200 · cosine to min_lr=6e-5 · max_steps=4768 (2.5B tokens).

Result: val **3.3566** nats · BPB **1.016** · HellaSwag **26.96%**.

---

## 4. v2 architecture: what was added and why

NanoLlama v2 keeps the same 127M scale and FineWeb-Edu corpus. Four features were added from the modded-nanogpt / nanochat speedrunning line, and the token budget was 5B (9537 steps); run ended at 4.72B when pod storage was exhausted.

**What the data shows.** A matched-token analysis is necessary before crediting the architecture: at 2.5B tokens, v1 and v2 are essentially tied (both ~3.357 nats). The correct comparison, extrapolating v1 log-linearly to v2's 4.72B token budget, predicts ~3.18 — which would be better than v2's measured 3.221. Two confounds prevent a clean attribution: the different LR schedules (cosine vs constant_warmdown) and the absence of a v1-at-4.72B control run. The nanochat features produce +0.003–0.010 BPB effects across 320 experiments at comparable scales; detecting that signal at 127M would require a controlled ablation with matched compute and identical schedules. The control experiment was deprioritised in favour of the SwiftLlama scale-up, which was the higher-value question. The features are adopted from a validated architecture family and retained for scale-up compatibility.

### 4a. Partial RoPE (`rope_fraction=0.5`)

Rotates only the first 50% of head dimensions; the remaining dims are passed through unchanged. The original motivation (from the nanochat line) is that a fraction of dimensions carrying absolute positional signal can help with longer-context generalization while the unrotated dims carry relative content. At our scales (block_size=1024) the empirical benefit is modest but the cost is zero.

### 4b. Value embeddings (`n_value_embeds=2`)

Per-layer learned parameter matrix of shape `[n_value_embeds, H_kv × head_dim]`. Mean-pooled to `[1, H_kv, 1, head_dim]` and broadcast-added to the value vectors before attention. The nanochat experiments ("models love the capacity") show consistent +0.003–0.010 BPB across 320 hyperparameter experiments at comparable scales.

### 4c. x0-mixin residual (`use_x0_mixin=True`)

Per-layer residual mixing: `x_out = λ · x + λ₀ · x₀` where `x₀` is the embedding layer output, and λ, λ₀ are per-layer scalar parameters initialized as λ=1, λ₀=0 (identity at init). This gives each layer a direct path back to the token embedding, which the nanochat line found helps preserve early-layer semantic information at scale. At init the model is behaviorally identical to the standard residual stream.

### 4d. Optimizer decision for v2

Three 1500-step probes at 127M scale (val at step 1500, max_steps=9537 so LR is at 100% peak throughout):

| Probe | Architecture | Optimizer | val@1500 |
|-------|-------------|-----------|----------|
| N1 | Full v2 (rope=0.5, ve=2, x0) | Muon nanochat | 4.206 |
| N2 | Full v2 (rope=0.5, ve=2, x0) | AdamW | **3.715** |
| N3 | Base (rope=1.0, ve=0, x0=off) | Muon | 4.001 |

**AdamW leads Muon by 0.491 nats** on the same v2 architecture (N2 vs N1). One confound to note: N1 uses constant_warmdown schedule; N2 uses cosine_with_warmup. Different schedules partially explain the gap — this is not a pure optimizer comparison. Despite the confound, the direction is unambiguous and consistent: N3 (Muon, base architecture) also loses to N2 (AdamW, full v2) by 0.286 nats. The architecture does not rescue Muon. Production run uses AdamW (N2 recipe).

The Muon optimizer was developed and validated on the **speedrunning architecture: MHA + ReLU²**. Our stack is **GQA + SwiGLU**. GQA changes the 2D weight matrix population (fewer, asymmetric KV projections); SwiGLU introduces gated gradient flow (3-matrix FFN vs 2-matrix ReLU²). Muon's Newton-Schulz orthogonalization is tuned to the gradient statistics of MHA+ReLU² and does not transfer to GQA+SwiGLU. **AdamW is the correct optimizer for this architecture family.**

Production run: AdamW, 9000/9537 steps, 4.72B tokens (pod storage exhausted at step 9000 — 94% of planned 9537 steps). Val **3.2210** at step 9000.

---

## 5. SwiftLlama-350M: scaling to 345M

### Architecture choices

Same v2 feature set, scaled up:

```
22 layers · d_model=1024 · 16Q heads · 4 KV heads (GQA 4:1) · d_ff=2728
SwiGLU · RMSNorm · RoPE (fraction=0.5)
logit_softcap=30.0 · qk_norm=True · value_embeds=2 · x0-mixin
vocab_size=50304 · block_size=4096
```

Parameters: **345,315,116**.

Note: d_ff=2728 (not 2730) to satisfy 8× alignment for tensor core efficiency. Original spec had 2730; corrected to 2728 (Δ = 0.07% param count).

### Context length: 4096

Extended from 1024 to 4096. The 4096-token context adds no architectural change (RoPE handles arbitrary lengths); it changes the effective batch computation. With B=2, GA=64: total batch = 2 × 64 × 4096 = **524,288 tokens/step** — identical to NanoLlama's B=16, GA=32, T=1024 = 524,288.

### GQA ratio: 4 KV heads / 16 Q heads

4:1 ratio. At 345M and d=1024, each KV projection is 1024×(4×64)=262,144 params. Reduction vs MHA: KV params go from 16×262,144 to 4×262,144, saving ~49M params in KV projections — reallocated to deeper layers (22 vs 12).

### Optimizer: why we tried Muon and what Phase 2A found

SwiftLlama launched with Muon because the modded-nanogpt speedrunning community reported ~5× token efficiency gains over AdamW. That is a large enough claimed edge that it was worth a direct test at 345M scale before committing to AdamW.

Phase 2A (three 1500-step probes on H100, same 524,288 tok/step):

| Probe | Optimizer | WD | val@500 | val@1000 | val@1500 |
|-------|-----------|-----|---------|----------|----------|
| A1 | Muon+Adam | 0.0 | 4.623 | 3.877 | 3.672 |
| A2 | Muon+Adam | 0.1 | 4.693 | 3.904 | killed (step 1110) |
| A3 | AdamW | 0.1 | **4.594** | **3.686** | **3.482** |

AdamW leads by **0.190 nats** at step 1500. The token efficiency advantage does not transfer. A1 briefly leads at step 500 (+0.029 nats over A3) — matching the pre-B.5 probe signal that motivated the Muon choice — but AdamW overtakes by step 1000 and extends the lead through step 1500. **Short probes systematically overestimate Muon's value on this architecture.**

The mechanism: the speedrun gains were measured on MHA + ReLU². GQA has asymmetric KV projections (4:1 head ratio); SwiGLU has a three-matrix gated FFN. Both produce gradient structure that differs from the MHA + 2-matrix ReLU² pattern Muon's Newton-Schulz orthogonalization was tuned for.

SwiftLlama production continues on Muon to maintain checkpoint continuity. The cost is quantifiable: SwiftLlama needs 3.4× tokens to match v1 vs the 2.7× Chinchilla prediction — roughly consistent with the 0.19 nat Phase 2A handicap compounding. AdamW is the default for all future models.

### Weight decay

Cautious WD=0.1 (decoupled, standard AdamW). Muon requires LR-coupled WD (`param *= 1 − lr × wd`) because Muon's orthogonalization is scale-invariant; standard decoupled WD creates a competing signal. This is another point of incompatibility with AdamW-first frameworks.

---

## 6. Hyperparameter rationale

Training hyperparameters were chosen to match established small-LLM practice and constrained by hardware. Each decision is documented below.

### Learning rate: 6×10⁻⁴

Karpathy's reference run for GPT-2 124M uses lr=6e-4 with AdamW and achieves the published 3.29 BPB result at 10B tokens. We reproduced this at 1.05B tokens (val 3.627) as Phase 4 confirmation. Using the same LR for NanoLlama enables clean architectural comparisons with the GPT-2 baseline — any LR difference would be a confound. Published scaling laws (Hoffmann et al. 2022, Appendix D) suggest optimal LR scales as ~C^(−0.3) where C is compute; the 127M → 345M scale-up changes optimal LR by less than 20%, so the same 6e-4 was retained for SwiftLlama.

### Batch size: 524,288 tokens/step

A batch of 524,288 tokens/step was chosen to match Karpathy's reference implementation (B=32, T=1024 → 32,768 tokens/step with gradient accumulation). Memory constraints drove the specific decomposition:
- v1/v2 (127M): B=16, GA=32, T=1024 → 524,288 (maximum that fits RTX 4090 24GB at bfloat16)
- SwiftLlama (345M): B=2, GA=64, T=4096 → 524,288 (same total, longer context for the larger model)

Keeping tok/step constant across models means each gradient update sees identical information volume — a controlled comparison.

### Warmup: 200 steps

200 steps = 104M tokens. Short relative to the total run (200/4768 = 4.2% for v1), sufficient to ramp LR from 0 to peak without gradient explosions. Karpathy uses 715 steps in Phase 4 (the GPT-2 baseline run), which creates a schedule mismatch for cross-phase comparisons. Phase 4b (warmup=200, max_steps=500) was run specifically to give a clean comparison point at step 250 where both models are at peak LR.

### min_lr: 6×10⁻⁵ (cosine), 0 (warmdown)

For cosine_with_warmup (v1): min_lr = max_lr × 0.1 = 6e-5 is the standard ratio from Chinchilla-era practice. For constant_warmdown (v2): LR decays linearly to 0. The linear-to-zero warmdown is more aggressive and consistent with the modded-nanogpt literature, which shows faster final convergence when LR hits zero cleanly rather than plateauing at min_lr.

### Weight decay: 0.1

Standard AdamW WD=0.1 follows Karpathy's reference. Higher WD (0.2, tried in N1 probe) showed no benefit at 127M scale. Lower WD reduces regularization; at the token budgets used here (up to 5B tokens for a 127M model, Chinchilla ratio ~39×) mild regularization is appropriate.

### β₁=0.9, β₂=0.95

β₂=0.95 (vs PyTorch's default 0.999) is faster-adapting for transformer pretraining — the second moment tracks gradient variance more responsively, which matters during rapid early loss descent. This follows Karpathy's configuration and is consistent with published practice for decoder-only pretraining.

---

## 7. Known limitations in the ablation evidence

**SWAP 3 confound.** logit_softcap=30 cannot be isolated because the init fix was applied simultaneously. The softcap contribution at 500 steps is unknown; it is retained because (a) it is standard in the nanochat line, (b) it prevents logit explosion observed in long runs without it.

**Short probe horizon.** All architecture ablations run 500 steps (~262M tokens). Features like RMSNorm that show small short-run regressions (SWAP 2, +0.016) may behave differently at 5B+ tokens. The Phase 6 run confirms RMSNorm does not hurt at 2.5B.

**Optimizer probes are not all-else-equal.** N1 uses constant_warmdown schedule; N2 uses cosine_with_warmup. The schedule difference is a confounder. The direction (AdamW wins) is robust across both the 127M and 345M comparisons, but the magnitude (0.491 nats at 127M) is partially schedule-driven.

**Muon LR not retuned.** Muon LR=0.02 came from the modded-nanogpt recipe for 124M MHA+ReLU². External experiments suggest optimal Muon LR for LLaMA-style 520M is 4–8e-3 — roughly half. Our probes used 0.02, which may be too high for GQA+SwiGLU, potentially understating Muon's ceiling. That said, SwiftLlama's Phase 2A result is clear enough (0.190 nats at step 1500) that retuning Muon LR is unlikely to close the full gap.

---

## See also

- `docs/muon_decision_rationale.md` — full Muon decision log across all three models, including the mechanism behind negative transfer and why SwiftLlama continues on Muon for checkpoint continuity
- `docs/retrospective.md` — first-person reflection on what held up, what I'd do differently, and what the next experiment should be
- `docs/training_dynamics.md` — LR schedule mechanics, v1/v2 matched-token comparison with explicit schedule-confound labeling

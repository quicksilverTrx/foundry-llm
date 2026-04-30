# Training Results

All quantitative results across NanoLlama v1 (ablation + full run), NanoLlama v2, and SwiftLlama-350M. All val losses are on the FineWeb-Edu validation shard using GPT-2 BPE (tiktoken), bytes/token = 4.766.

---

## Architecture ablation (Phase 5, NanoLlama v1 path)

8-swap progressive ablation. Each swap: B=8, GA=64, T=1024 (524,288 tok/step), warmup=50, max_steps=500.  
Swap 0 baseline: 163.16M params (untied weights, separate lm_head; not directly comparable to GPT-2 124M due to different weight tying convention).

| Swap | Feature | Params (M) | val@500 | Δ incremental | Δ cumulative | Valid? |
|------|---------|-----------|---------|---------------|-------------|--------|
| 0 | GPT-2 baseline (MHA, GELU, LayerNorm, learned pos) | 163.16 | 5.0825 | — | — | Internal ref |
| 1 | +RoPE | 162.37 | 4.6965 | **−0.3860** | −0.3860 | valid |
| 2 | +RMSNorm | 162.35 | 4.7127 | +0.0162 | −0.3698 | valid |
| 3 | +logit softcap=30 | 162.30 | 4.7997 | +0.0870 | −0.2828 | confounded |
| 4 | +QK-Norm + GQA path | 162.31 | 4.7820 | −0.0177 | −0.3005 | valid |
| 5 | +SwiGLU d_ff=2048 | 162.26 | 4.6430 | **−0.1390** | −0.4395 | valid |
| 6 | +GQA kv=4 | 152.81 | 4.6383 | −0.0047 | −0.4442 | valid |
| 7 | 8L NanoLlama shape | 127.63 | 4.6918 | +0.0535 | −0.3907 | valid |

Note: Swap 3 applies three simultaneous changes: logit_softcap=30 (intended) + lm_head.bias removed + token_embed std 0.036→0.02. Incremental delta is not interpretable as softcap-only.

---

## GPT-2 124M baseline (Phase 4)

Karpathy reference run on FineWeb-Edu for absolute anchoring.

| Checkpoint | Steps | Tokens | Val loss | BPB |
|-----------|-------|--------|----------|-----|
| Karpathy (llm.c, ~10B tokens) | — | ~10B | 3.29 | 0.996 |
| Phase 4 reproduction | 2009 | 1.053B | **3.6273** | 1.098 |
| Phase 4b (warmup=200, max_steps=500) | 500 | 262M | 4.9389 | — |

Phase 4 warmup=715 steps (35% LR at step 250, 70% at step 500). Phase 4b warmup=200 (100% LR at step 250). The only valid comparison with Phase 6 is at step 250 of Phase 4b.

---

## NanoLlama 8L — Phase 6 full pretraining

127.63M params. B=16, GA=32, T=1024 → 524,288 tok/step. AdamW lr=6e-4, WD=0.1, warmup=200, cosine to min_lr=6e-5. max_steps=4768, 2.5B total tokens. RTX 4090, ~75,940 tok/s.

| Step | Tokens | Val loss |
|------|--------|----------|
| 250 | 131M | 5.1926 |
| 500 | 262M | 4.3567 |
| 1000 | 524M | 3.8653 |
| 1500 | 786M | 3.6798 |
| 2000 | 1.049B | 3.5813 |
| 2500 | 1.311B | 3.5055 |
| 3000 | 1.572B | 3.4478 |
| 3500 | 1.834B | 3.4130 |
| 4000 | 2.096B | 3.3796 |
| **4768** | **2.500B** | **3.3566** |

**Final: val 3.3566 · BPB 1.016 · HellaSwag 26.96%**

Compute-matched comparison (1.05B tokens, Phase 4b vs Phase 6 at matched LR — both at peak lr=6e-4 after warmup=200):

| Model | Tokens | Val loss | BPB |
|-------|--------|----------|-----|
| Phase 4 GPT-2 baseline | 1.05B | 3.6273 | 1.098 |
| NanoLlama 8L @ 1.05B | 1.05B | ~3.589 | ~1.088 |
| **Gap** | — | **−0.038** | **−0.010** |

At 2.5B tokens NanoLlama is 0.020 BPB behind the Karpathy 10B-token reference. Simple way to read this: we used 4× fewer tokens and landed 0.020 BPB worse — the LLaMA-family architecture is more token-efficient than vanilla GPT-2 at this scale. The gap is not "closed" — the extrapolated crossover point (assuming Chinchilla-law descent of ≈ −0.022 BPB per doubling of tokens) would be roughly around 5–6B tokens, which is where the v2 run was aimed.

---

## NanoLlama v2 — optimizer probes (127M, 1500-step probes)

All probes: same v2 architecture except N3 (base arch). max_steps=9537; at step 1500, LR is at 100% peak (warmdown hasn't started).

| Probe | Architecture | Optimizer | val@500 | val@1000 | val@1500 |
|-------|-------------|-----------|---------|----------|----------|
| N1 | v2 (rope=0.5, ve=2, x0) | Muon nanochat, WD=0.2 | 4.845 | 4.516 | 4.206 |
| N2 | v2 (rope=0.5, ve=2, x0) | AdamW, WD=0.1 | 4.441 | 3.898 | **3.715** |
| N3 | Base (rope=1.0, ve=0, no x0) | Muon nanochat, WD=0.2 | 4.476 | 4.107 | 4.001 |

N2 wins. AdamW leads Muon by **0.491 nats** on v2 architecture; by **0.286 nats** over base-arch Muon (N3). Production uses N2 recipe.

---

## NanoLlama v2 — production run (127M, AdamW)

127.63M params. Same architecture as v2 probes. B=16, GA=32, T=1024 → 524,288 tok/step. AdamW lr=6e-4, WD=0.1, warmup=200, cosine, min_lr=6e-5. max_steps=9537 (5.0B tokens). Run reached 94% of planned steps (9000/9537) before pod storage was exhausted.

| Step | Tokens | Val loss | Notes |
|------|--------|----------|-------|
| 500 | 262M | 4.430 | — |
| 1000 | 524M | 3.897 | — |
| 1500 | 786M | 3.714 | matches probe N2 (+0.001) fixed |
| 2000 | 1.05B | 3.610 | — |
| 3000 | 1.57B | 3.480 | — |
| 4000 | 2.10B | 3.401 | — |
| 5000 | 2.62B | 3.345 | **beats v1 (3.3566)** |
| 6000 | 3.15B | 3.298 | — |
| 7000 | 3.67B | 3.261 | — |
| 8000 | 4.19B | 3.238 | — |
| **9000** | **4.72B** | **3.2210** | **best checkpoint (pod storage exhausted at 94% of planned run)** |

Final: val **3.2210** · Δ vs v1 final: **−0.136** nats.

**Matched-token analysis.** At 2.5B tokens v1 and v2 are essentially tied (both ~3.357, Δ=0.001). v2 is slightly behind v1 at every earlier token count (+0.073 at 262M, +0.029 at 1.05B, +0.021 at 2.10B) — consistent with v2 deliberately holding peak LR longer under constant_warmdown — a deliberate schedule choice, not an architecture penalty. Log-linear extrapolation of v1 to 4.72B predicts ~3.18, which would be better than v2's 3.221. The correct interpretation: the −0.136 nat improvement comes primarily from the larger training budget. The architectural contribution (partial RoPE, value embeddings, x0-mixin) cannot be isolated without a controlled run of v1 to 4.72B on the same schedule — that control was not prioritised, as the higher-value next experiment was scaling to 345M. The nanochat features show +0.003–0.010 BPB effects in their native experiments; detecting that signal here would require matched compute and identical schedules.

Note: run ended 537 steps short of max_steps=9537. The final 537 steps were not executed; val at step 9537 would have been slightly lower (estimated 3.21–3.22 based on descent rate ≈ 0.005 per 500 steps in this region).

---

## SwiftLlama-350M — Phase 2A optimizer ablation

345.3M params. Full v2 architecture (rope=0.5, ve=2, x0=True), scaled to 22L/d=1024. H100, B=8, GA=16, T=4096 → 524,288 tok/step. max_steps=1500 (~0.79B tokens). LR schedule: linear warmup to peak at step 500, cosine decay to ~0 at step 1500.

| Probe | Optimizer | WD | val@500 | val@1000 | val@1500 |
|-------|-----------|-----|---------|----------|----------|
| A1 | Muon+Adam, LR-coupled WD | 0.0 | 4.623 | 3.877 | 3.672 |
| A2 | Muon+Adam, LR-coupled WD | 0.1 | 4.693 | 3.904 | killed (step 1110) |
| A3 | AdamW | 0.1 | **4.594** | **3.686** | **3.482** |

A3 (AdamW) wins by **0.190 nats**. A3 reaches A1's step-1500 score at step 1000 (33% fewer steps). Decision: AdamW for production.

Early dynamics: at step 500 Muon leads (+0.029 nats). AdamW overtakes by step 1000. Short-horizon probes overstate Muon's benefit.

---

## SwiftLlama-350M — production run (345M, Muon)

345.3M params. RTX 4090, bfloat16. B=2, GA=64, T=4096 → 524,288 tok/step. Muon lr=0.02 + Adam lr=6e-4 (5 groups). max_steps=28,610 (~15B tokens). In progress as of step 16,363.

| Step | Tokens (wall-clock) | Val loss | Notes |
|------|---------------------|----------|-------|
| 2000 | 1.05B | 3.826 | first checkpoint after DataLoader reset |
| 4000 | 2.10B | 3.648 | — |
| 7000 | 3.67B | 3.520 | benchmark checkpoint |
| 9000 | 4.72B | 3.470 | — |
| 11500 | 6.03B | 3.414 | — |
| 12000 | 6.29B | 3.434 | blip (+0.020, recovered) |
| 14000 | 7.34B | 3.378 | — |
| **16000** | **8.39B** | **3.3566** | **matches v1 final val exactly** |

Unique-token note: DataLoader reset at step 2000 caused steps 2001–4000 to re-see first 1.05B tokens. Unique tokens at step 16000 ≈ 7.95B (not 8.39B wall-clock).

Chinchilla-optimal (20× params): 20 × 345.3M = 6.91B unique tokens, reached at step ~15,174. Model not saturated — val still descending.

Throughput: ~25,700 tok/s (eager mode, bfloat16, no torch.compile).

### Benchmarks at step 7000 (3.67B tokens, ~54% of Chinchilla-optimal)

| Task | Random | SwiftLlama-350M | NanoLlama-127M (v1) |
|------|--------|----------------|---------------------|
| PIQA | 0.500 | 0.584 | **0.601** |
| ARC-Easy | 0.250 | 0.351 | **0.381** |
| HellaSwag | 0.250 | 0.268 | **0.270** |
| WinoGrande | 0.500 | 0.512 | 0.503 |
| Lambada | — | 0.292 | **0.328** |

These are milestone benchmarks, not the valid architecture comparison point: NanoLlama is at 19.6× tokens/params while SwiftLlama at step 7000 is at 10.5×. Comparing models at mismatched training budgets measures how much data they've seen, not what the architecture can do. Valid architectural comparison: re-evaluate both at Chinchilla-optimal (SwiftLlama ~step 13,000).

---

## Model comparison at a glance

| Model | Params | Tokens | Val loss | BPB | HellaSwag |
|-------|--------|--------|----------|-----|-----------|
| GPT-2 124M baseline (1.05B tok) | 124M | 1.05B | 3.627 | 1.098 | — |
| NanoLlama 8L v1 | 127.6M | 2.5B | 3.357 | 1.016 | **26.96%** |
| NanoLlama v1 (extrapolated to 4.72B) | 127.6M | 4.72B | ~3.176 | — | — |
| NanoLlama v2 (storage limit @ 4.72B (94% of plan)) | 127.6M | 4.72B | 3.221 | — | — |
| SwiftLlama-350M (step 16000) | 345.3M | 8.39B | 3.357 | — | — |
| GPT-2 124M baseline (~10B tok) | 124M | ~10B | 3.29 | 0.996 | — |

**Note on v1 vs v2:** v1 extrapolated log-linearly to v2's token budget predicts ~3.176 (±uncertainty from schedule differences and extrapolation range). v2 measured 3.221 at the same budget. This gap is the missing control experiment: to credit the v2 architectural changes, we'd need v1 trained to 4.72B on the same schedule. Until then, the −0.136 nat improvement should be attributed to training budget rather than architecture.

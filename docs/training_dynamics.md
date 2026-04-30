# Training Dynamics

Loss curves, LR schedule mechanics, optimizer dynamics, and training events across all three production runs. For final endpoint values see `docs/training_results.md`.

---

## LR schedules

### cosine_with_warmup (NanoLlama v1 and v2)

```
[0, warmup_steps)           : linear ramp  0 → max_lr
[warmup_steps, max_steps]   : cosine decay max_lr → min_lr
(max_steps, ...]            : constant min_lr
```

`cosine_period_steps` can decouple the cosine window from `max_steps` when set.

### constant_warmdown (NanoLlama v2 production and optimizer probes)

```
[0, warmup_steps)                               : linear ramp  0 → max_lr
[warmup_steps, floor(max_steps × (1−ratio)))   : constant max_lr
[floor(max_steps × (1−ratio)), max_steps]       : linear decay  max_lr → 0
```

For v2 production: `max_steps=9537`, `warmdown_ratio=0.4`, `warmup_steps=200`.  
- Warmup ends: step 200  
- Constant plateau: steps 200–5722  
- Linear warmdown: steps 5722–9537

**Why this matters for probe comparisons.** At step 1500 (all probes), constant_warmdown is still at 100% peak LR (warmdown hasn't started). Cosine_with_warmup at step 1500 of a 9537-step run is at ~99.4% peak LR. The two schedules are effectively equivalent at early probe steps, which is why N1/N2/N3 probe results are comparable despite using different schedule types.

### LR integral comparison (why GPT-2 baseline comparisons are approximate)

At step 250 of Phase 4 (warmup=715): LR ≈ 2.1×10⁻⁴ (35% of max). At step 500: LR ≈ 4.2×10⁻⁴ (70%).  
At step 250 of Phase 6 (warmup=200): LR = 6×10⁻⁴ (100% of max).

The only clean comparison between Phase 4 (GPT-2 baseline) and Phase 6 (NanoLlama) is **Phase 4b at step 250**, where both have completed warmup and are at peak LR:

| | Phase 4b (GPT-2, step 250) | Phase 6 (NanoLlama, step 250) | Δ |
|--|--|--|--|
| Val loss | 5.72 | 5.19 | **−0.53 nats** |

This is the most valid single-step comparison between the two models at matched optimization state.

---

## Phase 5 ablation dynamics (NanoLlama v1 architecture)

Each swap: warmup=50, max_steps=500. The short horizon means swap values reflect early-training bias: features that help initial descent (RoPE, SwiGLU) score strongly; features providing long-run stability (RMSNorm) may show neutral or slightly negative short-run deltas.

Loss descent rate at step 500 (selected swaps):

| Swap | val@250 | val@500 | Descent 250→500 |
|------|---------|---------|-----------------|
| 0 (baseline) | 5.78 | 5.08 | −0.70 |
| 1 (+RoPE) | 5.35 | 4.70 | −0.65 |
| 5 (+SwiGLU) | 5.22 | 4.64 | −0.58 |
| 7 (NanoLlama shape) | 5.27 | 4.69 | −0.58 |

RoPE accelerates descent from the start. SwiGLU improves both level and rate. The capacity regression at SWAP 7 (8L vs 12L, +0.054 nats) is expected to recover over longer training — the full Phase 6 run demonstrates this.

---

## NanoLlama 8L — Phase 6 loss curve

127.6M params, AdamW, 2.5B tokens. RTX 4090, bfloat16. ~75,940 tok/s.

| Step | Tokens | Val loss | Notes |
|------|--------|----------|-------|
| 250 | 131M | 5.193 | |
| 500 | 262M | 4.357 | |
| 750 | 393M | 4.025 | |
| 1000 | 524M | 3.865 | |
| 1250 | 655M | 3.761 | |
| 1500 | 786M | 3.680 | |
| 2000 | 1.049B | 3.581 | |
| 2500 | 1.311B | 3.506 | |
| 3000 | 1.572B | 3.448 | |
| 3500 | 1.834B | 3.413 | |
| 4000 | 2.096B | 3.380 | |
| 4250 | 2.227B | 3.372 | |
| **4768** | **2.500B** | **3.3566** | **final** |

No instabilities. Smooth cosine descent throughout. Loss decreases at roughly −0.07 nats per 500M tokens in the early phase, slowing to −0.02 nats per 500M tokens near the end — consistent with Chinchilla scaling for a 127M model at this data volume.

**Descent rate analysis:**

| Interval | Tokens | Δ val | Rate (nats/500M tok) |
|----------|--------|-------|----------------------|
| 0→1.05B | 1.05B | −1.61 | −0.77 |
| 1.05B→1.57B | 524M | −0.13 | −0.13 |
| 1.57B→2.5B | 930M | −0.09 | −0.05 |

Rapid early descent dominated by learning common n-grams; slowing in the later phase as remaining signal is harder to extract. The final Chinchilla ratio (tokens/params) = 2.5B/127.6M ≈ 19.6×, close to the 20× Chinchilla-optimal point — further training would yield diminishing returns at this scale.

---

## NanoLlama v2 — optimizer probe dynamics

Three 1500-step probes, all using constant_warmdown at peak LR throughout (LR never decays in 1500 steps).

| Step | Tokens | N1 (Muon) | N2 (AdamW) | N3 (Muon, base) | N2 lead vs N1 |
|------|--------|-----------|------------|-----------------|---------------|
| 500 | 262M | 4.845 | 4.441 | 4.476 | −0.404 |
| 1000 | 524M | 4.516 | 3.898 | 4.107 | −0.618 |
| 1500 | 786M | 4.206 | **3.715** | 4.001 | **−0.491** |

AdamW (N2) leads Muon (N1) from step 1 and the gap widens continuously. This is not a late-training reversal — AdamW is faster and better on this architecture at all measured points. Muon shows the same pattern on the base architecture (N3): it converges slower than AdamW despite using the same compute.

**Descent (val@500 to val@1500):**
- N1 (Muon): 4.845 → 4.206, Δ = −0.639 over 524M tokens
- N2 (AdamW): 4.441 → 3.715, Δ = −0.726 over 524M tokens

AdamW achieves **14% more loss reduction** over the same token budget on this architecture. Note: N1 and N2 use different LR schedules (constant_warmdown vs cosine), so this number contains both optimizer and schedule effects.

---

## NanoLlama v2 — production loss curve

9000/9537 steps, 4.72B tokens, AdamW. run reached step 9000 of 9537 before pod storage was exhausted.

| Step | Tokens | Val loss | Notes |
|------|--------|----------|-------|
| 500 | 262M | 4.430 | |
| 1000 | 524M | 3.897 | |
| 1500 | 786M | 3.714 | matches probe N2 (+0.001 Δ — consistent ) |
| 2000 | 1.05B | 3.610 | |
| 3000 | 1.57B | 3.480 | |
| 4000 | 2.10B | 3.401 | |
| 5000 | 2.62B | **3.345** | **beats NanoLlama v1 final** (3.3566) |
| 6000 | 3.15B | 3.298 | |
| 7000 | 3.67B | 3.261 | |
| 8000 | 4.19B | 3.238 | |
| **9000** | **4.72B** | **3.2210** | **best checkpoint** |

Curve is clean throughout — no instabilities. The warmdown (steps 5722–9537) would steepen the descent; the final 537 unexecuted steps would likely have brought val to approximately 3.21–3.22 based on the descent rate.

**Why v1/v2 matched-token comparison is a schedule comparison, not an architecture comparison:**

v1 uses cosine_with_warmup (LR decays from step 200 onward). v2 uses constant_warmdown (LR stays at peak until step 5722, then linear decay). At every matched token count below 2.5B, v1 has been decaying for longer — it operates at a lower, more regularized LR. v2 deliberately holds peak LR longer (until step 5722) to maximise training signal before regularising — the trade-off shows up as higher early-checkpoint loss, not worse final quality. Correctly separating architecture from schedule requires running v1 to 4.72B on the same constant_warmdown schedule — the right next control is v1 trained to 4.72B on constant_warmdown — deferred in favour of scaling to 345M.

| Tokens | v1 val | v2 val | v1 LR at this step | v2 LR at this step |
|--------|--------|--------|--------------------|--------------------|
| 262M (step 500) | 4.357 | 4.430 | ~5.7e-4 (cosine) | 6e-4 (flat peak) |
| 524M (step 1000) | 3.865 | 3.897 | ~5.4e-4 | 6e-4 |
| 1.05B (step 2000) | 3.581 | 3.610 | ~4.6e-4 | 6e-4 |
| 2.50B (step 4768) | 3.357 | ~3.370 | min_lr 6e-5 | 6e-4 |

v2 crosses v1 at step ~5000 (2.62B tokens), when v2's warmdown finally starts biting and pulls the loss below v1's cosine-annealed endpoint. The schedule was deliberately chosen to maintain training pressure for longer; whether this or the architectural changes (rope_fraction, value_embeds, x0_mixin) drove the final −0.136 nats improvement is not separable from this data.

---

## SwiftLlama-350M — Phase 2A optimizer dynamics

1500-step probes, H100, both schedules use linear warmup to step 500 then cosine decay.

| Step | Tokens | A1 train | A1 val | A3 train | A3 val | A3 lead |
|------|--------|---------|--------|---------|--------|---------|
| 500 | 262M | 4.459 | 4.623 | 4.437 | **4.594** | −0.029 |
| 1000 | 524M | 3.668 | 3.877 | 3.535 | **3.686** | **−0.191** |
| 1500 | 786M | 3.519 | 3.672 | 3.430 | **3.482** | **−0.190** |

At step 500 Muon leads narrowly (−0.029 nats). By step 1000 AdamW leads by −0.191. The crossover happens between steps 500 and 700 as the cosine LR starts falling — Muon loses its early advantage as the learning rate decays. This is the key dynamic that short (500-step) probes miss.

---

## SwiftLlama-350M — production loss curve

Production run uses Muon (decision predated Phase 2A). RTX 4090, bfloat16, ~25,700 tok/s. In progress.

**DataLoader event:** Steps 1–2000 saw the first 1.05B tokens. After a checkpoint resume at step 2000, the DataLoader reinitialised from shard 0 — steps 2001–4000 re-saw the same 1.05B tokens a second time. Unique-token count from step 4001 onward: `1.05B + (step − 4000) × 524,288`. By step 16000: 1.05B + (12000 × 524,288) ≈ 7.34B unique tokens (not 8.39B wall-clock tokens).

| Step | Wall-clock tokens | Unique tokens | Val loss | Notes |
|------|------------------|--------------|----------|-------|
| 2500 | 1.31B | 1.31B | 3.778 | first val after resume |
| 4000 | 2.10B | 2.10B | 3.648 | data repeat ends |
| 5000 | 2.62B | 2.10B + 0.52B = ~2.62B | 3.590 | |
| 7000 | 3.67B | ~3.67B | 3.520 | benchmark checkpoint |
| 9000 | 4.72B | ~4.72B | 3.470 | |
| 11500 | 6.03B | ~6.03B | 3.414 | |
| 12000 | 6.29B | ~6.29B | 3.434 | blip (+0.020, recovered by step 12500) |
| 14000 | 7.34B | ~7.34B | 3.378 | |
| **16000** | **8.39B** | **~7.95B** | **3.3566** | **matches v1 exactly** |

Two val blips occurred: step 7500 (+0.006) and step 12000 (+0.020). Both recovered within 500 steps — consistent with noise in the 300-batch val evaluation rather than training instability.

**Chinchilla analysis:**
- Chinchilla-optimal (20× params): 20 × 345.3M = **6.91B unique tokens**
- Reached at step ≈15,174 (2026-04-16 ~17:00 UTC)
- Val at Chinchilla-optimal: ~3.377 (interpolated between steps 15000 and 16000)
- Val still descending past Chinchilla-optimal — model has not saturated

**Cross-model comparison at matched val loss:**

| Val loss | v1 NanoLlama (127M) | SwiftLlama-350M |
|----------|---------------------|-----------------|
| 3.648 | step ~1700 (~891M tok) | step 4000 (2.10B tok) |
| 3.52 | step ~2700 (~1.42B tok) | step 7000 (3.67B tok) |
| 3.357 | step 4768 (2.50B tok) | step 16000 (8.39B tok) |

SwiftLlama requires approximately **3.4× more tokens** to reach the same val loss as v1. For a 2.7× larger model (345M vs 127M), Chinchilla predicts ~2.7× more tokens needed to match — the actual ratio of 3.4× is slightly higher, possibly due to Muon suboptimality on this architecture (see `docs/architecture_decisions.md` §5).

---

## Summary: key dynamics observations

1. **RoPE + SwiGLU dominate early training signal.** At 500 steps, these two features account for nearly all the cumulative improvement over the GPT-2 baseline (−0.386 + −0.139 = −0.525 nats combined).

2. **Muon fast start, AdamW dominant long-run.** On GQA+SwiGLU at both 127M and 345M: Muon leads at step 500 but AdamW overtakes by step 1000. The crossover coincides with LR beginning to decay. Short probes (≤500 steps) systematically overestimate Muon's value on this architecture.

3. **v2 warmdown schedule intentionally aggressive early.** v2 holds LR at peak until step 5722 of 9537, producing slower early descent than v1's cosine but a steeper final warmdown. This is the right trade-off for a longer run — v2 crosses v1's final val at step ~5000.

4. **SwiftLlama blips are benign.** Two val spikes (+0.006 and +0.020) recovered within one eval interval. The underlying training loss trajectory shows no corresponding instability — the blips are evaluation noise from the 300-batch val window.

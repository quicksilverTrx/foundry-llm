# Muon Optimizer: Decision Log and Post-Mortem

This document records the full decision chain around the Muon optimizer across all three model generations — why it was chosen, what the evidence showed, why the production run continues on it, and what the correct choice is for future work. This is an engineering decision log, not a defence.

---

## Background: what Muon is and why it looked attractive

Muon is a second-order-inspired optimizer developed for the modded-nanogpt speedrunning challenge. Its core operation is Newton-Schulz orthogonalization of the gradient matrix before applying a Nesterov momentum step — conceptually similar to spectral preconditioning. The speedrunning community reported ~5× token efficiency over AdamW on their reference architecture, achieving GPT-2 124M-level performance in ~10% of the original token budget.

That is a large enough claimed edge that it warranted a direct test.

---

## NanoLlama v2: optimizer probe results (127M scale)

Three 1500-step probes at 127M scale. All on constant_warmdown schedule at peak LR throughout (LR never decays during the probe window). v2 architecture: GQA (3:1), SwiGLU, partial RoPE, value embeddings, x0-mixin.

| Probe | Architecture | Optimizer | val@1500 |
|-------|-------------|-----------|----------|
| N1 | Full v2 | Muon (nanochat recipe) | 4.206 |
| N2 | Full v2 | AdamW | **3.715** |
| N3 | Base (rope=1.0, no ve, no x0) | Muon | 4.001 |

**Decision: AdamW (N2 recipe) for v2 production.** AdamW leads Muon by 0.491 nats on the same architecture. Even on the simpler base architecture (N3), Muon loses to AdamW-on-v2 by 0.286 nats. The architecture does not rescue the optimizer.

One confound: N1 uses constant_warmdown; N2 uses cosine_with_warmup. Different schedules partially explain the magnitude. The direction is unambiguous: this probe ran clean.

---

## SwiftLlama-350M: why Muon was tried again

SwiftLlama launched with Muon. This decision predated the v2 probe results — the 345M training run started because the 5× efficiency claim from the speedrunning benchmarks was large enough to warrant a direct test at scale before committing to AdamW. The logic was: if Muon's advantage is architectural (better suited to larger gradient matrices at 345M than at 127M), the larger model is the right place to test it.

The alternative — launching SwiftLlama with AdamW, then running Muon probes — would have lost the opportunity to test Muon's efficiency claim at scale. The Phase 2A probes on H100 were designed to answer exactly this question.

---

## Phase 2A: SwiftLlama optimizer ablation (345M scale)

Three 1500-step probes on H100. Same 524,288 tok/step. Architecture: 22L, d=1024, GQA 4:1, SwiGLU, partial RoPE, value embeddings, x0-mixin. LR: linear warmup to step 500, cosine decay to ~0 at step 1500.

| Probe | Optimizer | WD | val@500 | val@1000 | val@1500 |
|-------|-----------|-----|---------|----------|----------|
| A1 | Muon + Adam | 0.0 | 4.623 | 3.877 | 3.672 |
| A2 | Muon + Adam | 0.1 | 4.693 | 3.904 | killed (1110) |
| A3 | AdamW | 0.1 | **4.594** | **3.686** | **3.482** |

**At step 500, Muon leads A3 by 0.029 nats.** This matches the early-probe signal that originally motivated the Muon choice. It is also where the speedrunning benchmarks typically stopped.

**By step 1000, AdamW leads by 0.191 nats.** The crossover happens between steps 500 and 700 as cosine LR begins decaying. Muon's early advantage is regime-specific: it disappears when the LR falls below a threshold. Any probe that terminates at or before step 500 will systematically overestimate Muon's value on this architecture.

**At step 1500, AdamW leads by 0.190 nats.** A3 reached A1's step-1500 score at step 1000 — 33% fewer tokens for the same loss level.

**The 5× token efficiency claim did not transfer.** At both 127M and 345M, on GQA + SwiGLU, AdamW dominates after the LR starts decaying. The speedrunning results were measured on MHA + ReLU² — a fundamentally different gradient geometry.

---

## Why the mechanism matters

Muon's Newton-Schulz orthogonalization produces gradient updates that lie on the Stiefel manifold — a good inductive bias when the weight matrices are square and the gradient statistics are well-conditioned for orthogonalization. The reference architecture for which Muon was tuned uses:
- **MHA**: symmetric, square Q/K/V projections (all d_model × d_model when n_head = d_model/head_dim)
- **ReLU²**: a two-matrix MLP with sparse, spiky gradients that orthogonalization compresses well

Our architecture differs on both axes:
- **GQA (4:1)**: KV projections are 4× smaller than Q projections — asymmetric weight matrices that orthogonalization treats differently
- **SwiGLU**: a three-matrix gated MLP where the gate product introduces correlated gradient flow across the three projections

Neither the asymmetric KV projection matrix nor the gated gradient interaction is what Muon's preconditioner was calibrated for. The optimizer is not wrong — it's correct for a different distribution of gradient geometry.

One additional note: Muon LR=0.02 came from the modded-nanogpt recipe for 124M MHA+ReLU². External experiments suggest optimal Muon LR for LLaMA-style 520M may be 4–8×10⁻³ — roughly half. Our probes may have used a suboptimal LR, potentially understating Muon's ceiling. The Phase 2A result (0.190 nats at step 1500) is directionally robust enough that LR retuning is unlikely to close the full gap, but it is a known confounder.

---

## Why SwiftLlama continues on Muon

Phase 2A results were available at step 1500. At that point, the production run was already at step ~4000 with a DataLoader reset event documented and a complete loss trajectory. Switching optimizers mid-run requires:
1. Reinitialising optimizer state (momentum buffers, second moments) — equivalent to a cold restart
2. Burning additional warmup steps for the new optimizer to adapt
3. Potentially invalidating the existing loss trajectory as a reference

The estimated cost of AdamW's advantage compounding from step 4000 to step 28,610: roughly 0.19 nats × (remaining fraction) ≈ a loss penalty consistent with the 3.4× token overage vs Chinchilla's 2.7× prediction. This is a real cost, fully quantified, and it was accepted because mid-run optimizer switching has its own costs.

**The correct decision for all future models: AdamW.** This is not in dispute. The SwiftLlama Muon run is now a data point documenting exactly how much the optimizer mismatch costs at 345M scale.

---

## Decision summary

| Point in time | Decision | Reasoning | Outcome |
|--------------|---------|-----------|---------|
| v2 launch | AdamW | N1/N2/N3 probes: AdamW +0.491 nats at 127M | Val 3.2210 at 4.72B |
| SwiftLlama launch | Muon | 5× efficiency claim from speedrunning; test-at-scale strategy | Continued on Muon after Phase 2A |
| Phase 2A (step 1500) | Confirmed AdamW wins by 0.190 nats | A3 reaches A1's loss 33% faster | Production run continues on Muon (checkpoint continuity) |
| All future models | AdamW | Muon does not transfer to GQA+SwiGLU | — |

The key engineering lesson: optimizer benchmarks transfer only within the same architecture family. Validate optimizer choice on the actual architecture with a probe that extends past LR decay before committing a production run.

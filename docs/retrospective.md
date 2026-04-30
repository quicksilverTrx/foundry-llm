# Retrospective: What We Built, What We Learned, What's Next

Three models, ~17B total tokens processed, two hardware platforms, one optimizer mistake turned into a documented lesson. This page is a first-person reflection on the decisions that shaped this project — what held up, what I'd do differently, and what the next experiment should be.

---

## What worked well

**Ablation-first architecture selection.** Running Phase 5 as a systematic 8-swap ablation before committing to the full pretraining run was the right call. It cost ~2.8B tokens of compute (8 × 350M each) and returned clear signal on the two features that actually matter: RoPE (−0.386 nats, dominant) and SwiGLU (−0.139 nats, significant). Everything else — RMSNorm, QK-Norm, logit softcap — showed negligible or confounded short-run effects. Without the ablation I would have adopted all of them as a block and couldn't have attributed the gain to anything specific.

**Short probes as a scouting mechanism.** 500-step probes at 262M tokens per feature are not predictive of long-run convergence — I learned that the hard way with Muon — but they are predictive of early descent dynamics and eliminate non-starters cheaply. The right mental model is "noise filter, not oracle." Features that fail at 500 steps definitively fail; features that pass still need a longer validation.

**Chinchilla ratio tracking.** Keeping explicit tokens/params ratios (19.6× for v1, 10.5× at the SwiftLlama benchmark checkpoint) prevented two bad comparisons from making it into the docs. The benchmark tables in the README flag mismatched-budget comparisons explicitly rather than presenting raw numbers that would mislead.

**KV-cache engineering discipline.** The serving layer was instrumented to measure prefill vs decode separately (TTFT at 100ms/189ms/463ms for ctx=256/512/1024) with explicit batch size in every number. That level of specificity makes the results reproducible and falsifiable, which is the only thing that makes serving benchmarks meaningful.

---

## What I'd do differently

**Run the v1-at-4.72B control before starting v2.** This is the clearest methodological gap in the project. The v2 architectural additions (partial RoPE, value embeddings, x0-mixin) cannot be credited or debited without a matched-schedule v1 run at the same token budget. The log-linear extrapolation suggesting v2 underperforms v1 at 4.72B is directionally concerning; the honest answer is that the control was deprioritised because SwiftLlama felt like the higher-ROI experiment at the time. In hindsight, 537 more steps on v1 (the gap between v2's best and its planned endpoint) would have been cheap insurance against the ambiguity. I'd run that control before ever committing to the v2 feature set in production.

**Validate Muon at 1000+ steps before using it in any production run.** The speedrunning community benchmarks were measured on MHA + ReLU², a completely different architecture from our GQA + SwiGLU stack. I launched SwiftLlama with Muon because the 5× token-efficiency claim looked compelling on paper without verifying that the optimizer's gradient geometry assumptions transferred. Phase 2A (three 1500-step probes) confirmed they don't: AdamW leads by 0.190 nats at step 1500 and the gap extends. SwiftLlama continues on Muon for checkpoint continuity — a sunk-cost decision with a quantifiable cost (~3.4× tokens vs Chinchilla's 2.7× prediction, roughly attributable to the optimizer penalty). The rule going forward: never pick an optimizer that was benchmarked on a different architecture family without running a 1000-step probe on the actual architecture first.

**Set up automatic checkpoint upload to object storage from day one.** Two runs ended prematurely due to local pod storage exhaustion: v2 at 94% of planned steps (step 9000/9537) and SwiftLlama still in progress when last checked. Both had no off-host backup. The cost of the missed steps in v2 (estimated 3.21–3.22 extrapolated) is minor; the cost of losing a 12-hour run to disk overflow is not minor. S3/GCS sync every 500 steps would have added ~15 minutes of setup and prevented both events.

**Use constant_warmdown for all runs from the start.** The cosine_with_warmup schedule used in v1 starts decaying LR from step 201, so every matched-token comparison between v1 and v2 is simultaneously a schedule comparison. The constant_warmdown schedule (plateau at peak LR through 60% of training, then linear decay) is strictly better for long runs because it maximises training signal before regularising. Had v1 used constant_warmdown, the v1/v2 matched-token comparison would be clean.

---

## What surprised me

**RMSNorm shows a slight regression at 500 steps.** I expected it to be neutral or mildly positive. The theoretical argument (omitting mean normalization saves a few FLOPs, no quality penalty) is standard and cited everywhere. At 500 steps it was +0.016 nats (worse). Phase 6 at 2.5B shows no measurable difference — the regression is transient — but I'd underestimated how long "transient" is for normalization layers. The practical lesson: short-probe signal on normalization and init features is especially unreliable.

**AdamW's lead over Muon appears between steps 500 and 1000, not at the beginning.** At step 500, Muon leads AdamW by 0.029 nats on the 345M model. This is why the 500-step probes that motivated the Muon launch looked credible. The overtake happens between steps 500 and 1000 as cosine LR begins decaying — Muon's early advantage is LR-regime-sensitive and disappears once the learning rate falls. Any probe that stops at step 500 will overstate Muon's value on this architecture. This is a subtle confound that required a 1500-step probe to detect.

**v2's warmdown schedule advantage only shows up past 2.62B tokens.** The whole point of holding peak LR longer (constant_warmdown) is to maximise training signal. But v2's loss doesn't drop below v1's cosine-annealed value until step ~5000 (2.62B tokens). That's a very long time to run at a loss deficit before the schedule's benefit materialises. In hindsight the reason is obvious: v1's cosine schedule is regularising aggressively by then, pulling it toward a shallow minimum. v2 only crosses because warmdown finally bites and drives the loss to a deeper basin. The signal is real; the timeline is longer than intuition suggests.

---

## The next experiment

**Controlled v1/v2 architectural comparison at matched compute.** The single highest-value experiment this project hasn't run: train NanoLlama v1 to 4.72B tokens on constant_warmdown schedule with the same hyperparameters as v2. This closes the control gap and finally answers whether partial RoPE + value embeddings + x0-mixin add any measurable value at 127M scale. If v1 at 4.72B matches or beats v2 at 3.221, the nanochat features are neutral and can be dropped. If v2 wins by a meaningful margin (>0.01 nats), the architecture changes are worth carrying into future scale-ups.

**Estimated cost:** 537 additional steps at 524,288 tok/step on RTX 4090 at ~75,940 tok/s ≈ 3.5 GPU-hours.

**Estimated cost of not running it:** perpetual ambiguity about whether the entire v2 architectural investment was worthwhile. That's not a good trade.

After that: SwiftLlama at Chinchilla-optimal (~step 13,000, 6.91B unique tokens) with a complete benchmark sweep — HellaSwag, PIQA, ARC, WinoGrande, Lambada — to get a clean architectural comparison between 127M and 345M at matched token efficiency. That's the number that tells you whether scaling from 127M to 345M was worth the compute, and the benchmark this project is missing.

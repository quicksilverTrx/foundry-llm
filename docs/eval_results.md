# Evaluation Results — NanoLlama 8L v1

**Model:** NanoLlama 8L — 127.63M params, 8 layers, GQA (12Q/4KV heads), d_model=768, SwiGLU, RoPE, RMSNorm  
**Training:** 4,768 steps · 2.5B tokens · FineWeb-Edu · RTX 4090  
**Final checkpoint:** val loss **3.3566** · BPB **1.016** · HellaSwag **26.96%**  
**Full training trajectory:** `results/nanollama_8l_training.csv`

> For NanoLlama v2 (4.72B tokens, val 3.2210) and SwiftLlama-350M results, see `docs/training_results.md`.  
> For generation samples on the v2 model, see `docs/qualitative_eval.md`.

---

## 1. Pipeline Integration: ShardLoader, LR Schedule, and ShardTrainer

### 1.1 Bug Fix: `llm_lab/core/train/optim.py`

**Problem:** `fused=True` was passed to `torch.optim.AdamW` unconditionally whenever the parameter existed in the PyTorch signature. PyTorch requires all parameters to be on CUDA for fused AdamW — it crashes with `RuntimeError` on CPU and MPS.

**Fix:**
```python
# Before (broken on CPU/MPS):
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
opt = torch.optim.AdamW(..., fused=fused_available)

# After (correct):
fused_available = (
    'fused' in inspect.signature(torch.optim.AdamW).parameters
    and all(p.device.type == 'cuda' for p in decay + no_decay)
)
opt = torch.optim.AdamW(..., fused=fused_available)
```

**Impact:** Allows `Trainer`, `ShardTrainer`, and all tests to run on CPU and MPS without error.
Previously all CPU/MPS-based tests and local dev workflows were silently blocked by this.

---

### 1.2 New: `llm_lab/core/data/shard_loader.py`

**What it is:** A stateful sequential loader for pre-tokenized `.npy` shard files in the
FineWeb-Edu / karpathy/build-nanogpt format. Extracted and generalized from the pretraining script.

**Two classes:**

**`ShardLoader`** — step-based direct interface (used by ShardTrainer):
```python
loader = ShardLoader(data_dir, split="train", B=16, T=1024, device="cuda")
x, y = loader.next_batch()   # [B, T] long tensors on device
loader.reset()               # rewind to shard 0, pos 0 — used before every val eval
```

**`ShardIterableDataset`** — wraps ShardLoader as `torch.utils.data.IterableDataset`:
```python
ds = loader.as_iterable_dataset()
dl = DataLoader(ds, batch_size=16)  # plugs into existing Trainer
```

**Key properties:**
- Loads one shard at a time (avoids loading all 19GB into RAM)
- Wraps around shards cyclically — no training run can "run out" of data
- `uint16 → int32 → torch.long` cast avoids overflow for vocab sizes >32767
- `file_pattern` is configurable so non-FineWeb naming works too
- `reset()` is idempotent — safe to call multiple times before val eval

**Relationship to the pretokenized shard dataset (`Sp16kPretokIterableDataset`):**
- NOT the same: that loads `.bin` files with sidecar manifests, SHA256 checks, and epoch sampling plans
- ShardLoader loads raw `.npy` shards (FineWeb-Edu format) — a different format entirely
- Both are needed; they serve different data pipelines

---

### 1.3 New: `llm_lab/core/train/lr_schedule.py`

**What it is:** A standalone `cosine_with_warmup` function extracted from the pretraining script.
Previously duplicated inline in `run_swap.py` and `pretrain_nanollama.py`.

```python
def cosine_with_warmup(step, *, warmup_steps, max_steps, max_lr, min_lr) -> float
```

**Regime breakdown:**
- `step < warmup_steps`: linear ramp `max_lr × (step+1) / warmup_steps`
- `step ∈ [warmup_steps, max_steps]`: cosine decay `min_lr + coeff × (max_lr − min_lr)`
- `step > max_steps`: constant `min_lr`

**NanoLlama 8L config:** `warmup=200, max_steps=4768, max_lr=6e-4, min_lr=6e-5`

**Relationship to existing `Trainer._lr_for_step()`:**
The Trainer already implements the same formula inline. This standalone function is useful for:
- ShardTrainer (step-based, no Trainer inheritance)
- Scripts / notebooks that need an LR without instantiating a Trainer
- Testing in isolation
There is some overlap, but the single canonical function is better than two diverging implementations.

---

### 1.4 New: `llm_lab/core/train/shard_trainer.py`

**What it is:** A step-based pretraining loop for FineWeb-Edu `.npy` shard data, matching
the pretraining script behaviour exactly but using foundry-llm components.

**Two classes:**

**`ShardTrainerConfig`** (dataclass with all run hyperparameters):
```python
cfg = ShardTrainerConfig(
    out_dir="./out",
    data_dir="./data/edu_fineweb10B",
    B=16, T=1024, grad_accum=32,
    max_lr=6e-4, min_lr=6e-5,
    warmup_steps=200, max_steps=4768,
    device="cuda",
)
```

**`ShardTrainer`** (training loop):
```python
trainer = ShardTrainer(model, model_cfg.__dict__, cfg)
trainer.fit()          # full run from start_step to max_steps
trainer.save_checkpoint(path, step=N)    # full: model + optimizer state
trainer.save_model_only(path, step=N)    # slim: model weights only (~487MB)
step = trainer.load_checkpoint(path)     # resume — returns step number
```

**Checkpoint format** (compatible with NanoLlama 8L checkpoints):
```python
{
    "step": int,
    "val_loss": float,
    "config": dict,               # MiniGPTConfig.__dict__ — needed to rebuild model
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,  # absent in slim checkpoints
}
```

**Feature comparison vs existing `Trainer`:**

| Feature | ShardTrainer | Trainer (existing) |
|---------|-------------|-------------------|
| Data source | ShardLoader (.npy) | DataLoader (any Dataset) |
| bf16 autocast | CUDA-only (fp32 on cpu/mps) | CUDA-only |
| Gradient accumulation | Done | Done |
| Cosine LR + warmup | cosine_with_warmup | _lr_for_step |
| Val eval with reset | val_loader.reset() | eval_every_n_steps |
| Checkpoint save | includes config dict | no config dict |
| Checkpoint resume | load_checkpoint() | load_checkpoint() |
| Fused AdamW | CUDA-only (post bug-fix) | CUDA-only (post bug-fix) |
| Grad clip | Done | Done |
| Trajectory CSV | NanoLlama 8L column layout | different layout |
| Step-based (not epoch) | Done | Epoch-based (max_steps supported) |
| Slim checkpoint export | save_model_only() | No |
| Model config in checkpoint | Done | No |

**What ShardTrainer does NOT reinvent:**
- Optimizer construction → delegates to `build_adamw_with_decay_groups` (existing)
- LR schedule → delegates to `cosine_with_warmup` (new, but thin)
- Model → MiniGPT (existing)
- The main loop structure mirrors the existing Trainer's `train_epoch` logic

---

### 1.5 Eval harness

An 11-test evaluation harness for NanoLlama 8L using the foundry-llm codebase.

**Key design decisions:**
- Uses `llm_lab.core.decode.sampling` for all generation (greedy_decode, sample_with_temperature, sample_top_k, sample_top_p) — not custom code
- Loads checkpoint via `MiniGPT` + `MiniGPTConfig` from foundry-llm
- Runs on MPS (Apple Silicon) locally
- Tests individually runnable: `python eval_suite.py --test N`

---

## 2. Test Results

### 2.1 New Tests: 29 pass, 0 fail

```
tests/core/test_lr_schedule.py    10 tests   0.03s
tests/core/test_shard_loader.py    9 tests   0.8s
tests/core/test_shard_trainer.py  10 tests   2.2s
Total: 29 passed in 2.20s
```

**`test_lr_schedule.py` — key assertions:**
- `step=0` returns `max_lr / warmup_steps` (not 0 — first ramp value)
- `step=warmup_steps-1` returns exactly `max_lr`
- `step=warmup_steps` (cosine boundary) returns exactly `max_lr`
- `step=max_steps` returns exactly `min_lr`
- Midpoint returns `(max_lr + min_lr) / 2`
- Decay is monotone non-increasing
- `step > max_steps` → constant `min_lr`
- No-warmup edge case: `step=0` → `max_lr`
- All post-warmup values in `[min_lr, max_lr]` range
- Warmup values in `[0, max_lr]` range (can be below min_lr — by design)

**`test_shard_loader.py` — key assertions:**
- Output shape `(B, T)` for both `x` and `y`, dtype `torch.long`
- `y == x + 1` for sequential data (LM target shift verified)
- `reset()` reproducibility: same batch returned after rewind
- Shard wrap-around: no crash after exhausting all shards (cyclic)
- `FileNotFoundError` on missing data dir
- `num_shards` property correct
- `__repr__` contains key fields
- `ShardIterableDataset` yields `[T]` rows (not `[B,T]` batches)
- DataLoader with `batch_size=B` correctly reassembles to `[B, T]`

**`test_shard_trainer.py` — key assertions:**
- `total_batch = B × T × grad_accum`, `total_tokens = total_batch × max_steps`
- `fit()` completes without error (5 CPU steps on synthetic data)
- `trajectory.csv` created with correct header: `step,train_loss,lr,grad_norm,tok_per_sec,val_loss`
- `config.json` created with `B`, `max_steps` keys
- Checkpoint file exists in `ckpts/` after `ckpt_every` steps
- **Loss decreases**: avg of last 5 steps < avg of first 5 steps on repeating data at lr=5e-3 (overfit sanity)
- `load_checkpoint()` returns correct step number
- Model-only checkpoint smaller than full checkpoint
- Full checkpoint contains `config`, `model_state_dict`, `optimizer_state_dict`, `step`

### 2.2 Existing Test Suite: No regressions

Pre-existing 65 tests pass after `optim.py` bug fix. No regressions introduced.

---

## 3. Model Evaluation: NanoLlama 8L (step 4768)

**Model:** NanoLlama 8L v1 — 127.63M params, val_loss=3.3566, ppl=28.69  
**Checkpoint:** step 4,768 · 2.5B tokens  
**Device:** MPS (Apple Silicon M-series)  
**Sampling backend:** `llm_lab.core.decode.sampling`

---

### Test 1 — Sanity Check

```
Checkpoint step  : 4768
Val loss (stored): 3.3566  →  ppl=28.69
Parameters       : 127.63M
n_layers         : 8    n_heads / kv : 12 / 4 (GQA)
d_model / d_ff   : 768 / 2048
logit_softcap    : 30.0    qk_norm : True
lm_head bias     : None    token_embed std : 0.0324
```

Note: `token_embed std=0.0324` vs expected 0.0200 — normal after 2.5B tokens of training (embeddings drift from init).

---

### Test 2 — Greedy Generation (deterministic)

**Observation:** All 9 prompts show repetitive loops under greedy decoding — a universal behaviour for autoregressive LMs without diversity forcing. The model converges to the highest-probability continuation and then loops.

**Notable:**
- [science]: loops "The theory of general relativity explains how the universe is made up of a series of events"
- [code]: quicksort prompt doesn't complete cleanly — greedy collapses on whitespace tokens after the `return arr` line
- [history]: loops "The French government declared war on the French people" (plausible French Revolution context, but repetitive)
- [chat]: "The capital of France is the capital of France" — correct answer but immediately loops
- [eol]: handles `<|endoftext|>` token cleanly, generates coherent start, then loops

**Verdict:** Expected behaviour for a pretrained-only model. Instruction tuning / RLHF would fix the loop problem.

---

### Test 3 — Temperature Sweep

Prompt: "The theory of general relativity explains how"

| Temp | 4gram-rep | Observation |
|------|-----------|-------------|
| 0.3  | 0.404 | High repetition, physically plausible but loops |
| 0.7  | 0.070 | Good quality, some repetition — **sweet spot** |
| 1.0  | 0.000 | Zero repetition, creative but incoherent ("presonics", "messageboxes") |
| 1.2  | 0.000 | Interesting structure but factually unreliable |
| 1.5  | 0.000 | Mostly noise ("transcophers", "Drohard storage FORS COL219") |

**Verdict:** temp=0.7 is the practical operating point. Clear temperature→diversity scaling.
The model transitions from deterministic loops (temp≤0.4) to zero-rep creative outputs (temp≥1.0).

---

### Test 4 — Nucleus (top-p) Sampling (temp=0.8)

| top_p | 4gram-rep | Sample |
|-------|-----------|--------|
| 0.5 | 0.468 | "the sun god, the sun god, was the son of the sun god" — highly repetitive, limited vocabulary |
| 0.9 | 0.026 | French history narrative — fluent, coherent paragraph |
| 0.95 | 0.013 | Carcharodont birds — factually grounded, low repetition |

**Verdict:** top_p≥0.9 is required for good outputs. top_p=0.5 collapses the distribution too aggressively, causing loops. top_p=0.95 at temp=0.8 gives best balance.

---

### Test 5 — Repetition Analysis (temp=0.7 vs 1.0, top_p=0.95, 200 tokens)

| Prompt | Temp | 4gram-rep | Unique tok% |
|--------|------|-----------|-------------|
| science | 0.7 | 0.310 | 26.0% |
| science | 1.0 | 0.005 | 59.5% |
| code | **0.7** | **0.777** | 5.5% |
| code | 1.0 | 0.208 | 45.5% |
| history | 0.7 | 0.330 | 23.0% |
| history | 1.0 | 0.000 | 59.5% |
| math | 0.7 | 0.731 | 12.0% |
| math | 1.0 | 0.091 | 45.5% |
| story | 0.7 | 0.122 | 36.5% |
| story | 1.0 | 0.005 | 61.5% |

**Key findings:**
- **Code is the most repetitive domain** (0.777 at temp=0.7) — model learned code structure but loops on patterns
- **Story is least repetitive** at temp=0.7 (0.122) — narrative context provides diversity naturally
- Temp=1.0 dramatically reduces repetition across all domains (4gram-rep→0 for most)
- Code unique tokens at temp=0.7: only 5.5% — model uses a very narrow vocabulary for code continuations

---

### Test 6 — Perplexity on Held-out Texts

| Text | Loss | PPL | Note |
|------|------|-----|------|
| wiki_gravity | 2.957 | **19.2** | (low) Done |
| wiki_python | 3.029 | **20.7** | (low) Done |
| numbers (1–25) | 2.322 | **10.2** | Strong numerical pattern learning |
| code_snippet (numpy) | 2.436 | **11.4** | Strong code competence |
| dialogue | 3.123 | 22.7 | Reasonable |
| random_junk | 6.867 | 960 | (high) Done |
| repetitive ("the"×20) | 7.461 | **1738** | Surprising: model does NOT predict "the" loops |

**Notable insight on repetitive text:** PPL=1738 on "the the the the..." means the model assigns very low probability to repetitive token sequences even though it *generates* them under greedy decoding. This reveals an important asymmetry: greedy mode exploits the highest single-step probability token (which at any given context may be "the"), but the *sequence* "the the the..." as a whole is assigned low joint probability. The repetition bug is a decoding artifact, not a model belief.

**Comparison to training val loss:**
- Training val loss = 3.3566 on FineWeb-Edu data
- In-distribution wiki text: 2.957–3.029 (lower — these are well-formed educational texts)
- This validates the model is well-calibrated — ppl is ~20 on clean prose, as expected

---

### Test 7 — Next-Token Entropy (bits)

| Prompt | Entropy | Interpretation |
|--------|---------|----------------|
| science | 6.61 | Moderately uncertain — many valid physics continuations |
| code | **4.78** | Most constrained — code has strong syntactic structure |
| history | 5.83 | Moderate |
| math | 5.49 | Moderate |
| story | 7.69 | High uncertainty — open-ended narrative |
| news | 8.42 | High — news topics are diverse |
| list | **9.91** | Highest — numbered list items are almost unconstrained |
| chat | 7.01 | Moderate — "Answer:" has many valid completions |
| eol | 8.18 | High — post-EOT context is fully open |

Max possible: log2(50304) = 15.62 bits.

**Key insight:** Code (4.78 bits) and science (6.61 bits) have the lowest entropy — these domains have the most learnable structure from FineWeb-Edu. The model assigns meaningful probability mass to a small set of likely next tokens. `list` at 9.91 bits is nearly uniform over the vocabulary — consistent with the model not having strong priors for "what comes after item 1 in a top-5 list."

---

### Test 8 — Coherence Battery (temp=0.8, top_p=0.92)

| Prompt | Response | Score |
|--------|----------|-------|
| Capital of Japan | "Kashi" | No (Tokyo) |
| 7 × 8 = ? | "3!4, 3!4..." (garbled) | No |
| Hot opposite of | "a cold" |  (almost) |
| Dog, cat, bird are... | "animal behavior" |  (semantically correct) |
| Monday→Thursday... | repeats "Saturday" | No (partial) |
| Python reverse_string | Incomplete function body | partial |
| Photosynthesis | "process by which plants use energy from light" |  correct |
| Einstein born in | "Dijon" (then non-sequitur) | No (Ulm) |

**Assessment:** 2 correct, 4 wrong, 2 partial out of 8. This is below what a well-tuned model should achieve but consistent with a pure pretrain on educational text with no instruction tuning. The model has latent factual knowledge (photosynthesis is correct, animal categories are correct) but factual recall is unreliable under sampling.

**Expected improvement path:** RLHF / instruction fine-tuning would dramatically improve these scores without retraining from scratch.

---

### Test 9 — HellaSwag-Style MCQ (5 custom items, 4 choices each)

**Note:** This is a 5-item hand-crafted probe, not the full HellaSwag benchmark. The full 10,042-item HellaSwag evaluation gives **26.96%** (random baseline 25%). The 5-item probe below is directionally useful but statistically noisy.

**Probe result: 3/5 = 60%** (random baseline = 25%)

| Item | Result | Margin (pred vs gold loss) |
|------|--------|---------------------------|
| Kitchen/baking | PASS | 0.139 nats |
| Gym/barbell | PASS | 0.260 nats |
| Teacher/chalkboard | **FAIL** | −0.150 nats (too close) |
| Terminal/compile | PASS | 0.662 nats |
| Dog/door | **FAIL** | −0.284 nats |

**Analysis:**
- Items 1, 2, 4 had clear loss gaps (0.14–0.66 nats) → easy for the model
- Item 3 fail: "copy homework" vs "ask for clarification" are both plausible continuations of a teacher writing on the board — even humans might disagree here
- Item 5 fail: "jumped into ocean" (pred) has lower loss than "barked twice" (gold) — the model assigns low cost to fantastical events, suggesting its prior from FineWeb-Edu doesn't strongly distinguish physically plausible vs implausible dog behaviour

**Note:** 60% on a 5-item custom MCQ is noisy. The real HellaSwag eval would require ~10K items for a reliable estimate. This is directionally positive (above random by 35pp) but not statistically conclusive.

---

### Test 10 — Vocabulary Coverage (500 generation steps)

```
Distinct top-1 tokens: 169 / 50,304  (0.3%)
```

**Top 20 most generated tokens:**
```
21x  '\n'       21x  ' to'      18x  '.'        17x  ' the'
13x  ' of'      13x  ' and'     11x  ','         10x  ' may'
 9x  ' or'       9x  ' social'   8x  '-'          8x  ' a'
 7x  ' an'       7x  ' child'    7x  ' are'        6x  ' develop'
 6x  ' is'       5x  ' be'       5x  ' have'       5x  ' inability'
```

**Analysis:**
- Only 0.3% of the 50K vocabulary ever becomes top-1 across 500 diverse positions
- This is extremely concentrated — consistent with a model trained on formal educational prose
- Top tokens are all common English function words + newline
- "social", "child", "develop", "inability" in top-20 reflect FineWeb-Edu's child development / educational psychology content bias
- The 169 distinct top-1 tokens at temp=0.9 suggests the model's learned distribution is sharp — most of the mass sits on very few tokens per context

**Implication for decoding:** This explains why greedy/low-temp generation loops — only ~100–200 tokens ever dominate top-1, so once the context steers toward a high-probability attractor (e.g. "the universe is"), the model keeps predicting the same next token.

---

### Test 11 — GPT-2 Baseline vs NanoLlama 8L Val Loss Comparison

| Model | Tokens | Val Loss | PPL |
|-------|--------|----------|-----|
| GPT-2 124M baseline | 1.053B | 3.6273 | 37.6 |
| NanoLlama 8L v1 | 2.500B | 3.3566 | 28.7 |
| NanoLlama 8L v1 @ 1B tokens (step ~2009) | 1.053B | 3.5892 | 36.2 |

**Compute-matched gap (1B tokens):** 3.6273 − 3.5892 = **0.038 nats**

**Full-run gap:** 3.6273 − 3.3566 = **0.271 nats**

**Verdict:** At compute-matched 1B tokens, NanoLlama has only a 0.038 nat edge over Karpathy GPT-2 — nearly the same. The 0.271 nat full gap is almost entirely explained by NanoLlama training on 2.38× more tokens, not by architectural superiority.

The architectural features (RoPE +0.386, SwiGLU +0.139) provide real improvements in the architecture ablations, but at the GPT-2 baseline → NanoLlama 8L scale the additional tokens dominate the final loss gap.

---

## 4. Cross-model quantitative results

### 4.1 Val loss comparison

All models trained on FineWeb-Edu with GPT-2 BPE tokenization (tiktoken). Val loss measured on the FineWeb-Edu validation shard.

| Model | Params | Tokens | Val loss | BPB | HellaSwag |
|-------|--------|--------|----------|-----|-----------|
| GPT-2 124M (Karpathy reference) | 124M | ~10B | 3.29 | 0.996 | — |
| GPT-2 124M (reproduced @ 1.05B tok) | 124M | 1.05B | 3.627 | 1.098 | — |
| NanoLlama 8L v1 | 127.6M | 2.5B | **3.357** | **1.016** | **26.96%** |
| NanoLlama v2 | 127.6M | 4.72B | **3.221** | — | — |
| SwiftLlama-350M (step 16,000) | 345.3M | 8.39B | **3.357** | — | — |

NanoLlama v2 eval suite (temperature sweep, perplexity profiling, entropy analysis) has not been run — only val loss is available. The v1 suite in sections 2–3 above applies architecturally; v2 differences are partial RoPE, value embeddings, and x0-mixin, which are unlikely to change the qualitative decoding behaviour.

### 4.2 SwiftLlama-350M benchmark scores (step 7,000, 3.67B tokens)

These are **milestone benchmarks at 54% of Chinchilla-optimal**, not the valid architectural comparison point. NanoLlama v1 at this point had seen 19.6× tokens/param vs SwiftLlama's 10.5×. The valid comparison is at Chinchilla-optimal (~step 13,000) — not yet run at time of writing.

| Task | Random baseline | SwiftLlama-350M | NanoLlama v1 (2.5B tok) |
|------|----------------|----------------|-------------------------|
| PIQA | 0.500 | 0.584 | **0.601** |
| ARC-Easy | 0.250 | 0.351 | **0.381** |
| HellaSwag | 0.250 | 0.268 | **0.270** |
| WinoGrande | 0.500 | 0.512 | 0.503 |
| Lambada | — | 0.292 | **0.328** |

SwiftLlama trails v1 on all five tasks at this checkpoint. As noted in `docs/training_dynamics.md`, SwiftLlama requires ~3.4× more tokens than Chinchilla's 2.7× prediction to match v1 val loss — attributable to Muon's negative transfer on GQA+SwiGLU. Scores at Chinchilla-optimal will be reported once step ~13,000 is reached.

### 4.3 What the eval suite would add for v2 / SwiftLlama

The 11-test suite above (temperature sweep, nucleus sampling, repetition analysis, perplexity on held-out texts, next-token entropy, MCQ probe, vocabulary coverage) has only been run on v1. Running the same suite on v2 would isolate whether the architectural additions (partial RoPE, value embeddings, x0-mixin) change qualitative generation behaviour — expected to be small at 127M scale.

---

## 5. Component Status

| Component | Module | Status |
|-----------|--------|--------|
| ShardLoader (.npy) | `llm_lab/core/data/shard_loader.py` | Done |
| ShardIterableDataset | Included in shard_loader.py | Done |
| LR schedule | `llm_lab/core/train/lr_schedule.py` | Done |
| ShardTrainer + config | `llm_lab/core/train/shard_trainer.py` | Done |
| Checkpoint save/resume | `ShardTrainer.save_checkpoint/load_checkpoint` | Done |
| fused AdamW (CPU/MPS safe) | `llm_lab/core/train/optim.py` | Done |
| Unit tests | 29 tests passing | Done |

**Note:** `Trainer` expects `DataLoader`, not `ShardLoader` directly. The `ShardIterableDataset` bridge exists for use with the existing `Trainer`. `ShardTrainer` is the preferred path for shard-based pretraining.

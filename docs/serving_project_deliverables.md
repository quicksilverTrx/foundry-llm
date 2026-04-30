# Production-ish LLM Serving — Master Deliverables Document

This is the self-contained reference document for the foundry-llm serving project. It captures what was planned, what was built, how it was tested, and what the measured results are. All code paths, experiment artifacts, and test receipts trace back to this document.

## 0. Current state (cold-start summary)

**What exists:** A complete serving layer (`llm_lab/serving/`, 17 files, ~2,200 LOC) and eval stack (`llm_lab/eval/`, 5 files, ~1,300 LOC) on top of the foundry-llm core transformer.

**What it can do right now:**
```bash
# Serve NanoLlama 8L over HTTP with SSE streaming
python scripts/serving/serve.py \
  --package <path/to/nanollama_checkpoint.pt> \
  --loader nanollama --device cpu

# Benchmark cache vs recompute
python scripts/serving/bench_inference.py --package <ckpt> --loader nanollama --mode both

# Run quant sweep (fp32/int8)
python scripts/serving/quant_sweep.py --package <ckpt> --loader nanollama --text-path data/serving_eval/prompts.jsonl

# Run prompt suite eval
python scripts/eval/eval_prompt_suite.py --package <ckpt> --loader nanollama --backend engine
```

**Key numbers (NanoLlama 8L, 127.6M params, CPU M1 Pro):**

| Metric | Value |
|--------|-------|
| KV cache decode speedup | **7.53x** (24.2 ms/tok cached vs 182.1 ms/tok recompute) |
| TTFT @ ctx 256 / 512 / 1024 (B=1) | 100ms / 189ms / 463ms |
| Steady-state TPS @ ctx 256 / 512 / 1024 | 56 / 49 / 42 |
| int8 decode speedup vs fp32 | **1.27x** (13.9 vs 17.6 ms/tok) |
| int8 memory savings | **70%** (155 MB vs 511 MB) |
| int8 PPL drift | +6.0% (268 → 284) |
| Test suite | **152 pass, 0 fail** |

**Two models supported:**
- `--loader package` → sp16k SubwordTokenizer packages (8.4M toy model for testing)
- `--loader nanollama` → raw `.pt` checkpoints with tiktoken GPT-2 (NanoLlama 8L for production)

**Key files to read first:** `llm_lab/serving/engine.py` (the engine), `llm_lab/serving/api.py` (HTTP surface), `tests/serving/conftest.py` (test fixtures and ToyTokenizer/ToyScheduleModel).

---

## 1. Project objective

Build a production-grade LLM inference serving layer on top of the foundry-llm core transformer stack (MiniGPT). The serving layer must:

1. **Load any trained model package and serve it** — both the sp16k SubwordTokenizer packages from earlier experiments and tiktoken-based NanoLlama checkpoints
2. **Separate prefill from decode** — the fundamental architectural split that enables KV-cache reuse and meaningful latency decomposition (TTFT vs inter-token latency)
3. **Prove KV-cache correctness before optimizing** — cached decode logits must be equivalent to full recompute logits, verified step-by-step with a first-divergence diagnostic
4. **Expose a real HTTP API** — `/generate` (sync), `/stream` (SSE), `/health`, `/metrics` — curlable, with request validation, rate limiting, and structured logging
5. **Measure and report** — TTFT, tokens/sec, prefill_ms, decode_ms/token across context lengths, batch sizes, and precision modes; perplexity regression under quantization
6. **Implement safety guardrails** — PII/profanity heuristics, refusal templates, privacy-preserving logs (no raw prompts by default)
7. **Keep architecture modular** — serving wraps core; core never imports serving

**Hard boundary:** `llm_lab/core/` has zero imports from `llm_lab/serving/`. All serving logic lives in `llm_lab/serving/` and `llm_lab/eval/`.

### What this project is NOT

- Not a full vLLM reimplementation (no paged attention, no continuous batching scheduler)
- Not SFT/RLHF/DPO (that's a separate project scope)
- Not multi-node serving (no autoscaling, no service mesh, no auth)
- Not a chat interface (NanoLlama is a base language model with no chat template)

---

## 2. Win conditions (what "done" means)

These were defined at project start and all have been met:

| Win condition | Status | Evidence |
|--------------|--------|----------|
| `load_model_package(...)` from core and serve it | Done | sp16k packages load via `build_engine_from_package(loader="package")`; NanoLlama loads via `loader="nanollama"` |
| `/generate` and `/stream` endpoints work | Done | FastAPI app with SSE streaming, tested with TestClient and CLI |
| Step-by-step cache equivalence with first-divergence diagnostic | Done | `test_cache_equivalence_stepwise.py` + `diagnostics.py` |
| Measured decode speedup from KV cache (2x+ on GPU target; any on M1) | Done | **7.53x on CPU M1** (NanoLlama 8L, prompt=256, gen=128) |
| fp16/bf16 and at least one quant mode (int8 required) | Done | fp32/fp16/bf16 casting with device fallback; int8 via qnnpack dynamic quantization |
| Benchmark report + eval suite + safety regressions | Done | Quant report, TTFT/TPS grid, prompt suite (6/6 pass), safety filters with PII refusal |
| Architecture: serving wraps core; core doesn't import serving | Done | Verified by import analysis; zero cross-boundary imports |

---

## 3. Concept coverage (spec tier mapping)

### Tier A (non-negotiable for serving) — all implemented

| Concept | Where |
|---------|-------|
| Prefill vs decode separation | `engine.py`: `prefill()` called once, `decode_step()` per token |
| KV cache integration (per-layer K/V tensors) | `kv_cache.py` + `attention.py::_append_past()` |
| Streaming generation (SSE) | `stream.py::iter_token_sse()`, `api.py::/stream` |
| Latency metrics: TTFT + inter-token + tokens/sec; prefill vs decode breakdown | `engine.py::generate()` returns full metrics dict; `metrics.py::MetricsStore` |
| Stop/EOS/context handling (max tokens, stop sequences, truncation) | `decode_controls.py`: 4-way stop taxonomy (eos, stop_token, stop_string, max_new_tokens) |
| Variable-length batching: right padding + attention_mask | `batching.py::right_pad_and_mask()`, `engine.py::prefill_batch()` |
| Precision modes: fp16/bf16 inference | `precision.py`: normalize → validate → cast, device-aware fallback |
| Perplexity eval on heldout (regression + quant drift) | `eval/ppl.py::evaluate_streaming_nll()` — sliding-window NLL with stride |
| Safety basics: PII/profanity heuristics + refusal template | `safety.py`: regex PII (email/phone/SSN), profanity filter, `should_refuse_prompt()` + `postprocess_generated_text()` |
| Logging with privacy guardrails | `logging.py`: prompt hash (no raw text), configurable debug opt-in |
| Health/readiness endpoint | `api.py::/health` returns block_size + cache config |

### Tier B (high leverage) — all implemented

| Concept | Where |
|---------|-------|
| top-k, top-p (nucleus), temperature, multinomial sampling | `sampling.py`: full pipeline in `select_next_token_id()` |
| Repetition + frequency penalty | `sampling.py::apply_repetition_penalty()`, `apply_frequency_penalty()` |
| Rate limiting + request validation | `rate_limit.py::RateLimiter` (fixed window), `schemas.py` Pydantic validation |

### Tier C (stretch / optional) — partial

| Concept | Status |
|---------|--------|
| int8 quantization | Done — `quant.py` via `torch.quantization.quantize_dynamic`, qnnpack backend |
| int4 quantization | Not attempted — int8 was sufficient for CPU; int4 needs bitsandbytes + CUDA |
| Prefix caching (prompt reuse) | Not implemented |
| Continuous/dynamic batching | Not implemented — prefill batching only; decode is B=1 |
| Speculative decoding | Not implemented |

---

## 4. Models served

### 4a. RoPE MHA toy model (development + testing)

All unit tests and integration tests run against this model. It loads fast, has deterministic behavior, and exercises all serving code paths without requiring large memory.

| Field | Value |
|-------|-------|
| Package | `experiments/p1_pos_enc/runs/rope/package` |
| Architecture | MHA (4 heads), RoPE, LayerNorm, GELU MLP |
| Config | 4 layers, d_model=256, d_ff=1024, block_size=512 |
| Parameters | 8,420,624 (~8.4M) |
| Vocab | 10,000 tokens |
| Tokenizer | sp16k SubwordTokenizer (BPE: `merges.txt` + `vocab.json`) |
| Loader | `build_engine_from_package(loader="package")` |

### 4b. NanoLlama 8L (real production model)

Trained from scratch on FineWeb-Edu. This is the model all benchmark numbers are reported for.

| Field | Value |
|-------|-------|
| Checkpoint | NanoLlama 8L best val `.pt` (see `docs/replication.md` for checkpoint paths) |
| Architecture | GQA (12 Q heads, 4 KV heads), RoPE, RMSNorm, SwiGLU, logit_softcap=30.0, qk_norm=True |
| Config | 8 layers, d_model=768, d_ff=2048, block_size=1024 |
| Parameters | 127.6M |
| Vocab | 50,304 (tiktoken GPT-2 50,257 + 47 padding IDs for hardware alignment) |
| Tokenizer | tiktoken GPT-2 via `TiktokenWrapper` (eos_token_id=50256, pad_token_id=50256) |
| Loader | `build_engine_from_package(loader="nanollama")` |
| Training | 4,768 steps, 2.5B tokens, RTX 4090, ~9.2h, 75,940 tok/s |
| Final val loss | 3.3566 nats (BPB 1.016) |
| HellaSwag | 0.2696 (random baseline 0.25) |
| Checkpoint keys | `"config"` (dict), `"model_state_dict"`, `"step"`, `"val_loss"` |

**Tiktoken integration gaps addressed:**
- `TiktokenWrapper.decode()` filters IDs 50257–50303 (model padding range) to prevent `KeyError`
- `eos_token_id=50256` explicitly set so `resolve_eos_token_id()` works correctly
- `pad_token_id=50256` (EOS as pad, standard convention for models without dedicated pad token)
- `encode()` passes `allowed_special="all"` so `<|endoftext|>` encodes as single ID

---

## 5. Phase-by-phase deliverables

### Phase 0 — Gap Closure: KV-Cache ABI + Attention Mask + Serving Package (mandatory prerequisite)

**Why this phase exists:** The core model project (Project 1) gave us a working transformer with RoPE and GQA, but the forward path didn't return KV cache, attention_mask didn't handle right-padding for batching, and no serving-ready model package existed. These are hard prerequisites for every subsequent phase.

**What was delivered:**

1. **KV-cache ABI finalized:** `MiniGPT.forward(input_ids, attention_mask=None, past_key_values=None, use_cache=False) -> (logits, past_key_values)`. Cache layout is canonical: `list[tuple[K, V]]` where K, V are `[B, n_kv_heads, T, head_dim]`, time on axis 2.

2. **Attention mask correctness for right-padding:** Mask shape `[B, T]` with 1=keep, 0=pad. In decode mode where query length < key length (Tq < Tk), the mask broadcasts correctly. SDPA path builds `causal_bias [1,1,Tq,Tk] + padding_bias [B,1,1,Tk]` as additive float bias.

3. **RoPE position offset for cached decode:** `apply_rope()` uses `position_offset` so decode step gets correct absolute positions. Relative `position_ids = arange(0, T)` + past_len offset. SingleHeadAttention appends past KV before computing attention, using K_total/V_total for the full context window.

4. **Serving-ready model package:** The `experiments/p1_pos_enc/runs/rope/package` directory loads via `load_model_package()` and generates text. Legacy checkpoint keys (`pos_embed.weight`, `lm_head.bias`) from older architectures are filtered gracefully.

**Remaining known TODOs (documented, not blocking):**
- GQA RoPE rotation: verify that RoPE is applied to base KV heads before `_append_past` in the nanollama path
- The attention.py `assert` for `(B, Tq, Tk)` is rectangular (done), but the GQA past_len indexing could be cleaner

**Key files:**
| File | What it does |
|------|-------------|
| `llm_lab/core/model/attention.py` | KV-cache append via `_append_past()`, rectangular mask handling, SDPA causal+padding bias |
| `llm_lab/core/model/gpt.py` | Forward signature with `past_key_values` + `use_cache` return |
| `llm_lab/core/model/pos_encodings.py` | `apply_rope()` with `position_offset` parameter |
| `llm_lab/core/package/io.py` | `load_model_package()` with legacy key filtering |

**Tests (6 tests, all pass):**
| Test | What it verifies |
|------|-----------------|
| `test_package_smoke.py` | Package loads, forward pass produces finite logits |
| `test_attention_mask_padding.py` | Padded batch logits match unpadded individual logits at non-pad positions; all-pad rows don't NaN |
| `test_kv_cache_abi.py` | past_key_values shape is `[B, H, T, D]`, grows by +1 on decode step |
| `test_rope_offset_progression.py` | Cached decode with RoPE offset matches full recompute at last step |
| `test_past_key_values_shapes.py` | Prefill returns cache with correct number of layers and shape |
| `test_rope_position_offset_progression.py` | Position offset progression is monotonic across decode steps |

---

### Phase 1 — KV Cache Correctness First, Then Speed

**Why correctness before speed:** Cache bugs (RoPE offset off-by-one, mask mismatch, append ordering) are invisible — shapes match, generation "works", but logits drift. The equivalence test is the gate: no optimization work until cached decode matches recompute exactly.

**What was delivered:**

1. **KV cache operations:** `kv_cache.py` with `kv_append()` (concatenate on time axis), `kv_truncate()` (keep last N), `apply_sliding_window()` (per-layer truncation). All preserve dtype/device.

2. **Engine prefill/decode split:** `engine.py::Engine` with:
   - `prefill(input_ids, attention_mask)` → `(last_logits, CacheState, meta)` — called exactly once
   - `decode_step(next_input_id, state)` → `(step_logits, CacheState)` — called per generated token
   - `decode_step` enforces B=1 and applies sliding window for max-cache

3. **Cache equivalence invariant (the correctness gate):**
   - For a fixed prompt, run generation two ways: (a) full recompute each step, (b) prefill once + decode steps
   - Assert per step: logits close within tolerance, greedy token matches
   - fp32 cached-vs-recompute max_abs_diff: **6.68e-06** at atol 1e-4

4. **First-divergence diagnostic:** When cache divergence occurs, `diagnostics.py` reports the exact step, layer-wise output norm diffs, and which layer first exceeded tolerance. This was critical during development — almost all first failures were RoPE offset off-by-one or mask mismatch.

5. **Measured speedup:**

   **NanoLlama 8L (127.6M params, CPU M1 Pro, fp32, prompt=256, gen=128, warmup=5, iters=20):**

   | Metric | Cached decode | Full recompute | Speedup |
   |--------|--------------|----------------|---------|
   | prefill_ms | 136.1 | 151.1 | — |
   | TTFT_ms | 136.2 | — | — |
   | decode_ms/token | 24.2 | 182.1 | **7.53x** |
   | tokens/sec | 41.3 | 5.5 | **7.53x** |

   The recompute path is O(T) per token where T grows each step (reattends entire context). Cached decode is O(1) per token (attend only the new token against cached keys). The 7.53x speedup reflects the quadratic-vs-linear cost difference at gen_len=128.

   **TTFT / TPS grid (cache mode, ctx 256/512/1024, batch 1/2/4, gen=64):**

   | ctx_len | B=1 | B=2 | B=4 |
   |---------|-----|-----|-----|
   | 256 | 100ms / 56.0 TPS | 170ms / 55.7 TPS | 323ms / 56.4 TPS |
   | 512 | 189ms / 49.4 TPS | 375ms / 50.2 TPS | 675ms / 51.0 TPS |
   | 1024 | 463ms / 41.8 TPS | 872ms / 40.3 TPS | 2015ms / 42.6 TPS |

   TTFT scales ~quadratically with context (prefill O(T²)) and linearly with batch size. Decode TPS is stable across batch sizes (~56 at ctx=256, ~42 at ctx=1024) — memory bandwidth bound, not compute bound.

**Key files:**
| File | What it does |
|------|-------------|
| `llm_lab/serving/engine.py` | `Engine` class: prefill/decode split, generate loop, prefill_batch |
| `llm_lab/serving/kv_cache.py` | `kv_append`, `kv_truncate`, `apply_sliding_window` |
| `llm_lab/serving/diagnostics.py` | `write_cache_divergence_report()` — first-divergence JSON |
| `scripts/serving/bench_inference.py` | Benchmark harness: single-point + grid sweep, writes cache.json/recompute.json |

**Tests (5 tests, all pass):**
| Test | What it verifies |
|------|-----------------|
| `test_engine_prefill_decode_structure.py` | Prefill called once, decode called N times, counter bookkeeping correct |
| `test_kv_cache_ops.py` | Append grows T, truncate keeps last N, sliding window applies per-layer |
| `test_cache_equivalence_stepwise.py` | Cached logits ≈ recompute logits per step, greedy tokens match |
| `test_cache_divergence_diagnostic.py` | Diagnostic report identifies divergence step and layer |
| `test_bench_outputs_schema.py` | Benchmark script writes required JSON keys, thresholds enforced (CPU ≤ 1.75x) |

---

### Phase 2 — Decode Controls: Stop, Sample, Penalize

**What was delivered:**

1. **Stop/EOS/context handling:**
   - Stop taxonomy: `eos` (EOS token seen), `stop_token` (custom stop token IDs), `stop_string` (text-level match), `max_new_tokens` (length limit)
   - Stop markers are **excluded** from visible completion — EOS and stop tokens are detected but not appended to output
   - Stop-string detection uses earliest-match: if multiple stop strings match, the first boundary wins
   - Context truncation keeps last `block_size` tokens across prompt+generated, with coupled KV-cache truncation maintaining the invariant `prompt_len + gen_len ≤ block_size` and `cache_len ≤ block_size`

2. **Sampling pipeline (strict order):**
   ```
   greedy/temperature==0 → short-circuit to argmax
   repetition penalty → penalize previously generated tokens (CTRL-style: divide/multiply logits)
   frequency penalty → subtract penalty × count(token) from logits
   temperature → scale logits by 1/T
   top-k → zero out all but top-k logits
   top-p (nucleus) → keep smallest prefix with cumulative probability ≥ p
   multinomial → sample from resulting distribution
   ```

3. **Determinism:** Per-step local seed `seed + step_idx` ensures reproducible generation without global RNG state pollution.

4. **Known failure modes (documented):**
   - Stop-string overlap at token boundaries: a stop string that spans two tokens may be detected one token late
   - Truncation/cache drift: after truncation, the KV cache is a sliding window approximation of the full context
   - Token vs text stop precedence: token-level stops (EOS, stop_token) have higher precedence than text-level stops (stop_string)

**Key files:**
| File | What it does |
|------|-------------|
| `llm_lab/serving/decode_controls.py` | `should_stop_tokens()`, `should_stop_text()`, `apply_context_truncation()`, `truncate_kv_cache_to_block_size()` |
| `llm_lab/serving/sampling.py` | `apply_temperature()`, `top_k_filter()`, `top_p_filter()`, `apply_repetition_penalty()`, `apply_frequency_penalty()`, `select_next_token_id()` |

**Tests (3 tests, all pass):**
| Test | What it verifies |
|------|-----------------|
| `test_engine_integration.py` | Full generate() with greedy, temperature, top-k, stop tokens — cache equivalence gate stays green |
| `test_sampling_and_penalties.py` | Each sampling transform changes distribution in expected direction; deterministic seed reproducibility |
| `test_stop_eos_truncation.py` | EOS stops generation, stop strings trim correctly, context truncation maintains block_size invariant, cache-vs-recompute consistency after truncation |

**Metrics contract:** `generated_tokens` in output excludes stop markers. `cache_len` reflects post-truncation state. `all_token_ids` reflects the current sliding window after truncation, not the full history.

---

### Phase 3 — HTTP API + SSE Streaming + Metrics + Batching + Rate Limiting

**What was delivered:**

1. **FastAPI endpoints:**
   - `POST /generate` — synchronous generation, returns JSON with completion_text, token_ids, stop_reason, metrics, safety_flags
   - `POST /stream` — SSE token streaming: incremental `event: token` chunks with `{index, token_id, token_text}`, terminal `event: final` with full metrics payload. Stop-string holdback prevents leaking partial marker tails during streaming.
   - `GET /health` — returns `{status: "ok", block_size, max_cache_len}`
   - `GET /metrics` — Prometheus-style text format aggregating requests_total, prompt/completion token counts, TTFT histogram, error categories

2. **Latency instrumentation (every response includes):**
   - `ttft_ms` — time from request start to first token selected (includes prefill + first decode)
   - `prefill_ms` — model forward time for the prompt
   - `decode_ms_total` — total decode loop time
   - `decode_ms_per_token` — average per-token decode latency
   - `tokens_per_sec` — steady-state decode throughput (1000 / decode_ms_per_token)

3. **Privacy-preserving logging:**
   - Every request logged as JSON with `request_id`, `prompt_hash` (SHA-256), `prompt_len`, `completion_len`, all latency fields, `stop_reason`, `error_category`
   - Raw prompt text is **not logged by default** — requires explicit `log_raw_prompts=True` in config
   - `request_id` derived from `X-Request-Id` header (if valid) or UUID4

4. **Prefill batching (mandatory, right-padding):**
   - `engine.prefill_batch(batch_prompt_ids)` takes ragged token-id lists, right-pads with attention mask, runs one batched forward pass
   - Per-sequence cache slices are extracted post-forward for independent decode
   - Equivalence verified: batched prefill per-sequence logits ≈ individual prefill logits

5. **Rate limiting:**
   - Fixed-window token bucket per client key (derived from `X-Request-Id` header → `X-Forwarded-For` → client IP)
   - Returns HTTP 429 with `Retry-After` header when limit exceeded
   - Rate-limited requests are still observed in metrics/logging for visibility

6. **Safety integration in API:**
   - Pre-filter: `should_refuse_prompt()` short-circuits generation on PII-containing prompts, returns refusal text
   - Post-filter: `postprocess_generated_text()` checks generated output for PII/profanity, replaces with refusal if triggered
   - Safety flags and `refusal_applied` boolean in every response

**Key files:**
| File | What it does |
|------|-------------|
| `llm_lab/serving/api.py` | FastAPI app factory: `/generate`, `/stream`, `/health`, `/metrics` with rate-limit + safety + metrics wiring |
| `llm_lab/serving/schemas.py` | `GenerateRequest` (Pydantic: prompt, max_new_tokens, temperature, top_k, top_p, stop_strings, seed, return_logprobs) + `GenerateResponse` |
| `llm_lab/serving/stream.py` | `iter_token_sse()` — decode loop yielding SSE chunks, holdback logic for stop-string safety, `sse_encode()` |
| `llm_lab/serving/batching.py` | `right_pad_and_mask()` — right-pad ragged sequences, build attention mask |
| `llm_lab/serving/metrics.py` | `RequestMetrics` dataclass, `MetricsStore` (thread-safe append + Prometheus text render) |
| `llm_lab/serving/logging.py` | `build_privacy_log_record()` — SHA-256 prompt hash, no raw text by default |
| `llm_lab/serving/rate_limit.py` | `RateLimiter` — fixed-window per-key with clock injection for testing |
| `llm_lab/serving/config.py` | `ServingConfig` — rate limit params, log settings, clock injection |
| `scripts/serving/serve.py` | CLI: `python scripts/serving/serve.py --package <path> --loader nanollama` |
| `scripts/serving/serving_client.py` | CLI streaming client for manual testing |

**Tests (5 tests, all pass):**
| Test | What it verifies |
|------|-----------------|
| `test_api_generate.py` | /generate returns valid response schema, stop reasons work, safety refusal triggers, error codes correct |
| `test_api_stream_sse.py` | /stream emits token events + final event, SSE format correct, safety refusal in streaming |
| `test_metrics_and_logging.py` | MetricsStore counts requests/tokens correctly, /metrics endpoint returns text, log records contain required fields without raw prompts |
| `test_prefill_batching_equivalence.py` | Batched prefill per-sequence logits match individual prefill, no NaN in batch |
| `test_rate_limit.py` | Rate limiter returns 429 after threshold, Retry-After header present, FakeClock advances correctly |

---

### Phase 4 — Precision & Quantization

**Why this matters:** Memory dominates serving cost. A 127M model at fp32 is 510MB; at int8 it's 155MB. On memory-constrained devices (edge, mobile, shared GPU), the 3.3x reduction is the difference between "fits" and "doesn't fit". The trade-off is PPL drift and latency change.

**What was delivered:**

1. **Precision policy:**
   - `normalize_requested_dtype()` maps aliases (float16→fp16, bfloat16→bf16, etc.)
   - `runtime_precision_decision()` checks device support and falls back with reason string (e.g., "fp16 unavailable on cpu, fell back to fp32")
   - `cast_model_for_inference()` casts model to target dtype but keeps norm layers (LayerNorm, RMSNorm) in fp32 for numerical stability

2. **int8 dynamic quantization:**
   - `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` with qnnpack backend
   - Applied to all Linear layers; Embedding layers untouched
   - No calibration dataset needed (dynamic quantization quantizes weights, not activations)
   - `describe_quant_runtime()` reports availability, backend, coverage, and fallback reason

3. **Measured results (NanoLlama 8L, CPU M1 Pro):**

   | Mode | prefill_ms | TTFT_ms | decode_ms/tok | tokens/sec | PPL | Memory |
   |------|-----------|---------|---------------|------------|-----|--------|
   | fp32 | 118.1 | — | 17.6 | 56.7 | 268.2 | 511 MB |
   | int8 | 341.1 | — | 13.9 | 72.1 | 284.3 | 155 MB |

   **Analysis:**
   - int8 **decode is 1.27x faster** (13.9 vs 17.6 ms/tok) — smaller weights → less memory bandwidth
   - int8 **prefill is 2.9x slower** (341 vs 118 ms) — dynamic quantization overhead on the compute-bound prefill path
   - int8 **memory is 70% smaller** (155 vs 511 MB)
   - int8 **PPL drift is +6.0%** (268.2 → 284.3) — acceptable for most use cases
   - **Recommendation:** int8 is the CPU default for memory-constrained deployment; fp32 for latency-sensitive prefill-heavy workloads
   - fp16/bf16 are not natively supported on CPU M1 and normalize to fp32; they would show real benefit on CUDA

   **Comparison with toy model (int8 scaling insight):**

   | Metric | Toy MHA (8.4M) | NanoLlama (127.6M) |
   |--------|---------------|-------------------|
   | int8 decode vs fp32 | 0.65x (slower) | 1.27x (faster) |
   | int8 PPL drift | +0.3% | +6.0% |

   int8 dynamic quantization hurts the tiny model (compute overhead dominates) but helps NanoLlama (memory bandwidth savings dominate at 127.6M params).

4. **Cache correctness under quantization:**
   - fp32: cached-vs-recompute max_abs_diff **6.68e-06** at atol 1e-4 (well within tolerance)
   - int8: cached-vs-recompute max_abs_diff **0.09075** at atol 0.1 with one near-threshold case; greedy exact match fp32 vs int8 on short prompt: **true**

**Key files:**
| File | What it does |
|------|-------------|
| `llm_lab/serving/precision.py` | Dtype normalization, device support check, model casting with norm-layer exceptions |
| `llm_lab/serving/quant.py` | `maybe_quantize_model()`, `quant_backend_is_available()`, `describe_quant_runtime()` |
| `scripts/serving/quant_sweep.py` | Sweep harness: runs fp32/fp16/bf16/int8, writes precision_matrix + results + bench/ppl JSONs |
| `scripts/serving/quant_report.py` | Generates quant_report.md with field definitions, memory/latency/PPL tables, recommendation |
| `scripts/serving/quant_audit.py` | Full audit orchestrator: runs tests, sweep, report, writes traceability CSV + completion report |

**Tests (5 tests, all pass):**
| Test | What it verifies |
|------|-----------------|
| `test_precision_correctness.py` | fp16/bf16 greedy matches fp32 (or drift within tolerance), int8 stability, cache equivalence holds across precisions |
| `test_precision_policy.py` | Engine loads requested precision or explains fallback, service surface unchanged across dtypes |
| `test_quant_backend_adapter.py` | int8 backend reports unavailable cleanly on non-CPU, tokenizer artifacts preserved through quantization |
| `test_quant_audit_script.py` | Audit script writes all 13 required artifacts, module status has correct rationale, metrics summary has run_state |
| `test_quant_eval_report.py` | Streaming NLL outputs required schema, quant sweep writes all artifacts with correct run_state semantics |

---

### Phase 5 — Eval Suite + Safety + Evidence Pack

**Why this matters:** Running a model is table stakes. The investigation + evidence layer — cache divergence diagnostics, latency curves across operating points, memory economics with formula + measurements — is what makes the serving claims falsifiable. The evidence pack is the deliverable, not the code.

**What was delivered:**

1. **Safety filters (pre + post):**
   - **Pre-filter** (`should_refuse_prompt`): scans prompt for PII patterns (email via `\b[A-Za-z0-9._%+-]+@...`, phone, SSN-like `\d{3}-\d{2}-\d{4}`). If triggered, generation is short-circuited and a refusal template is returned.
   - **Post-filter** (`postprocess_generated_text`): scans generated text for PII + profanity (intentionally narrow list: 4 terms for deterministic, low-false-positive regression). If triggered, output is replaced with refusal text.
   - Both filters integrated into `/generate` and `/stream` endpoints.

2. **Prompt suite runner:**
   - `data/serving_eval/prompts.jsonl` — 6 test cases bucketed by: short_prompt, long_prompt, repetition_trap, stop_trap, code_like, safety_probe
   - `eval/prompt_suite.py` — loads cases, validates schema (case_id, bucket, prompt, max_new_tokens), runs against Engine or HTTP backend, compares outputs for backend parity
   - `data/serving_eval/rubric.md` — evaluation criteria (not exact answers)
   - Results: **6/6 cases pass, 0 errors, 0 false-positive safety flags**

3. **Streaming perplexity eval:**
   - `eval/ppl.py::evaluate_streaming_nll()` — sliding-window NLL computation with configurable stride and max_seq_len
   - Handles out-of-vocabulary tokens gracefully (fallback tokenization)
   - Used for quant drift measurement: PPL(fp32) vs PPL(int8) to quantify quality impact

4. **Evidence pack artifacts:**
   - **Cache equivalence report:** step-by-step max |logit_diff|, greedy token match per step, first-divergence diagnostic
   - **TTFT vs TPS curves:** grid of context lengths × batch sizes × precision modes with measured latency breakdown
   - **KV-cache memory economics:** formula `2 × B × T × n_layers × n_kv_heads × head_dim × bytes_per_elem` + measured values for MHA vs GQA architectures

5. **Report builder:**
   - `eval/report.py` — builds markdown/JSON reports from raw benchmark data
   - `build_ttft_tps_table()`, `build_kv_memory_economics_table()`, `estimate_kv_cache_bytes()`, `build_eval_report()`, `build_evidence_pack_manifest()`

**Key files:**
| File | What it does |
|------|-------------|
| `llm_lab/serving/safety.py` | PII regex (email/phone/SSN), profanity filter, `should_refuse_prompt()`, `postprocess_generated_text()`, `apply_safety_policy()` |
| `llm_lab/eval/ppl.py` | `evaluate_streaming_nll()` — sliding-window NLL with stride |
| `llm_lab/eval/prompt_suite.py` | Case loader/validator, Engine + HTTP backends, backend parity comparison |
| `llm_lab/eval/report.py` | TTFT/TPS table, KV memory economics, quant report, evidence pack manifest |
| `llm_lab/eval/research_lane.py` | Data integrity checks, stability analysis, text quality metrics |
| `data/serving_eval/prompts.jsonl` | 6 prompt cases across 6 buckets |
| `data/serving_eval/rubric.md` | Evaluation rubric |
| `scripts/eval/eval_prompt_suite.py` | CLI: runs prompt suite, writes reports |
| `scripts/eval/eval_perplexity.py` | CLI: runs streaming PPL eval |
| `scripts/eval/make_evidence_pack.py` | CLI: assembles evidence pack from bench/quant/eval artifacts |

**Tests (8 tests, all pass):**
| Test | What it verifies |
|------|-----------------|
| `test_safety_filters.py` | PII detection (email triggers, phone triggers, SSN triggers), profanity detection, refusal template, apply_safety_policy pipeline, privacy_meta contains hash not raw text |
| `test_eval_report.py` | Bucket/safety/latency summaries computed correctly, report contains required sections, report writes to file |
| `test_eval_suite_script.py` | Script runs both engine+HTTP backends, writes parity artifacts, detects mismatches |
| `test_evidence_pack.py` | Grid validator fails on missing points/accepts full grid, evidence pack manifest points to real files, cache equivalence report surfaces receipt fields |
| `test_prompt_suite_data_contract.py` | prompts.jsonl schema validation (required fields, bucket constraints, field types) |
| `test_prompt_suite_runner.py` | Case execution, output schema, error handling |
| `test_research_lane_guards.py` | Data integrity checks, stability analysis guards |
| `test_quant_eval_report.py` | Streaming NLL schema, quant sweep artifacts, precision recommendation |

---

### Tiktoken Integration (cross-cutting, zero modification to existing serving code)

**Why needed:** NanoLlama was trained with tiktoken GPT-2, not the sp16k SubwordTokenizer. The serving layer was built for sp16k. Rather than modifying the serving layer, a thin adapter pattern was used.

**Gaps addressed:**
1. **G1 (Critical):** `tiktoken.decode()` raises `KeyError` on IDs 50257–50303 (model padding range). `TiktokenWrapper.decode()` filters these before calling tiktoken.
2. **G2 (Critical):** Raw tiktoken `Encoding` has no `token_to_id()` or `eos_token_id`. Without the wrapper, `resolve_eos_token_id()` returns `None` and generation never stops on EOS.
3. **G3 (High):** tiktoken has no `pad_token_id`. Without the wrapper, batch padding uses ID 0 (the `!` token in GPT-2) instead of a proper pad.

**What was delivered:**
- `TiktokenWrapper` (35 lines) — satisfies the full serving tokenizer interface
- `load_nanollama_checkpoint()` (42 lines) — loads raw `.pt` format with different checkpoint keys
- `build_engine_from_package(loader="nanollama")` — explicit loader parameter, no auto-detection magic
- `--loader nanollama` CLI flag on all 5 serving/eval scripts

**Key files:**
| File | What it does |
|------|-------------|
| `llm_lab/core/tokenization/tiktoken_wrapper.py` | `TiktokenWrapper`: encode, decode (with OOV filter), token_to_id, eos/pad attributes |
| `llm_lab/core/package/nanollama_loader.py` | `load_nanollama_checkpoint()`: loads config from `ckpt["config"]`, weights from `ckpt["model_state_dict"]` |

**Tests (11 tests, all pass):**
| Test | What it verifies |
|------|-----------------|
| `test_tiktoken_wrapper.py` | Encode/decode roundtrip, eos=50256, pad=50256, token_to_id for endoftext, OOV ID filtering, empty list decode, special token encoding, vocab_size=50257 |

---

## 6. Test summary

| Category | Files | Tests | Status |
|----------|-------|-------|--------|
| Serving (Phase 0–4 + tiktoken) | 24 | 143 | All pass |
| Eval (Phase 5) | 8 | 9 | All pass |
| **Total** | **32** | **152** | **152 pass, 0 fail** |

---

## 7. Experiment artifacts

### Toy model results (development scaffolding)

All ran on RoPE MHA 8.4M model, CPU fp32. See `experiments/serving_model_provenance.md`.

| Directory | Contents |
|-----------|----------|
| `experiments/serving_bench/` | Cache vs recompute, grid sweep (ctx 128/256/512, B 1-8) |
| `experiments/serving_quant/` | fp32/int8 audit, precision matrix, full traceability |
| `experiments/serving_reports/` | API smoke, cache equivalence, TTFT/TPS, prompt suite, eval report |
| `experiments/serving_eval/` | Validation by context length |

### NanoLlama 8L results (production benchmarks)

All ran on NanoLlama 8L 127.6M model, CPU M1 Pro. See `experiments/nanollama_serving_provenance.md`.

Benchmark configs chosen for this model: prompt=256 gen=128 for single-point (block_size=1024 means longer sequences are realistic); ctx 256/512/1024 for grid (tests full block_size range); gen=64 for grid and quant (enough for steady-state decode).

| Directory | Contents |
|-----------|----------|
| `experiments/nanollama_serving_bench/` | cache.json + recompute.json (7.53x speedup at prompt=256/gen=128), cache_grid.json (9 rows: ctx 256/512/1024 × B 1/2/4, gen=64), recompute_grid.json, cache_equivalence_receipts.json |
| `experiments/nanollama_serving_quant/` | fp32 vs int8 at prompt=256/gen=64: 70% memory savings, 1.27x decode speedup, +6.0% PPL drift |
| `experiments/nanollama_serving_reports/` | Prompt suite (6/6 pass, 0 errors), eval report |
| `experiments/nanollama_serving_provenance.md` | Full provenance, benchmark configs, key results, toy-vs-NanoLlama comparison |

---

## 8. Scripts reference

```
scripts/
├── serving/
│   ├── serve.py                    # python scripts/serving/serve.py --package <path> --loader nanollama
│   ├── serving_client.py           # Streaming CLI client (engine or HTTP)
│   ├── bench_inference.py          # Cache vs recompute + grid sweep (--context-lens, --batch-sizes)
│   ├── quant_sweep.py              # fp32/fp16/bf16/int8 sweep (--package, --loader)
│   ├── quant_report.py             # Generate quant_report.md from sweep results
│   ├── quant_audit.py              # Full audit: tests + sweep + report + traceability
│   └── run_nanollama_benchmarks.sh # One-command: all NanoLlama benchmarks
├── eval/
│   ├── eval_prompt_suite.py        # Prompt suite with safety checks (--backend engine/http/both)
│   ├── eval_perplexity.py          # Streaming PPL (--package, --text-path, --max-seq-len)
│   ├── eval_hellaswag.py           # HellaSwag 10K benchmark
│   ├── eval_suite.py               # 11-test eval suite
│   └── make_evidence_pack.py       # Assemble evidence pack from artifacts
└── interact.py                     # Interactive REPL (tiktoken + NanoLlama reference impl)
```

All serving/eval scripts accept `--loader nanollama` for NanoLlama checkpoints (default: `--loader package` for sp16k).

---

## 9. Git structure

**Remote:** `origin` → `https://github.com/quicksilverTrx/foundry-llm`

**Commit conventions:**
- `feat(serving):` — new serving/eval functionality
- `feat(core):` — core model/training changes
- `fix(core):` — bug fixes in core
- Commits are grouped by logical phase, not chronological order
- Each commit should leave all 152 tests passing

---

## 10. Code structure (module dependency map)

```
┌──────────────────────────────────────────────────────────┐
│  scripts/serving/                                         │
│  (serve.py, bench_inference.py, quant_sweep.py, ...)     │
└──────────────────┬───────────────────────────────────────┘
                   │ imports
┌──────────────────▼───────────────────────────────────────┐
│  llm_lab/serving/                                         │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ api.py  │→ │ engine.py│→ │kv_cache.py│  │ safety.py │ │
│  │ stream.py│→ │          │→ │sampling.py│  │ metrics.py│ │
│  │ schemas │  │ _shared  │  │decode_ctrl│  │ logging.py│ │
│  └─────────┘  └────┬─────┘  └──────────┘  └───────────┘ │
│                     │ batching.py, config.py, rate_limit  │
│                     │ precision.py, quant.py, diagnostics │
└─────────────────────┼────────────────────────────────────┘
                      │ imports (one direction only)
┌─────────────────────▼────────────────────────────────────┐
│  llm_lab/core/                                            │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ model/     │  │ package/     │  │ tokenization/    │  │
│  │  gpt.py    │  │  io.py       │  │  subword_tok.py  │  │
│  │  attention │  │  nanollama_  │  │  tiktoken_wrap.py│  │
│  │  blocks    │  │   loader.py  │  │  char_tok.py     │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌────────────┐  ┌──────────────┐                        │
│  │ train/     │  │ data/        │                        │
│  │  trainer   │  │  shard_loader│                        │
│  │  shard_tr  │  │  pretok_*    │                        │
│  └────────────┘  └──────────────┘                        │
└──────────────────────────────────────────────────────────┘
                      ↑ NEVER imports serving
┌─────────────────────┼────────────────────────────────────┐
│  llm_lab/eval/      │ imports serving (for benchmarking)  │
│  ppl.py, prompt_suite.py, report.py, research_lane.py    │
└──────────────────────────────────────────────────────────┘
```

**Hard boundary:** `core/` has zero imports from `serving/`. `eval/` imports from `serving/` only for `build_engine_from_package` in `report.py`. `serving/` imports from `core/` for model loading and the `PastKeyValues` type.

**Tokenizer interface contract** (what `Engine` requires from any tokenizer object):
```python
# Required methods:
def encode(self, text: str) -> list[int]: ...
def decode(self, ids: Sequence[int]) -> str: ...

# Optional (probed by resolve_eos_token_id):
def token_to_id(self, token: str) -> int: ...

# Optional attributes (with fallback defaults):
eos_token_id: int    # fallback: None (EOS never triggers)
pad_token_id: int    # fallback: 0
```

Both `SubwordTokenizer` (sp16k) and `TiktokenWrapper` (GPT-2) satisfy this interface. The Engine is tokenizer-agnostic.

**Checkpoint format contract:**

| Format | Loader | Config source | Weights key | Tokenizer |
|--------|--------|--------------|-------------|-----------|
| sp16k package dir | `load_model_package()` | `model_config.json` file | `"model_state"` | `tokenizer/` dir (SentencePiece artifacts) |
| NanoLlama raw `.pt` | `load_nanollama_checkpoint()` | `ckpt["config"]` dict | `"model_state_dict"` | tiktoken GPT-2 (constructed, not serialized) |

`build_engine_from_package(loader="package"|"nanollama")` dispatches to the correct loader. Both return `(MiniGPTConfig, tokenizer, MiniGPT)`.

---

## 11. Known limitations and non-goals

| Limitation | Why | Impact |
|-----------|-----|--------|
| Decode is B=1 only | Batched decode requires careful cache management across sequences with different lengths; prefill batching gets 80% of throughput gain for 20% complexity | Single-request decode; throughput limited by sequential generation |
| No continuous/dynamic batching | Would need a request queue, scheduler, and preemption logic — full serving framework scope | Cannot pack multiple concurrent requests into one decode batch |
| No speculative decoding | Requires a draft model + acceptance/rejection logic; research project scope | No latency reduction from parallel token speculation |
| No prefix caching | Each request prefills from scratch; would need a prompt-keyed cache with eviction | Repeated prompts re-prefill every time |
| int8 prefill is 2.9x slower | Dynamic quantization overhead on compute-bound path; weight-only quant doesn't help prefill | Use fp32 for prefill-heavy workloads, int8 for memory-constrained decode |
| fp16/bf16 falls back to fp32 on CPU | M1 MPS doesn't support bf16; CPU fp16 is not natively accelerated | Need CUDA GPU for real fp16/bf16 benefit |
| Safety is heuristic | Regex PII, 4-word profanity list — not a learned classifier | Low false-positive rate but not comprehensive; suitable for demonstration, not production safety |
| No chat template | NanoLlama is a base LM; no role tokens in training data | Use plain text prompting with in-context delimiters |

---

## 12. Repo organization

Scripts are organized into `scripts/{core, serving, eval, data, research, pretrain}/` subdirectories.

---

## 13. Related documents

| Document | Purpose |
|----------|---------|
| `docs/serving_nanollama_tiktoken.md` | Detailed gap analysis for tiktoken integration: vocab mismatch, EOS resolution, checkpoint format differences |
| `experiments/serving_model_provenance.md` | Provenance for toy model experiment runs (8.4M RoPE MHA) |
| `experiments/nanollama_serving_provenance.md` | Provenance for NanoLlama experiment runs (127.6M) with full results + toy-vs-real comparison |
| `CLAUDE.md` | Repo conventions, script layout, quick-start commands |
| `README.md` | Public-facing overview with evolution log, capabilities, serving API examples |

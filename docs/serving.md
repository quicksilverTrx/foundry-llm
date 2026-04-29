# Serving foundry-llm models

A short guide to loading a trained model and serving it with KV-cache decoding, streaming, and quantization.

The serving layer wraps the core MiniGPT and is fully tokenizer-agnostic. It supports two model formats out of the box: **sp16k packages** (SubwordTokenizer artifacts) and **raw NanoLlama checkpoints** (tiktoken GPT-2). Adding a third format means writing a thin loader; no serving code needs to change.

---

## Two model formats

| Format | Loader | Config source | Weights key | Tokenizer |
|--------|--------|--------------|-------------|-----------|
| sp16k package directory | `load_model_package(pkg_dir)` | `model_config.json` file | `"model_state"` | `tokenizer/` directory (SentencePiece artifact) |
| Raw NanoLlama `.pt` | `load_nanollama_checkpoint(ckpt_path)` | `ckpt["config"]` dict | `"model_state_dict"` | `TiktokenWrapper` (constructed, not serialized) |

`build_engine_from_package(loader="package"|"nanollama")` dispatches to the right loader. Both return `(MiniGPTConfig, tokenizer, MiniGPT)`.

---

## Serve NanoLlama 8L over HTTP

```bash
python scripts/serving/serve.py \
  --package experiments/tinyllama_pretrain_2026-03-31/phase6/ckpts/step_04768_model_only.pt \
  --loader nanollama --device cpu --dtype fp32
```

Endpoints:

```bash
# Sync generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_new_tokens": 64, "temperature": 0.8, "top_k": 40}'

# Server-sent event streaming
curl -N -X POST http://127.0.0.1:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 128, "temperature": 0.7}'

# Health + metrics
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

Every response carries `ttft_ms`, `prefill_ms`, `decode_ms_per_token`, `tokens_per_sec`, `stop_reason`, and `safety_flags`.

---

## Programmatic use

```python
import torch
from llm_lab.core.package.nanollama_loader import load_nanollama_checkpoint
from llm_lab.serving.engine import Engine

cfg, tokenizer, model = load_nanollama_checkpoint(
    "experiments/tinyllama_pretrain_2026-03-31/phase6/ckpts/step_04768_model_only.pt",
    device="cpu",
)

engine = Engine(
    model=model,
    tokenizer=tokenizer,
    block_size=cfg.block_size,
    max_cache_len=cfg.block_size,
)

out = engine.generate(
    prompt_ids=tokenizer.encode("Once upon a time"),
    attention_mask=None,
    max_new_tokens=64,
    temperature=0.7,
    top_k=40,
    eos_token_id=tokenizer.eos_token_id,
)
print(out["completion_text"])
print(out["metrics"])  # ttft_ms, decode_ms_per_token, tokens_per_sec, ...
```

---

## Tokenizer interface

The Engine requires only:

```python
def encode(text: str) -> list[int]
def decode(ids: Sequence[int]) -> str
```

Optional, used by EOS resolution and batching:

```python
def token_to_id(token: str) -> int    # probed for "<|endoftext|>"
eos_token_id: int | None              # falls back to None if absent
pad_token_id: int                     # falls back to 0 if absent
```

`SubwordTokenizer` (sp16k) and `TiktokenWrapper` both satisfy this. Any custom tokenizer that implements these methods plugs in directly.

### TiktokenWrapper notes

The wrapper handles three real gaps in raw `tiktoken.Encoding`:

- **Decode safety**: filters IDs ≥ `n_vocab` (NanoLlama's lm_head emits 50,304 logits; tiktoken GPT-2 only knows 0–50,256). Prevents `KeyError` when the model's padding-range IDs reach `decode()`.
- **EOS resolution**: exposes `eos_token_id = 50256` and `token_to_id("<|endoftext|>")`.
- **Padding**: sets `pad_token_id = eos_token_id` (standard convention for models without a dedicated pad).

---

## Code structure

```
llm_lab/serving/
├── engine.py          prefill + decode loop, batched prefill, generate()
├── kv_cache.py        append, truncate, sliding window
├── decode_controls.py stop taxonomy (eos, stop_token, stop_string, max_new_tokens) + truncation
├── sampling.py        temperature, top-k, top-p, repetition + frequency penalty
├── api.py             FastAPI: /generate, /stream, /health, /metrics
├── stream.py          SSE iterator with stop-string holdback
├── batching.py        right-pad + attention mask
├── precision.py       fp16/bf16/fp32 with device-aware fallback, norm layers kept fp32
├── quant.py           int8 dynamic quantization (qnnpack)
├── safety.py          PII/profanity filters + refusal template
├── metrics.py         per-request counters, Prometheus text rendering
├── logging.py         privacy-preserving logs (SHA-256 prompt hash)
├── rate_limit.py      fixed-window rate limiter
├── schemas.py         Pydantic request/response
└── _shared.py         shared helpers (sampling utils, EOS resolve, device family)

llm_lab/eval/
├── ppl.py             streaming sliding-window NLL
├── prompt_suite.py    case loader/runner with engine and HTTP backends
├── report.py          TTFT/TPS table, KV memory economics, evidence pack manifest
└── research_lane.py   data integrity and stability guards
```

The hard boundary `llm_lab/core/` never imports `llm_lab/serving/`. The serving layer wraps core; the model code does not depend on the engine.

---

## Performance notes (NanoLlama 8L, CPU M1 Pro)

Measured with `scripts/serving/bench_inference.py` and `scripts/serving/quant_sweep.py`. See `experiments/nanollama_serving_provenance.md` for full configs and environment.

**KV cache speedup** (prompt_len=256, gen_len=128, fp32):

| | Cached decode | Full recompute |
|---|---|---|
| decode_ms/token | 24.2 | 182.1 |
| tokens/sec | 41.3 | 5.5 |

→ **7.53× decode speedup**.

**TTFT vs context length** (cache mode, gen_len=64, batch_size=1):

| ctx_len | TTFT | TPS |
|---------|------|-----|
| 256 | 100 ms | 56 |
| 512 | 189 ms | 49 |
| 1024 | 463 ms | 42 |

TTFT scales ~quadratically with context (prefill is O(T²)). Steady-state decode TPS degrades modestly as the KV cache grows.

**int8 quantization** (prompt_len=256, gen_len=64):

| Mode | decode_ms/tok | TPS | PPL | Memory |
|------|---------------|-----|-----|--------|
| fp32 | 17.6 | 56.7 | 268.2 | 511 MB |
| int8 | 13.9 | 72.1 | 284.3 | 155 MB |

→ **70% memory reduction**, **1.27× decode speedup**, **+6% PPL drift**. int8 prefill is 2.9× *slower* (dynamic quantization overhead on the compute-bound path); the overall trade-off favors int8 for memory-constrained deployment and fp32 for prefill-heavy workloads.

---

## Known limits

- Decode is B=1 only. Prefill batching works; batched decode would need cache management across sequences with different lengths.
- No continuous batching, speculative decoding, or prefix caching.
- fp16/bf16 fall back to fp32 on CPU. Real benefit needs CUDA.
- Safety filters are heuristic (regex PII, narrow profanity list) — suitable for demonstration, not production.
- NanoLlama is a base LM with no chat template; use plain text prompting.

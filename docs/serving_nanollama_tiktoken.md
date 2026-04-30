# Serving NanoLlama with tiktoken

Context document for wiring `llm_lab/serving/` to the NanoLlama 8L checkpoint.
The serving layer was originally written for the sp16k `SubwordTokenizer`; NanoLlama
was trained with tiktoken GPT-2. This document captures every gap, the correct ID
values, and the minimal adapter work required.

---

## 1. The model

**NanoLlama 8L** — trained from scratch on FineWeb-Edu (10B tokens), not a
fine-tuned HuggingFace model.

Architecture (`configs/nanollama_8l.json`):

```json
{
  "vocab_size": 50304,
  "d_model": 768,
  "n_layers": 8,
  "n_heads": 12,
  "num_kv_heads": 4,
  "d_ff": 2048,
  "block_size": 1024,
  "norm_type": "rmsnorm",
  "mlp_type": "swiglu",
  "attention_type": "gqa",
  "pos_encoding_type": "rope",
  "arch_family": "nanollama",
  "logit_softcap": 30.0,
  "qk_norm": true
}
```

Model class: `llm_lab.core.model.gpt.MiniGPT` / `MiniGPTConfig`.

Forward signature:
```python
logits, past_key_values = model(input_ids, attention_mask, past_key_values, use_cache)
# logits: [B, T, vocab_size=50304]
```

---

## 2. Tokenizer: tiktoken GPT-2

All current training and inference scripts use `tiktoken.get_encoding("gpt2")`.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode(text, allowed_special="all")
text = enc.decode(list(ids))
```

Key facts:

| Property | Value |
|---|---|
| Vocab size (tiktoken) | 50,257 |
| Model `vocab_size` | 50,304 (padded to nearest 64 for hardware alignment) |
| Unused tail IDs | 50,257 – 50,303 (47 IDs, never seen during training) |
| `<|endoftext|>` ID | **50256** |
| `allowed_special` | Must pass `"all"` to avoid tiktoken raising on `<|endoftext|>` |

The 47 padding IDs are harmless: after training on GPT-2 tokens only, their
embedding rows and lm_head columns receive near-zero gradient and will never
be top-1 sampled. No special masking is required.

---

## 3. Control tokens: what exists and what does not

### 3a. sp16k reserved tokens (NOT for NanoLlama)

`llm_lab/core/tokenization/tokenizer_shared.py` defines:

```python
RESERVED_SPECIAL_TOKENS = {
    "<|pad|>":        0,
    "<|user|>":       1,
    "<|assistant|>":  2,
    "<|endoftext|>":  3,
}
```

These IDs belong to the **sp16k SentencePiece pipeline** — a prior experiment
to build a custom 16K vocab tokenizer. They have **no relationship** to NanoLlama.
The sp16k tokenizer was never used to train NanoLlama.

### 3b. tiktoken GPT-2 special tokens

| Token | ID | Exists in tiktoken GPT-2? |
|---|---|---|
| `<|endoftext|>` | 50256 | Yes — only special token |
| `<|pad|>` | — | **No** |
| `<|user|>` | — | **No** |
| `<|assistant|>` | — | **No** |
| `<|system|>` | — | **No** |

NanoLlama is a **pure language model** — no chat/role control tokens were
present in training data or the tokenizer. Using `<|user|>` etc. as string
delimiters in a prompt would encode them character-by-character as regular
text, not as single control token IDs.

### 3c. Adding chat tokens to tiktoken (future path)

tiktoken supports extending encodings with custom special tokens:

```python
import tiktoken
base = tiktoken.get_encoding("gpt2")
custom_enc = tiktoken.Encoding(
    name="gpt2_chat",
    pat_str=base._pat_str,
    mergeable_ranks=base._mergeable_ranks,
    special_tokens={
        **base._special_tokens,
        "<|user|>":       50257,
        "<|assistant|>":  50258,
        "<|system|>":     50259,
        "<|pad|>":        50260,
    },
)
```

**Critical constraint**: IDs 50257+ have no trained meaning in the base
NanoLlama weights. This approach only makes sense after supervised fine-tuning
with a dataset that uses these tokens.

---

## 4. The serving layer

### 4a. What it expects

`llm_lab/serving/engine.py` — `Engine.__init__` takes a pre-loaded model and
tokenizer. `Engine.from_package()` calls `load_model_package()` which expects
the **sp16k package directory format**:

```
package_dir/
  model_config.json
  tokenizer/          ← sp16k artifact (sentencepiece model + reserved_tokens.json)
  checkpoints/
    best_val.pt       ← checkpoint key: "model_state"
```

NanoLlama checkpoints are **raw `.pt` files** saved directly by the training
script. They are NOT in this package format. Checkpoint keys differ too:

| Key | sp16k package format | NanoLlama raw `.pt` |
|---|---|---|
| model weights | `"model_state"` | `"model_state_dict"` |
| config | loaded from `model_config.json` | `ckpt["config"]` (dict) |
| step | `ckpt.get("global_step")` | `ckpt.get("step")` |
| val loss | `ckpt.get("best_val_loss")` | `ckpt.get("val_loss")` |

### 4b. EOS resolution (the critical gap)

Both `api.py` and `stream.py` resolve EOS like this:

```python
def _resolve_eos_token_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "token_to_id"):           # sp16k path → returns 3
        return int(tokenizer.token_to_id("<|endoftext|>"))
    value = getattr(tokenizer, "eos_token_id", None) # fallback
    return int(value) if value is not None else None
```

With a raw tiktoken `Encoding` object, `token_to_id` does not exist and
`eos_token_id` is not an attribute — so this returns `None`, and generation
never stops on EOS. The correct EOS ID for NanoLlama is **50256**.

### 4c. Serving layer files

```
llm_lab/serving/
  engine.py          — Engine class: prefill + KV-cache decode loop
  api.py             — FastAPI REST endpoints (/generate, /stream, /health)
  stream.py          — SSE token streaming
  sampling.py        — top-k, top-p, temperature, frequency/repetition penalty
  decode_controls.py — EOS stop, stop-string stop, context truncation
  kv_cache.py        — sliding window KV cache
  precision.py       — bf16/fp16/fp32 casting policy
  quant.py           — quantisation stubs
  schemas.py         — Pydantic request/response schemas
  config.py          — ServingConfig dataclass
  rate_limit.py      — per-client rate limiting
  safety.py          — content filtering stubs
  logging.py         — structured request logging
  metrics.py         — latency/throughput counters
  diagnostics.py     — health + debug endpoints
  batching.py        — right-pad + attention mask helpers
  __init__.py
```

---

## 5. Required adapter work

### 5a. Minimal tiktoken tokenizer wrapper

The serving layer probes `token_to_id` and `eos_token_id`. A thin wrapper:

```python
import tiktoken

class TiktokenWrapper:
    def __init__(self, encoding: tiktoken.Encoding, eos_token_id: int = 50256):
        self._enc = encoding
        self.eos_token_id = eos_token_id
        self.vocab_size = self._enc.n_vocab  # 50257 for gpt2

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def token_to_id(self, token: str) -> int:
        # Handles the _resolve_eos_token_id probe in api.py / stream.py
        ids = self._enc.encode(token, allowed_special="all")
        if len(ids) != 1:
            raise ValueError(f"{token!r} does not map to a single token ID")
        return ids[0]
```

With this wrapper, `_resolve_eos_token_id(wrapper)` calls `token_to_id("<|endoftext|>")`
and correctly returns **50256**.

### 5b. Loading NanoLlama into Engine

```python
import torch
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.serving.engine import Engine
from llm_lab.serving.precision import runtime_precision_decision

ckpt_path = "path/to/nanollama.pt"
device = "cuda"

ckpt = torch.load(ckpt_path, map_location="cpu")
cfg = MiniGPTConfig(**ckpt["config"])
model = MiniGPT(cfg).to(device).eval()
model.load_state_dict(ckpt["model_state_dict"])

import tiktoken
tokenizer = TiktokenWrapper(tiktoken.get_encoding("gpt2"))

requested_dtype = "bf16"
runtime_dtype, fallback_reason = runtime_precision_decision(requested_dtype, device)

engine = Engine(
    model=model,
    tokenizer=tokenizer,
    block_size=cfg.block_size,      # 1024
    requested_dtype=requested_dtype,
    runtime_dtype=runtime_dtype,
    runtime_fallback_reason=fallback_reason,
)
```

### 5c. No chat template (current state)

There is no chat template. For prompting, use plain text. If you want a
lightweight role separator that the model will statistically follow (without
fine-tuning), a common approach is to use distinctive string delimiters and
rely on in-context learning — e.g. `"User: ...\nAssistant:"`. The model has
no semantic attachment to these strings beyond what appears in pre-training
data.

---

## 6. Vocab size mismatch note

Model `vocab_size=50304`, tiktoken GPT-2 generates IDs 0–50256. The
lm_head output has 50304 logits; the tail 47 are padding. Options:

- **Do nothing** — they compete but never win (near-zero logit after training).
- **Mask at sampling time** — pass `logits[:, :50257]` before softmax if you
  want strict containment. Only needed if you observe degenerate output.

---

## 7. Reference implementation

`scripts/interact.py` is the working reference for tiktoken + NanoLlama
end-to-end inference. Read it before modifying the serving layer — it shows
the exact encode/decode calls, checkpoint loading keys, and model forward call.

---

## 8. Relevant file index

| Path | Purpose |
|---|---|
| `llm_lab/core/model/gpt.py` | `MiniGPT`, `MiniGPTConfig` |
| `llm_lab/serving/engine.py` | `Engine` — prefill + KV-cache decode |
| `llm_lab/serving/api.py` | FastAPI REST (`/generate`, `/stream`, `/health`) |
| `llm_lab/serving/stream.py` | SSE streaming, EOS resolution |
| `llm_lab/serving/sampling.py` | top-k, top-p, temperature, penalties |
| `llm_lab/serving/config.py` | `ServingConfig` dataclass |
| `llm_lab/serving/schemas.py` | Pydantic request/response schemas |
| `llm_lab/core/tokenization/tokenizer_shared.py` | sp16k `RESERVED_SPECIAL_TOKENS` (NOT for NanoLlama) |
| `llm_lab/core/package/io.py` | `load_model_package` (sp16k package format only) |
| `scripts/interact.py` | Working tiktoken + NanoLlama inference reference |
| `configs/nanollama_8l.json` | Model architecture config |

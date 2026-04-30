# Model Architecture Reference

Formal specification for all three model families in this repository. For the reasoning behind each design choice, see `docs/architecture_decisions.md`.

---

## MiniGPTConfig field reference

All models are instantiated from `MiniGPTConfig` in `llm_lab/core/model/gpt.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vocab_size` | int | required | Token vocabulary size. All trained models use 50304 (GPT-2 BPE + 47 padding tokens). |
| `d_model` | int | required | Residual stream / embedding dimension. |
| `n_layers` | int | required | Number of transformer blocks. |
| `n_heads` | int | required | Number of query heads in attention. |
| `d_ff` | int | required | FFN inner dimension. For SwiGLU: 3 matrices of shape `[d_model, d_ff]`; set so that `3 × d_model × d_ff` matches the GELU param budget `2 × d_model × d_ff_gelu`. |
| `block_size` | int | required | Maximum sequence length (tokens). |
| `dropout` | float | 0.0 | Dropout probability. Set to 0 for all production runs. |
| `norm_type` | `"layernorm"\|"rmsnorm"` | `"layernorm"` | Pre-norm type. All production models use `"rmsnorm"`. |
| `mlp_type` | `"gelu"\|"swiglu"\|"relu_squared"` | `"gelu"` | FFN activation. Production: `"swiglu"`. `"relu_squared"` is available for research (see `llm_lab/core/model/mlp.py`). |
| `attention_type` | `"mha"\|"gqa"` | `"mha"` | Attention variant. `"gqa"` requires `num_kv_heads`. |
| `num_kv_heads` | int \| None | None | KV head count for GQA. Must divide `n_heads` evenly. |
| `pos_encoding_type` | `"learned"\|"sinusoidal"\|"rope"` | `"learned"` | Positional encoding. Production: `"rope"`. |
| `rope_scaling_type` | `"none"\|"linear"` | `"none"` | RoPE frequency scaling for longer-context extension. |
| `rope_scaling_factor` | float | 1.0 | Scale factor for linear RoPE scaling. Must be ≥ 1.0. |
| `arch_family` | `"miniGPT"\|"nanollama"\|"swiftllama"` | `"miniGPT"` | Validates constraints: `"nanollama"` enforces GQA + RoPE + RMSNorm in `__post_init__`. |
| `tie_weights` | bool | False | Share `lm_head.weight` ↔ `token_embed.weight`. Not used in production (separate weights). |
| `use_sdpa` | bool | False | Use `F.scaled_dot_product_attention` (Flash Attention path). |
| `logit_softcap` | float \| None | None | Gemma-style cap: `lm_head_out = cap * tanh(lm_head_out / cap)`. Production: 30.0. |
| `qk_norm` | bool | False | RMSNorm on Q and K before dot-product (GQA path only). |
| `rope_fraction` | float | 1.0 | Fraction of head dims to rotate. `0.5` = rotate first half, pass-through second half. |
| `n_value_embeds` | int | 0 | Per-layer learned bias. Shape `[n, H_kv × head_dim]`; mean-pooled and added to V. |
| `use_x0_mixin` | bool | False | Per-layer residual mixing: `x = λ·x + λ₀·x₀` where `x₀` is embedding output. λ init=1, λ₀ init=0. |

---

## Initialization

All parameters use normal initialization. Key values (from `_init_weights`):

| Parameter | std |
|-----------|-----|
| `token_embed.weight` | 0.02 |
| `lm_head.weight` | 0.02 |
| Residual projections (`out_proj`, `w_down`, `fc2`) | `0.02 / √(2 × n_layers)` |
| All other 2D weights | `1 / √d_model` |
| 1D params (biases, norms) | zero (unchanged) |

Biases: no bias on `lm_head`. Attention projections also bias=False in the GQA path.

---

## Model family comparison

### Architectures

| Field | NanoLlama 8L (v1) | NanoLlama v2 | SwiftLlama-350M |
|-------|-------------------|-------------|-----------------|
| `arch_family` | `nanollama` | `nanollama` | `swiftllama` |
| `vocab_size` | 50304 | 50304 | 50304 |
| `n_layers` | 8 | 8 | 22 |
| `d_model` | 768 | 768 | 1024 |
| `n_heads` | 12 | 12 | 16 |
| `num_kv_heads` | 4 | 4 | 4 |
| GQA ratio | 3:1 | 3:1 | 4:1 |
| `d_ff` | 2048 | 2048 | 2730 |
| `block_size` | 1024 | 1024 | 4096 |
| `norm_type` | rmsnorm | rmsnorm | rmsnorm |
| `mlp_type` | swiglu | swiglu | swiglu |
| `pos_encoding_type` | rope | rope | rope |
| `rope_fraction` | 1.0 | **0.5** | **0.5** |
| `logit_softcap` | 30.0 | 30.0 | 30.0 |
| `qk_norm` | True | True | True |
| `n_value_embeds` | 0 | **2** | **2** |
| `use_x0_mixin` | False | **True** | **True** |
| `dropout` | 0.0 | 0.0 | 0.0 |
| `tie_weights` | False | False | False |

### Parameter counts

| Component | NanoLlama 8L (v1) | NanoLlama v2 | SwiftLlama-350M |
|-----------|-------------------|-------------|-----------------|
| `token_embed` | 38.6M | 38.6M | 51.5M |
| Attention (all layers) | 50.3M¹ | 50.3M¹ | 190.5M¹ |
| MLP (all layers) | 0 | 0 | 0 |
| FFN (all layers) | 100.7M² | 100.7M² | 261.4M² |
| Value embeds | — | 0.05M | 0.13M |
| x0-mixin scalars | — | 0.002M | 0.004M |
| Final norm + lm_head | 38.6M | 38.6M | 51.5M |
| **Total** | **127.6M** | **127.6M** | **345.3M** |

¹ Per layer (GQA): `d_model×(n_heads×head_dim)` + `d_model×(2×num_kv_heads×head_dim)` + `(n_heads×head_dim)×d_model`  
² Per layer (SwiGLU): `3 × d_model × d_ff`

### Training configs

| Field | NanoLlama 8L (v1) | NanoLlama v2 | SwiftLlama-350M |
|-------|-------------------|-------------|-----------------|
| Optimizer | AdamW | AdamW | Muon+Adam |
| Batch size B | 16 | 16 | 2 |
| Grad accum | 32 | 32 | 64 |
| Sequence length T | 1024 | 1024 | 4096 |
| Effective tok/step | 524,288 | 524,288 | 524,288 |
| LR (Adam) | 6e-4 | 6e-4 | 6e-4 |
| LR (Muon) | — | — | 0.02 |
| LR (embed) | — | — | 0.3 |
| LR (unembed) | — | — | 0.004 |
| LR (scalars) | — | — | 0.5 |
| Weight decay | 0.1 | 0.1 | 0.1 |
| Warmup steps | 200 | 200 | 500 |
| LR schedule | cosine | cosine + warmdown | cosine |
| Warmdown ratio | — | 0.4 | — |
| Grad clip | 1.0 | 1.0 | 1.0 |
| max_steps | 4768 | 9537 | 28610 |
| Target tokens | 2.5B | 5.0B | 15.0B |
| Hardware | RTX 4090 | RTX 4090 | RTX 4090 |
| Precision | bfloat16 | bfloat16 | bfloat16 |
| Throughput | 75,940 tok/s | ~415,000 tok/s (compiled) | ~25,700 tok/s |

NanoLlama v2 uses `constant_warmdown` LR schedule: constant at peak for `(1 − warmdown_ratio) × max_steps`, then linear decay to 0 over the final `warmdown_ratio × max_steps` steps.

---

## Key mechanics

### GQA: grouped-query attention

`n_heads=12` query heads share `num_kv_heads=4` KV heads (3:1 ratio for v1/v2, 4:1 for SwiftLlama). Each KV head serves 3 (or 4) query heads via `repeat_interleave`.

Head dimension: `head_dim = d_model / n_heads`
- NanoLlama: 768/12 = **64**
- SwiftLlama: 1024/16 = **64**

KV cache shape per layer: `[B, num_kv_heads, T, head_dim]`

### Partial RoPE (`rope_fraction=0.5`)

`rotary_dim = int(head_dim × rope_fraction)`, rounded down to nearest even, minimum 2.

At `head_dim=64`, `rope_fraction=0.5`: `rotary_dim=32`.

Forward: split query/key as `[q_rot | q_pass]` along last dim. Apply RoPE rotation to `q_rot` (first 32 dims), concatenate unchanged `q_pass` (last 32 dims). Full vector participates in attention.

### Value embeddings (`n_value_embeds=2`)

Parameter: `value_embed` shape `[n_value_embeds, num_kv_heads × head_dim]`.  
At inference: `bias = value_embed.mean(0).view(1, num_kv_heads, 1, head_dim)`.  
Applied: `v = v_proj(x) + bias` before KV cache append.

### x0-mixin

Parameters per layer: `x0_lambda` (scalar, init=1), `x0_lambda0` (scalar, init=0).  
`x₀` = `token_embed(input_ids)` captured before the block stack.  
Each block output: `x = x0_lambda[i] * x + x0_lambda0[i] * x₀`.  
At init (λ=1, λ₀=0): identical to standard residual — no cold-start penalty.

### Logit softcap

Applied post-lm_head: `logits = cap × tanh(logits / cap)`.  
Bounds output to `(−cap, +cap)`. Prevents logit explosion in long runs without clamping gradients.

### Weight initialization: residual projections

Residual-path projections (`out_proj`, `w_down` / `fc2`) use scaled std: `0.02 / √(2 × n_layers)`.  
For NanoLlama (n_layers=8): `0.02 / √16 = 0.005`.  
For SwiftLlama (n_layers=22): `0.02 / √44 ≈ 0.003`.  
This keeps the residual stream variance stable with depth (Wang & Komatsuzaki, GPT-J).

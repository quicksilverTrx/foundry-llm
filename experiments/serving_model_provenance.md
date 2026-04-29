# Serving Experiment Model Provenance

All serving experiment results in this directory tree were produced from a single model package.

## Model

| Field | Value |
|-------|-------|
| Package path | `experiments/p1_pos_enc/runs/rope/package` |
| Architecture | MHA (Multi-Head Attention) |
| Positional encoding | RoPE |
| Layers | 4 |
| Heads | 4 |
| d_model | 256 |
| d_ff | 1024 |
| vocab_size | 10,000 |
| block_size | 512 |
| Parameters | 8,420,624 |
| Tokenizer | sp16k SubwordTokenizer (BPE, `merges.txt` + `vocab.json`) |
| Model config hash | `903fb239a49426809523a0772eb05be11c721c94431d3dbec72cfb681f6be602` |
| Tokenizer hash | `464d717b26e6e69ba739c66e1d7f44cfb3af474a6b6eecaeb4e626dc474c08c9` |

## Environment

| Field | Value |
|-------|-------|
| Device | CPU |
| Hardware | Apple M1 Pro |
| OS | macOS 26.3.1 arm64 |
| Python | 3.11.11 |
| PyTorch | 2.2.2 |
| Quant engine | qnnpack |

## Precision modes measured

| Mode | Run state | Effective dtype |
|------|-----------|-----------------|
| fp32 | Executed | fp32 |
| int8 | Executed | fp32 compute, int8 weights (dynamic quantization) |
| fp16 | Not run separately | Falls back to fp32 on CPU |
| bf16 | Not run separately | Falls back to fp32 on CPU |

## What this model is NOT

This is a **toy RoPE model** trained on a 10K-vocab character/subword corpus for serving layer development and testing. It is **not** the NanoLlama 8L (127.6M params, 50K vocab, trained on FineWeb-Edu). All metrics (TTFT, TPS, PPL, memory) reflect this toy model and should not be compared to NanoLlama or production LLMs.

## Experiment directories using this model

- `experiments/serving_bench/` — KV cache vs recompute benchmarks, grid sweeps
- `experiments/serving_quant/` — Quantization audit (fp32 vs int8), PPL, memory
- `experiments/serving_reports/` — API smoke, cache equivalence, TTFT/TPS curves, prompt suite, eval report
- `experiments/serving_eval/` — Validation by context length

# foundry-llm

An experimental, minimal GPT-style language model lab for learning and iterating on
decoder-only Transformers.

## Highlights
- MiniGPT: decoder-only Transformer with multi-head attention (MHA)
- Positional encodings: learned, sinusoidal, or RoPE
- Normalization: LayerNorm or RMSNorm
- MLP: GELU or SwiGLU variants
- Tokenization: character-level and a simple BPE subword tokenizer
- Training utilities with CSV logging and optional sample callbacks
- Sampling: greedy, temperature, top-k, and top-p (nucleus)
- Model packaging: save/load config + tokenizer + checkpoint

## Repository layout
- `llm_lab/core/model`: attention, blocks, MLP, norms, MiniGPT, positional encodings
- `llm_lab/core/tokenization`: `CharTokenizer` and `SubwordTokenizer` (BPE)
- `llm_lab/core/data`: `CharDataset` and `LanguageModelingDataset`
- `llm_lab/core/train`: `Trainer` + `TrainerConfig`
- `llm_lab/core/decode`: sampling helpers
- `llm_lab/core/package`: package layout + IO helpers
- `llm_lab/utils`: config loading and misc utilities
- `scripts/`: runnable experiments and sampling entry points
- `configs/`: example model config JSON
- `tests/`: unit and smoke tests
- `data/`: local text corpora (expected by training scripts)
- `experiments/`: local run outputs (logs, checkpoints, analysis)
- `artifacts/`: packaged models (config + tokenizer + checkpoints)

## Setup
Requirements:
- Python 3.11+
- PyTorch (install separately; not pinned in `pyproject.toml`)

Suggested local setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install -e .
```

Or install exact versions from the current `requirements.txt`:
```bash
pip install -r requirements.txt
```

If you use conda, align your env with `requirements.txt` (pip subset) or
install PyTorch via your preferred conda channel.

## Quickstart (toy char model)
```bash
python scripts/p1_train_char.py
```

## Training scripts
Character-level:
- `python scripts/p1_train_char.py`: tiny "hello world" example
- `python scripts/p1_train_char_real.py`: real-text training on `data/tiny_shakespeare.txt`
- `python scripts/p1_train_char_sanity_check.py`: quick sanity run with a small slice
- `python scripts/p1_env_sanity.py`: quick device sanity check (CPU/MPS/CUDA)

Subword (BPE):
- `python scripts/p2_train_bpe.py`: train BPE tokenizer, build LM datasets, train MiniGPT, save package

Positional encodings:
- `python scripts/p3_train_posenc.py`: compare learned, sinusoidal, and RoPE configs

Benchmarks / analysis:
- `python scripts/p1_bench_attention.py`: attention performance micro-benchmark
- `python scripts/p1_bench_step.py`: single-step throughput benchmark
- `python scripts/p1_collect_phase4_numbers.py`: collect run metrics for reporting

## Sampling scripts
Character-level:
- `python scripts/p1_sample_char.py` (expects a run directory + vocab file)

Subword (BPE):
- `python scripts/p2_sample_bpe.py` (expects a saved model package)

Tip: several scripts have hard-coded paths (for example, `experiments/...` and
`artifacts/...`). Update those paths if you use a different output layout.

## Data expectations
Most training scripts assume a text corpus at:
```
data/tiny_shakespeare.txt
```
You can replace this with any plain-text file; update the scripts as needed.

## Model API (MiniGPT)
`MiniGPT.forward(...)` returns a tuple:
```python
logits, past = model(input_ids)
```
`past_key_values` and `attention_mask` are not implemented yet, so pass
`None` (the training and sampling helpers already do this).

## Packaging
Use `llm_lab/core/package/io.py` to save/load a package:
```python
from llm_lab.core.package.io import save_model_package, load_model_package

save_model_package("artifacts/my_run", config, tokenizer, model, is_best=True)
cfg, tok, model = load_model_package("artifacts/my_run")
```

Package layout:
```
<package>/
  config.json
  tokenizer/
    vocab.json
    merges.txt
  checkpoints/
    best_val.pt
```

## Configuration
Example configs:
- `configs/p1/gpt_small_subword.json`
- `configs/p1/gpt_small_rope.json`
- `configs/p1/gpt_small_sinusoidal.json`

You can load JSON configs into dataclasses with:
```python
from llm_lab.utils.config_loader import load_json_config
from llm_lab.core.model.gpt import MiniGPTConfig

cfg = load_json_config("configs/p1/gpt_small_subword.json", MiniGPTConfig)
```

## Tests
Run the test suite with:
```bash
pytest
```

## Known limitations (current)
- `attention_mask` and KV cache are stubbed; they raise `NotImplementedError`
- Only MHA is implemented (`attention_type` is a placeholder for now)
- BPE tokenizer is a simple, educational implementation (not SentencePiece)

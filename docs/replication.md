# Replication Guide — NanoLlama 8L Pretraining

Exact steps to reproduce the full NanoLlama 8L pretraining result from scratch.

> **Already have the local run?** All artifact locations (checkpoints, logs, CSVs) are indexed in [`docs/local_artifacts.md`](local_artifacts.md).
> Final checkpoint is at `experiments/tinyllama_pretrain_2026-03-31/phase6/ckpts/step_04768_model_only.pt`.

---

## Hardware requirements

| Item | Minimum | Used in original run |
|------|---------|---------------------|
| GPU | Any CUDA device | RTX 4090 (24 GB VRAM) |
| Disk (data + checkpoints) | 50 GB | 75 GB persistent |
| RAM | 32 GB | 64 GB |
| Training time | ~9 h (RTX 4090) | 9.2 h wall time |

### Persistent storage breakdown (measured)
| Item | Size |
|------|------|
| FineWeb-Edu HF download cache (temporary) | ~28 GB — delete after tokenization |
| Tokenized shards (99 train + 1 val `.npy`) | ~19 GB — keep |
| Model-only checkpoints (every 500 steps) | ~1 GB |
| Full optimizer checkpoints (optional) | ~2–3 GB each |
| **Peak** | ~48 GB (after HF cache cleanup) |

---

## Setup

```bash
# Python 3.11+, PyTorch 2.4+
git clone https://github.com/<your-org>/foundry-llm
cd foundry-llm
pip install -e .
pip install tiktoken datasets tqdm numpy
```

---

## Step 1 — Download and tokenise FineWeb-Edu (~45 min, ~20 GB)

```bash
python data/prepare_dataset.py
# Writes to: ./data/edu_fineweb10B/
#   edufineweb_val_000000.npy         (100M uint16 tokens — validation)
#   edufineweb_train_000001.npy       (100M uint16 tokens)
#   ...
#   edufineweb_train_000099.npy       (99 training shards total)
```

Custom output path:
```bash
python data/prepare_dataset.py --out_dir /fast_disk/edu_fineweb10B
```

**Tokenizer:** GPT-2 BPE via tiktoken (`"gpt2"` encoding), vocab=50257, padded to 50304.

---

## Step 2 — Train NanoLlama 8L (~9 h on RTX 4090)

```bash
python scripts/pretrain_nanollama.py
```

The script:
- Auto-detects CUDA / MPS / CPU
- Reads config from `configs/nanollama_8l.json`
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically (prevents OOM at B=16 on 24 GB)
- Saves model-only checkpoints every 500 steps to `./out/ckpts/`
- Writes `trajectory.csv` and `run.log` to `./out/`

Override config at the command line:
```bash
python scripts/pretrain_nanollama.py --max_steps 5 --device cpu   # smoke test
python scripts/pretrain_nanollama.py --data_dir /fast_disk/edu_fineweb10B
```

---

## Expected results

| Step | Tokens | Val loss |
|------|--------|----------|
| 250 | 131M | ~5.19 |
| 500 | 262M | ~4.36 |
| 1000 | 524M | ~3.87 |
| 2000 | 1.05B | ~3.58 |
| 3000 | 1.57B | ~3.45 |
| 4000 | 2.10B | ~3.38 |
| **4768** | **2.50B** | **3.3566** |

Full trajectory: `results/nanollama_8l_training.csv`

---

## Step 3 — Evaluate

```bash
# 11-test eval suite (sanity, generation, perplexity, entropy, HellaSwag-style)
python scripts/eval_suite.py --ckpt out/ckpts/step_04768_model_only.pt

# Real HellaSwag benchmark (10,042 items, ~10 min on CPU)
python scripts/eval_hellaswag.py --ckpt out/ckpts/step_04768_model_only.pt

# Interactive sampling REPL
python -i scripts/interact.py --ckpt out/ckpts/step_04768_model_only.pt
```

**Validated results:**
- HellaSwag normalized accuracy: **0.2696** (random=0.25, GPT-2 124M @ 1.05B ref ≈ 0.238)
- Val BPB: **1.016**

---

## Architecture (from `configs/nanollama_8l.json`)

```
n_layers=8  d_model=768  n_heads=12  num_kv_heads=4 (GQA)
d_ff=2048 (SwiGLU)  block_size=1024  norm=RMSNorm  pos=RoPE
logit_softcap=30  qk_norm=True  lm_head bias=False
vocab_size=50304  params=127.6M
```

Training:
```
B=16  T=1024  grad_accum=32 → 524,288 tokens/step
max_lr=6e-4  min_lr=6e-5  warmup=200  max_steps=4768
AdamW: weight_decay=0.1  betas=(0.9, 0.95)  fused (CUDA only)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Cloud GPU notes (Vast.ai / RunPod)

- Use a **persistent volume** for data and checkpoints — container disk is wiped on restart
- Delete the HuggingFace download cache (`~/.cache/huggingface/`) after tokenization to recover ~28 GB
- Monitor with `watch -n2 nvidia-smi` and `tail -f out/run.log`
- If OOM: the script sets `expandable_segments:True` automatically; if still OOM reduce `B` in the config
- Checkpoint every 500 steps is the default; increase `ckpt_every` for less disk usage

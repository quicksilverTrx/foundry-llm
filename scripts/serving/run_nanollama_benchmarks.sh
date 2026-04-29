#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

CKPT="experiments/tinyllama_pretrain_2026-03-31/phase6/ckpts/step_04768_model_only.pt"
BENCH_OUT="experiments/nanollama_serving_bench"
QUANT_OUT="experiments/nanollama_serving_quant"
EVAL_OUT="experiments/nanollama_serving_reports"

mkdir -p "$BENCH_OUT" "$QUANT_OUT" "$EVAL_OUT"

echo "=== [$(date)] Single-point benchmark (prompt=256, gen=128) ==="
$PYTHON scripts/serving/bench_inference.py \
  --mode both \
  --prompt-len 256 \
  --gen-len 128 \
  --warmup 5 \
  --iters 20 \
  --out-dir "$BENCH_OUT" \
  --package-dir "$CKPT" \
  --loader nanollama

echo "=== [$(date)] Grid sweep (ctx 256/512/1024, batch 1/2/4, gen=64) ==="
$PYTHON scripts/serving/bench_inference.py \
  --mode both \
  --context-lens 256,512,1024 \
  --batch-sizes 1,2,4 \
  --gen-len 64 \
  --warmup 3 \
  --iters 10 \
  --repeats 2 \
  --out-dir "$BENCH_OUT" \
  --package-dir "$CKPT" \
  --loader nanollama

echo "=== [$(date)] Quant sweep (prompt=256, gen=64) ==="
$PYTHON scripts/serving/quant_sweep.py \
  --package "$CKPT" \
  --text-path data/serving_eval/prompts.jsonl \
  --device cpu \
  --prompt-len 256 \
  --gen-len 64 \
  --max-seq-len 256 \
  --out-dir "$QUANT_OUT" \
  --loader nanollama

echo "=== [$(date)] Prompt suite eval ==="
$PYTHON scripts/eval/eval_prompt_suite.py \
  --backend engine \
  --package "$CKPT" \
  --loader nanollama \
  --prompts data/serving_eval/prompts.jsonl \
  --out-dir "$EVAL_OUT" \
  --device cpu \
  --dtype fp32 \
  --seed 42 \
  --ppl-artifact "$QUANT_OUT/ppl_fp32.json"

echo "=== [$(date)] All NanoLlama benchmarks complete ==="

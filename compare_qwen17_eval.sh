#!/usr/bin/env bash
# Apples-to-apples: Qwen3-1.7B zero-shot vs GRPO best-by-loss.
#
# DISK: The finetuned weights are ~6.9GB. You need ~8–10GB free on the VOLUME you use
# (project folder OR ~/.cache/huggingface). If `hf download` warns "not enough free disk
# space", free space first, or use MODE=hub below (cache may be on a roomier volume).
#
# TIME: Full --limit 30 on a MacBook Air can take a long time (many LLM forward passes).
# Quick check: LIMIT=5 bash compare_qwen17_eval.sh
#
# Run from repo root: bash compare_qwen17_eval.sh
# Modes: MODE=local (default) or MODE=hub (no clone under this repo; uses Hub + subfolder)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
FT="${ROOT}/releaseops-grpo-1.7b-best/best_by_loss"
LIMIT="${LIMIT:-30}"
MODE="${MODE:-local}"

mkdir -p "${ROOT}/outputs"
cd "${ROOT}"

echo "=== Zero-shot: Qwen/Qwen3-1.7B (limit=$LIMIT) ==="
python training/evaluate_llm_baseline.py \
  --backend torch \
  --torch-model "Qwen/Qwen3-1.7B" \
  --limit "$LIMIT" \
  --output-json outputs/eval_zeroshot_qwen1.7b.json

if [[ "$MODE" == "hub" ]]; then
  echo "=== Finetuned: Hub hiitsesh/releaseops-grpo-1.7b-best (subfolder best_by_loss) ==="
  python training/evaluate_llm_baseline.py \
    --backend torch \
    --torch-model "hiitsesh/releaseops-grpo-1.7b-best" \
    --torch-subfolder "best_by_loss" \
    --limit "$LIMIT" \
    --output-json outputs/eval_finetuned_rl.json
else
  if [[ ! -f "${FT}/model.safetensors" ]]; then
    echo "Missing ${FT}/model.safetensors. Either finish hf download, or run:"
    echo "  MODE=hub LIMIT=$LIMIT bash compare_qwen17_eval.sh"
    exit 1
  fi
  echo "=== Finetuned: local ${FT} ==="
  python training/evaluate_llm_baseline.py \
    --backend torch \
    --torch-model "${FT}" \
    --limit "$LIMIT" \
    --output-json outputs/eval_finetuned_rl.json
fi

echo "Done. Compare: outputs/eval_zeroshot_qwen1.7b.json vs outputs/eval_finetuned_rl.json"

#!/usr/bin/env bash
# Hugging Face Space base URL (host only). Do NOT use .../outputs/ls as the base:
#   - List files:  ${HF_SPACE_BASE}/outputs/ls
#   - Training:    ${HF_SPACE_BASE}/train/pilot?...
export HF_SPACE_BASE="${HF_SPACE_BASE:-https://hiitsesh-new-gpu-space.hf.space}"
# Optional: remote inference / env from this machine
export RELEASEOPS_SPACE_URL="${RELEASEOPS_SPACE_URL:-$HF_SPACE_BASE}"

# Examples (uncomment):
# curl -sS "${HF_SPACE_BASE}/outputs/ls"
# curl -sS "${HF_SPACE_BASE}/train/smoke"
# curl -sS "${HF_SPACE_BASE}/train/pilot?model_name=Qwen%2FQwen3-1.7B&max_steps=100&bf16=true"
# curl -sS "${HF_SPACE_BASE}/train/logs?lines=200"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Serve a merged HF-format checkpoint. LoRA-only folders must be merged first (LlamaFactory export).
MODEL_PATH="${1:-${ROOT}/artifacts/checkpoints/qwen25_7b_lora_dpo}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM CLI not found. Install separately (CUDA-specific), e.g.: pip install vllm"
  exit 1
fi

exec vllm serve "${MODEL_PATH}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --trust-remote-code

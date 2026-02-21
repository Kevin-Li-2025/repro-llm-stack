#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
CFG="${1:-${ROOT}/configs/train/llamafactory_qwen25_7b_lora_dpo.yaml}"
if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli not found. Install training stack: pip install -e '.[train]'"
  exit 1
fi
if [[ ! -f "${ROOT}/artifacts/data/dataset_info.json" ]]; then
  echo "Missing ${ROOT}/artifacts/data/dataset_info.json — run scripts/data/prepare.sh first."
  exit 1
fi
exec llamafactory-cli train "${CFG}"

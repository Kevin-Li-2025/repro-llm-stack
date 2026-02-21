#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ $# -ge 1 ]]; then
  CFG="$1"
  shift
else
  CFG="${ROOT}/configs/train/llamafactory_qwen25_7b_lora_cpt_smoke.yaml"
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli not found. Install: pip install -e '.[train]'"
  exit 1
fi
if [[ ! -f "${ROOT}/artifacts/cpt_data/dataset_info.json" ]]; then
  echo "Missing CPT bundle. Run first:"
  echo "  ./scripts/data/prepare_cpt_smoke.sh"
  echo "See docs/CPT_AND_PRETRAIN.md for scope and production guidance."
  exit 1
fi
if [[ ! -s "${ROOT}/artifacts/cpt_data/cpt_smoke.jsonl" ]]; then
  echo "Missing or empty ${ROOT}/artifacts/cpt_data/cpt_smoke.jsonl"
  exit 1
fi

exec llamafactory-cli train "${CFG}" "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[cpt] Phase-2 continued pretrain is intentionally NOT wired to LlamaFactory in this repo."
echo "[cpt] Add a dedicated NeMo/Megatron (or HF CPT) subtree when you have the compute budget."
echo "[cpt] Repo root: ${ROOT}"

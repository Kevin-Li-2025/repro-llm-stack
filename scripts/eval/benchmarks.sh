#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ $# -ge 1 && "${1}" != -* ]]; then
  TASKS_FILE="$1"
  shift
else
  TASKS_FILE="${ROOT}/configs/eval/lm_eval_tasks.txt"
fi

OUT_DIR="${ROOT}/artifacts/eval"
mkdir -p "${OUT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_JSON="${OUT_DIR}/lm_eval_${STAMP}.json"

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "lm_eval not found. Install: pip install -e '.[eval]'"
  exit 1
fi
if [[ ! -f "${TASKS_FILE}" ]]; then
  echo "Task list not found: ${TASKS_FILE}"
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True}"

TASKS="$(
  grep -v '^\s*#' "${TASKS_FILE}" | grep -v '^\s*$' | tr '\n' ',' | sed 's/,$//'
)"
TASKS="${TASKS//[[:space:]]/}"

if [[ -z "${TASKS}" ]]; then
  echo "No tasks found in ${TASKS_FILE} (check comments and blank lines)."
  exit 1
fi

echo "[eval] tasks=${TASKS}"
echo "[eval] model_args=${MODEL_ARGS}"
echo "[eval] output=${OUT_JSON}"

lm_eval \
  --model hf \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size auto \
  --output_path "${OUT_JSON}" \
  "$@"

# lm-eval writes ${stem}_${ISO_timestamp}.json, not the literal --output_path when it ends in .json.
STEM="$(basename "${OUT_JSON}" .json)"
ACTUAL_JSON="$(ls -t "${OUT_DIR}/${STEM}"_*.json 2>/dev/null | head -1)"
if [[ -z "${ACTUAL_JSON}" || ! -f "${ACTUAL_JSON}" ]]; then
  echo "[eval] could not find lm_eval output matching ${OUT_DIR}/${STEM}_*.json" >&2
  ls -la "${OUT_DIR}" >&2 || true
  exit 1
fi
python3 tools/summarize_lm_eval.py "${ACTUAL_JSON}" --out "${OUT_DIR}/SUMMARY.md"
echo "[eval] wrote ${OUT_DIR}/SUMMARY.md (${ACTUAL_JSON})"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
if [[ $# -ge 1 && "${1}" != -* ]]; then
  RECIPE="$1"
  shift
else
  RECIPE="recipes/cpt_smoke.yaml"
fi
exec python3 tools/prepare_cpt_smoke.py --recipe "${RECIPE}" --root "${ROOT}" "$@"

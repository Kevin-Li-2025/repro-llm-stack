#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
if [[ $# -ge 1 && "${1}" != -* ]]; then
  RECIPE="$1"
  shift
else
  RECIPE="recipes/default.yaml"
fi
exec python3 tools/prepare_data.py --recipe "${RECIPE}" --root "${ROOT}" "$@"

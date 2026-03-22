#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if ! command -v repro-beir-retrieval-compare >/dev/null 2>&1; then
  echo "repro-beir-retrieval-compare not found. Install: pip install -e '.[retrieval]'"
  exit 1
fi

exec repro-beir-retrieval-compare --root "${ROOT}" "$@"

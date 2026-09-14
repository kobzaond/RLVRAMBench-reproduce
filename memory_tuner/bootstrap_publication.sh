#!/usr/bin/env bash
set -euo pipefail
ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLVRAM_BOOTSTRAP_PYTHON="${RLVRAM_BOOTSTRAP_PYTHON:-python3.11}"
RLVRAM_VENV="${RLVRAM_VENV:-${ARTIFACT_ROOT}/.venv-reproduce}"
"${RLVRAM_BOOTSTRAP_PYTHON}" -c \
  'import sys; assert sys.version_info[:2] == (3, 11), "Use CPython 3.11 for the pinned publication environment"'
if [[ ! -x "${RLVRAM_VENV}/bin/python" ]]; then
    "${RLVRAM_BOOTSTRAP_PYTHON}" -m venv "${RLVRAM_VENV}"
fi
"${RLVRAM_VENV}/bin/python" -m pip install -r "${ARTIFACT_ROOT}/publication-requirements.txt"
"${RLVRAM_VENV}/bin/python" -m pip check
echo "Publication environment ready: ${RLVRAM_VENV}/bin/python"

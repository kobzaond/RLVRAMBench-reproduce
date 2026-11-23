#!/usr/bin/env bash
# Single reproduction procedure for release rlvrambench-2026-11.
#
# Creates (if missing) the two pinned CPU environments inside this checkout and
# runs reproduce_all.py with all arguments passed through, for example
#
#   bash reproduce_all.sh --revision <40-hex dataset commit> --work-dir reproduce-all-output
#   bash reproduce_all.sh --archives-dir downloads --work-dir reproduce-all-output
#
# Interpreters: RLVRAM_BOOTSTRAP_PYTHON (CPython 3.11, default python3.11) and
# RLVRAM_REANALYSIS_PYTHON (CPython 3.12, default python3.12). Environments:
# RLVRAM_VENV (default .venv-reproduce) and RLVRAM_REANALYSIS_VENV
# (default .venv-reanalysis). Needs tar, zstd and bubblewrap (bwrap) on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLVRAM_BOOTSTRAP_PYTHON="${RLVRAM_BOOTSTRAP_PYTHON:-python3.11}"
export RLVRAM_VENV="${RLVRAM_VENV:-${ROOT}/.venv-reproduce}"
REANALYSIS_PYTHON="${RLVRAM_REANALYSIS_PYTHON:-python3.12}"
REANALYSIS_VENV="${RLVRAM_REANALYSIS_VENV:-${ROOT}/.venv-reanalysis}"

bash "${ROOT}/memory_tuner/bootstrap_publication.sh"

"${REANALYSIS_PYTHON}" -c \
  'import sys; assert sys.version_info[:2] == (3, 12), "Use CPython 3.12 for the re-analysis environment"'
if [[ ! -x "${REANALYSIS_VENV}/bin/python" ]]; then
  "${REANALYSIS_PYTHON}" -m venv "${REANALYSIS_VENV}"
fi
"${REANALYSIS_VENV}/bin/python" -m pip install -r "${ROOT}/revision_2026_11/cpu_reanalyses/requirements.txt"
"${REANALYSIS_VENV}/bin/python" -m pip check

exec "${RLVRAM_VENV}/bin/python" -I "${ROOT}/reproduce_all.py" \
  --reanalysis-python "${REANALYSIS_VENV}/bin/python" "$@"

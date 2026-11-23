#!/usr/bin/env bash
# RETROSPECTIVE C2/C3/C8 analyses; CPU only, read-only on the tables root.
# Requires RLVRAMBENCH_TABLES_ROOT (curated benchmark/ and profiles/ tables) and
# RLVRAMBENCH_ANALYSIS_OUTPUT (new output directory); see ../README.md.
# reproduce_all.py sets both. Runs isolated Python from the output directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RLVRAMBENCH_TABLES_ROOT:?set RLVRAMBENCH_TABLES_ROOT}"
: "${RLVRAMBENCH_ANALYSIS_OUTPUT:?set RLVRAMBENCH_ANALYSIS_OUTPUT}"
mkdir -p "${RLVRAMBENCH_ANALYSIS_OUTPUT}"
cd "${RLVRAMBENCH_ANALYSIS_OUTPUT}"
for script in a_reproduce.py b_hurdle.py c_coverage.py d_margins.py; do
  echo "== ${script}"
  "${PYTHON:-python3}" -I -B "${HERE}/${script}"
done

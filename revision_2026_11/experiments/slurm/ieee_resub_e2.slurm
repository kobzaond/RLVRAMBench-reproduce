#!/usr/bin/env bash
# Outer allocation for study e2 (E2 ten fresh repeats of four near-boundary configurations, 4-GPU allocation).
# One array task = one frozen same-device block; the runner never submits jobs.
#SBATCH --job-name=ieee-resub-e2
#SBATCH --account=FTA-26-70
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=01:40:00
#SBATCH --array=0-9%3
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj1/fta-26-70/rlvram/logs/ieee-resub-e2-%A_%a.out
#SBATCH --error=/mnt/proj1/fta-26-70/rlvram/logs/ieee-resub-e2-%A_%a.err
set -euo pipefail
: "${IEEE_RESUB_SOURCE_ROOT:?immutable source snapshot required}"
: "${SOURCE_COMMIT:?frozen commit required}"
: "${PROTOCOL_PATH:?frozen protocol required}"
: "${PROTOCOL_SHA256:?frozen protocol hash required}"
[[ "$IEEE_RESUB_SOURCE_ROOT" == /mnt/proj1/fta-26-70/rlvram/src/* ]]
[[ "$PROTOCOL_PATH" == /mnt/proj1/fta-26-70/rlvram/protocols/e2/* ]]
# Blocks that start under a hash-chained amendment (PLAN.md section 9) pass it on.
AMENDMENT_ARGS=()
if [[ -n "${AMENDMENT_PATH:-}" ]]; then
  [[ "$AMENDMENT_PATH" =~ ^/mnt/proj1/fta-26-70/rlvram/protocols/e2/amendment-[0-9]{2}\.json$ ]]
  [[ "${AMENDMENT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]
  AMENDMENT_ARGS=(--amendment "$AMENDMENT_PATH" --amendment-sha256 "$AMENDMENT_SHA256")
fi
export TZ=UTC PYTHONDONTWRITEBYTECODE=1
export PATH="/mnt/proj1/fta-26-70/rlvram/venv/bin:$PATH"
cd "$IEEE_RESUB_SOURCE_ROOT"
exec python -I -B experiments/ieee_resubmission_2026-11/ieee_resub/runner.py run --study e2 \
  --protocol "$PROTOCOL_PATH" --protocol-sha256 "$PROTOCOL_SHA256" \
  --source-commit "$SOURCE_COMMIT" --index "$SLURM_ARRAY_TASK_ID" "${AMENDMENT_ARGS[@]}"

#!/usr/bin/env bash
# Amendment reruns for study e3 (PLAN.md section 9): one array task = one listed slot.
# The runner never submits jobs; submit_ieee_resub.sh --rerun passes --array.
#SBATCH --job-name=ieee-resub-e3-amend
#SBATCH --account=FTA-26-70
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=11:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj1/fta-26-70/rlvram/logs/ieee-resub-e3-amend-%A_%a.out
#SBATCH --error=/mnt/proj1/fta-26-70/rlvram/logs/ieee-resub-e3-amend-%A_%a.err
set -euo pipefail
: "${IEEE_RESUB_SOURCE_ROOT:?immutable source snapshot required}"
: "${SOURCE_COMMIT:?frozen commit required}"
: "${PROTOCOL_PATH:?frozen protocol required}"
: "${PROTOCOL_SHA256:?frozen protocol hash required}"
: "${AMENDMENT_PATH:?recorded amendment required}"
: "${AMENDMENT_SHA256:?amendment hash required}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array over the amendment reruns}"
[[ "$IEEE_RESUB_SOURCE_ROOT" == /mnt/proj1/fta-26-70/rlvram/src/* ]]
[[ "$PROTOCOL_PATH" == /mnt/proj1/fta-26-70/rlvram/protocols/e3/* ]]
[[ "$AMENDMENT_PATH" =~ ^/mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-[0-9]{2}\.json$ ]]
[[ "$AMENDMENT_SHA256" =~ ^[0-9a-f]{64}$ ]]
export TZ=UTC PYTHONDONTWRITEBYTECODE=1
export PATH="/mnt/proj1/fta-26-70/rlvram/venv/bin:$PATH"
cd "$IEEE_RESUB_SOURCE_ROOT"
exec python -I -B experiments/ieee_resubmission_2026-11/ieee_resub/runner.py rerun --study e3 \
  --protocol "$PROTOCOL_PATH" --protocol-sha256 "$PROTOCOL_SHA256" \
  --source-commit "$SOURCE_COMMIT" --index "$SLURM_ARRAY_TASK_ID" \
  --amendment "$AMENDMENT_PATH" --amendment-sha256 "$AMENDMENT_SHA256"

#!/usr/bin/env bash
# Step S0 of PLAN.md: create the FTA-26-70 storage layout, copy the frozen inputs
# from the read-only OPEN-35-44 project storage, and check out the immutable
# source snapshot. NOT executed during preparation; run on a Karolina login node
# after approval. CPU/IO only; no Slurm job. Without --execute it prints actions.
#
# usage: setup_fta_storage.sh SOURCE_COMMIT BUNDLE_PATH [--execute] [--snapshot-only]
#   BUNDLE_PATH: git bundle of the approved commit, copied (scp) by the user to
#                /mnt/proj1/fta-26-70/rlvram/src/bundles/ (no git remote involved).
#   --snapshot-only: step 6 only, for an amended commit (e.g. e3-amendment-01):
#                no layout, copies or venv, and no protocol freeze (the amendment
#                carries the amended source hashes; the frozen protocols stay).
set -euo pipefail
umask 007
# Copies keep symlinks and timestamps but not the OPEN-35-44 group, so that the
# setgid FTA-26-70 project group (and its quota) applies to every new file.

COMMIT=${1:?40-hex approved commit}
BUNDLE=${2:?git bundle path}
MODE=print SCOPE=all
for option in "${@:3}"; do
  case "$option" in
    --execute) MODE=--execute ;;
    --snapshot-only) SCOPE=snapshot ;;
    *) echo "unknown option: $option" >&2; exit 2 ;;
  esac
done
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
OLD=/mnt/proj3/open-35-44/verl
PROJECT=/mnt/proj1/fta-26-70/rlvram
SCRATCH=/scratch/project/fta-26-70/rlvram
SNAPSHOT=$PROJECT/src/ieee-resub-${COMMIT:0:12}
IMAGE_SHA256=3dc581ca780cd19d565445b8a02e5daa98b350b3d2116d4d64476a64a2614912
TRANSFER_QUEUE_COMMIT=5cb184efd99addd4b40ff25a700aa260ed2e9eed
MODELS=(
  models--ibm-granite--granite-3.3-2b-instruct
  models--microsoft--Phi-4-mini-instruct
  models--Qwen--Qwen2.5-3B-Instruct
  models--Qwen--Qwen2.5-7B-Instruct
  models--mistralai--Mistral-7B-Instruct-v0.3
)

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "$MODE" == "--execute" ]]; then "$@"; fi
}

if [[ "$SCOPE" == all ]]; then
# 1. Layout (proj1: inputs, evidence, protocols; scratch: runtime and checkpoints).
run mkdir -p "$PROJECT"/{images,python-packages,data,src/bundles,protocols,output,logs,cache/apptainer} \
  "$PROJECT/runtime/$USER/cache/huggingface/hub" "$SCRATCH"/{r/port-reservations,checkpoints}

# 2. Container (12.6 GB), verified against the documented SHA-256.
run cp --preserve=mode,timestamps "$OLD/images/verl-vllm018-dev1.sif" "$PROJECT/images/"
run bash -c "echo '$IMAGE_SHA256  $PROJECT/images/verl-vllm018-dev1.sif' | sha256sum -c -"
run chmod 0444 "$PROJECT/images/verl-vllm018-dev1.sif"

# 3. Base VERL distribution metadata used on PYTHONPATH (39 MB) and datasets.
run cp -dR --preserve=mode,timestamps "$OLD/python-packages/." "$PROJECT/python-packages/"
run cp -dR --preserve=mode,timestamps "$OLD/rl/data/gsm8k" "$OLD/rl/data/math" "$PROJECT/data/"
run mkdir -p "$PROJECT/data/codecontests"
run cp -dR --preserve=mode,timestamps "$OLD/rl/data/codecontests/heavy_tail" "$OLD/rl/data/codecontests/standard" "$PROJECT/data/codecontests/"

# 4. Hugging Face cache entries (blobs + snapshots + refs, ~47 GB). Symlinks preserved.
for model in "${MODELS[@]}"; do
  run cp -dR --preserve=mode,timestamps "$OLD/runtime/$USER/cache/huggingface/hub/$model" "$PROJECT/runtime/$USER/cache/huggingface/hub/"
done

# 5. Host-side runner environment (same versions as the previous .venv-reproduce).
run bash -lc "module load Anaconda3/2023.07-2 && python3.11 -m venv '$PROJECT/venv'"
run "$PROJECT/venv/bin/pip" install --no-input psutil==5.9.0 numpy==1.24.3 scipy==1.10.1 pytest==7.4.0 PyYAML==6.0 scikit-learn==1.3.0

fi

# 6. Immutable source snapshot of the approved commit (from a bundle; no remote).
# git 2.43 on Karolina can only verify a bundle inside a repository.
run bash -c "T=\$(mktemp -d) && git init -q \"\$T\" && git -C \"\$T\" bundle verify '$BUNDLE' && rm -rf \"\$T\""
run git clone --no-checkout "$BUNDLE" "$SNAPSHOT"
run git -C "$SNAPSHOT" checkout --detach "$COMMIT"
run git -C "$SNAPSHOT" config submodule.third_party/TransferQueue.url "$OLD/rl/third_party/TransferQueue"
run git -C "$SNAPSHOT" -c protocol.file.allow=always submodule update --init third_party/TransferQueue
run bash -c "test \"\$(git -C '$SNAPSHOT/third_party/TransferQueue' rev-parse HEAD)\" = $TRANSFER_QUEUE_COMMIT"
run bash -c "test -z \"\$(git -C '$SNAPSHOT' status --porcelain --untracked-files=normal)\""
run chmod -R a-w "$SNAPSHOT"

if [[ "$SCOPE" == snapshot ]]; then
  echo "mode: $MODE (snapshot only)"
  exit 0
fi
# 7. Hash verification against the earlier frozen records and the final protocols
#    (each refuses to write if a model/data/container/source hash differs).
for study in smoke1 smoke2 e1 e2 e3; do
  run mkdir -p "$PROJECT/protocols/$study"
  run bash -c "cd '$SNAPSHOT' && '$PROJECT/venv/bin/python' -I -B experiments/ieee_resubmission_2026-11/ieee_resub/plan.py freeze-final --study $study --out '$PROJECT/protocols/$study'"
done
echo "mode: $MODE"

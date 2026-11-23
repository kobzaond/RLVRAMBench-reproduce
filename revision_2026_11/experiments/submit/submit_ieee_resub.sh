#!/usr/bin/env bash
# Submit ONE frozen study of the IEEE resubmission panel (E1-E3 or a smoke test).
#
# NOT executed during preparation. Run on a Karolina login node only after the
# user approved PLAN.md, the setup step S0 finished, and the frozen protocol was
# written by `plan.py freeze-final`. Without --submit it only prints the command.
#
# usage: submit_ieee_resub.sh STUDY PROTOCOL_PATH PROTOCOL_SHA256 SOURCE_COMMIT [--submit]
#          [--amendment AMENDMENT_PATH AMENDMENT_SHA256 (--indices LIST | --rerun [--indices LIST])]
#
# With a hash-chained amendment (PLAN.md section 9; protocols/STUDY/amendment-NN.json,
# read-only), SOURCE_COMMIT is the amended commit whose snapshot the amendment hashes:
#   --indices LIST  submits only these array indices (e.g. 5-11); the amendment must
#                   list each as never launched, and none may be claimed yet;
#   --rerun         submits the amendment's rerun slots (one array task each; with
#                   --indices only these positions in its "reruns" list); each
#                   original slot must be terminal and unresolved, its claim unused.
set -euo pipefail
umask 007

STUDY=${1:?study (smoke1|smoke2|e1|e2|e3)}
PROTOCOL_PATH=${2:?frozen protocol path}
PROTOCOL_SHA256=${3:?frozen protocol sha256}
SOURCE_COMMIT=${4:?40-hex source commit}
shift 4
MODE=print AMENDMENT_PATH='' AMENDMENT_SHA256='' INDICES='' RERUN=0
while (( $# )); do
  case "$1" in
    --submit) MODE=--submit; shift ;;
    --amendment) AMENDMENT_PATH=${2:?amendment path}; AMENDMENT_SHA256=${3:?amendment sha256}; shift 3 ;;
    --indices) INDICES=${2:?array indices}; shift 2 ;;
    --rerun) RERUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
PROJECT=/mnt/proj1/fta-26-70/rlvram
SOURCE=$PROJECT/src/ieee-resub-${SOURCE_COMMIT:0:12}
SCRIPT=$SOURCE/experiments/ieee_resubmission_2026-11/slurm/ieee_resub_${STUDY}.slurm

[[ "$STUDY" =~ ^(smoke1|smoke2|e1|e2|e3)$ ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$PROTOCOL_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$PROTOCOL_PATH" == "$PROJECT/protocols/$STUDY/${STUDY}_protocol.json" ]]
[[ "$(hostname -f)" == login*.karolina.it4i.cz ]] || { echo "run on a Karolina login node" >&2; exit 2; }
if [[ -n "$INDICES" || "$RERUN" == 1 || -n "$AMENDMENT_PATH" ]]; then
  [[ -n "$AMENDMENT_PATH" ]] || { echo "--indices and --rerun need --amendment" >&2; exit 2; }
  if [[ -z "$INDICES" && "$RERUN" == 0 ]]; then
    echo "an amendment submits --indices LIST (blocks) or --rerun [--indices LIST]" >&2; exit 2
  fi
  [[ "$AMENDMENT_PATH" =~ ^$PROJECT/protocols/$STUDY/amendment-[0-9]{2}\.json$ ]]
  [[ "$AMENDMENT_SHA256" =~ ^[0-9a-f]{64}$ ]]
  [[ -f "$AMENDMENT_PATH" && ! -L "$AMENDMENT_PATH" && ! -w "$AMENDMENT_PATH" ]] \
    || { echo "missing or writable amendment: $AMENDMENT_PATH" >&2; exit 3; }
  [[ "$(sha256sum "$AMENDMENT_PATH" | cut -d' ' -f1)" == "$AMENDMENT_SHA256" ]] \
    || { echo "amendment hash differs" >&2; exit 3; }
  [[ -z "$INDICES" || "$INDICES" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]
fi

# Immutable, clean source at the frozen (or amended) commit.
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git -C "$SOURCE" status --porcelain --untracked-files=normal)" ]]
[[ "$(sha256sum "$PROTOCOL_PATH" | cut -d' ' -f1)" == "$PROTOCOL_SHA256" ]]
# Prints the --array value for amended submissions (empty otherwise); any failed
# check stops the script before sbatch.
ARRAY=$("$PROJECT/venv/bin/python" -I -B - "$PROTOCOL_PATH" "$STUDY" "$PROTOCOL_SHA256" "$SOURCE" \
  "$AMENDMENT_PATH" "$INDICES" "$RERUN" <<'PY'
import json, sys
from pathlib import Path
protocol_path, study, protocol_sha, source, amendment_path, indices, rerun = sys.argv[1:]
protocol = json.load(open(protocol_path))
assert protocol["study"] == study and protocol["status"] == "frozen", "not a frozen protocol for this study"
assert protocol["account"] == "FTA-26-70" and protocol["partition"] == "qgpu"
if not amendment_path:
    print("")
    raise SystemExit(0)
sys.path[:0] = [source + "/experiments/ieee_resubmission_2026-11", source]
from ieee_resub import amendment as AM
from ieee_resub.plan import source_hashes
record = AM.validate(json.load(open(amendment_path)), protocol, protocol_sha, study)
assert Path(amendment_path).name == AM.tag(record) + ".json", "amendment file name differs from its number"
assert source_hashes() == record["source_files_sha256"], "snapshot is not the amended source of this amendment"
group = Path("/mnt/proj1/fta-26-70/rlvram/output") / protocol["group"]
parallel = protocol["caps"]["array_parallelism"]
wanted = []
for part in indices.split(",") if indices else []:
    first, _, last = part.partition("-")
    wanted += list(range(int(first), int(last or first) + 1))
assert len(set(wanted)) == len(wanted), "duplicate index"
if rerun == "0":
    launch = record["amended_launch_blocks"]
    assert set(wanted) <= set(launch["array_indices"]), "an index is not listed by the amendment as never launched"
    for index in wanted:
        block = launch["block_ids"][launch["array_indices"].index(index)]
        assert not (group / "blocks" / block).exists(), f"block {block} was already claimed"
    print(f"{indices}%{parallel}")
else:
    wanted = wanted or list(range(len(record["reruns"])))
    assert set(wanted) <= set(range(len(record["reruns"]))), "unknown rerun index"
    for entry in (record["reruns"][i] for i in wanted):
        block_dir = group / "blocks" / entry["block_id"]
        receipt_path = block_dir / f"period-{int(entry['period'])}.json"
        assert receipt_path.is_file(), f"{entry['experiment_id']} has no terminal receipt yet"
        receipt = json.load(open(receipt_path))
        assert receipt["status"] == entry["required_original_status"], f"{entry['experiment_id']} is not unresolved"
        assert (block_dir / "summary.json").is_file(), f"original block {entry['block_id']} has not finished"
        assert not (group / entry["claim"]).exists(), f"{entry['claim']} was already claimed"
        if "restore_source" in entry:
            assert (Path(entry["restore_source"]["folder"]) / "data.pt").is_file(), "preserved checkpoint is absent"
    print(",".join(map(str, wanted)) + f"%{parallel}")
PY
)

# Gates: formal studies start only after the corresponding smoke test was reviewed
# and the user recorded a pass decision (PLAN.md section 10).
case "$STUDY" in
  e1) GATES=(smoke1) ;;
  e2) GATES=(smoke1 smoke2) ;;
  e3) GATES=(smoke2) ;;
  *) GATES=() ;;
esac
for gate in "${GATES[@]}"; do
  test -s "$PROJECT/protocols/$gate/${gate}_gate.json" || { echo "missing reviewed gate: $gate" >&2; exit 3; }
done

KIND=study
EXPORT=ALL,IEEE_RESUB_SOURCE_ROOT="$SOURCE",SOURCE_COMMIT="$SOURCE_COMMIT",PROTOCOL_PATH="$PROTOCOL_PATH",PROTOCOL_SHA256="$PROTOCOL_SHA256"
ARRAY_ARGS=()
if [[ -n "$AMENDMENT_PATH" ]]; then
  EXPORT=$EXPORT,AMENDMENT_PATH="$AMENDMENT_PATH",AMENDMENT_SHA256="$AMENDMENT_SHA256"
  ARRAY_ARGS=(--array="$ARRAY")
  KIND=amended-blocks
  if [[ "$RERUN" == 1 ]]; then
    KIND=amendment-rerun
    SCRIPT=$SOURCE/experiments/ieee_resubmission_2026-11/slurm/ieee_resub_${STUDY}_amendment.slurm
  fi
fi
test -s "$SCRIPT"
COMMAND=(sbatch --parsable "${ARRAY_ARGS[@]}" --export="$EXPORT" "$SCRIPT")
printf 'command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
if [[ "$MODE" != "--submit" ]]; then
  echo "print-only mode; nothing submitted"
  exit 0
fi
JOB=$("${COMMAND[@]}")
RECEIPT=$PROJECT/protocols/$STUDY/submission-$JOB.json
set -o noclobber
printf '{"study":"%s","kind":"%s","job_id":"%s","array":"%s","source_commit":"%s","source_root":"%s","protocol_path":"%s","protocol_sha256":"%s","amendment_path":"%s","amendment_sha256":"%s","sbatch_script_sha256":"%s","submitted_at_utc":"%s"}\n' \
  "$STUDY" "$KIND" "$JOB" "$ARRAY" "$SOURCE_COMMIT" "$SOURCE" "$PROTOCOL_PATH" "$PROTOCOL_SHA256" \
  "$AMENDMENT_PATH" "$AMENDMENT_SHA256" "$(sha256sum "$SCRIPT" | cut -d' ' -f1)" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RECEIPT"
chmod 0444 "$RECEIPT"
echo "submitted $STUDY ($KIND) as job $JOB; receipt $RECEIPT"

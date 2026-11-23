#!/usr/bin/env python3
"""Render the outer Slurm scripts and a complete local dry run (never submits).

`slurm` writes slurm/ieee_resub_<study>.slurm (source files, hashed into the
protocol). `dryrun` additionally executes the real payload launcher for every
planned invocation inside ieee_resub.sandbox (stub python/apptainer), checks
the trial record against the matrix and the Hydra keys against the patched
VERL configuration, and writes everything under dryrun/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieee_resub import common as C  # noqa: E402

C.bootstrap()

SLURM_TEMPLATE = """#!/usr/bin/env bash
# Outer allocation for study {study} ({description}).
# One array task = one frozen same-device block; the runner never submits jobs.
#SBATCH --job-name=ieee-resub-{study}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{last_index}%{parallelism}
#SBATCH --no-requeue
#SBATCH --output={logs}/ieee-resub-{study}-%A_%a.out
#SBATCH --error={logs}/ieee-resub-{study}-%A_%a.err
set -euo pipefail
: "${{IEEE_RESUB_SOURCE_ROOT:?immutable source snapshot required}}"
: "${{SOURCE_COMMIT:?frozen commit required}}"
: "${{PROTOCOL_PATH:?frozen protocol required}}"
: "${{PROTOCOL_SHA256:?frozen protocol hash required}}"
[[ "$IEEE_RESUB_SOURCE_ROOT" == {source_parent}/* ]]
[[ "$PROTOCOL_PATH" == {protocols}/{study}/* ]]
# Blocks that start under a hash-chained amendment (PLAN.md section 9) pass it on.
AMENDMENT_ARGS=()
if [[ -n "${{AMENDMENT_PATH:-}}" ]]; then
  [[ "$AMENDMENT_PATH" =~ ^{protocols}/{study}/amendment-[0-9]{{2}}\\.json$ ]]
  [[ "${{AMENDMENT_SHA256:-}}" =~ ^[0-9a-f]{{64}}$ ]]
  AMENDMENT_ARGS=(--amendment "$AMENDMENT_PATH" --amendment-sha256 "$AMENDMENT_SHA256")
fi
export TZ=UTC PYTHONDONTWRITEBYTECODE=1
export PATH="{venv}/bin:$PATH"
cd "$IEEE_RESUB_SOURCE_ROOT"
exec python -I -B {runner} run --study {study} \\
  --protocol "$PROTOCOL_PATH" --protocol-sha256 "$PROTOCOL_SHA256" \\
  --source-commit "$SOURCE_COMMIT" --index "$SLURM_ARRAY_TASK_ID" "${{AMENDMENT_ARGS[@]}}"
"""

# Amendment reruns: one array task = one slot listed in the amendment's reruns;
# the same allocation shape and wall cap as the study; the array range is
# passed by submit_ieee_resub.sh from the amendment.
AMENDMENT_SLURM_TEMPLATE = """#!/usr/bin/env bash
# Amendment reruns for study {study} (PLAN.md section 9): one array task = one listed slot.
# The runner never submits jobs; submit_ieee_resub.sh --rerun passes --array.
#SBATCH --job-name=ieee-resub-{study}-amend
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --no-requeue
#SBATCH --output={logs}/ieee-resub-{study}-amend-%A_%a.out
#SBATCH --error={logs}/ieee-resub-{study}-amend-%A_%a.err
set -euo pipefail
: "${{IEEE_RESUB_SOURCE_ROOT:?immutable source snapshot required}}"
: "${{SOURCE_COMMIT:?frozen commit required}}"
: "${{PROTOCOL_PATH:?frozen protocol required}}"
: "${{PROTOCOL_SHA256:?frozen protocol hash required}}"
: "${{AMENDMENT_PATH:?recorded amendment required}}"
: "${{AMENDMENT_SHA256:?amendment hash required}}"
: "${{SLURM_ARRAY_TASK_ID:?submit as an array over the amendment reruns}}"
[[ "$IEEE_RESUB_SOURCE_ROOT" == {source_parent}/* ]]
[[ "$PROTOCOL_PATH" == {protocols}/{study}/* ]]
[[ "$AMENDMENT_PATH" =~ ^{protocols}/{study}/amendment-[0-9]{{2}}\\.json$ ]]
[[ "$AMENDMENT_SHA256" =~ ^[0-9a-f]{{64}}$ ]]
export TZ=UTC PYTHONDONTWRITEBYTECODE=1
export PATH="{venv}/bin:$PATH"
cd "$IEEE_RESUB_SOURCE_ROOT"
exec python -I -B {runner} rerun --study {study} \\
  --protocol "$PROTOCOL_PATH" --protocol-sha256 "$PROTOCOL_SHA256" \\
  --source-commit "$SOURCE_COMMIT" --index "$SLURM_ARRAY_TASK_ID" \\
  --amendment "$AMENDMENT_PATH" --amendment-sha256 "$AMENDMENT_SHA256"
"""
AMENDMENT_RERUN_STUDIES = ("e3",)

DESCRIPTIONS = {
    "smoke1": "development smoke: fixed-length path and shape identity, 2 GPUs",
    "smoke2": "development smoke: validation, checkpoints, retention and restore, 4 GPUs",
    "e1": "E1 controlled-length micro-batch x logging contrast, 2 GPUs",
    "e2": "E2 ten fresh repeats of four near-boundary configurations, 4-GPU allocation",
    "e3": "E3 100-step runs with validation, checkpoints, screens and restore, 4 GPUs",
}


def hhmmss(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def slurm_text(study: str, blocks: int, template: str = SLURM_TEMPLATE) -> str:
    caps = C.CAPS[study]
    return template.format(
        study=study, description=DESCRIPTIONS[study], account=C.ACCOUNT, partition=C.PARTITION,
        gpus=caps["allocation_gpus"], cpus=caps["cpus"], mem=caps["mem"], time=hhmmss(caps["wall_seconds"]),
        last_index=blocks - 1, parallelism=caps["array_parallelism"], logs=C.CLUSTER_LOG_ROOT,
        source_parent=C.CLUSTER_SOURCE_PARENT, protocols=C.CLUSTER_PROJECT_ROOT / "protocols", venv=C.CLUSTER_VENV,
        runner=C.EXPERIMENT_RELATIVE / "ieee_resub/runner.py")


def write_slurm(matrices: Path) -> dict:
    out = {}
    for study in C.ALL_STUDIES:
        rows = C.read_rows(matrices / f"{study}_matrix.csv")
        blocks = len({r["block_id"] for r in rows})
        path = C.EXPERIMENT_DIR / "slurm" / f"ieee_resub_{study}.slurm"
        path.parent.mkdir(exist_ok=True)
        path.write_text(slurm_text(study, blocks))
        path.chmod(0o755)
        out[study] = str(path.relative_to(C.REPO_ROOT))
        if study in AMENDMENT_RERUN_STUDIES:
            path = C.EXPERIMENT_DIR / "slurm" / f"ieee_resub_{study}_amendment.slurm"
            path.write_text(slurm_text(study, blocks, AMENDMENT_SLURM_TEMPLATE))
            path.chmod(0o755)
            out[f"{study}_amendment"] = str(path.relative_to(C.REPO_ROOT))
    return out


def config_keys():
    import yaml
    config = yaml.safe_load((C.REPO_ROOT / C.INSTRUMENTED_PACKAGES / "verl/trainer/config/_generated_ppo_trainer.yaml").read_text())

    # Hydra's own run settings are not part of VERL's schema; the launcher sets
    # these two so that Hydra writes into the per-job runtime, not the read-only
    # source snapshot.
    hydra_keys = {"hydra.run.dir", "hydra.output_subdir"}

    def has(dotted):
        if dotted in hydra_keys:
            return True
        node = config
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True
    return has


def wall_budget(rows, caps):
    """Worst case the runner admits: each period needs timeout+60 s; idle gates <= 2 x 180 s."""
    need = sum(int(r["per_process_timeout_seconds"]) + 60 + 360 for r in rows) + 300
    restore = sum(r["role"] == "restore" for r in rows) * 900
    return dict(required_seconds=need + restore, wall_seconds=caps["wall_seconds"],
                fits=need + restore <= caps["wall_seconds"])


def private_tmp_issues(result, row, paths, job="9000001"):
    """e3-amendment-01: every container gets this invocation's own /tmp."""
    options = result.get("apptainer_options", [])
    runtime = paths["scratch"] / "r" / job / f"matched-{row['experiment_id']}-j{job}"
    binds = [options[i + 1] for i, option in enumerate(options[:-1]) if option == "--bind"]
    targets = [b for b in binds if b.split(":")[1:2] == ["/tmp"]]
    issues = [] if targets == [f"{runtime}/tmp:/tmp"] else [f"container /tmp is not private: {targets}"]
    tokens = [o.split("=", 1)[1] for o in options if o.startswith("RLVRAM_PRIVATE_TMP_TOKEN=")]
    if len(tokens) != 1 or not tokens[0] or result.get("private_tmp_marker") != tokens[0]:
        issues.append("private /tmp token is not visible at the bound /tmp source")
    command = result.get("trainer_command", [])
    if command[:2] != ["bash", "-c"] or "/tmp/.rlvram-private-tmp" not in command[2] or "exit 97" not in command[2]:
        issues.append("trainer does not start behind the in-container private /tmp check")
    return issues


def cluster_mapping(root, paths, row):
    model = C.CLUSTER_HF_HUB / ("models--" + row["model"].replace("/", "--")) / "snapshots" / row["model_revision"]
    return sorted([
        (str(root / "snapshot"), str(model)),
        (str(root / "data"), str(C.CLUSTER_DATA_ROOT)),
        (str(paths["source"]), str(C.CLUSTER_SOURCE_PARENT / "ieee-resub-<SOURCE_COMMIT_12>")),
        (str(C.REPO_ROOT), str(C.CLUSTER_SOURCE_PARENT / "ieee-resub-<SOURCE_COMMIT_12>")),
        (str(paths["project"]), str(C.CLUSTER_PROJECT_ROOT)),
        (str(paths["scratch"]), str(C.CLUSTER_SCRATCH_ROOT)),
        ("9000001", "<SLURM_JOB_ID>"), ("9000002", "<SLURM_JOB_ID>")], key=lambda item: -len(item[0]))


def render_amendment(path, matrices, paths, root, out, has_key):
    """Sandbox launcher run of every amendment rerun, plus its exact submission commands."""
    from ieee_resub import sandbox
    from ieee_resub.amendment import tag
    from ieee_resub.assess import TRIAL_FIELDS, normalized

    amendment = json.loads(path.read_text())
    study = amendment["study"]
    rows = {r["experiment_id"]: r for r in C.read_rows(matrices / f"{study}_matrix.csv")}
    preserved_root = str(C.CLUSTER_SCRATCH_ROOT / "preserved-checkpoints") + "/"
    problems, rendered, budgets = [], [], []
    for entry in amendment["reruns"]:
        row = rows[entry["experiment_id"]]
        source = entry.get("restore_source", {}).get("folder")
        relative = source.removeprefix(preserved_root) if source else None
        result = sandbox.render(row, paths, job="9000002", preserved_source=relative)
        issues = [] if result["returncode"] == 0 else [f"launcher exit {result['returncode']}"]
        trial = result.get("trial") or {}
        issues += [f"trial field {f}" for f in TRIAL_FIELDS if normalized(trial.get(f)) != normalized(row.get(f))]
        command = result.get("trainer_command", [])
        overrides = dict(a.split("=", 1) for a in command if "=" in a and not a.startswith("-"))
        issues += [f"Hydra key {k} absent" for k in overrides if not has_key(k)]
        issues += private_tmp_issues(result, row, paths, job="9000002")
        if source and overrides.get("trainer.resume_from_path") != str(paths["scratch"] / "preserved-checkpoints" / relative):
            issues.append("restore does not resume from the preserved checkpoint")
        if issues:
            problems.append(dict(experiment_id=row["experiment_id"], issues=issues))
        budgets.append(wall_budget([row], C.CAPS[study]))
        rendered.append(relabel(dict(experiment_id=row["experiment_id"], claim=entry["claim"],
                                     launcher_environment=result["environment"],
                                     apptainer_options=result.get("apptainer_options"), trainer_command=command,
                                     trial_record=trial, issues=issues), cluster_mapping(root, paths, row)))
    with (out / "invocations" / f"{path.stem}.jsonl").open("w") as handle:
        for item in rendered:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    caps, sha = C.CAPS[study], C.digest(path)
    protocol = f"{C.CLUSTER_PROJECT_ROOT}/protocols/{study}/{study}_protocol.json"
    target = f"{C.CLUSTER_PROJECT_ROOT}/protocols/{study}/{tag(amendment)}.json"
    launch = amendment["amended_launch_blocks"]["array_indices"]
    indices = f"{min(launch)}-{max(launch)}" if launch == list(range(min(launch), max(launch) + 1)) else ",".join(map(str, launch))
    submit = f"{C.EXPERIMENT_RELATIVE}/submit/submit_ieee_resub.sh {study} {protocol} {amendment['protocol_sha256']} <AMENDED_SOURCE_COMMIT>"
    export = (f"--export=ALL,IEEE_RESUB_SOURCE_ROOT={C.CLUSTER_SOURCE_PARENT}/ieee-resub-<AMENDED_SOURCE_COMMIT_12>,"
              f"SOURCE_COMMIT=<AMENDED_SOURCE_COMMIT>,PROTOCOL_PATH={protocol},PROTOCOL_SHA256={amendment['protocol_sha256']},"
              f"AMENDMENT_PATH={target},AMENDMENT_SHA256={sha}")
    lines = [
        f"# {amendment['amendment_id']} (sha256 {sha} as rendered; recompute after any edit), copied read-only to {target}",
        f"# (a) blocks never launched, array indices {indices}, under the amended snapshot",
        f"{submit} --amendment {target} {sha} --indices {indices} --submit",
        f"#   which runs: sbatch --parsable --array={indices}%{caps['array_parallelism']} {export} "
        f"<SNAPSHOT>/{C.EXPERIMENT_RELATIVE}/slurm/ieee_resub_{study}.slurm",
        f"# (b) amendment reruns, each submitted once its original has ended unresolved:"]
    for index, entry in enumerate(amendment["reruns"]):
        lines += [
            f"#   rerun {index}: {entry['experiment_id']} (original job {entry['original_job_id']})"
            + (f"; after {entry['submit_after']}" if entry.get("submit_after") else ""),
            f"{submit} --amendment {target} {sha} --rerun --indices {index} --submit",
            f"#   which runs: sbatch --parsable --array={index}%{caps['array_parallelism']} {export} "
            f"<SNAPSHOT>/{C.EXPERIMENT_RELATIVE}/slurm/ieee_resub_{study}_amendment.slurm"]
    lines.append("")
    return dict(study=study, amendment_sha256=sha, reruns=len(rendered), problems=problems,
                wall_budget_ok=all(b["fits"] for b in budgets),
                worst_rerun_required_seconds=max((b["required_seconds"] for b in budgets), default=0)), lines


def relabel(value, mapping):
    """Show sandbox paths as the cluster paths they stand for (longest prefix first)."""
    if isinstance(value, str):
        for old, new in mapping:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [relabel(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: relabel(v, mapping) for k, v in value.items()}
    return value


def dryrun(matrices: Path, out: Path) -> dict:
    from ieee_resub import sandbox
    from ieee_resub.assess import TRIAL_FIELDS, normalized

    if out.exists():
        shutil.rmtree(out)
    (out / "sbatch").mkdir(parents=True)
    (out / "invocations").mkdir()
    has_key = config_keys()
    slurm = write_slurm(matrices)
    report = dict(submitted_jobs=0, note="Dry run: nothing was submitted; no cluster path was touched.",
                  studies={}, slurm_scripts=slurm)
    submit_lines = ["#!/usr/bin/env bash", "# Rendered submission commands (NOT executed). Replace <...> after approval.",
                    "set -euo pipefail"]
    root = sandbox.new_root()
    try:
        paths = sandbox.prepare(root)
        for study in C.ALL_STUDIES:
            rows = C.read_rows(matrices / f"{study}_matrix.csv")
            shutil.copy2(C.REPO_ROOT / slurm[study], out / "sbatch" / Path(slurm[study]).name)
            problems, rendered = [], []
            blocks = {}
            for row in rows:
                blocks.setdefault(row["block_id"], []).append(row)
                result = sandbox.render(row, paths)
                issues = []
                if result["returncode"] != 0:
                    issues.append(f"launcher exit {result['returncode']}: {result['stderr_tail'][-300:]}")
                trial = result.get("trial") or {}
                for field in TRIAL_FIELDS:
                    if normalized(trial.get(field)) != normalized(row.get(field)):
                        issues.append(f"trial field {field}: {trial.get(field)!r} != {row.get(field)!r}")
                command = result.get("trainer_command", [])
                overrides = [a.split("=", 1)[0] for a in command if "=" in a and not a.startswith("-")]
                missing = [k for k in overrides if not has_key(k)]
                if missing:
                    issues.append(f"Hydra keys absent from patched config: {missing}")
                options = " ".join(result.get("apptainer_options", []))
                if row["shape_trace"] == "True" and "RLVRAM_SHAPE_TRACE=/" not in options:
                    issues.append("shape trace not enabled in container environment")
                if row["canonical_batch_order"] == "True" and "RLVRAM_CANONICAL_BATCH_ORDER=1" not in options:
                    issues.append("canonical order not enabled in container environment")
                if f"actor_rollout_ref.rollout.fixed_response_length={row['fixed_response_length']}" not in command:
                    issues.append("fixed-length flag not forwarded")
                issues += private_tmp_issues(result, row, paths)
                if issues:
                    problems.append(dict(experiment_id=row["experiment_id"], issues=issues))
                mapping = cluster_mapping(root, paths, row)
                rendered.append(relabel(dict(
                    experiment_id=row["experiment_id"], block_id=row["block_id"], period=row["period"],
                    launcher_environment=result["environment"], apptainer_options=result.get("apptainer_options"),
                    trainer_command=command, trial_record=trial, issues=issues,
                    note="GPU UUIDs are placeholders; the runner substitutes the allocation's physical UUIDs."),
                    mapping))
            with (out / "invocations" / f"{study}.jsonl").open("w") as handle:
                for item in rendered:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
            budgets = {b: wall_budget(members, C.CAPS[study]) for b, members in blocks.items()}
            report["studies"][study] = dict(
                invocations=len(rows), blocks=len(blocks), rendered=len(rendered), problems=problems,
                wall_budget_ok=all(v["fits"] for v in budgets.values()),
                worst_block_required_seconds=max(v["required_seconds"] for v in budgets.values()),
                wall_seconds=C.CAPS[study]["wall_seconds"])
            submit_lines += [
                f"# {study}: {len(blocks)} array tasks x {C.CAPS[study]['allocation_gpus']} GPUs, "
                f"--time={hhmmss(C.CAPS[study]['wall_seconds'])}, at most {C.CAPS[study]['array_parallelism']} concurrent",
                f"{C.EXPERIMENT_RELATIVE}/submit/submit_ieee_resub.sh {study} "
                f"{C.CLUSTER_PROJECT_ROOT}/protocols/{study}/{study}_protocol.json <PROTOCOL_SHA256_{study.upper()}> "
                f"<SOURCE_COMMIT> --submit",
                f"#   which runs: sbatch --parsable --export=ALL,IEEE_RESUB_SOURCE_ROOT={C.CLUSTER_SOURCE_PARENT}/"
                f"ieee-resub-<SOURCE_COMMIT_12>,SOURCE_COMMIT=<SOURCE_COMMIT>,PROTOCOL_PATH=...,PROTOCOL_SHA256=... "
                f"{C.EXPERIMENT_RELATIVE}/slurm/ieee_resub_{study}.slurm", ""]
        for path in sorted((C.EXPERIMENT_DIR / "amendments").glob("*-amendment-[0-9][0-9].json")):
            entry, lines = render_amendment(path, matrices, paths, root, out, has_key)
            report.setdefault("amendments", {})[path.stem] = entry
            submit_lines += lines
    finally:
        shutil.rmtree(root, ignore_errors=True)
    (out / "submit_commands.sh").write_text("\n".join(submit_lines) + "\n")
    entries = list(report["studies"].values()) + list(report.get("amendments", {}).values())
    report["all_rendered_without_problems"] = all(not s["problems"] for s in entries)
    report["all_wall_budgets_ok"] = all(s["wall_budget_ok"] for s in entries)
    C.write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("slurm")
    s.add_argument("--matrices", type=Path, default=C.EXPERIMENT_DIR / "matrices")
    d = sub.add_parser("dryrun")
    d.add_argument("--matrices", type=Path, default=C.EXPERIMENT_DIR / "matrices")
    d.add_argument("--out", type=Path, default=C.EXPERIMENT_DIR / "dryrun")
    args = parser.parse_args()
    if args.command == "slurm":
        print(json.dumps(write_slurm(args.matrices), indent=2))
        return 0
    report = dryrun(args.matrices, args.out)
    summary = {k: dict(invocations=v["invocations"], blocks=v["blocks"], problems=len(v["problems"]),
                       wall_budget_ok=v["wall_budget_ok"]) for k, v in report["studies"].items()}
    summary.update({k: dict(reruns=v["reruns"], problems=len(v["problems"]), wall_budget_ok=v["wall_budget_ok"],
                            amendment_sha256=v["amendment_sha256"]) for k, v in report.get("amendments", {}).items()})
    print(json.dumps(dict(summary=summary, all_ok=report["all_rendered_without_problems"] and report["all_wall_budgets_ok"],
                          submitted_jobs=0), indent=2))
    return 0 if report["all_rendered_without_problems"] and report["all_wall_budgets_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

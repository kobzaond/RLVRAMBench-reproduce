#!/usr/bin/env python3
"""Create and verify a cryptographic manifest for publication artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from memory_tuner.aggregate_online import (
    RETAINABLE_FAILURE_KINDS,
    classify_failed_attempt,
)
from memory_tuner.rl_trial_attempts import select_scientific_rl_attempt
from memory_tuner.trial_attempts import select_trial_attempts
from memory_tuner.artifact_paths import recorded_path


REFERENCED_FILE_FIELDS = (
    "artifact_path",
    "phase_memory_csv",
    "gpu_telemetry_csv",
    "environment_json",
    "run_log",
)
CHECKPOINT_DIRECTORY_PATTERN = re.compile(r"^global_step_[0-9]+$")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(repo_root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return recorded_path(text, repo_root)


def add_file(
    files: dict[Path, str],
    path: Path | None,
    category: str,
    missing: list[str],
) -> None:
    if path is None:
        return
    if not path.is_file():
        missing.append(f"{category}: {path}")
        return
    files[path.absolute()] = category


def is_checkpoint_payload(path: Path, trial_root: Path) -> bool:
    """Return whether *path* is a model-checkpoint payload.

    Temporal studies retain the small checkpoint marker and phase telemetry
    at the trial root, but the multi-gigabyte model/optimizer shards beneath
    ``global_step_<n>`` are neither analysis inputs nor redistributable
    publication artifacts.
    """

    relative = path.relative_to(trial_root)
    return any(
        CHECKPOINT_DIRECTORY_PATTERN.fullmatch(part)
        for part in relative.parts
    )


def add_trial_tree(
    files: dict[Path, str],
    trial_root: Path,
    category: str,
) -> None:
    for path in trial_root.rglob("*"):
        if path.is_file() and not is_checkpoint_payload(path, trial_root):
            # Preserve canonical relative trial links as well as their immutable
            # attempt targets; matrix reconstruction discovers the former.
            files[path.absolute()] = category


def collect_files(
    repo_root: Path,
    *,
    require_complete_transfer: bool = False,
    require_complete_multistep: bool = False,
    require_complete_cross_family: bool = False,
    require_complete_major_revision: bool = False,
    require_complete_strengthening: bool = False,
) -> tuple[dict[Path, str], list[str]]:
    files: dict[Path, str] = {}
    missing: list[str] = []
    corpus = read_csv(repo_root / "profiles/benchmark/benchmark-corpus.csv")
    artifact_manifest = read_csv(
        repo_root / "profiles/benchmark/artifact-manifest.csv"
    )

    for row in corpus:
        for field in REFERENCED_FILE_FIELDS:
            add_file(
                files,
                resolve_path(repo_root, row.get(field)),
                f"scientific_{field}",
                missing,
            )
        trace_dir = resolve_path(repo_root, row.get("allocator_trace_dir"))
        if trace_dir is not None:
            if not trace_dir.is_dir():
                missing.append(f"allocator_trace_dir: {trace_dir}")
            else:
                for path in trace_dir.rglob("*"):
                    if path.is_file():
                        files[path.resolve()] = "allocator_trace"
        data_dir = resolve_path(repo_root, row.get("data_dir"))
        if data_dir is not None:
            for name in ("train.parquet", "test.parquet", "profile.json"):
                path = data_dir / name
                if path.is_file():
                    files[path.resolve()] = "benchmark_dataset"

    for row in artifact_manifest:
        if str(row.get("scientific_valid", "1")) != "0":
            continue
        trial_path = resolve_path(repo_root, row.get("artifact_path"))
        add_file(
            files,
            trial_path,
            "excluded_trial_json",
            missing,
        )
        if trial_path is not None and trial_path.parent.is_dir():
            add_trial_tree(
                files,
                trial_path.parent,
                "excluded_trial_evidence",
            )
            if trial_path.is_file():
                files[trial_path.resolve()] = "excluded_trial_json"

    standalone_matrices = (
        repo_root / "memory_tuner/rlvram_exact_transfer_vllm.csv",
        repo_root
        / "memory_tuner/rlvram_exact_transfer_vllm_repeats.csv",
    )
    standalone_ids = {
        row["experiment_id"]
        for matrix in standalone_matrices
        for row in read_csv(matrix)
    }
    if standalone_ids:
        observed = set()
        for trial_path in (repo_root / "profiles/vllm").glob("*/trial.json"):
            trial = json.loads(trial_path.read_text())
            if trial.get("experiment_id") not in standalone_ids:
                continue
            observed.add(trial["experiment_id"])
            for path in trial_path.parent.iterdir():
                if path.is_file():
                    files[path.resolve()] = "standalone_transfer_raw"
        if require_complete_transfer:
            for experiment_id in sorted(standalone_ids - observed):
                missing.append(
                    f"standalone_transfer_experiment: {experiment_id}"
                )

    online_matrix = (
        repo_root / "memory_tuner/rlvram_exact_transfer_online.csv"
    )
    online_ids = {
        row["experiment_id"] for row in read_csv(online_matrix)
    }
    if online_ids:
        observed = set()
        for trial_path in (
            repo_root / "profiles/vllm-online"
        ).glob("*/trial.json"):
            trial = json.loads(trial_path.read_text())
            if trial.get("experiment_id") not in online_ids:
                continue
            observed.add(trial["experiment_id"])
            for path in trial_path.parent.iterdir():
                if path.is_file():
                    files[path.resolve()] = "online_transfer_raw"
        if require_complete_transfer:
            for experiment_id in sorted(online_ids - observed):
                missing.append(f"online_transfer_experiment: {experiment_id}")

    multistep_matrix = (
        repo_root / "profiles/multistep/matrix-2gpu.csv"
    )
    multistep_ids = {
        row["experiment_id"] for row in read_csv(multistep_matrix)
    }
    if require_complete_multistep and not multistep_ids:
        missing.append(
            "multistep_matrix: profiles/multistep/matrix-2gpu.csv"
        )
    if multistep_ids:
        observed = set()
        for experiment_id in sorted(multistep_ids):
            trial_dir = repo_root / "output/multistep" / experiment_id
            trial_paths = sorted(trial_dir.glob("trial-*.json"))
            try:
                selected, _ = select_scientific_rl_attempt(trial_paths)
            except ValueError as error:
                missing.append(
                    f"multistep_experiment: {experiment_id}: {error}"
                )
                continue
            if selected is None:
                continue
            observed.add(experiment_id)
            add_trial_tree(files, trial_dir, "multistep_raw")
        if require_complete_multistep:
            for experiment_id in sorted(multistep_ids - observed):
                missing.append(f"multistep_experiment: {experiment_id}")

    cross_family_matrix = (
        repo_root / "memory_tuner/rlvram_cross_family_phi4_2gpu.csv"
    )
    cross_family_ids = {
        row["experiment_id"] for row in read_csv(cross_family_matrix)
    }
    if require_complete_cross_family and not cross_family_ids:
        missing.append(
            "cross_family_matrix: "
            "memory_tuner/rlvram_cross_family_phi4_2gpu.csv"
        )
    if cross_family_ids:
        observed = set()
        for experiment_id in sorted(cross_family_ids):
            trial_dir = repo_root / "output/cross_family_phi4" / experiment_id
            trial_paths = sorted(trial_dir.glob("trial-*.json"))
            try:
                selected, _ = select_scientific_rl_attempt(trial_paths)
            except ValueError as error:
                missing.append(
                    f"cross_family_experiment: {experiment_id}: {error}"
                )
                continue
            if selected is None:
                continue
            observed.add(experiment_id)
            add_trial_tree(files, trial_dir, "cross_family_raw")
        if require_complete_cross_family:
            for experiment_id in sorted(cross_family_ids - observed):
                missing.append(
                    f"cross_family_experiment: {experiment_id}"
                )

    v2_trial_tables = (
        repo_root
        / "profiles/benchmark_v2/cross-family-confirmatory-trials.csv",
        repo_root / "profiles/benchmark_v2/sleep-expansion-trials.csv",
        repo_root / "profiles/benchmark_v2/telemetry-trials.csv",
        repo_root / "profiles/benchmark_v2/temporal-confirmatory-trials.csv",
    )
    for table in v2_trial_tables:
        for row in read_csv(table):
            artifact = resolve_path(repo_root, row.get("artifact_path"))
            add_file(files, artifact, "benchmark_v2_raw", missing)
            if artifact is not None and artifact.parent.is_dir():
                add_trial_tree(files, artifact.parent, "benchmark_v2_raw")

    major_revision_rl_groups = (
        (
            repo_root
            / "memory_tuner/rlvram_major_revision_cross_model_smoke.csv",
            repo_root / "output/major_revision_cross_model_compatibility",
            "major_revision_cross_model_compatibility",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_major_revision_cross_model_confirmatory.csv",
            repo_root / "output/major_revision_cross_model_confirmatory",
            "major_revision_cross_model",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_major_revision_transfer_colocated.csv",
            repo_root / "output/major_revision_transfer_colocated",
            "major_revision_transfer_colocated",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_major_revision_residency_period1.csv",
            repo_root / "output/major_revision_residency_counterbalanced",
            "major_revision_residency",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_major_revision_residency_period2.csv",
            repo_root / "output/major_revision_residency_counterbalanced",
            "major_revision_residency",
        ),
    )
    for matrix, trial_root, category in major_revision_rl_groups:
        identifiers = {
            row["experiment_id"] for row in read_csv(matrix)
        }
        if require_complete_major_revision and not identifiers:
            missing.append(f"major_revision_matrix: {matrix}")
            continue
        observed = set()
        for experiment_id in sorted(identifiers):
            trial_dir = trial_root / experiment_id
            trial_paths = sorted(trial_dir.glob("trial-*.json"))
            try:
                selected, _ = select_scientific_rl_attempt(trial_paths)
            except ValueError as error:
                missing.append(
                    f"major_revision_experiment: {experiment_id}: {error}"
                )
                continue
            if selected is None:
                continue
            observed.add(experiment_id)
            add_trial_tree(files, trial_dir, category)
        if require_complete_major_revision:
            for experiment_id in sorted(identifiers - observed):
                missing.append(
                    f"major_revision_experiment: {experiment_id}"
                )

    major_revision_serving_groups = (
        (
            repo_root
            / "memory_tuner/rlvram_major_revision_transfer_offline.csv",
            repo_root / "profiles/vllm",
            "major_revision_transfer_offline",
            False,
        ),
        (
            repo_root
            / "memory_tuner/rlvram_major_revision_transfer_online.csv",
            repo_root / "profiles/vllm-online",
            "major_revision_transfer_online",
            True,
        ),
    )
    for (
        matrix,
        trial_root,
        category,
        retain_scientific_failures,
    ) in major_revision_serving_groups:
        identifiers = {
            row["experiment_id"] for row in read_csv(matrix)
        }
        if require_complete_major_revision and not identifiers:
            missing.append(f"major_revision_matrix: {matrix}")
            continue
        observed = set()
        relevant_paths = []
        for trial_path in trial_root.glob("*/trial.json"):
            trial = json.loads(trial_path.read_text())
            experiment_id = str(trial.get("experiment_id", ""))
            if experiment_id not in identifiers:
                continue
            relevant_paths.append(trial_path)
            add_trial_tree(files, trial_path.parent, category)
        invalid_reasons = {}
        try:
            attempts = select_trial_attempts(relevant_paths)
        except ValueError as error:
            attempts = {}
            if require_complete_major_revision:
                missing.append(
                    f"major_revision_serving_attempts: {error}"
                )
        for experiment_id, attempt in attempts.items():
            if attempt["successful"]:
                observed.add(experiment_id)
                continue
            failure_kind = classify_failed_attempt(attempt)
            if (
                retain_scientific_failures
                and failure_kind in RETAINABLE_FAILURE_KINDS
            ):
                observed.add(experiment_id)
            else:
                invalid_reasons[experiment_id] = failure_kind
        if require_complete_major_revision:
            for experiment_id in sorted(identifiers - observed):
                reason = invalid_reasons.get(experiment_id)
                detail = f" ({reason})" if reason else ""
                missing.append(
                    "major_revision_serving_experiment: "
                    f"{experiment_id}{detail}"
                )

    for table in (repo_root / "profiles/major_revision").glob("*.csv"):
        files[table.resolve()] = "derived_major_revision"

    strengthening_rl_groups = (
        (
            repo_root
            / "memory_tuner/rlvram_strengthening_matched_colocated.csv",
            repo_root / "output/strengthening_matched_transfer_colocated",
            "strengthening_matched_rl",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_strengthening_same_node_period1.csv",
            repo_root / "output/strengthening_same_node_residency",
            "strengthening_same_node",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_strengthening_same_node_period2.csv",
            repo_root / "output/strengthening_same_node_residency",
            "strengthening_same_node",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_strengthening_granite_smoke.csv",
            repo_root / "output/strengthening_granite_compatibility",
            "strengthening_granite_compatibility",
        ),
        (
            repo_root
            / "memory_tuner/rlvram_strengthening_granite_confirmatory.csv",
            repo_root / "output/strengthening_granite_confirmatory",
            "strengthening_granite",
        ),
        (
            repo_root / "memory_tuner/rlvram_strengthening_temporal.csv",
            repo_root / "output/strengthening_temporal_phases",
            "strengthening_temporal",
        ),
        (
            repo_root / "memory_tuner/rlvram_strengthening_factorial.csv",
            repo_root / "output/strengthening_factorial",
            "strengthening_factorial",
        ),
    )
    for matrix, trial_root, category in strengthening_rl_groups:
        identifiers = {row["experiment_id"] for row in read_csv(matrix)}
        if require_complete_strengthening and not identifiers:
            missing.append(f"strengthening_matrix: {matrix}")
            continue
        observed = set()
        for experiment_id in sorted(identifiers):
            trial_dir = trial_root / experiment_id
            trial_paths = sorted(trial_dir.glob("trial-*.json"))
            try:
                selected, _ = select_scientific_rl_attempt(trial_paths)
            except ValueError as error:
                missing.append(
                    f"strengthening_experiment: {experiment_id}: {error}"
                )
                continue
            if selected is None:
                continue
            observed.add(experiment_id)
            add_trial_tree(files, trial_dir, category)
        if require_complete_strengthening:
            for experiment_id in sorted(identifiers - observed):
                missing.append(
                    f"strengthening_experiment: {experiment_id}"
                )

    matched_serving_matrix = (
        repo_root / "memory_tuner/rlvram_strengthening_matched_serving.csv"
    )
    matched_serving_ids = {
        row["experiment_id"] for row in read_csv(matched_serving_matrix)
    }
    matched_serving_paths: dict[str, list[Path]] = {}
    for trial_path in (repo_root / "profiles/vllm-matched").glob(
        "*/trial.json"
    ):
        trial = json.loads(trial_path.read_text())
        experiment_id = str(trial.get("experiment_id", ""))
        if experiment_id not in matched_serving_ids:
            continue
        matched_serving_paths.setdefault(experiment_id, []).append(trial_path)
        add_trial_tree(
            files,
            trial_path.parent,
            "strengthening_matched_serving",
        )
    if require_complete_strengthening:
        if not matched_serving_ids:
            missing.append(
                f"strengthening_matrix: {matched_serving_matrix}"
            )
        for experiment_id in sorted(matched_serving_ids):
            count = len(matched_serving_paths.get(experiment_id, []))
            if count != 1:
                missing.append(
                    "strengthening_matched_serving_experiment: "
                    f"{experiment_id} ({count} artifacts)"
                )

    for table in (repo_root / "profiles/strengthening").glob("*.csv"):
        files[table.resolve()] = "derived_strengthening"
    for path in (
        repo_root / "profiles/strengthening/matched_requests"
    ).glob("*.jsonl"):
        files[path.resolve()] = "strengthening_matched_request_corpus"
    for path in (repo_root / "output/infrastructure_retries").glob("**/*"):
        if path.is_file():
            files[path.resolve()] = "infrastructure_retry_evidence"

    for study in ("instrumentation", "temporal", "topology"):
        add_trial_tree(files, repo_root / "output" / f"revision_{study}",
                       f"scientific_revision_{study}")

    static_globs = (
        ("README.md", "review_entry_point"),
        ("memory_tuner/README.md", "review_entry_point"),
        ("logs/revision-*.out", "scientific_revision_scheduler_log"),
        ("logs/revision-*.err", "scientific_revision_scheduler_log"),
        ("memory_tuner/*.py", "analysis_code"),
        ("memory_tuner/*.sh", "analysis_code"),
        ("memory_tuner/*.csv", "experiment_matrix"),
        ("memory_tuner/*.slurm", "job_launcher"),
        ("*.slurm", "job_launcher"),
        ("sppo_replay/*.py", "training_code"),
        ("instrumented_python_packages/verl/**/*", "instrumented_runtime_source"),
        ("third_party/TransferQueue/**/*", "pinned_runtime_dependency"),
        ("paper/*.md", "paper_artifact"),
        ("paper/*.json", "paper_artifact"),
        ("paper/*.bib", "paper_artifact"),
        ("paper/generated/*.md", "paper_artifact"),
        ("paper/generated/*.json", "paper_artifact"),
        ("paper/tables/*", "paper_table"),
        ("paper/figures/*", "paper_figure"),
        ("paper/submission/*", "submission_package"),
        ("profiles/benchmark/*.csv", "derived_benchmark"),
        ("profiles/benchmark/*.sha256", "provenance_digest"),
        ("profiles/benchmark_v2/*.csv", "derived_benchmark_v2"),
        ("profiles/standard_grpo/*.csv", "derived_standard_grpo"),
        ("profiles/standard_grpo/*.json", "derived_standard_grpo"),
        ("profiles/major_revision/*.txt", "derived_major_revision"),
        ("profiles/strengthening/*.csv", "derived_strengthening"),
        ("profiles/strengthening/same_node/**/*", "same_allocation_provenance"),
        (
            "profiles/strengthening/matched_requests/*.jsonl",
            "strengthening_matched_request_corpus",
        ),
        ("profiles/phaseguard/*.csv", "derived_phaseguard"),
        ("profiles/screening/*.csv", "derived_screening"),
        ("profiles/multistep/*.csv", "derived_multistep"),
        ("profiles/multistep/*.txt", "derived_multistep"),
        ("analysis-requirements.txt", "environment_specification"),
        ("publication-requirements.txt", "environment_specification"),
        (".gitmodules", "source_provenance"),
    )
    output_manifest = (
        repo_root / "profiles/benchmark/publication-artifact-manifest.csv"
    ).resolve()
    output_summary = (repo_root / "paper/artifact_manifest.md").resolve()
    for pattern, category in static_globs:
        for path in repo_root.glob(pattern):
            resolved = path.resolve()
            if (
                path.is_file()
                and resolved not in {output_manifest, output_summary}
                and path.name != ".git"
                and "__pycache__" not in path.parts
                and ".pytest_cache" not in path.parts
            ):
                files[resolved] = category
    return files, missing


def manifest_rows(repo_root: Path, files: dict[Path, str]) -> list[dict]:
    rows = []
    for path, category in sorted(files.items(), key=lambda item: str(item[0])):
        try:
            display = str(path.relative_to(repo_root))
        except ValueError:
            display = str(path)
        rows.append(
            {
                "path": display,
                "category": category,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "category", "size_bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest(repo_root: Path, manifest: Path) -> list[str]:
    errors = []
    for row in read_csv(manifest):
        path = resolve_path(repo_root, row["path"])
        if path is None or not path.is_file():
            errors.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {row['path']}")
            continue
        if file_sha256(path) != row["sha256"]:
            errors.append(f"sha256 mismatch: {row['path']}")
    return errors


def render_summary(rows: list[dict], missing: list[str]) -> str:
    categories = Counter(row["category"] for row in rows)
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    lines = [
        "# Publication artifact manifest",
        "",
        f"- Files hashed: {len(rows)}",
        f"- Total bytes: {total_bytes}",
        f"- Missing referenced artifacts: {len(missing)}",
        "",
        "## Categories",
        "",
    ]
    lines.extend(
        f"- {category}: {count}"
        for category, count in sorted(categories.items())
    )
    if missing:
        lines.extend(["", "## Missing", ""])
        lines.extend(f"- {item}" for item in missing)
    lines.extend(
        [
            "",
            "The CSV manifest stores each file size and SHA-256 digest. Run "
            "`python -m memory_tuner.artifact_manifest --verify` after copying "
            "or releasing the artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "profiles/benchmark/publication-artifact-manifest.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("paper/artifact_manifest.md"),
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--require-complete-transfer",
        action="store_true",
        help=(
            "fail unless every frozen exact offline/online transfer trial "
            "is present; use only after the prospective arrays finish"
        ),
    )
    parser.add_argument(
        "--require-complete-multistep",
        action="store_true",
        help=(
            "fail unless the gated multi-step matrix exists and every "
            "declared trial is present"
        ),
    )
    parser.add_argument(
        "--require-complete-cross-family",
        action="store_true",
        help=(
            "fail unless every frozen Phi-4-mini external-validation "
            "configuration has one scientific outcome"
        ),
    )
    parser.add_argument(
        "--require-complete-major-revision",
        action="store_true",
        help=(
            "fail unless every frozen major-revision RL, offline-serving, "
            "and online-serving row has a scientific outcome"
        ),
    )
    parser.add_argument(
        "--require-complete-strengthening",
        action="store_true",
        help=(
            "fail unless every frozen strengthening RL and matched-serving "
            "row has exactly one admissible scientific artifact"
        ),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    manifest = resolve_path(repo_root, args.manifest)
    if manifest is None:
        raise SystemExit("manifest path is empty")
    if args.verify:
        errors = verify_manifest(repo_root, manifest)
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"verified {len(read_csv(manifest))} artifact files")
        return

    files, missing = collect_files(
        repo_root,
        require_complete_transfer=args.require_complete_transfer,
        require_complete_multistep=args.require_complete_multistep,
        require_complete_cross_family=args.require_complete_cross_family,
        require_complete_major_revision=args.require_complete_major_revision,
        require_complete_strengthening=args.require_complete_strengthening,
    )
    if missing:
        raise SystemExit("\n".join(missing))
    rows = manifest_rows(repo_root, files)
    write_manifest(manifest, rows)
    summary = resolve_path(repo_root, args.summary)
    if summary is None:
        raise SystemExit("summary path is empty")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(render_summary(rows, missing))
    print(f"hashed {len(rows)} publication artifact files")


if __name__ == "__main__":
    main()

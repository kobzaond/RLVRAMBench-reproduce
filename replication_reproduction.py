"""Optional, fail-closed CPU replay of the two archived replication panels.

Release contract (all digests are lowercase SHA256):
replication_reproduction = {
  "schema_version": 1,
  "code_sha256": {"reproduce.py": "...", "replication_reproduction.py": "..."},
  "panels": {
    "completion_replication": {
      "source_tree_sha256": {"original": "...", "amended": "..."},
      "freeze_sha256": "...", "runtime_amendment_sha256": "...",
      "amendment_filename": "completion-runtime-amendment-01.json",
      "allocation_accounting_sha256": "..."
    },
    "replication_control": {
      "source_tree_sha256": {"amended": "..."},
      "execution_seal_sha256": "...", "runtime_amendment_sha256": "..."
      # Optional together, only when the accounting input is staged:
      # "allocation_accounting_sha256": "...",
      # "accounting_code_sha256": {name: "..." for name in ACCOUNTING_CODE_FILES}
    }
  }
}

tree_sha256 hashes the canonical JSON {relative_posix_path: file_sha256}, using
sort_keys=True and separators=(",", ":"). The control amended tree includes its
nested original snapshot. No source code is imported into this parent process.
Completion also requires the manifest-bound source-export-report.json, validated
as an export receipt rather than scientific output. Optional control accounting
executes a separately pinned additive helper from the artifact root, never a
replacement collector or code added to either frozen source snapshot.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


LIVE_PARENT = Path("/mnt/proj3/open-35-44/verl")
PANELS = ("completion_replication", "replication_control")
OUTPUTS = {
    "completion_replication": {"summary.json", "slots.csv", "configurations.csv", "pairs.csv"},
    "replication_control": {
        "replication_control_summary.json", "replication_control_trials.csv",
        "replication_control_blocks.csv", "replication_control_cases.csv"},
}
CONTROL_DIRECTORY = "benchmark/replication_control"
CONTROL_SEAL = "replication_control_execution_seal.json"
CONTROL_AMENDMENT = "replication_control_runtime_amendment.json"
EXPORT_REPORT = "source-export-report.json"
TIMING_FIELD = "collection_cpu_wall_seconds"
ACCOUNTING_CODE_FILES = frozenset(
    "memory_tuner/" + name + ".py" for name in
    ("__init__", "core", "device_contract", "capture_allocation_accounting", "replication_control_accounting"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def contract_sha256(contract):
    """Bind the entire validated release contract, not a projection of panel settings."""
    require(isinstance(contract, dict), "Replication release contract must be an object")
    return hashlib.sha256(canonical(contract).encode()).hexdigest()


def confined(root, relative):
    root, relative = Path(root).absolute(), Path(relative)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts,
            f"Unsafe replication path: {relative}")
    path = root / relative
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            f"Linked replication path: {path}")
    require(path.resolve().is_relative_to(root.resolve()), "Replication path escapes root")
    return path


def inventory(root):
    root = Path(root)
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
            f"Missing or linked replication tree: {root}")
    files, pending = {}, [root]
    while pending:
        path = pending.pop()
        require(not path.is_symlink(), f"Linked replication artifact: {path}")
        if path.is_dir():
            pending.extend(path.iterdir())
        else:
            require(path.is_file(), f"Nonregular replication artifact: {path}")
            files[path.relative_to(root).as_posix()] = sha256(path)
    require(files, f"Empty replication tree: {root}")
    return dict(sorted(files.items()))


def tree_sha256(root):
    return hashlib.sha256(canonical(inventory(root)).encode()).hexdigest()


def checked_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected),
            f"Missing/invalid replication digest: {path}")
    require(path.is_file() and sha256(path) == expected, f"Replication hash differs: {path}")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON field: {path}: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def manifest_entries(manifest):
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        require(row["path"] not in result, "Duplicate evidence manifest path")
        result[row["path"]] = row
    return result


def bind_file(artifact, path, manifest, bound):
    name = path.relative_to(artifact).as_posix()
    digest = sha256(path)
    entry = manifest.get(name)
    require(entry is not None and entry["sha256"] == digest
            and int(entry["size_bytes"]) == path.stat().st_size,
            f"Replication file missing/mismatched in evidence manifest: {name}")
    bound[name] = digest


def bind_tree(artifact, root, manifest, bound):
    for relative in inventory(root):
        bind_file(artifact, confined(root, relative), manifest, bound)


def validate_completion_export(container, settings):
    """Read-only receipt/source validation; no loader, raw audit, or model IO."""
    report = read_json(confined(container, EXPORT_REPORT))
    amendment = read_json(confined(container, settings["amendment_filename"]))
    checked_hash(confined(container, settings["amendment_filename"]),
                 settings["runtime_amendment_sha256"])
    require(set(report) == {
        "schema_version", "status", "amendment_file", "amendment_sha256",
        "original_freeze_sha256", "initial_freeze_sha256", "trees", "cpu_validation",
        "excludes", "notice_provenance", "raw_root_policy", "helper_sha256"},
        "Completion export receipt fields differ")
    require(report["schema_version"] == "completion-source-export-1"
            and report["status"] == "verified_git_free_cpu_export"
            and report["amendment_file"] == settings["amendment_filename"]
            and report["amendment_sha256"] == settings["runtime_amendment_sha256"]
            and report["original_freeze_sha256"] == settings["freeze_sha256"]
            == amendment["original_freeze_sha256"]
            and report["initial_freeze_sha256"] == amendment["initial_freeze_sha256"]
            and set(report["trees"]) == {"original", "amended"},
            "Completion export receipt anchor/status differs")
    require(re.fullmatch(r"[a-f0-9]{64}", report["helper_sha256"] or "")
            and report["excludes"] == ["Git metadata", "model weights/cache", "containers", "raw target attempts"]
            and all(isinstance(report[k], str) and report[k] for k in ("notice_provenance", "raw_root_policy")),
            "Invalid non-scientific export receipt provenance")
    for role, tree in report["trees"].items():
        source = confined(container, role)
        files = inventory(source)
        actual = {name: {"sha256": digest, "size_bytes": confined(source, name).stat().st_size}
                  for name, digest in files.items()}
        require(set(tree) == {"source_commit", "runtime_files", "fitting_planning_input_entries",
                              "freeze_files", "notices", "files", "file_count", "total_bytes"}
                and canonical(tree["files"]) == canonical(actual)
                and type(tree["file_count"]) is int and tree["file_count"] == len(actual)
                and type(tree["total_bytes"]) is int
                and tree["total_bytes"] == sum(item["size_bytes"] for item in actual.values())
                and tree["source_commit"] == amendment[role + "_source_commit"],
                f"Completion export receipt file inventory differs: {role}")
        runtime = amendment[role + "_runtime_sha256"]
        protocol = read_json(source / "benchmark/completion_replication/protocol.json")
        inputs = protocol["input_sha256"]
        expected = {}
        for mapping in (runtime, inputs, tree["notices"]):
            for name, digest in mapping.items():
                require(name not in expected or expected[name] == digest, "Conflicting export receipt hashes")
                expected[name] = digest
        require({"LICENSE-RLVRAMBench", "instrumented_python_packages/verl/LICENSE",
                 "third_party/TransferQueue/LICENSE"} <= set(tree["notices"]),
                "Completion export receipt omits notices")
        for directory, anchor in (("completion_replication", settings["freeze_sha256"]),
                                  ("completion_replication_initial", amendment["initial_freeze_sha256"])):
            top = confined(source, "benchmark/" + directory)
            names = {"freeze.json", "protocol.json", "matrix.csv", "predictions.json"}
            require({p.name for p in top.iterdir()} == names, "Exported freeze file inventory differs")
            checked_hash(top / "freeze.json", anchor)
            frozen = read_json(top / "freeze.json")["files_sha256"]
            require(set(frozen) == names - {"freeze.json"}, "Exported freeze content inventory differs")
            for name in names:
                relative = f"benchmark/{directory}/{name}"
                expected[relative] = anchor if name == "freeze.json" else frozen[name]
        require(expected == files and type(tree["runtime_files"]) is int
                and tree["runtime_files"] == len(runtime)
                and type(tree["fitting_planning_input_entries"]) is int
                and tree["fitting_planning_input_entries"] == len(inputs)
                and type(tree["freeze_files"]) is int and tree["freeze_files"] == 8,
                f"Completion export receipt closure differs: {role}")
    source = container / "original/benchmark/completion_replication"
    with (source / "matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    protocol = read_json(source / "protocol.json")
    cpu = report["cpu_validation"]
    expected = {
        "status": "passed", "loader": "amended/memory_tuner/completion_replication_protocol.py",
        "amendment_sha256": settings["runtime_amendment_sha256"], "freeze_sha256": settings["freeze_sha256"],
        "planned_slots": len(rows), "planned_pairs": len({r["pair_id"] for r in rows}),
        "target_configurations": len(protocol["targets"]),
        "prediction_rows": len(read_json(source / "predictions.json")),
    }
    versions = {"python", "numpy", "scipy", "scikit_learn"}
    require(set(cpu) == set(expected) | versions
            and canonical({k: cpu[k] for k in expected}) == canonical(expected)
            and all(isinstance(cpu[k], str) and cpu[k] for k in versions),
            "Completion export CPU receipt differs from its frozen inputs")
    return report


def verify_accounting_code(artifact, code, hashes, manifest=None):
    require(isinstance(hashes, dict) and set(hashes) == ACCOUNTING_CODE_FILES,
            "Control accounting requires its exact five-file additive code closure")
    for name, digest in hashes.items():
        for root in (code, artifact):
            checked_hash(confined(root, name), digest)
        if manifest is not None:
            bind_file(artifact, confined(artifact, name), manifest, {})


def prepare_panels(artifact, code, release, manifest, *, isolated):
    """Verify all new input/source/reference inventories before any analysis."""
    present = {
        panel for panel in PANELS if any(
            (root / prefix / panel).exists() or (root / prefix / panel).is_symlink()
            for root in (artifact, code) for prefix in ("reproduction_sources", "benchmark", "output"))
    }
    metadata = release.get("replication_reproduction")
    if metadata is None and not present:
        return {}
    require(isinstance(metadata, dict) and set(metadata) == {"schema_version", "code_sha256", "panels"}
            and type(metadata["schema_version"]) is int and metadata["schema_version"] == 1,
            "Replication sources/references require complete release metadata")
    require(isolated, "Replication reconstruction requires --isolate-analysis")
    require(isinstance(metadata["panels"], dict) and set(metadata["panels"]) == present and present,
            "Replication release panels and archived cohorts disagree")
    require(set(metadata["code_sha256"]) == {"reproduce.py", "replication_reproduction.py"},
            "Replication wrapper code inventory differs")
    for name, digest in metadata["code_sha256"].items():
        checked_hash(confined(code, name), digest)
    require(release.get("benchmark_protocol_version"), "Replication needs hidden benchmark references")
    entries, prepared = manifest_entries(manifest), {}
    for panel in PANELS:
        if panel not in present:
            continue
        settings = metadata["panels"][panel]
        common = {"source_tree_sha256", "runtime_amendment_sha256"}
        fields = ({"freeze_sha256", "amendment_filename", "allocation_accounting_sha256"}
                  if panel == "completion_replication" else {"execution_seal_sha256"})
        optional = {"allocation_accounting_sha256", "accounting_code_sha256"} if panel == "replication_control" else set()
        require(isinstance(settings, dict) and set(settings) in (common | fields, common | fields | optional),
                f"Incomplete replication panel metadata: {panel}")
        container = confined(artifact, "reproduction_sources/" + panel)
        reference = confined(artifact, f"benchmark/{panel}/results")
        raw = confined(artifact, "output/" + panel)
        bound = {}
        for tree in (container, reference, raw):
            bind_tree(artifact, tree, entries, bound)
        require(set(inventory(reference)) == OUTPUTS[panel], f"Unexpected {panel} reference outputs")
        if panel == "completion_replication":
            filename = settings["amendment_filename"]
            require(isinstance(filename, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", filename),
                    "Completion amendment must be a safe top-level JSON basename")
            sources = {name: confined(container, name) for name in ("original", "amended")}
            amendment_path = confined(container, filename)
            require({p.name for p in container.iterdir()} == {"original", "amended", filename, EXPORT_REPORT},
                    "Unexpected completion snapshot entries")
            accounting = confined(artifact, "benchmark/completion_replication/allocation_accounting.json")
            checked_hash(accounting, settings["allocation_accounting_sha256"])
            entry = entries.get(accounting.relative_to(artifact).as_posix())
            require(entry is not None and entry["sha256"] == settings["allocation_accounting_sha256"]
                    and int(entry["size_bytes"]) == accounting.stat().st_size,
                    "Completion accounting is not manifest-bound")
            bound[accounting.relative_to(artifact).as_posix()] = settings["allocation_accounting_sha256"]
            for source in sources.values():
                checked_hash(confined(source, "benchmark/completion_replication/freeze.json"),
                             settings["freeze_sha256"])
            original = sources["original"]
        else:
            sources = {"amended": container}
            original = confined(container, "replication_control_base_source")
            require(original.is_dir(), "Original control source cohort is missing")
            amendment_path = confined(container, CONTROL_DIRECTORY + "/" + CONTROL_AMENDMENT)
            for source in (container, original):
                checked_hash(confined(source, CONTROL_DIRECTORY + "/" + CONTROL_SEAL),
                             settings["execution_seal_sha256"])
            accounting = confined(artifact, "benchmark/replication_control/allocation_accounting.json")
            has_accounting = "allocation_accounting_sha256" in settings
            require(accounting.exists() == has_accounting,
                    "Control accounting input and optional release fields disagree")
            if has_accounting:
                checked_hash(accounting, settings["allocation_accounting_sha256"])
                bind_file(artifact, accounting, entries, bound)
                verify_accounting_code(artifact, code, settings["accounting_code_sha256"], entries)
        require(isinstance(settings["source_tree_sha256"], dict)
                and set(settings["source_tree_sha256"]) == set(sources),
                "Missing original/amended source-tree anchors")
        for cohort, source in sources.items():
            require(tree_sha256(source) == settings["source_tree_sha256"][cohort],
                    f"Archived replication source-tree digest differs: {panel}/{cohort}")
        for source in {*sources.values(), original}:
            for other in PANELS:
                for relative in ("output/" + other, f"benchmark/{other}/results"):
                    require(not (source / relative).exists(), "Target raw/references leaked into source snapshot")
        checked_hash(amendment_path, settings["runtime_amendment_sha256"])
        amendment = read_json(amendment_path)
        if panel == "completion_replication":
            require(amendment["original_freeze_sha256"] == settings["freeze_sha256"],
                    "Completion amendment/freeze anchors disagree")
            require(re.fullmatch(r"[a-f0-9]{40}", amendment["amended_source_commit"] or ""),
                    "Missing amended completion execution commit")
            validate_completion_export(container, settings)
        else:
            require(amendment["base_seal_sha256"] == settings["execution_seal_sha256"],
                    "Control amendment/seal anchors disagree")
        prepared[panel] = dict(settings=settings, sources=sources, original=original,
                               amendment_path=amendment_path, amendment=amendment, bound=bound, code_root=code)
    return prepared


def restore_panel_inputs(artifact, reference, panels):
    """Only explicit accounting is restored; all forecasts live in sealed snapshots."""
    for name, panel in panels.items():
        if "allocation_accounting_sha256" not in panel["settings"]:
            continue
        relative = name + "/allocation_accounting.json"
        source = confined(reference, relative)
        target = confined(artifact, "benchmark/" + relative)
        expected = panel["settings"]["allocation_accounting_sha256"]
        checked_hash(source, expected)
        target.parent.mkdir(parents=True, exist_ok=False)
        with target.open("xb") as writer, source.open("rb") as reader:
            shutil.copyfileobj(reader, writer)
        checked_hash(target, expected)


def verify_panel_inputs(name, panel, artifact, reference):
    """Recheck inventories, not just previously discovered files, around children."""
    actual = {}
    for root, prefix in (
        (artifact / "reproduction_sources" / name, f"reproduction_sources/{name}/"),
        (artifact / "output" / name, f"output/{name}/"),
        (reference / name / "results", f"benchmark/{name}/results/"),
    ):
        actual.update({prefix + key: value for key, value in inventory(root).items()})
    if "allocation_accounting_sha256" in panel["settings"]:
        relative = name + "/allocation_accounting.json"
        expected = panel["settings"]["allocation_accounting_sha256"]
        for root, path in ((reference, relative), (artifact, "benchmark/" + relative)):
            checked_hash(confined(root, path), expected)
        actual["benchmark/" + relative] = expected
    if name == "replication_control" and "accounting_code_sha256" in panel["settings"]:
        verify_accounting_code(artifact, panel["code_root"], panel["settings"]["accounting_code_sha256"])
    require(actual == panel["bound"], "Replication source/raw/reference input inventory changed")


def normalize(text, aliases):
    for source, target in sorted(aliases.items(), key=lambda pair: len(pair[0]), reverse=True):
        require(source.startswith("/") and len(Path(source).parts) > 2 and ".." not in Path(source).parts,
                "Unsafe recorded normalization root")
        # A path prefix must end at a path/token boundary, never inside a label.
        text = re.sub(re.escape(source) + r'(?=/|["\s,;)}\]]|$)', lambda _: target, text)
    return text


def compare_outputs(panel, reference, output, aliases):
    require(set(inventory(reference)) == set(inventory(output)) == OUTPUTS[panel],
            f"Reconstructed {panel} output inventory differs")
    timing = {}
    for name in sorted(OUTPUTS[panel]):
        old = normalize((reference / name).read_text(), aliases)
        new = normalize((output / name).read_text(), aliases)
        if panel == "completion_replication" and name == "summary.json":
            values = [json.loads(old), json.loads(new)]
            for label, value in zip(("published_seconds", "reconstructed_seconds"), values):
                seconds = value.pop(TIMING_FIELD, None)
                require(type(seconds) in (float, int) and math.isfinite(seconds) and seconds > 0,
                        "Completion collection CPU timing must be present, positive and finite")
                timing[label] = seconds
            old, new = (canonical(value) for value in values)
        require(old == new, f"Scientific replication reconstruction differs: {panel}/{name}")
    return timing


def replay_completion(panel, artifact, work, hidden, run_child):
    source = panel["original"]
    path = work / "completion-replication-prediction-verification.json"
    script = (
        "import json,sys; from pathlib import Path; "
        "from memory_tuner.completion_replication_protocol import load_frozen; "
        "root=Path.cwd(); f=load_frozen(root,root/'benchmark/completion_replication',sys.argv[1]); "
        "r=dict(status='passed',freeze_sha256=f['freeze_sha256'],"
        "protocol_sha256=f['protocol_sha256'],matrix_sha256=f['matrix_sha256'],"
        "predictions=f['predictions'],fresh_refit_performed=False,prospective_outcomes_read=False); "
        "h=Path(sys.argv[2]).open('x'); json.dump(r,h,sort_keys=True,allow_nan=False); h.close()"
    )
    replay_hidden = list(dict.fromkeys([
        *hidden, artifact / "output", artifact / "logs", artifact / "paper",
        artifact / "benchmark/completion_replication", artifact / "benchmark/replication_control",
        artifact / "reproduction_sources/replication_control", panel["sources"]["amended"],
        panel["amendment_path"], panel["amendment_path"].parent / EXPORT_REPORT,
        work / "replication_control-results", work / "completion_replication-results",
        work / "replication_control-accounting.json",
    ]))
    run_child(["-c", script, panel["settings"]["freeze_sha256"], str(path)],
              work, replay_hidden, cwd=source)
    actual = read_json(path)
    freeze = read_json(source / "benchmark/completion_replication/freeze.json")
    expected = dict(status="passed", freeze_sha256=panel["settings"]["freeze_sha256"],
                    protocol_sha256=freeze["files_sha256"]["protocol.json"],
                    matrix_sha256=freeze["files_sha256"]["matrix.csv"],
                    predictions=read_json(source / "benchmark/completion_replication/predictions.json"),
                    fresh_refit_performed=False, prospective_outcomes_read=False)
    require(canonical(actual) == canonical(expected), "Completion frozen prediction replay differs")
    return dict(prediction_replay_passed=True, prediction_replay_outcomes_hidden=True,
                fresh_refit_performed=False, prediction_hidden_paths=list(map(str, replay_hidden)))


def reconstruct_panels(artifact, reference, work, hidden, panels, run_child):
    if not panels:
        return {}
    require(hidden, "Replication reconstruction cannot run without namespace isolation")
    hidden = list(dict.fromkeys([*hidden, LIVE_PARENT, reference]))
    reports = {}
    for name in PANELS:
        if name not in panels:
            continue
        panel = panels[name]
        source, settings = panel["sources"]["amended"], panel["settings"]
        verify_panel_inputs(name, panel, artifact, reference)
        report = {}
        if name == "completion_replication":
            report.update(replay_completion(panel, artifact, work, hidden, run_child))
            verify_panel_inputs(name, panel, artifact, reference)
        output = work / (name + "-results")
        aliases = {str(artifact): "<ARTIFACT_ROOT>", str(LIVE_PARENT / "rl"): "<ARTIFACT_ROOT>"}
        if name == "completion_replication":
            args = [
                "-m", "memory_tuner.collect_completion_replication", "--root", str(artifact),
                "--source-root", str(source), "--protocol-dir", str(source / "benchmark/completion_replication"),
                "--freeze-sha256", settings["freeze_sha256"],
                "--source-commit", panel["amendment"]["amended_source_commit"],
                "--allocation-accounting", str(artifact / "benchmark/completion_replication/allocation_accounting.json"),
                "--runtime-amendment", str(panel["amendment_path"]),
                "--amendment-sha256", settings["runtime_amendment_sha256"],
                "--original-source-root", str(panel["original"]), "--output", str(output),
            ]
            for cohort in ("original", "amended"):
                aliases[str(panel["sources"][cohort])] = f"<{cohort.upper()}_SOURCE>"
                aliases[panel["amendment"][cohort + "_source_root"]] = f"<{cohort.upper()}_SOURCE>"
        else:
            args = [
                "-m", "memory_tuner.collect_replication_control",
                "--design-dir", str(source / CONTROL_DIRECTORY),
                "--results-root", str(artifact / "output/replication_control"),
                "--seal-sha256", settings["execution_seal_sha256"],
                "--amendment", str(panel["amendment_path"]),
                "--amendment-sha256", settings["runtime_amendment_sha256"],
                "--base-source", str(panel["original"]), "--allow-relocation", "--output", str(output),
            ]
            aliases[panel["amendment"]["results_root"]] = "<CONTROL_RAW>"
            aliases[str(artifact / "output/replication_control")] = "<CONTROL_RAW>"
        other = next(p for p in PANELS if p != name)
        collector_hidden = list(dict.fromkeys([
            *hidden, artifact / "output" / other, artifact / "reproduction_sources" / other,
            work / (other + "-results"),
        ]))
        run_child(args, work, collector_hidden, cwd=source)
        timing = compare_outputs(name, reference / name / "results", output, aliases)
        verify_panel_inputs(name, panel, artifact, reference)
        report.update(
            panel_files_matched=4, archived_source_only=True, source_and_manifest_verified=True,
            independent_raw_reconstruction_passed=True, reference_outcomes_hidden=True,
            child_namespace_verified=True, runtime_amendment_sha256=settings["runtime_amendment_sha256"],
            source_tree_sha256=settings["source_tree_sha256"],
            collector_hidden_paths=list(map(str, collector_hidden)), archived_source_cwd=str(source),
            scientific_comparison=("Exact after path normalization; only summary collection CPU time excluded."
                                   if name == "completion_replication" else
                                   "All four files exact after path normalization; no field exclusions."))
        if timing:
            report["collection_cpu_wall_seconds"] = timing
            report["allocation_accounting_sha256"] = settings["allocation_accounting_sha256"]
            report["source_export_receipt_verified"] = True
            report["source_export_receipt_sha256"] = sha256(panel["amendment_path"].parent / EXPORT_REPORT)
        elif "allocation_accounting_sha256" in settings:
            accounting_output = work / "replication_control-accounting.json"
            run_child([
                "-m", "memory_tuner.replication_control_accounting",
                "--results-root", str(artifact / "output/replication_control"),
                "--accounting", str(artifact / "benchmark/replication_control/allocation_accounting.json"),
                "--output", str(accounting_output),
            ], work, collector_hidden, cwd=artifact)
            cost = read_json(accounting_output)
            require(type(cost["schema_version"]) is int and cost["schema_version"] == 1
                    and type(cost["allocation_count"]) is int and 0 < cost["allocation_count"] <= 36
                    and len(cost["allocations"]) == cost["allocation_count"],
                    "Invalid reconstructed control allocation count/schema")
            costs = [item["whole_allocation_gpu_seconds"] for item in cost["allocations"]]
            require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        for value in [cost["whole_allocation_gpu_seconds"], *costs])
                    and sum(costs) == cost["whole_allocation_gpu_seconds"]
                    and len({item["job_id"] for item in cost["allocations"]}) == cost["allocation_count"],
                    "Invalid reconstructed whole-allocation control costs")
            verify_panel_inputs(name, panel, artifact, reference)
            # The additive cost helper must not alter any of the four scientific files.
            compare_outputs(name, reference / name / "results", output, aliases)
            report.update(
                allocation_accounting_sha256=settings["allocation_accounting_sha256"],
                accounting_code_sha256=settings["accounting_code_sha256"],
                whole_allocation_accounting_reconstruction_passed=True,
                whole_allocation_accounting=cost)
        else:
            report["whole_allocation_accounting"] = "Separate parent report; not reconstructed or invented here"
        reports[name] = report
    return {"replication_reproduction": reports}

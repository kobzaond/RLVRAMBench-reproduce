#!/usr/bin/env python3
"""Regenerate every reported RLVRAMBench result from the five released archives.

One procedure for the release ``rlvrambench-2026-11``. It obtains the five
raw-evidence archives listed in ``release.json`` (from a local directory, or
downloaded from the Hugging Face dataset at an explicit commit), verifies
their size and SHA-256, runs the documented CPU procedure for each archive,
reruns the retrospective CPU re-analyses, renders the paper figures under
their manuscript file names, and writes ``REPORT.md`` and ``report.json``
mapping every table and figure of the paper to its archive, step, and output.

Run it through ``reproduce_all.sh``, which creates the two pinned CPU
environments. Nothing here needs a GPU, Slurm, model weights, or network
access other than the optional archive download. No step reruns training.

    bash reproduce_all.sh --revision <dataset commit> --work-dir reproduce-all-output
    bash reproduce_all.sh --archives-dir DIR --work-dir reproduce-all-output
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

CODE = Path(__file__).resolve().parent
RELEASE = json.loads((CODE / "release.json").read_text())
REPRODUCTION = CODE / "reproduction"
EXPECTED = json.loads((REPRODUCTION / "expected_outputs.json").read_text())
PAPER_MAP = json.loads((REPRODUCTION / "paper_map.json").read_text())
REVISION_CODE = CODE / "revision_2026_11"
CLUSTER_OUTPUT = "/mnt/proj1/fta-26-70/rlvram/output"
HEX40 = re.compile(r"[0-9a-f]{40}")
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
           "SOURCE_DATE_EPOCH": "1761177600"}

# Child bootstrap: isolated interpreter, explicit sys.path, optional audit hook
# that denies reads of saved derived results and network connections, and an
# optional log of every file opened below a root (input provenance).
BOOT = r'''
import atexit, json, os, runpy, sys
from pathlib import Path
spec = json.loads(sys.argv[1])
deny = [Path(p) for p in spec.get("deny_read", [])]
log_root = Path(spec["log_reads_under"]) if spec.get("log_reads_under") else None
seen = set()
def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        mode = args[1]
        reading = not isinstance(mode, str) or "r" in mode or "+" in mode
        if reading and any(path.is_relative_to(d) for d in deny):
            raise PermissionError("Saved derived result read denied: " + str(path))
        if reading and log_root is not None and path.is_relative_to(log_root) and path.is_file():
            seen.add(str(path.relative_to(log_root)))
    elif event == "socket.connect" and spec.get("deny_network"):
        raise PermissionError("Reconstruction must not use network access")
if log_root is not None:
    def dump():
        with open(spec["log_file"], "a") as handle:
            for item in sorted(seen):
                handle.write(item + "\n")
    atexit.register(dump)
sys.addaudithook(audit)
for entry in reversed(spec["path"]):
    sys.path.insert(0, entry)
if spec.get("chdir"):
    os.chdir(spec["chdir"])
sys.argv = [spec["target"], *spec["args"]]
if spec["kind"] == "module":
    runpy.run_module(spec["target"], run_name="__main__", alter_sys=True)
else:
    runpy.run_path(spec["target"], run_name="__main__")
'''


# --------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


class StepFailed(Exception):
    pass


def require(condition, message):
    if not condition:
        raise StepFailed(message)


class Runner:
    def __init__(self, args):
        self.args = args
        self.work = args.work_dir.resolve()
        self.logs = self.work / "logs"
        self.tmp = self.work / "tmp"
        self.python = Path(sys.executable)
        self.reanalysis_python = args.reanalysis_python
        self.steps: list[dict] = []
        self.state: dict[str, dict] = {}
        self.archives: dict[str, Path] = {}

    # -- process execution -------------------------------------------------
    def env(self, extra=None):
        env = dict(os.environ)
        env.update(THREADS)
        env["TMPDIR"] = str(self.tmp)
        env.update(extra or {})
        return env

    def short(self, text: str) -> str:
        text = text.replace(str(self.work), "$WORK").replace(str(CODE), "$CODE")
        if self.args.archives_dir:
            text = text.replace(str(self.args.archives_dir.resolve()), "$ARCHIVES")
        return text

    def run(self, step, command, *, cwd=None, env=None, label=None):
        log = self.logs / f"{step}.log"
        printable = " ".join(map(str, command))
        shown = self.short(printable if label is None else label)
        self.current["commands"].append(shown)
        with log.open("a") as handle:
            handle.write(f"\n$ {shown}\n")
            handle.flush()
            started = time.monotonic()
            completed = subprocess.run(list(map(str, command)), cwd=cwd or self.work, env=self.env(env),
                                       stdout=handle, stderr=subprocess.STDOUT, text=True)
            handle.write(f"[exit {completed.returncode} after {time.monotonic() - started:.1f} s]\n")
        if completed.returncode:
            tail = log.read_text().splitlines()[-25:]
            raise StepFailed(f"command failed ({completed.returncode}): {shown}\n" + "\n".join(tail))

    def run_python(self, step, kind, target, args, *, python=None, path=(), deny_read=(), deny_network=True,
                   log_reads_under=None, log_file=None, chdir=None, env=None, display=None):
        spec = dict(kind=kind, target=str(target), args=[str(a) for a in args], path=[str(p) for p in path],
                    deny_read=[str(p) for p in deny_read], deny_network=deny_network, chdir=str(chdir or self.work),
                    log_reads_under=str(log_reads_under) if log_reads_under else None,
                    log_file=str(log_file) if log_file else None)
        python = python or self.python
        shown = display or (f"{Path(python).name} -I " + ("-m " if kind == "module" else "") + f"{target} "
                            + " ".join(map(str, args)))
        self.run(step, [python, "-I", "-B", "-c", BOOT, json.dumps(spec)], cwd=chdir or self.work, env=env,
                 label=shown)

    def extract(self, step, archive: Path, target: Path):
        require(not target.exists(), f"extraction target exists: {target}")
        target.mkdir(parents=True)
        self.run(step, ["tar", "--use-compress-program=unzstd", "--no-same-owner", "-xf", archive, "-C", target])
        links = [p for p in target.rglob("*") if p.is_symlink()]
        require(not links, f"archive contains symbolic links: {links[:3]}")

    # -- step bookkeeping --------------------------------------------------
    def step(self, name, title, function, *, needs=(), informational=False):
        record = dict(step=name, title=title, status="running", started_utc=now(), commands=[], checks=[],
                      outputs={}, notes=[], informational=informational)
        self.current = record
        self.steps.append(record)
        missing = [n for n in needs if self.state.get(n, {}).get("status") != "passed"]
        started = time.monotonic()
        print(f"[{now()}] {name}: {title}", flush=True)
        try:
            if missing:
                record["status"] = "not_run"
                record["notes"].append("Not run because these required steps did not pass: " + ", ".join(missing))
            else:
                function(record)
                failed = [c for c in record["checks"] if c["ok"] is False and not c.get("informational")]
                record["status"] = "failed" if failed else ("informational" if informational else "passed")
        except StepFailed as error:
            record["status"] = "failed"
            record["error"] = str(error)
        except Exception as error:  # noqa: BLE001 - report every unexpected failure
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()[-3000:]}"
        record["seconds"] = round(time.monotonic() - started, 1)
        record["finished_utc"] = now()
        self.state[name] = record
        print(f"[{now()}] {name}: {record['status']} ({record['seconds']} s)", flush=True)

    def check(self, name, ok, detail="", informational=False):
        self.current["checks"].append(dict(check=name, ok=bool(ok), detail=detail, informational=informational))
        return ok

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).relative_to(self.work))
        except ValueError:
            return str(path)


# --------------------------------------------------------------------------- comparisons
def files_under(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def same_bytes(a: Path, b: Path, transform=None) -> bool:
    if not (a.is_file() and b.is_file()):
        return False
    left = a.read_bytes()
    if transform:
        left = transform(left)
    return left == b.read_bytes()


def relocation(prefix_from: str, prefix_to: str):
    old, new = prefix_from.encode(), prefix_to.encode()
    return lambda data: data.replace(old, new)


def normalized_raw_path(value, key=None):
    """The A100 follow-up verifier's rule: only raw_path values are relocated."""
    if isinstance(value, list):
        return [normalized_raw_path(v) for v in value]
    if isinstance(value, dict):
        return {k: normalized_raw_path(v, k) for k, v in value.items()}
    if isinstance(value, str) and key == "raw_path" and "/output/" in value:
        return "output/" + value.split("/output/", 1)[1]
    return value


def read_normalized(path: Path):
    if path.suffix == ".json":
        return normalized_raw_path(json.loads(path.read_text()))
    with path.open(newline="") as handle:
        return normalized_raw_path(list(csv.DictReader(handle)))


def manifest_check(root: Path, manifest: Path, *, complete=True):
    """Verify a sha256sum-style manifest; with complete=True every file must be listed."""
    listed = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.lstrip("*").strip()
        require(not name.startswith("/") and ".." not in Path(name).parts, f"unsafe manifest entry {name}")
        listed[name] = digest
    bad = [name for name, digest in listed.items()
           if not (root / name).is_file() or sha256(root / name) != digest]
    extra = []
    if complete:
        present = set(files_under(root))
        extra = sorted(present - set(listed) - {str(manifest.relative_to(root))})
    return len(listed), bad, extra


# --------------------------------------------------------------------------- steps
def archive_entry(cohort):
    for entry in RELEASE["release_archives"]:
        if entry["cohort"] == cohort:
            return entry
    raise KeyError(cohort)


def step_archives(r: Runner, record):
    revision = r.args.revision
    if revision is None and HEX40.fullmatch(str(RELEASE["huggingface_revision"])):
        revision = r.args.revision = RELEASE["huggingface_revision"]
    if revision:
        require(HEX40.fullmatch(revision), "--revision must be a 40-hex Hugging Face commit")
    source_dir = r.args.archives_dir.resolve() if r.args.archives_dir else None
    download_dir = source_dir or (r.work / "archives")
    for entry in RELEASE["release_archives"]:
        name = Path(entry["path"]).name
        local = source_dir / name if source_dir else None
        if local is not None and local.exists():
            path, origin = local, "local"
        else:
            require(revision is not None,
                    f"{name} is not in --archives-dir and no --revision was given for download")
            download_dir.mkdir(parents=True, exist_ok=True)
            path = download_dir / name
            require(not path.exists(), f"refusing to overwrite {path}")
            url = (f"https://huggingface.co/datasets/{RELEASE['huggingface_repository']}/resolve/"
                   f"{revision}/{entry['path']}")
            record["commands"].append(f"download {url}")
            part = path.with_suffix(path.suffix + ".part")
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as handle:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    digest.update(chunk)
                    handle.write(chunk)
            require(digest.hexdigest() == entry["sha256"], f"downloaded {name} has a different SHA-256")
            part.rename(path)
            origin = f"downloaded at {revision}"
        size, digest = path.stat().st_size, sha256(path)
        ok = size == entry["size_bytes"] and digest == entry["sha256"]
        r.check(f"{name}: size and SHA-256", ok, f"{origin}; {size} bytes; {digest}")
        record["outputs"][entry["cohort"]] = r.short(str(path))
        r.archives[entry["cohort"]] = path


def step_historical(r: Runner, record):
    archive = r.archives["historical_a100"]
    target = r.work / "historical"
    r.run_python("historical", "path", CODE / "reproduce.py",
                 ["--archive", archive, "--work-dir", target, "--isolate-analysis"], path=[CODE], deny_network=False,
                 display=f"python reproduce.py --archive {archive.name} --work-dir {r.rel(target)} --isolate-analysis")
    report = json.loads((target / "verification.json").read_text())
    r.check("verification.json status", report.get("status") == "passed", report.get("status", "missing"))
    for key in ("derived_csv_json_files_matched", "revised_csv_json_files_matched", "curated_benchmark_files_matched",
                "decision_panel_files_matched", "estimation_panel_files_matched", "host_capacity_panel_files_matched",
                "figures_regenerated", "principal_processes", "manifest_entries_verified"):
        r.check(f"{key}", key in report, str(report.get(key)))
    record["outputs"].update(verification=r.rel(target / "verification.json"), results=r.rel(target / "results"),
                             review_results=r.rel(target / "review-results"), benchmark=r.rel(target / "benchmark"))


def step_offload(r: Runner, record):
    raw, out = r.work / "offload" / "raw", r.work / "offload" / "reconstructed"
    r.extract("offload", r.archives["independent_offload_a100"], raw)
    count, bad, extra = manifest_check(raw, raw / "MANIFEST.sha256")
    r.check("MANIFEST.sha256", not bad and not extra, f"{count} entries; bad={bad[:3]}; unlisted={extra[:3]}")
    procedure = archive_entry("independent_offload_a100")["procedure"]
    r.run_python("offload", "module", "memory_tuner.collect_offload_factorial",
                 ["--root", raw, "--source", raw / "source", "--freeze-sha256", procedure["freeze_sha256"],
                  "--source-commit", procedure["source_commit"], "--accounting", raw / "allocation_accounting.json",
                  "--output", out], path=[CODE])
    reference = CODE / "benchmark/offload_factorial/results"
    for name in ("slots.csv", "configurations.csv", "prediction_scores.csv", "offload_pairs.csv",
                 "block_contrasts.csv"):
        r.check(f"{name} byte-identical to benchmark/offload_factorial/results", same_bytes(out / name, reference / name))
    left, right = json.loads((out / "summary.json").read_text()), json.loads((reference / "summary.json").read_text())
    differing = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
    r.check("summary.json equal except raw_root/source_snapshot", set(differing) <= {"raw_root", "source_snapshot"},
            "differing keys: " + ", ".join(differing))
    record["outputs"]["reconstructed"] = r.rel(out)


A100_PANEL_FILES = {
    "adaptation": ("outcomes.csv", "outcomes.json", "cases.csv", "contrasts.csv", "summary.json",
                   "offload_contrasts.csv"),
    "reservation": ("outcomes.csv", "outcomes.json", "cases.csv", "contrasts.csv", "summary.json",
                    "predictions.csv", "prediction_scores.csv"),
    "release": ("outcomes.csv", "outcomes.json", "cases.csv", "contrasts.csv", "summary.json"),
    "timing": ("outcomes.csv", "outcomes.json", "cases.csv", "contrasts.csv", "summary.json"),
}


def step_a100(r: Runner, record):
    base = r.work / "a100"
    artifact, results = base / "artifact", base / "results"
    figures, tables = base / "figures", base / "generated_tables"
    r.extract("a100", r.archives["electronics_followups_a100"], artifact)
    # 1. The archive's own documented verifier (manifest + fresh reconstruction in a temporary directory).
    report = base / "verification.json"
    r.run("a100", [r.python, "-I", "-B", artifact / "paper/verify_electronics_reconstruction.py", artifact,
                     "--report", report])
    verified = json.loads(report.read_text())
    r.check("archive verifier: manifest files verified", verified.get("manifest_files_verified") == 112689,
            str(verified.get("manifest_files_verified")))
    r.check("archive verifier: matched outputs", len(verified.get("matched_outputs", [])) == 46,
            str(len(verified.get("matched_outputs", []))))
    # 2. The archive's own analysis code (as run by its verifier) into a persistent directory, with the
    #    saved derived results of the archive unreadable; then this repository's paper renderers.
    derived = artifact / "profiles/electronics_revision"
    guard = dict(path=[artifact], deny_read=[derived], chdir=artifact)
    render = dict(path=[CODE], deny_read=[derived], chdir=artifact)
    for panel in A100_PANEL_FILES:
        r.run_python("a100", "module", "memory_tuner.electronics_followup_analysis",
                     ["--root", artifact, "--panel", panel, "--output-root", artifact / "output", "--results", results],
                     **guard)
    r.run_python("a100", "module", "memory_tuner.collect_llmem_native",
                 ["--directory", artifact / "benchmark/electronics_revision_20261016/llmem_native",
                  "--output", artifact / "output/llmem_native_validation_20261016",
                  "--results", results / "llmem_native"], **guard)
    receipt = json.loads((artifact / "benchmark/electronics_revision_20261016/pilot/submission.json").read_text())
    r.run_python("a100", "module", "memory_tuner.collect_electronics",
                 ["--directory", artifact / "benchmark/electronics_revision_20261016/pilot",
                  "--output-root", artifact / "output", "--protocol-sha256", receipt["protocol_sha256"],
                  "--source-commit", receipt["source_commit"], "--results", results / "pilot"], **guard)
    r.run_python("a100", "module", "memory_tuner.electronics_resource_accounting",
                 ["--root", artifact, "--snapshot",
                  artifact / "benchmark/electronics_revision_20261016/resource_accounting/sacct.psv",
                  "--results", results / "resource_accounting", "--outcomes-root", results], **guard)
    r.run_python("a100", "module", "memory_tuner.electronics_review_analysis",
                 ["--root", artifact, "--output", results / "historical", "--figures", figures], **render)
    r.run_python("a100", "module", "memory_tuner.electronics_followup_figures",
                 ["--root", artifact, "--results", results, "--figures", figures], **render)
    r.run_python("a100", "module", "memory_tuner.electronics_revision_tables",
                 ["--root", artifact, "--results", results, "--output", tables], **render)
    record["notes"].append("Analysis modules: the archive's own memory_tuner/ (analysis commit a53af63). "
                           "Renderers (review analysis figures, follow-up figures, SI tables): this repository's "
                           "versions, whose layout matches the manuscript; their analysis outputs are compared too.")
    curated = CODE / "profiles/electronics_revision"
    mismatched = []
    compared = 0
    pairs = [(f"{p}/{n}", results / p / n, curated / p / n) for p, names in A100_PANEL_FILES.items() for n in names]
    pairs += [(f"llmem_native/{n}", results / "llmem_native" / n, curated / "llmem_native" / n)
              for n in ("pairs.csv", "cases.csv", "summary.json")]
    pairs += [(f"pilot/{n}", results / "pilot" / n, curated / "pilot" / n)
              for n in ("outcomes.csv", "outcomes.json", "summary.json")]
    pairs += [(n, results / n, curated / n) for n in ("adaptation_contrast_summary.csv", "timing_condition_summary.csv")]
    pairs += [(f"resource_accounting/{n}", results / "resource_accounting" / n, curated / "resource_accounting" / n)
              for n in ("summary.json", "allocations.csv")]
    pairs += [(f"historical/{p.name}", p, curated / p.name) for p in sorted((results / "historical").iterdir())]
    for name, produced, reference in pairs:
        compared += 1
        if not (produced.is_file() and reference.is_file()) or read_normalized(produced) != read_normalized(reference):
            mismatched.append(name)
    r.check("persistent reconstruction equals profiles/electronics_revision (raw_path relocation only)",
            not mismatched and compared == 42, f"{compared} files compared; mismatched: {mismatched}")
    for name in sorted(EXPECTED["generated_tables"]):
        entry = EXPECTED["generated_tables"][name]
        if entry["step"] != "a100":
            continue
        r.check(f"generated table {name} equals the released generator output",
                (tables / name).is_file() and sha256(tables / name) == entry["sha256"])
    record["outputs"].update(results=r.rel(results), figures=r.rel(figures), generated_tables=r.rel(tables),
                             verifier_report=r.rel(report))


def step_mi250x(r: Runner, record):
    raw, out = r.work / "mi250x" / "raw", r.work / "mi250x" / "reconstructed"
    r.extract("mi250x", r.archives["mi250x"], raw)
    count, bad, _ = manifest_check(raw, raw / "release/MANIFEST.sha256", complete=False)
    r.check("release/MANIFEST.sha256", not bad, f"{count} entries; bad={bad[:3]}")
    script = CODE / "amd_mi250x"
    r.run_python("mi250x", "path", script / "collect.py", ["--root", raw, "--output", out], path=[script])
    for name in ("trials.csv", "configurations.csv", "summary.json"):
        r.check(f"{name} byte-identical to amd_mi250x/results", same_bytes(out / name, script / "results" / name))
    ledger = raw / "release/attempt_ledger.csv"
    r.check("raw attempt ledger equals amd_mi250x/results/attempt_ledger.csv",
            same_bytes(ledger, script / "results/attempt_ledger.csv"))
    shutil.copy2(ledger, out / "attempt_ledger.csv")
    record["notes"].append("make_supplement.py also needs attempt_ledger.csv; it is copied from the raw archive's "
                           "release/attempt_ledger.csv (collect.py does not write it).")
    r.run_python("mi250x", "path", script / "analyze.py", ["--results", out], path=[script])
    reference = script / "results/analysis"
    for name in sorted(p.name for p in reference.iterdir()):
        r.check(f"analysis/{name} byte-identical", same_bytes(out / "analysis" / name, reference / name))
    tables = r.work / "mi250x" / "generated_tables"
    tables.mkdir()
    r.run_python("mi250x", "path", script / "make_supplement.py",
                 ["--results", out, "--output", tables / "supplement_mi250x.tex"], path=[script])
    entry = EXPECTED["generated_tables"]["supplement_mi250x.tex"]
    r.check("supplement_mi250x.tex equals the released generator output",
            sha256(tables / "supplement_mi250x.tex") == entry["sha256"])
    record["outputs"].update(reconstructed=r.rel(out), generated_tables=r.rel(tables))


def step_estimator(r: Runner, record):
    out = r.work / "estimator" / "reconstructed"
    regenerated = r.work / "historical" / "benchmark"
    names = files_under(regenerated)
    differing = [n for n in names if not same_bytes(regenerated / n, CODE / "benchmark" / n)]
    r.check("curated benchmark/ inputs equal the files regenerated by the historical step",
            names and not differing, f"{len(names)} files; differing: {differing}")
    script = CODE / "estimator_comparison"
    r.run_python("estimator", "path", script / "run.py", ["--output", out], path=[CODE])
    reference = script / "results"
    produced, expected = files_under(out), files_under(reference)
    differing = [n for n in expected if not same_bytes(out / n, reference / n)]
    r.check("estimator_comparison/results reproduced byte for byte (incl. figures)",
            produced == expected and not differing, f"{len(expected)} files; differing: {differing[:5]}")
    tables = r.work / "estimator" / "generated_tables"
    tables.mkdir()
    r.run_python("estimator", "path", script / "make_supplement.py",
                 ["--results", out, "--output", tables / "supplement_deepspeed.tex"], path=[CODE])
    entry = EXPECTED["generated_tables"]["supplement_deepspeed.tex"]
    r.check("supplement_deepspeed.tex equals the released generator output",
            sha256(tables / "supplement_deepspeed.tex") == entry["sha256"])
    record["notes"].append("Input: benchmark/ of this repository (checked above against the regenerated files).")
    record["outputs"].update(reconstructed=r.rel(out), generated_tables=r.rel(tables))


def step_revision(r: Runner, record):
    base = r.work / "revision"
    raw, out = base / "raw", base / "results"
    entry = archive_entry("revision_2026_11")
    r.extract("revision", r.archives["revision_2026_11"], raw)
    count, bad, extra = manifest_check(raw, raw / "MANIFEST.sha256")
    r.check("MANIFEST.sha256 covers every member", not bad and not extra and count == entry["manifest_entries"],
            f"{count} entries; bad={bad[:3]}; unlisted={extra[:3]}")
    meta = json.loads((raw / "RELEASE.json").read_text())
    procedure = entry["procedure"]
    curated = REVISION_CODE / "results"
    relocate = relocation(str(raw / "output"), CLUSTER_OUTPUT)
    for study, settings in procedure["collect"].items():
        commit = settings["source"]
        experiment = raw / "source" / commit / "experiments/ieee_resubmission_2026-11"
        protocol = raw / "protocols" / study / f"{study}_protocol.json"
        r.check(f"{study}: protocol SHA-256", sha256(protocol) == settings["protocol_sha256"])
        arguments = ["--study", study, "--protocol", protocol, "--protocol-sha256", settings["protocol_sha256"],
                     "--source-commit", procedure["frozen_source_commit"], "--output-root", raw / "output",
                     "--results", out / study]
        if settings.get("amendment"):
            amendment = raw / settings["amendment"]
            r.check(f"{study}: amendment SHA-256", sha256(amendment) == settings["amendment_sha256"])
            arguments += ["--amendment", amendment, "--amendment-sha256", settings["amendment_sha256"],
                          "--amended-source-commit", settings["source"]]
        r.run_python("revision", "path", experiment / "ieee_resub/collect.py", arguments, path=[])
        names = sorted(p.name for p in (raw / "results" / study).iterdir()
                       if p.name in ("summary.json", "outcomes.json", "outcomes.csv", "attempts.json",
                                     "attempts.csv"))
        for name in names:
            r.check(f"{study}/{name} equals the archived collector output (raw_path relocation only)",
                    same_bytes(out / study / name, raw / "results" / study / name, relocate))
            r.check(f"{study}/{name} equals revision_2026_11/results",
                    same_bytes(out / study / name, curated / study / name, relocate))
    analysis_source = raw / "source" / procedure["analysis_source"] / "experiments/ieee_resubmission_2026-11"
    for study in ("e1", "e2", "e3"):
        r.run_python("revision", "path", analysis_source / "ieee_resub/analysis.py",
                     [study, "--outcomes", out / study / "outcomes.json", "--out", out / study / "analysis.json"],
                     path=[])
        r.check(f"{study}/analysis.json equals revision_2026_11/results",
                same_bytes(out / study / "analysis.json", curated / study / "analysis.json"))
        if study != "e1":
            r.check(f"{study}/analysis.json equals the archived analysis",
                    same_bytes(out / study / "analysis.json", raw / "results" / study / "analysis"))
    record["notes"].append("results/e1/analysis in the archive was written by the 9f6faf9 analysis.py before "
                           "amendment A7 defect 3; the reported E1 analysis is the cc685c0 re-run above.")
    # Reproduction checks of earlier panels (PLAN.md section 7) on tables regenerated by the earlier steps.
    root = base / "check-existing-root"
    shutil.copytree(raw / "source" / procedure["analysis_source"], root)
    inputs = {
        "benchmark/replication_control/results/replication_control_trials.csv":
            r.work / "historical/replication_control-results/replication_control_trials.csv",
        "profiles/review_revision/control-trials.csv": r.work / "historical/review-results/control-trials.csv",
        "profiles/standard_grpo/revision-temporal-trials.csv": r.work / "historical/results/revision-temporal-trials.csv",
        "profiles/electronics_revision/release/outcomes.csv": r.work / "a100/results/release/outcomes.csv",
    }
    for relative, source in inputs.items():
        require(source.is_file(), f"missing regenerated input {source}")
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, root / relative)
    reference = raw / "source" / procedure["analysis_source"] / "profiles/electronics_revision/stage_paired_contrasts.csv"
    r.check("E1 natural-length reference table equals the regenerated A100 historical table",
            same_bytes(reference, r.work / "a100/results/historical/stage_paired_contrasts.csv"))
    report = base / "check-existing.json"
    r.run_python("revision", "path", root / "experiments/ieee_resubmission_2026-11/ieee_resub/analysis.py",
                 ["check-existing", "--out", report], path=[])
    result = json.loads(report.read_text())
    r.check("analysis.py check-existing: all earlier reported values reproduced", result.get("all_ok") is True,
            f"{sum(c['ok'] for c in result['checks'].values())}/{len(result['checks'])} checks")
    record["outputs"].update(results=r.rel(out), check_existing=r.rel(report), release_json=meta.get("archive"))


CPU_GROUPS = (
    ("c1_realized_work", ("c1_realized_work.py",)),
    ("c2_predictors", ("a_reproduce.py", "b_hurdle.py", "c_coverage.py", "d_margins.py")),
    ("c4_sensitivity", ("c4_eligibility.py", "c4_predictors.py", "c5_uncertainty.py", "c6_gpu_time.py")),
)


def build_tables_root(r: Runner, root: Path):
    """Curated tables exactly as released, with the code files that frozen protocols hash."""
    root.mkdir(parents=True)
    shutil.copy2(CODE / "benchmark.py", root / "benchmark.py")
    shutil.copytree(CODE / "memory_tuner", root / "memory_tuner",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(CODE / "benchmark", root / "benchmark")
    (root / "profiles").mkdir()
    shutil.copytree(CODE / "profiles/electronics_revision", root / "profiles/electronics_revision")
    for name in ("standard_grpo", "review_revision"):
        shutil.copytree(r.work / "historical/reference-profiles" / name, root / "profiles" / name)


def run_cpu_groups(r: Runner, step, python, tables, out_root, log_file=None):
    for group, scripts in CPU_GROUPS:
        out = out_root / group
        out.mkdir(parents=True)
        env = dict(RLVRAMBENCH_TABLES_ROOT=str(tables), RLVRAMBENCH_ANALYSIS_OUTPUT=str(out),
                   RLVRAMBENCH_CODE_ROOT=str(CODE))
        for script in scripts:
            arguments = []
            if group == "c1_realized_work":
                arguments = ["--repo", tables, "--extract", r.work / "historical/artifact",
                             "--archive", r.archives["historical_a100"], "--out", out]
            r.run_python(step, "path", REVISION_CODE / "cpu_reanalyses" / group / script, arguments, python=python,
                         path=[REVISION_CODE / "cpu_reanalyses" / group], chdir=out, env=env,
                         log_reads_under=tables if log_file else None, log_file=log_file)


PATH_TOKEN = re.compile(r"`/[^`]*`")


def comparable(group, name, data: bytes) -> bytes:
    """Run-specific locations are the only permitted differences (C1 report and key results)."""
    if group == "c1_realized_work" and name == "key_results.json":
        value = json.loads(data)
        value["archive"]["path"] = "<archive>"
        return json.dumps(value, sort_keys=True).encode()
    if group == "c1_realized_work" and name == "REPORT.md":
        return PATH_TOKEN.sub("`<path>`", data.decode()).encode()
    return data


def compare_cpu(out_root: Path):
    differing, missing, compared = [], [], 0
    for group, _ in CPU_GROUPS:
        reference = REVISION_CODE / "cpu_reanalyses" / group / "reference"
        for name in sorted(p.name for p in reference.iterdir()):
            compared += 1
            produced = out_root / group / name
            if not produced.is_file():
                missing.append(f"{group}/{name}")
            elif comparable(group, name, produced.read_bytes()) != comparable(group, name,
                                                                                (reference / name).read_bytes()):
                differing.append(f"{group}/{name}")
    return compared, differing, missing


REGENERATED = (  # tables-root prefix -> directory regenerated by an earlier step
    ("profiles/standard_grpo/", "historical/results/", "historical (reproduce.py, 40 files matched)"),
    ("profiles/review_revision/", "historical/review-results/", "historical (reproduce.py, 27 files matched)"),
    ("benchmark/decision/results/", "historical/decision-results/", "historical"),
    ("benchmark/estimation/results/", "historical/estimation-results/", "historical"),
    ("benchmark/host_capacity/results/", "historical/host-capacity-results/", "historical"),
    ("benchmark/completion_replication/results/", "historical/completion_replication-results/", "historical"),
    ("benchmark/replication_control/results/", "historical/replication_control-results/", "historical"),
    ("benchmark/offload_factorial/results/", "offload/reconstructed/", "offload"),
    ("profiles/electronics_revision/", "a100/results/", "a100"),
)


def classify_inputs(r: Runner, tables: Path, names):
    rows = []
    for name in sorted(set(names)):
        row = dict(input=name, regenerated_by="", regenerated_file="", byte_identical="")
        candidates = []
        for prefix, target, step in REGENERATED:
            if name.startswith(prefix):
                candidates.append((r.work / (target + name[len(prefix):]), step))
        if name.startswith("profiles/electronics_revision/"):
            candidates.append((r.work / "a100/results/historical" / Path(name).name, "a100"))
        if name.startswith("benchmark/") and name.count("/") == 1:
            candidates.append((r.work / "historical/benchmark" / Path(name).name, "historical (curated benchmark)"))
        for path, step in candidates:
            if path.is_file():
                row.update(regenerated_by=step, regenerated_file=r.rel(path),
                           byte_identical=str(same_bytes(path, tables / name)))
                break
        if not row["regenerated_by"]:
            row["regenerated_by"] = ("code file" if name.endswith(".py") else
                                     "frozen or curated input (not an analysis output)")
        rows.append(row)
    return rows


def step_cpu(r: Runner, record):
    base = r.work / "cpu_reanalyses"
    tables = base / "tables-root"
    build_tables_root(r, tables)
    log = base / "inputs-read.txt"
    run_cpu_groups(r, "cpu_reanalyses", r.reanalysis_python, tables, base / "out", log_file=log)
    compared, differing, missing = compare_cpu(base / "out")
    r.check("all re-analysis outputs equal revision_2026_11/cpu_reanalyses/*/reference",
            not differing and not missing, f"{compared} files; differing={differing}; missing={missing}")
    rows = classify_inputs(r, tables, log.read_text().splitlines())
    with (base / "inputs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    regenerated = [x for x in rows if x["regenerated_file"]]
    identical = [x for x in regenerated if x["byte_identical"] == "True"]
    record["notes"].append(
        f"{len(rows)} input files read; {len(regenerated)} have a counterpart regenerated from raw evidence in this "
        f"run ({len(identical)} byte-identical; the rest were matched by reproduce.py after its documented "
        f"artifact-path normalization); the others are frozen protocols, matrices, predictions, metadata, or code.")
    version = subprocess.run([str(r.reanalysis_python), "-I", "-c",
                              "import sys,numpy,scipy,sklearn;print(sys.version.split()[0],numpy.__version__,"
                              "scipy.__version__,sklearn.__version__)"], capture_output=True, text=True).stdout.strip()
    record["notes"].append(f"Re-analysis environment: Python/numpy/scipy/scikit-learn = {version}")
    record["outputs"].update(outputs=r.rel(base / "out"), inputs=r.rel(base / "inputs.csv"))


def step_cpu_publication_env(r: Runner, record):
    """Informational: the same re-analyses in the historical publication environment (CPython 3.11)."""
    base = r.work / "cpu_reanalyses_publication_env"
    run_cpu_groups(r, "cpu_reanalyses_publication_env", r.python, r.work / "cpu_reanalyses/tables-root",
                   base / "out")
    compared, differing, missing = compare_cpu(base / "out")
    r.check("outputs under publication-requirements.txt equal the reference", not differing and not missing,
            f"{compared} files; differing={differing}; missing={missing}", informational=True)
    record["notes"].append("Informational only. Differences come from floating-point summation and the "
                           "logistic-regression solver of older numpy/scipy/scikit-learn releases.")
    record["outputs"]["outputs"] = r.rel(base / "out")


def step_figures(r: Runner, record):
    base = r.work / "paper"
    historical = base / "renderer-historical"
    r.run_python("figures", "module", "memory_tuner.electronics_historical_figures",
                 ["--root", CODE, "--data", r.work / "historical/results",
                  "--review-results", r.work / "historical/review-results", "--output", historical], path=[CODE])
    sources = {"historical": historical, "a100": r.work / "a100/figures"}
    figures, tables = base / "figures", base / "generated_tables"
    figures.mkdir(parents=True)
    tables.mkdir()
    for paper_name, entry in sorted(EXPECTED["figures"].items()):
        produced = sources[entry["renderer"]] / entry["renderer_output"]
        target = figures / paper_name
        if produced.is_file():
            shutil.copy2(produced, target)
            png = produced.with_suffix(".png")
            if png.is_file():
                shutil.copy2(png, target.with_suffix(".png"))
        r.check(f"{entry['paper']} ({paper_name}) byte-identical to the manuscript figure",
                target.is_file() and sha256(target) == entry["sha256"], entry["renderer_output"])
    for name, entry in sorted(EXPECTED["generated_tables"].items()):
        source = r.work / entry["step"] / "generated_tables" / name
        if source.is_file():
            shutil.copy2(source, tables / name)
    record["outputs"].update(figures=r.rel(figures), generated_tables=r.rel(tables))


def step_tests(r: Runner, record):
    junit = r.work / "tests" / "junit.xml"
    junit.parent.mkdir(parents=True)
    r.run_python("tests", "module", "pytest", ["-q", "-p", "no:cacheprovider", f"--junitxml={junit}", "tests"],
                 path=[CODE], chdir=CODE, deny_network=False)
    suite = re.search(r"<testsuite [^>]*>", junit.read_text())
    counts = dict(re.findall(r'(tests|failures|errors|skipped)="(\d+)"', suite.group(0))) if suite else {}
    record["notes"].append("pytest: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    r.check("pytest: no failures or errors", counts.get("failures") == "0" and counts.get("errors") == "0"
            and int(counts.get("tests", 0)) > 0, ", ".join(f"{k}={v}" for k, v in counts.items()))
    record["outputs"]["junit"] = r.rel(junit)


# --------------------------------------------------------------------------- report
def environment_record(r: Runner):
    def version(python):
        return subprocess.run([str(python), "-I", "-c", "import sys; print(sys.version.split()[0])"],
                              capture_output=True, text=True).stdout.strip()

    def freeze(python):
        return subprocess.run([str(python), "-m", "pip", "freeze", "--all"], capture_output=True,
                              text=True).stdout.split()

    tools = {}
    for tool in ("tar", "zstd", "bwrap"):
        location = shutil.which(tool)
        output = subprocess.run([tool, "--version"], capture_output=True, text=True) if location else None
        tools[tool] = (output.stdout or output.stderr).splitlines()[0] if output else "missing"
    git = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(CODE), "status", "--porcelain"], capture_output=True, text=True)
    return dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
                code_commit=git.stdout.strip() or "not a git checkout",
                code_tree_clean=(dirty.returncode == 0 and not dirty.stdout.strip()) if git.returncode == 0 else None,
                publication_python=str(r.python), publication_python_version=version(r.python),
                publication_packages=freeze(r.python), reanalysis_python=str(r.reanalysis_python),
                reanalysis_python_version=version(r.reanalysis_python), reanalysis_packages=freeze(r.reanalysis_python),
                tools=tools, thread_environment=THREADS)


def write_report(r: Runner, started, environment):
    status = {s["step"]: s["status"] for s in r.steps}
    total = round(time.monotonic() - started, 1)
    archives = [dict(file=Path(e["path"]).name, cohort=e["cohort"], size_bytes=e["size_bytes"], sha256=e["sha256"],
                     used=str(r.archives.get(e["cohort"], ""))) for e in RELEASE["release_archives"]]
    items = []
    for item in PAPER_MAP["items"]:
        steps = item.get("steps", [])
        state = ("descriptive" if not steps else
                 "passed" if all(status.get(s) == "passed" for s in steps) else "not reproduced")
        items.append(dict(item, status=state))
    passed = all(s["status"] == "passed" for s in r.steps if not s["informational"])
    report = dict(release=RELEASE["release"], release_tag=RELEASE["release_tag"],
                  dataset_repository=RELEASE["huggingface_repository"], dataset_revision_requested=r.args.revision,
                  dataset_revision_in_release_json=RELEASE["huggingface_revision"],
                  doi_in_release_json=RELEASE["doi"], code_repository=RELEASE["code_repository"],
                  started_utc=r.steps[0]["started_utc"] if r.steps else None, finished_utc=now(),
                  total_seconds=total, overall_status="passed" if passed else "failed", archives=archives,
                  environment=environment, steps=r.steps, paper_items=items,
                  numbering_source=PAPER_MAP["numbering_source"])
    (r.work / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [f"# RLVRAMBench reproduction report ({RELEASE['release_tag']})", "",
             f"- Overall status: **{report['overall_status']}**; total wall time {total / 60:.1f} min",
             f"- Code: `{environment['code_commit']}` (clean tree: {environment['code_tree_clean']})",
             f"- Work directory `$WORK` = `{r.work}`; code `$CODE` = `{CODE}`"
             + (f"; archives `$ARCHIVES` = `{r.args.archives_dir.resolve()}`" if r.args.archives_dir else ""),
             f"- Dataset: `{RELEASE['huggingface_repository']}`; archives "
             + (f"downloaded at `{r.args.revision}`" if r.args.revision else "from a local directory")
             + (f" and/or `{r.args.archives_dir}`" if r.args.archives_dir and r.args.revision else ""),
             f"- Publication environment: CPython {environment['publication_python_version']} "
             f"(`publication-requirements.txt`); re-analysis environment: CPython "
             f"{environment['reanalysis_python_version']} (`revision_2026_11/cpu_reanalyses/requirements.txt`)",
             f"- Started {report['started_utc']}, finished {report['finished_utc']}", "",
             "## Archives", "", "| File | Bytes | SHA-256 |", "|---|---:|---|"]
    lines += [f"| `{a['file']}` | {a['size_bytes']:,} | `{a['sha256']}` |" for a in archives]
    lines += ["", "## Steps", "", "| Step | Status | Seconds | Checks passed |", "|---|---|---:|---:|"]
    for s in r.steps:
        checks = [c for c in s["checks"] if not c.get("informational")]
        lines.append(f"| {s['step']}: {s['title']} | {s['status']} | {s.get('seconds', '')} | "
                     f"{sum(c['ok'] for c in checks)}/{len(checks)} |")
    for s in r.steps:
        lines += ["", f"### {s['step']} ({s['status']})", ""]
        lines += [f"    {c}" for c in s["commands"]]
        failed = [c for c in s["checks"] if not c["ok"]]
        if failed:
            lines += [""] + [f"- {'NOTE' if c.get('informational') else 'FAILED'}: {c['check']} {c['detail']}"
                             for c in failed]
        lines += [""] + [f"- {n}" for n in s["notes"]] if s["notes"] else []
        if s.get("error"):
            lines += ["", "```", s["error"][-2500:], "```"]
        if s["outputs"]:
            lines += [""] + [f"- output `{k}`: `{v}`" for k, v in s["outputs"].items()]
    lines += ["", "## Paper tables and figures", "",
              f"Numbering: {PAPER_MAP['numbering_source']}. Paths are relative to the work directory.", "",
              "| Item | Label | Archive(s) | Step / command | Output(s) | Status |", "|---|---|---|---|---|---|"]
    for item in items:
        lines.append(f"| {item['item']} | `{item['label']}` | {', '.join(item.get('archives', [])) or '-'} | "
                     f"{item.get('command', '-')} | {'<br>'.join('`' + o + '`' for o in item.get('outputs', [])) or '-'}"
                     f" | {item['status']} |")
    lines += ["", "Descriptive items have no data output. This is an author-performed reconstruction of recorded "
              "measurements, not an independent replication or a rerun of GPU training."]
    (r.work / "REPORT.md").write_text("\n".join(lines) + "\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True, help="new, nonexistent output directory")
    parser.add_argument("--revision", help="Hugging Face dataset commit (40 hex) for archives not in --archives-dir")
    parser.add_argument("--archives-dir", type=Path, help="directory with already downloaded archives")
    parser.add_argument("--reanalysis-python", type=Path, required=True,
                        help="interpreter of the re-analysis environment (CPython 3.12)")
    parser.add_argument("--skip-publication-env-comparison", action="store_true",
                        help="skip the informational rerun of the re-analyses in the 3.11 environment")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Run reproduce_all.py with the CPython 3.11 publication environment (reproduce_all.sh)")
    for tool in ("tar", "zstd", "unzstd", "bwrap"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Required tool not found on PATH: {tool}")
    work = args.work_dir.resolve()
    if work.exists():
        raise SystemExit(f"Work directory exists; choose a new one: {work}")
    if work.is_relative_to(CODE / "benchmark"):
        raise SystemExit("The work directory must not be inside the code repository's benchmark/")
    runner = Runner(args)
    for directory in (runner.work, runner.logs, runner.tmp):
        directory.mkdir(parents=True, exist_ok=directory != runner.work)
    started = time.monotonic()
    environment = environment_record(runner)
    runner.step("archives", "obtain and verify the five release archives", lambda rec: step_archives(runner, rec))
    runner.step("historical", "historical A100 archive: reproduce.py (isolated)",
                lambda rec: step_historical(runner, rec), needs=("archives",))
    runner.step("offload", "independent-offload archive: collect_offload_factorial",
                lambda rec: step_offload(runner, rec), needs=("archives",))
    runner.step("a100", "A100 follow-up archive: verifier, reconstruction, renderers",
                lambda rec: step_a100(runner, rec), needs=("archives",))
    runner.step("mi250x", "MI250X archive: collect, analyze, supplement", lambda rec: step_mi250x(runner, rec),
                needs=("archives",))
    runner.step("estimator", "estimator comparison (DeepSpeed replay, analytic budget)",
                lambda rec: step_estimator(runner, rec), needs=("historical",))
    runner.step("revision", "revision archive (2026-11): E1-E3 collectors, analyses, earlier-value checks",
                lambda rec: step_revision(runner, rec), needs=("archives", "historical", "a100"))
    runner.step("cpu_reanalyses", "retrospective CPU re-analyses C1, C2/C3/C8, C4-C6",
                lambda rec: step_cpu(runner, rec), needs=("historical", "offload", "a100"))
    if not args.skip_publication_env_comparison:
        runner.step("cpu_reanalyses_publication_env", "informational: re-analyses under publication-requirements",
                    lambda rec: step_cpu_publication_env(runner, rec), needs=("cpu_reanalyses",),
                    informational=True)
    runner.step("figures", "render the paper figures under their manuscript file names",
                lambda rec: step_figures(runner, rec), needs=("historical", "a100"))
    runner.step("tests", "pytest -q tests (CPU numbers checks)", lambda rec: step_tests(runner, rec))
    passed = write_report(runner, started, environment)
    print(f"Report: {runner.work / 'REPORT.md'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

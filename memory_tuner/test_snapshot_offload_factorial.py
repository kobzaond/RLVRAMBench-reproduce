"""Real frozen-source CPU replay tests; every mutation is confined to /tmp copies.

Run with the existing project scientific Python. Integration tests are skipped
only when the explicitly named execution source is unavailable; path/JSON tests
are independent of it. No GPU, scheduler, model, data, or new outcome is used.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from memory_tuner import snapshot_offload_factorial as snapshot


REAL_SOURCE = Path("/mnt/proj3/open-35-44/verl/experiments/offload-factorial-3d6ecac")
PYTHON = Path("/mnt/proj3/open-35-44/verl/rl/.venv-reproduce/bin/python")
FREEZE = "5df43f35e0b02c23e4816b7e860c5a613176da41f5adb27e57f79ac2becf1fb9"
COMMIT = "3d6ecac1852e16a0422c9c72a230187be663f8b5"


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(payload, bytes):
        payload = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)


class SnapshotPathTests(unittest.TestCase):
    def test_unsafe_names_and_non_source_payloads_are_rejected(self):
        for name in ("../outside.py", "/absolute.py", "a/../outside.py", "a//b.py",
                     "a\\b.py", "./a.py", ".git/config", ".venv/lib/a.py",
                     "a/__pycache__/b.pyc", "a/model.safetensors", "a/data.parquet",
                     "a/container.sif", "a/model.bin", "a/extension.so", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                snapshot.relative_name(name)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        for payload in (b'{"same": 1, "same": 2}', b'{"value": NaN}', b'{"value": Infinity}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                snapshot.parse_json(payload)

    def test_conflicting_or_invalid_hash_inventory_is_rejected(self):
        for changes in ({"a.py": "b" * 64}, {"b.py": "invalid"}, {"../outside.py": "a" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                snapshot.merge_hashes({"a.py": "a" * 64}, changes)

    def test_write_new_never_overwrites_existing_content(self):
        with tempfile.TemporaryDirectory(prefix="offload-snapshot-paths-", dir="/tmp") as directory:
            path = Path(directory) / "existing.txt"
            snapshot.write_new(path, b"first")
            with self.assertRaises(FileExistsError):
                snapshot.write_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_special_or_linked_text_files_are_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory(prefix="offload-snapshot-paths-", dir="/tmp") as directory:
            root = Path(directory)
            put(root / "original.py", b"# source\n")
            (root / "symlink.py").symlink_to(root / "original.py")
            os.link(root / "original.py", root / "hardlink.py")
            (root / "directory.py").mkdir()
            os.mkfifo(root / "fifo.py")
            for name in ("symlink.py", "hardlink.py", "directory.py", "fifo.py"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    snapshot.read_bytes(root, name)

    def test_binary_non_utf8_and_oversized_source_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="offload-snapshot-paths-", dir="/tmp") as directory:
            root = Path(directory)
            for payload in (b"\0binary", b"\xffnon-utf8"):
                put(root / "source.py", payload)
                with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "payload"):
                    snapshot.read_bytes(root, "source.py")
            put(root / "source.py", b"ninebytes")
            with patch.object(snapshot, "MAX_TEXT_BYTES", 8), self.assertRaisesRegex(ValueError, "Oversized"):
                snapshot.read_bytes(root, "source.py")


class RealOffloadSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REAL_SOURCE.is_dir() or not PYTHON.is_file():
            raise unittest.SkipTest("Explicit frozen source/project scientific Python unavailable")
        cls.scratch = tempfile.TemporaryDirectory(prefix="offload-snapshot-real-", dir="/tmp")
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.root = Path(cls.scratch.name)
        # Read the immutable real source; copy only the identified closure. All
        # following exports and adversarial mutations use this disposable copy.
        cls.plan = snapshot.source_plan(REAL_SOURCE, FREEZE, COMMIT)
        cls.source = cls.root / "real source copy"
        for name in cls.plan["files"]:
            target = cls.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REAL_SOURCE / name, target)
        for name, payload in {
            ".git/config": b"Not a usable Git checkout\n",
            "model.safetensors": b"\0not an exported model\n",
            "data/train.parquet": b"\0not exported workload data\n",
            "output/offload_factorial/never-read.txt": b"No target outcomes in this fixture\n",
            "memory_tuner/snapshot_offload_factorial.py": b"# unsealed, unused live helper\n",
        }.items():
            put(cls.source / name, payload)
        cls.output = cls.root / "public/reproduction_sources/offload_factorial"
        cls.report = snapshot.export_sources(cls.source, FREEZE, COMMIT, cls.output, PYTHON)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="offload-snapshot-case-", dir=self.root)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def copied_source(self):
        return Path(shutil.copytree(self.source, self.directory / "copied source"))

    def copied_export(self):
        return Path(shutil.copytree(self.output, self.directory / "relocated snapshot"))

    def test_exact_execution_closure_and_transitive_inputs_replay_from_real_source_copy(self):
        report = self.report
        self.assertEqual(report["source_commit"], COMMIT)
        self.assertEqual(report["freeze_sha256"], FREEZE)
        self.assertEqual(len(report["runtime_sha256"]), 584)
        self.assertEqual(len(report["input_sha256"]), 18)
        self.assertEqual(report["transitive_loader_input_sha256"], report["input_sha256"])
        self.assertEqual(report["source_file_count"], 605)
        self.assertEqual(report["source_total_bytes"], 9022588)
        checked = report["cpu_validation"]
        self.assertEqual(checked["status"], "passed")
        self.assertEqual(checked["planned_slots"], 144)
        self.assertEqual(checked["planned_blocks"], 18)
        self.assertEqual(checked["target_configurations"], 48)
        self.assertEqual(checked["prediction_rows"], 288)
        self.assertEqual(checked["prediction_replay"], "exact")
        self.assertFalse(checked["verify_assets"])
        self.assertFalse(checked["outcome_reconstruction_performed"])
        for name, record in report["files"].items():
            payload = (self.output / "source" / name).read_bytes()
            self.assertEqual(payload, (self.source / name).read_bytes())
            self.assertEqual(snapshot.sha256(payload), record["sha256"])
        for name in (".git", "data", "output", "model.safetensors", "memory_tuner/snapshot_offload_factorial.py"):
            self.assertFalse((self.output / "source" / name).exists())
        self.assertTrue((self.output / snapshot.HELPER_NAME).is_file())
        self.assertEqual(len([p for p in self.output.rglob("*") if p.is_file()]), 607)
        self.assertFalse(list(self.output.rglob("__pycache__")))
        stored = json.loads((self.output / snapshot.REPORT_NAME).read_text())
        self.assertEqual(stored, {k: v for k, v in report.items() if k != "report_sha256"})
        self.assertEqual(snapshot.sha256((self.output / snapshot.REPORT_NAME).read_bytes()), report["report_sha256"])

    def test_subprocess_pythonpath_is_snapshot_only_and_thread_settings_are_one(self):
        calls = []
        original_run = snapshot.subprocess.run

        def inspect(command, **kwargs):
            self.assertEqual(command[0], str(PYTHON))
            self.assertEqual(command[1:5], ["-B", "-s", "-P", "-c"])
            env = kwargs["env"]
            self.assertEqual(env["PYTHONPATH"], str(self.output / "source"))
            self.assertEqual(kwargs["cwd"], self.output / "source")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                self.assertEqual(env[name], "1")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertNotIn("LD_PRELOAD", env)
            self.assertNotIn("PYTHONHOME", env)
            calls.append(command)
            return original_run(command, **kwargs)

        with patch.dict(os.environ, {"PYTHONPATH": str(self.source), "OMP_NUM_THREADS": "19",
                                     "OPENBLAS_NUM_THREADS": "23", "MKL_NUM_THREADS": "29",
                                     "LD_PRELOAD": "/not/a/real/preload.so"}), \
                patch.object(snapshot.subprocess, "run", inspect):
            result = snapshot.verify_export(self.output, FREEZE, COMMIT, PYTHON,
                                            report_sha256=self.report["report_sha256"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["cpu_validation"]["status"], "passed")

    def test_standalone_verifier_runs_after_relocation_without_git_or_live_pythonpath(self):
        moved = self.copied_export()
        neutral = self.directory / "empty working directory"
        neutral.mkdir()
        command = [
            str(PYTHON), "-B", "-s", "-P", str(moved / snapshot.HELPER_NAME), "verify",
            "--export-root", str(moved), "--freeze-sha256", FREEZE, "--source-commit", COMMIT,
            "--report-sha256", self.report["report_sha256"], "--python", str(PYTHON),
        ]
        env = {key: value for key, value in os.environ.items()
               if key in ("LANG", "LC_ALL", "LD_LIBRARY_PATH")}
        env.update(PATH="", PYTHONPATH="/nonexistent/live/project", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(command, cwd=neutral, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        checked = json.loads(result.stdout)
        self.assertEqual(checked["cpu_validation"]["status"], "passed")
        self.assertEqual(checked["source_file_count"], 605)
        self.assertFalse(list(moved.rglob("__pycache__")))
        self.assertEqual(list(neutral.iterdir()), [])
        self.assertFalse(any(str(self.source) in str(value) for value in
                             checked["cpu_validation"]["source_modules"].values()))

    def test_existing_destination_or_execution_source_is_never_modified(self):
        before = (self.output / snapshot.REPORT_NAME).read_bytes()
        with self.assertRaisesRegex(ValueError, "Preserving existing"):
            snapshot.export_sources(self.source, FREEZE, COMMIT, self.output, PYTHON)
        with self.assertRaisesRegex(ValueError, "outside execution source"):
            snapshot.export_sources(self.source, FREEZE, COMMIT, self.source / "new-export", PYTHON)
        self.assertEqual((self.output / snapshot.REPORT_NAME).read_bytes(), before)
        self.assertFalse((self.source / "new-export").exists())

    def test_wrong_external_freeze_or_commit_rejected_before_export(self):
        output = self.directory / "new"
        for freeze, commit in (("0" * 64, COMMIT), (FREEZE, "0" * 40)):
            with self.subTest(freeze=freeze, commit=commit), self.assertRaises(ValueError):
                snapshot.export_sources(self.source, freeze, commit, output, PYTHON)
            self.assertFalse(output.exists())

    def test_changed_runtime_and_transitive_metadata_rejected_before_export(self):
        names = (
            "memory_tuner/offload_factorial_protocol.py", "memory_tuner/completion_replication_protocol.py",
            "benchmark/configurations.csv", "benchmark/outcomes.csv",
            "benchmark/estimation/prediction_freeze.json", "benchmark/estimation/fitted_model.json",
            "benchmark/completion_replication/freeze.json", "benchmark/completion_replication/protocol.json",
            "benchmark/offload_factorial/matrix.csv", "benchmark/offload_factorial/predictions.json",
            "benchmark/offload_factorial/freeze.json",
        )
        source = self.copied_source()
        for name in names:
            path = source / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash differs"):
                    snapshot.export_sources(source, FREEZE, COMMIT, self.directory / "new", PYTHON)
                self.assertFalse((self.directory / "new").exists())
            finally:
                path.write_bytes(original)

    def test_missing_required_source_input_or_notice_cannot_use_live_fallback(self):
        source = self.copied_source()
        for name in ("memory_tuner/resolve_hf_snapshot.py", "benchmark/estimation/amendment-01.json",
                     "benchmark/estimation/targets.csv", "third_party/TransferQueue/LICENSE"):
            path = source / name
            parked = self.directory / "parked-file"
            path.rename(parked)
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Cannot read required"):
                    snapshot.export_sources(source, FREEZE, COMMIT, self.directory / "new", PYTHON)
                self.assertFalse((self.directory / "new").exists())
            finally:
                parked.rename(path)

    def test_linked_source_parent_and_linked_output_parent_are_rejected(self):
        source = self.copied_source()
        (source / "memory_tuner").rename(source / "moved-memory-tuner")
        (source / "memory_tuner").symlink_to(source / "moved-memory-tuner", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Linked"):
            snapshot.export_sources(source, FREEZE, COMMIT, self.directory / "new", PYTHON)
        link = self.directory / "linked-parent"
        link.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Linked"):
            snapshot.export_sources(self.source, FREEZE, COMMIT, link / "new", PYTHON)
        self.assertFalse((self.directory / "new").exists())

    def test_linked_snapshot_file_or_parent_is_rejected(self):
        moved = self.copied_export()
        name = "memory_tuner/offload_factorial_protocol.py"
        path = moved / "source" / name
        path.unlink()
        path.symlink_to(self.source / name)
        with self.assertRaisesRegex(ValueError, "Linked"):
            snapshot.verify_export(moved, FREEZE, COMMIT, PYTHON)

    def test_receipt_hash_binds_code_notices_and_full_file_inventory(self):
        moved = self.copied_export()
        path = moved / snapshot.REPORT_NAME
        report = json.loads(path.read_text())
        report["source_commit"] = "0" * 40
        put(path, report)
        with self.assertRaisesRegex(ValueError, "hash differs"):
            snapshot.verify_export(moved, FREEZE, COMMIT, PYTHON, report_sha256=self.report["report_sha256"])
        with self.assertRaisesRegex(ValueError, "Receipt differs"):
            snapshot.verify_export(moved, FREEZE, COMMIT, PYTHON)

    def test_changed_exported_runtime_or_verifier_or_notice_is_rejected(self):
        moved = self.copied_export()
        for name in ("source/memory_tuner/offload_factorial_protocol.py", snapshot.HELPER_NAME,
                     "source/third_party/TransferQueue/LICENSE"):
            path = moved / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            try:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    snapshot.verify_export(moved, FREEZE, COMMIT, PYTHON)
            finally:
                path.write_bytes(original)

    def test_extra_files_or_empty_directories_are_rejected(self):
        moved = self.copied_export()
        extra = moved / "unexpected"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            snapshot.verify_export(moved, FREEZE, COMMIT, PYTHON)
        extra.rmdir()
        put(moved / "source/.git/config", b"Not allowed in the portable snapshot\n")
        with self.assertRaisesRegex(ValueError, "Unexpected snapshot file"):
            snapshot.verify_export(moved, FREEZE, COMMIT, PYTHON)

    def test_copied_byte_corruption_leaves_no_completed_receipt(self):
        output = self.directory / "new"
        original_write = snapshot.write_new

        def corrupt(path, payload):
            original_write(path, payload + b"\n" if path.name == "device_contract.py" else payload)

        with patch.object(snapshot, "write_new", corrupt), self.assertRaisesRegex(ValueError, "hash differs"):
            snapshot.export_sources(self.source, FREEZE, COMMIT, output, PYTHON)
        self.assertTrue(output.is_dir())
        self.assertFalse((output / snapshot.REPORT_NAME).exists())

    def test_receipt_byte_corruption_cannot_return_a_successful_export(self):
        output = self.directory / "new"
        original_write = snapshot.write_new

        def corrupt(path, payload):
            original_write(path, payload + b"\n" if path.name == snapshot.REPORT_NAME else payload)

        with patch.object(snapshot, "write_new", corrupt), self.assertRaisesRegex(ValueError, "hash differs"):
            snapshot.export_sources(self.source, FREEZE, COMMIT, output, PYTHON)

    def test_cpu_dependency_failure_is_actionable_and_leaves_no_success_receipt(self):
        output = self.directory / "new"
        failure = subprocess.CompletedProcess([], 1, "", "ModuleNotFoundError: No module named 'numpy'")
        with patch.object(snapshot.subprocess, "run", return_value=failure), \
                self.assertRaisesRegex(ValueError, "existing project scientific Python"):
            snapshot.export_sources(self.source, FREEZE, COMMIT, output, PYTHON)
        self.assertFalse((output / snapshot.REPORT_NAME).exists())
        self.assertFalse(list(output.rglob(".venv")))

    def test_cpu_timeout_leaves_no_success_receipt(self):
        output = self.directory / "new"
        with patch.object(snapshot.subprocess, "run", side_effect=subprocess.TimeoutExpired("python", 120)), \
                self.assertRaisesRegex(ValueError, "timed out"):
            snapshot.export_sources(self.source, FREEZE, COMMIT, output, PYTHON)
        self.assertFalse((output / snapshot.REPORT_NAME).exists())

    def test_cpu_guard_rejects_live_reads_writes_and_git_processes(self):
        moved = self.copied_export()
        outside = self.directory / "outside-model.txt"
        outside.write_text("Must not enter replay\n")
        probes = (
            f"Path({str(outside)!r}).read_text()",
            "(source / 'forbidden-write.txt').write_text('not allowed')",
            "import subprocess; subprocess.run(['git', '--version'], check=True)",
        )
        for probe in probes:
            code = snapshot.VALIDATE_CODE.replace(
                "frozen = design.load(source, freeze, verify_assets=False)",
                probe + "\nfrozen = design.load(source, freeze, verify_assets=False)")
            with self.subTest(probe=probe), patch.object(snapshot, "VALIDATE_CODE", code), \
                    self.assertRaisesRegex(ValueError, "forbidden|read-only|forbids"):
                snapshot.validate_cpu(moved, FREEZE, COMMIT, PYTHON)
        self.assertFalse((moved / "source/forbidden-write.txt").exists())

    def test_preview_and_cli_help_are_read_only(self):
        stream = io.StringIO()
        with patch.object(snapshot, "validate_cpu", side_effect=AssertionError("Preview must not execute loader")), \
                redirect_stdout(stream):
            snapshot.main(["preview", "--source-root", str(self.source),
                           "--freeze-sha256", FREEZE, "--source-commit", COMMIT])
        result = json.loads(stream.getvalue())
        self.assertFalse(result["writes_performed"])
        self.assertEqual(result["source_file_count"], 605)
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaises(SystemExit) as stopped:
            snapshot.main(["verify", "--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--report-sha256", stream.getvalue())
        self.assertNotIn("--raw-root", stream.getvalue())


if __name__ == "__main__":
    unittest.main()

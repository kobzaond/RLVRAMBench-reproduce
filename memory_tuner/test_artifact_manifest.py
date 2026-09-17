import tempfile
import unittest
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import patch

from memory_tuner.artifact_manifest import (
    add_trial_tree,
    collect_files,
    file_sha256,
    manifest_rows,
    publication_empty_directories,
    replication_empty_directories,
    verify_manifest,
    write_manifest,
)


class ArtifactManifestTests(unittest.TestCase):
    def test_replication_keeps_nested_retained_logs_and_every_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = [
                "output/completion_replication/pairs/cell/ledger.jsonl",
                "output/completion_replication/cell/attempt-1/.ray-log-paths.ABC123",
                "output/completion_replication/cell/attempt-1/trainer-metrics/step-1.json",
                "output/completion_replication/cell/attempt-1/ray-logs/ray/session_1/logs/worker-a-123.out",
                "output/replication_control/attempts/cell/ray-logs/ray/session_1/logs/worker-a-123.err",
                "output/completion_diagnostics/check/result.json",
                "output/revision_retention_recovery/first/ray-logs/ray/session_1/logs/worker-a-123.out",
                "benchmark/completion_replication/freeze.json",
                "benchmark/completion_replication_initial/predictions.json",
                "benchmark/completion_replication/results/configurations.csv",
                "benchmark/replication_control/replication_control_execution_seal.json",
                "benchmark/replication_control/runtime_amendment.json",
                "benchmark/revision_20260916/execution.json",
                "benchmark/revision_20260916/plan.md",
                "logs/completion-replication-123_0.out",
                "logs/replication_control-124_2.err",
            ]
            excluded = [
                "output/completion_replication/cell/attempt-1/ray/session_1/worker-a-123.out",
                "output/completion_replication/cell/attempt-1/ray-logs/ray/session_1/model.safetensors",
                "output/replication_control/attempts/cell/cache/token.json",
                "logs/completion-replication-other-123.out",
            ]
            for name in included + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("retained bytes")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {(root / name).absolute() for name in included})

    def test_source_snapshots_keep_model_source_directories_and_licenses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = [
                "reproduction_sources/completion/original/instrumented_python_packages/verl/models/model.py",
                "reproduction_sources/completion/amended/memory_tuner/collector.py",
                "reproduction_sources/control/original/benchmark/replication_control/protocol.json",
                "reproduction_sources/control/amended/third_party/TransferQueue/LICENSE",
            ]
            for name in included:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {(root / name).absolute() for name in included})
            bad = root / "reproduction_sources/weights.safetensors"
            bad.write_bytes(b"not source")
            with self.assertRaisesRegex(ValueError, "Model payload"):
                collect_files(root)

    def test_source_snapshots_refuse_linked_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "reproduction_sources"
            tree.mkdir()
            outside = root / "outside.py"
            outside.write_text("source")
            (tree / "alias.py").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "Nonregular reproduction source"):
                collect_files(root)

    def test_replication_empty_directory_inventory_does_not_import_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = root / "output/completion_replication/cell/attempt-1/allocator-traces"
            kept.mkdir(parents=True)
            cache = root / "output/completion_replication/cell/attempt-1/cache"
            cache.mkdir()
            (cache / "private").write_text("not evidence")
            irrelevant = root / "output/old_study/empty"
            irrelevant.mkdir(parents=True)
            self.assertEqual(replication_empty_directories(root), [kept])

    def test_publication_keeps_only_referenced_legacy_empty_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "output/legacy/cell/allocator-traces"
            leaf = trace / "rank-0"
            leaf.mkdir(parents=True)
            (root / "output/unrelated/empty").mkdir(parents=True)
            corpus = root / "profiles/benchmark/benchmark-corpus.csv"
            corpus.parent.mkdir(parents=True)
            corpus.write_text(
                "allocator_trace_dir\n"
                "/mnt/proj3/open-35-44/verl/rl/output/legacy/cell/allocator-traces\n"
                "/mnt/proj3/open-35-44/verl/rl/output/legacy/cell/allocator-traces\n")
            self.assertEqual(publication_empty_directories(root), [leaf])
            (leaf / "rank.jsonl").write_text("{}\n")
            self.assertEqual(publication_empty_directories(root), [])

    def test_publication_does_not_recreate_missing_or_follow_linked_allocator_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            corpus = root / "profiles/benchmark/benchmark-corpus.csv"
            corpus.parent.mkdir(parents=True)
            corpus.write_text("allocator_trace_dir\noutput/legacy/allocator-traces\n")
            with self.assertRaisesRegex(ValueError, "Missing recorded allocator"):
                publication_empty_directories(root)
            trace = root / "output/legacy/allocator-traces"
            trace.parent.mkdir(parents=True)
            outside = Path(directory) / "outside"
            outside.mkdir()
            trace.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Unsafe recorded allocator"):
                publication_empty_directories(root)

    @unittest.skipUnless(shutil.which("zstd"), "zstd archive compressor is required")
    def test_review_archive_preserves_empty_legacy_dirs_for_manifest_rebuilding(self):
        from memory_tuner.build_review_archives import build

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "source"
            trace = root / "output/legacy/cell/allocator-traces"
            trace.mkdir(parents=True)
            corpus = root / "profiles/benchmark/benchmark-corpus.csv"
            corpus.parent.mkdir(parents=True)
            corpus.write_text(
                "allocator_trace_dir\n"
                "/mnt/proj3/open-35-44/verl/rl/output/legacy/cell/allocator-traces\n")
            submission = root / "paper/submission/ieee_access_submission"
            submission.mkdir(parents=True)
            # Minimal archiver fixture; compilation and science auditing are
            # outside this test of directory preservation and reconstruction.
            stem = "RLVRAMBench_IEEE_Access_submission"
            verification = {
                "source_bundle_verified_after_extraction": True,
                "author_blockers_remain": [],
            }
            for name, field in (
                (stem + ".tex", "tex_sha256"), (stem + ".pdf", "pdf_sha256"),
                ("RLVRAMBench_supplement.tex", "supplement_tex_sha256"),
                ("RLVRAMBench_supplement.pdf", "supplement_pdf_sha256"),
            ):
                path = submission / name
                path.write_text("Authoritative submission manuscript. Edit this TeX directly")
                verification[field] = file_sha256(path)
            bundle = root / "paper/submission/RLVRAMBench_IEEE_Access_source.zip"
            bundle.write_bytes(b"archiver fixture")
            verification["source_bundle_sha256"] = file_sha256(bundle)
            (submission / "export_verification.json").write_text(json.dumps(verification))
            (root / "paper/artifact_manifest.md").write_text("Manifest fixture\n")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            expected = manifest_rows(root, files)
            write_manifest(root / "profiles/benchmark/publication-artifact-manifest.csv", expected)
            with patch("memory_tuner.build_review_archives.audit_package",
                       return_value=([], [], {})):
                report = build(root, base / "release", "test-empty-directories")
            self.assertEqual(report["empty_evidence_directories"],
                             ["output/legacy/cell/allocator-traces"])
            archive = Path(report["archives"][0]["path"])
            extracted = base / "extracted"
            extracted.mkdir()
            decompressor = subprocess.Popen(
                ["zstd", "-q", "-d", "-c", str(archive)], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=decompressor.stdout, mode="r|") as contents:
                contents.extractall(extracted)
            decompressor.stdout.close()
            self.assertEqual(decompressor.wait(), 0)
            rebuilt, missing = collect_files(extracted)
            self.assertEqual(missing, [])
            self.assertEqual(manifest_rows(extracted, rebuilt), expected)
            self.assertEqual(verify_manifest(
                extracted, extracted / "profiles/benchmark/publication-artifact-manifest.csv"), [])
            self.assertTrue((extracted / trace.relative_to(root)).is_dir())

    def test_estimation_preserves_partial_raw_records_without_a_completed_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                "pairs/cell/pair.json",
                "pairs/cell/ledger.jsonl",
                "pairs/cell/allocation-trace.jsonl",
                "pairs/cell/period-1.json",
                "pairs/cell/pair-summary.json",
                "cell-2gpu/attempt-123/dispatch.json",
                "cell-2gpu/attempt-123/launcher.log",
                "cell-2gpu/attempt-123/gpu-device-map.json",
                "cell-2gpu/attempt-123/device-evidence/trainer-cuda.json",
                "cell-2gpu/attempt-123/device-evidence/ray-resources.json",
                "cell-4gpu/attempt-123/trial-123.json",
                "cell-4gpu/attempt-123/environment-123.json",
                "cell-4gpu/attempt-123/training-123.log",
                "cell-4gpu/attempt-123/phase-memory-123.csv",
                "cell-4gpu/attempt-123/gpu-memory-123.csv",
                "cell-4gpu/attempt-123/gpu-telemetry-123.csv",
                "cell-4gpu/attempt-123/device-evidence/worker-error-1.json",
            ]
            trial_root = root / "output/prospective_estimation"
            for name in paths:
                path = trial_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                # Even a truncated record belongs in the archive; this is
                # inclusion, not parsing or successful-outcome selection.
                path.write_text('{"partial":')
            alias = trial_root / "cell-4gpu/trial-123.json"
            alias.symlink_to("attempt-123/trial-123.json")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(
                {path.relative_to(trial_root) for path in files},
                {Path(name) for name in paths} | {Path("cell-4gpu/trial-123.json")},
            )
            self.assertEqual(set(files.values()), {"prospective_estimation_evidence"})

    def test_estimation_prunes_ray_and_cache_trees_but_keeps_preflight_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = [
                "output/estimation-device-preflight/123/preflight-start.json",
                "output/estimation-device-preflight/123/preflight-passed.json",
                "output/estimation-device-preflight/123/task-2/preflight-ray.json",
                "output/estimation-device-preflight/123/task-4/preflight-ray.json",
                "output/prospective_estimation/cell/attempt-123/device-evidence/ray-resources.json",
            ]
            excluded = [
                "output/estimation-device-preflight/123/task-2/ray-runtime/session_1/logs/event.json",
                "output/estimation-device-preflight/123/task-2/ray-runtime/session_1/worker.log",
                "output/estimation-device-preflight/123/task-2/cache/metadata.json",
                "output/estimation-device-preflight/123/task-2/debug.log",
                "output/prospective_estimation/cell/attempt-123/r/ray/session_1/event.json",
                "output/prospective_estimation/cell/attempt-123/.cache/state.json",
                "output/prospective_estimation/cell/attempt-123/runtime/cache.json",
                "output/prospective_estimation/cell/attempt-123/torchinductor/kernel.py",
            ]
            for name in included + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {(root / name).absolute() for name in included})

    def test_estimation_excludes_model_payloads_but_preserves_checkpoint_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial_root = root / "output/prospective_estimation/cell/attempt-123"
            marker = trial_root / "latest_checkpointed_iteration.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text("1\n")
            for name in (
                "global_step_1/actor/model.pt",
                "global_step_1/actor/config.json",
                "checkpoints/weights.dat",
                "model/weights.dat",
                "models--Qwen--Qwen2.5-7B/snapshots/revision/config.json",
                "model.safetensors", "pytorch_model.bin", "optimizer.pt",
                "weights.pth", "weights.ckpt",
            ):
                path = trial_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not published")
            (trial_root / "payload-alias.json").symlink_to("optimizer.pt")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {marker.absolute()})

    def test_estimation_does_not_follow_external_runtime_links_or_proof_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            runtime = Path(directory) / "r/ray"
            runtime.mkdir(parents=True)
            runtime_file = runtime / "runtime.json"
            runtime_file.write_text("{}")
            task = root / "output/estimation-device-preflight/123/task-2"
            task.mkdir(parents=True)
            proof = task / "preflight-ray.json"
            proof.write_text(json.dumps({"runtime": str(runtime)}))
            (task / "ray-runtime").symlink_to(runtime, target_is_directory=True)
            (task / "other-runtime-link").symlink_to(runtime, target_is_directory=True)
            (task / "external-file.json").symlink_to(runtime_file)
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {proof.absolute()})

    def test_estimation_structured_inputs_results_and_notes_are_recursive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = [
                "protocol.json", "amendment-01.json", "model_metadata.json",
                "prediction_freeze.json", "fitted_model.json", "predictions.json",
                "retrospective.json", "matrix.csv", "targets.csv",
                "scheduler-accounting.psv",
                "results/summary.json", "results/partial-attempts.csv",
                "audits/numerical-threadpools/review.md",
            ]
            excluded = [
                "results/debug.log", "results/payload.bin",
                "cache/backend.json", "__pycache__/state.json",
                "global_step_1/actor/config.json",
            ]
            base = root / "benchmark/estimation"
            for name in included + excluded:
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {(base / name).absolute() for name in included})
            self.assertEqual(set(files.values()), {"estimation_benchmark"})

    def test_estimation_scheduler_log_selection_is_limited_to_exact_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            included = [
                "matched-gpu-123_0.out", "matched-gpu-123_0.err",
                "matched-preflight-124.out", "matched-preflight-124.err",
                "gpu-identity-125.out", "gpu-identity-125.err",
                "host-capacity-126_8.out", "host-capacity-126_8.err",
            ]
            excluded = [
                "matched-other-123.out", "matched-gpu-extra-123.out",
                "other-matched-gpu-123.out", "gpu-identity-unrelated.out",
                "matched-preflight-124.out.bak", "matched-gpu-123_0.json",
                "host-capacity-extra-126.out", "host-capacity-126_8.out.bak",
            ]
            for name in included + excluded:
                (logs / name).write_text("scheduler evidence")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {(logs / name).absolute() for name in included})
            self.assertEqual(set(files.values()), {"estimation_scheduler_log"})

    def test_existing_source_globs_include_estimation_helpers_and_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                "memory_tuner/estimation_baselines.py",
                "memory_tuner/model_memory_metadata.py",
                "memory_tuner/collect_estimation_study.py",
                "memory_tuner/run_matched_gpu.py",
                "memory_tuner/device_contract.py",
                "run_matched_gpu.slurm",
                "run_matched_gpu_preflight.slurm",
                "memory_tuner/host_capacity_protocol.py",
                "memory_tuner/run_host_capacity.py",
                "memory_tuner/collect_host_capacity.py",
                "memory_tuner/capture_allocation_accounting.py",
                "run_host_capacity.slurm",
            ]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# source")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {(root / name).absolute() for name in paths})

    def test_host_followup_preserves_separate_raw_and_design_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = [
                "output/estimation_host_capacity/pairs/pair/ledger.jsonl",
                "output/estimation_host_capacity/cell/attempt-123/dispatch.json",
                "output/estimation_host_capacity/cell/attempt-123/training-123.log",
                "benchmark/host_capacity/protocol.json",
                "benchmark/host_capacity/study_freeze.json",
                "benchmark/host_capacity/predictions.json",
                "benchmark/host_capacity/targets.csv",
                "benchmark/host_capacity/scheduler-accounting.psv",
                "benchmark/host_capacity/results/configurations_flat.csv",
                "benchmark/host_capacity/README.md",
            ]
            excluded = [
                "output/estimation_host_capacity/cell/attempt-123/model.safetensors",
                "output/estimation_host_capacity/cell/attempt-123/ray/session_1/log.json",
                "benchmark/host_capacity/cache/private.json",
            ]
            for name in included + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {(root / name).absolute() for name in included})
            self.assertEqual(set(files.values()),
                             {"host_capacity_evidence", "host_capacity_benchmark"})

    def test_uuid_compatibility_hook_is_explicit_pinned_runtime_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "instrumented_python_packages"
            hook = runtime / "sitecustomize.py"
            verl_init = runtime / "verl/__init__.py"
            unrelated = runtime / "unrelated.py"
            verl_init.parent.mkdir(parents=True)
            for path in (hook, verl_init, unrelated):
                path.write_text("# source")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(set(files), {hook.absolute(), verl_init.absolute()})
            self.assertEqual(set(files.values()), {"instrumented_runtime_source"})

    def test_decision_panel_preserves_protocol_attempts_and_not_model_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                "decision_benchmark.py",
                "benchmark/decision/protocol.json",
                "benchmark/decision/results/scores.json",
                "benchmark/decision/matrix.csv",
                "output/prospective_decision/pairs/slot/pair-1.json",
                "output/prospective_decision/slot/attempt-1/trial-1.json",
            ]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            checkpoint = root / "output/prospective_decision/slot/attempt-1/global_step_5/actor/model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"not published")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            for name in paths:
                self.assertIn((root / name).absolute(), files)
            self.assertNotIn(checkpoint.absolute(), files)

    def test_revision_scheduler_errors_are_preserved_without_unrelated_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            revision = logs / "revision-123_0.err"
            revision.write_text("startup failure before a training log exists")
            unrelated = logs / "unrelated-job.err"
            unrelated.write_text("not part of this artifact")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertIn(revision.absolute(), files)
            self.assertNotIn(unrelated.absolute(), files)

    def test_trial_links_and_allocation_provenance_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "output/revision_temporal/cell/attempt-1/trial-1.json"
            target.parent.mkdir(parents=True)
            target.write_text("{}")
            link = target.parent.parent / "trial-1.json"
            link.symlink_to("attempt-1/trial-1.json")
            provenance = root / "profiles/strengthening/same_node/1/pair-0.json"
            provenance.parent.mkdir(parents=True)
            provenance.write_text("{}")
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertIn(target.absolute(), files)
            self.assertIn(link.absolute(), files)
            self.assertIn(provenance.absolute(), files)

    def test_trial_tree_excludes_checkpoint_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = root / "trial"
            trial.mkdir()
            record = trial / "trial-1.json"
            record.write_text("{}")
            phase = trial / "phase-memory-1.csv"
            phase.write_text("timestamp,step,phase\n")
            checkpoint = (
                trial
                / "global_step_80"
                / "actor"
                / "model_world_size_2_rank_0.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            marker = trial / "latest_checkpointed_iteration.txt"
            marker.write_text("80\n")

            files = {}
            add_trial_tree(files, trial, "raw")

            self.assertEqual(files[record.resolve()], "raw")
            self.assertEqual(files[phase.resolve()], "raw")
            self.assertEqual(files[marker.resolve()], "raw")
            self.assertNotIn(checkpoint.resolve(), files)

    def test_manifest_verification_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("first")
            manifest = root / "manifest.csv"
            write_manifest(
                manifest,
                [
                    {
                        "path": "artifact.txt",
                        "category": "test",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": file_sha256(artifact),
                    }
                ],
            )
            self.assertEqual(verify_manifest(root, manifest), [])
            artifact.write_text("second")
            self.assertTrue(verify_manifest(root, manifest))

    def test_manifest_verification_rejects_missing_or_empty_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            self.assertTrue(verify_manifest(root, manifest))
            write_manifest(manifest, [])
            self.assertTrue(verify_manifest(root, manifest))

    def test_manifest_verification_rejects_duplicate_escaping_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("evidence")
            manifest = root / "manifest.csv"
            good = dict(path="artifact.txt", category="test",
                        size_bytes=artifact.stat().st_size, sha256=file_sha256(artifact))
            for rows in (
                [good, good],
                [{**good, "path": str(artifact)}],
                [{**good, "path": "../artifact.txt"}],
                [{**good, "size_bytes": "-1"}],
                [{**good, "sha256": "not-a-digest"}],
            ):
                write_manifest(manifest, rows)
                self.assertTrue(verify_manifest(root, manifest))

    def test_strict_multistep_requires_generated_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            _, missing = collect_files(
                Path(directory),
                require_complete_multistep=True,
            )
            self.assertIn(
                "multistep_matrix: profiles/multistep/matrix-2gpu.csv",
                missing,
            )

    def test_strict_cross_family_requires_every_frozen_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = (
                root / "memory_tuner/rlvram_cross_family_phi4_2gpu.csv"
            )
            matrix.parent.mkdir(parents=True)
            matrix.write_text("experiment_id\nphi-cell\n")
            _, missing = collect_files(
                root,
                require_complete_cross_family=True,
            )
            self.assertIn(
                "cross_family_experiment: phi-cell",
                missing,
            )

    def test_generated_paper_materials_are_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "paper/generated/scope.md"
            generated.parent.mkdir(parents=True)
            generated.write_text("generated")
            files, _ = collect_files(root)
            self.assertEqual(
                files[generated.resolve()],
                "paper_artifact",
            )

    def test_strict_major_revision_requires_every_frozen_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = (
                root
                / "memory_tuner/rlvram_major_revision_cross_model_smoke.csv"
            )
            matrix.parent.mkdir(parents=True)
            matrix.write_text("experiment_id\nsmoke-cell\n")
            _, missing = collect_files(
                root,
                require_complete_major_revision=True,
            )
            self.assertIn(
                "major_revision_experiment: smoke-cell",
                missing,
            )

    def test_prospective_corpus_digest_is_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = (
                root
                / "profiles/benchmark/prospective-method-corpus.sha256"
            )
            digest.parent.mkdir(parents=True)
            digest.write_text("a" * 64)
            files, _ = collect_files(root)
            self.assertEqual(
                files[digest.resolve()],
                "provenance_digest",
            )

    def test_strict_major_revision_rejects_infrastructure_only_online_attempt(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = (
                root
                / "memory_tuner/rlvram_major_revision_transfer_online.csv"
            )
            matrix.parent.mkdir(parents=True)
            matrix.write_text("experiment_id\nonline-cell\n")
            attempt = root / "profiles/vllm-online/1"
            attempt.mkdir(parents=True)
            (attempt / "trial.json").write_text(
                json.dumps(
                    {
                        "experiment_id": "online-cell",
                        "job_id": "1",
                        "exit_code": 1,
                        "elapsed_seconds": 0,
                    }
                )
            )
            (attempt / "server.err").write_text(
                "OSError: [Errno 98] Address already in use"
            )
            _, missing = collect_files(
                root,
                require_complete_major_revision=True,
            )
            self.assertIn(
                "major_revision_serving_experiment: online-cell "
                "(infrastructure_failure)",
                missing,
            )

    def test_excluded_trial_directory_evidence_is_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = root / "output/invalid/example/trial-1.json"
            trial.parent.mkdir(parents=True)
            trial.write_text("{}")
            log = trial.parent / "training-1.log"
            log.write_text("infrastructure failure")
            artifact_manifest = (
                root / "profiles/benchmark/artifact-manifest.csv"
            )
            artifact_manifest.parent.mkdir(parents=True)
            artifact_manifest.write_text(
                "artifact_path,scientific_valid\n"
                f"{trial},0\n"
            )
            files, missing = collect_files(root)
            self.assertEqual(missing, [])
            self.assertEqual(
                files[log.resolve()],
                "excluded_trial_evidence",
            )


if __name__ == "__main__":
    unittest.main()

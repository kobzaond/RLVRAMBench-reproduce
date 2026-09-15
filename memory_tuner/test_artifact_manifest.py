import tempfile
import unittest
import json
from pathlib import Path

from memory_tuner.artifact_manifest import (
    add_trial_tree,
    collect_files,
    file_sha256,
    verify_manifest,
    write_manifest,
)


class ArtifactManifestTests(unittest.TestCase):
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

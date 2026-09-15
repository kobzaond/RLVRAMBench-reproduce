import json
import tempfile
import unittest
from pathlib import Path

from memory_tuner.resolve_hf_snapshot import resolve_snapshot


class ResolveHFSnapshotTests(unittest.TestCase):
    def make_snapshot(self, root: Path, model: str, revision: str) -> Path:
        repo = root / "hub" / f"models--{model.replace('/', '--')}"
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(revision)
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (snapshot / name).write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"weights")
        return snapshot

    def test_resolves_complete_main_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.make_snapshot(root, "org/model", "abc")
            self.assertEqual(resolve_snapshot(root, "org/model"), snapshot.resolve())

    def test_rejects_incomplete_weight_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.make_snapshot(root, "org/model", "abc")
            (snapshot / "model.safetensors").unlink()
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer": "missing.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "no complete project-local"):
                resolve_snapshot(root, "org/model")


if __name__ == "__main__":
    unittest.main()

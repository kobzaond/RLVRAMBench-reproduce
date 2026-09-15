import tempfile
import unittest
from pathlib import Path

from memory_tuner.capture_environment import (
    data_metadata,
    file_sha256,
    resolve_model_revision,
)


class CaptureEnvironmentTests(unittest.TestCase):
    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value"
            path.write_bytes(b"abc")
            self.assertEqual(
                file_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_resolves_huggingface_main_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            hf_home = Path(directory)
            repo = hf_home / "hub" / "models--Org--Model"
            (repo / "refs").mkdir(parents=True)
            (repo / "snapshots" / "deadbeef").mkdir(parents=True)
            (repo / "refs" / "main").write_text("deadbeef\n")
            metadata = resolve_model_revision(hf_home, "Org/Model")
            self.assertEqual(metadata["resolved_revision"], "deadbeef")
            self.assertEqual(metadata["available_snapshots"], ["deadbeef"])

    def test_data_metadata_hashes_present_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.parquet").write_bytes(b"train")
            metadata = data_metadata(root)
            self.assertIn("train.parquet", metadata)
            self.assertNotIn("test.parquet", metadata)


if __name__ == "__main__":
    unittest.main()

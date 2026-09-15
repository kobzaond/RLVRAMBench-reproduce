import tempfile
import unittest
from pathlib import Path

from memory_tuner.gpu_monitor import read_marker


class GpuMonitorTests(unittest.TestCase):
    def test_read_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker"
            marker.write_text("123,7,weight_sync\n")
            self.assertEqual(read_marker(marker), ("7", "weight_sync"))

    def test_missing_marker(self):
        self.assertEqual(read_marker(Path("/definitely/missing")), ("0", "unknown"))


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import memory_profile

FOUR_GB = 4 * 1024 * 1024 * 1024


class TestMemoryProfile(unittest.TestCase):
    def test_peak_rss_is_positive(self):
        peak = memory_profile.measure_peak_rss_bytes(["python3", "-c", "print(1)"])
        self.assertGreater(peak, 0)

    def test_peak_rss_within_four_gb(self):
        peak = memory_profile.measure_peak_rss_bytes(["python3", "-c", "print(1)"])
        self.assertLessEqual(peak, FOUR_GB)

    def test_cli_output(self):
        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = memory_profile.main(["python3", "-c", "print(1)"])
        self.assertEqual(result, 0)
        self.assertIn("peak_rss_bytes=", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

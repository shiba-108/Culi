import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import experiment_runner


class TestExperimentRunner(unittest.TestCase):
    def test_successful_run_records_result(self):
        with tempfile.TemporaryDirectory() as directory:
            result = experiment_runner.run_experiment(
                ["python3", "-c", "print('ok')"],
                output_dir=directory,
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(result.task_success)
            self.assertGreater(result.runtime_seconds, 0)
            self.assertGreater(result.peak_rss_bytes, 0)
            result_file = Path(directory) / f"{result.experiment_id}.json"
            self.assertTrue(result_file.exists())
            stored = json.loads(result_file.read_text(encoding="utf-8"))
            self.assertEqual(stored["experiment_id"], result.experiment_id)
            self.assertIn("stdout", stored)

    def test_failed_run_records_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = experiment_runner.run_experiment(
                ["python3", "-c", "raise SystemExit(7)"],
                output_dir=directory,
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.exit_code, 7)
            self.assertFalse(result.task_success)

    def test_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            result = experiment_runner.run_experiment(
                ["python3", "-c", "import time; time.sleep(5)"],
                output_dir=directory,
                timeout=0.1,
            )
            self.assertEqual(result.status, "timeout")
            self.assertIsNone(result.exit_code)
            self.assertFalse(result.task_success)
            self.assertGreaterEqual(result.runtime_seconds, 0.1)

    def test_repeated_run_creates_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = experiment_runner.run_repeated(
                ["python3", "-c", "print('ok')"],
                repeat=2,
                output_dir=directory,
            )
            self.assertEqual(summary["repeat_count"], 2)
            self.assertEqual(summary["success_count"], 2)
            self.assertEqual(summary["task_success_rate"], 1.0)
            summary_file = Path(directory) / f"{summary['experiment_id']}.json"
            self.assertTrue(summary_file.exists())

    def test_cli_success(self):
        with tempfile.TemporaryDirectory() as directory:
            import io
            import contextlib

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = experiment_runner.main(
                    ["--output-dir", directory, "--", "python3", "-c", "print('ok')"]
                )
            self.assertEqual(code, 0)
            self.assertIn("experiment_id=", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

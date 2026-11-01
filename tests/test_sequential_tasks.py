"""Tests for synthetic sequential task generation."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequential_tasks import make_dataset, make_examples


class TestSequentialTasks(unittest.TestCase):
    def test_recall_last_example(self):
        examples = make_examples("recall_last", 5, 6, seed=7)
        self.assertEqual(len(examples), 5)
        for bits, label in examples:
            self.assertEqual(label, bits[-1])

    def test_make_dataset_structure(self):
        examples = make_examples("recall_last", 3, 4, seed=9)
        dataset = make_dataset("recall_last", examples, dataset_id="seq_recall_last_len4_v1")
        self.assertEqual(dataset["dataset_id"], "seq_recall_last_len4_v1")
        self.assertEqual(len(dataset["cases"]), 3)
        for case in dataset["cases"]:
            self.assertEqual(case["metadata"]["task"], "recall_last")


if __name__ == "__main__":
    unittest.main()
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import (
    EvaluationDataset,
    EvaluationRunner,
    TestCase,
    compare_evaluations,
    score_case,
)
from mock_model import MockModelAdapter


class GoodAdapter:
    name = "good-model"
    version = "1.0"

    def generate(self, prompt: str) -> str:
        return "ok"


class FailingAdapter:
    name = "failing-model"
    version = "1.0"

    def generate(self, prompt: str) -> str:
        raise RuntimeError("model failure")


class TestEvaluation(unittest.TestCase):
    def test_dataset_validation(self):
        dataset = EvaluationDataset(
            dataset_id="test",
            version="1",
            cases=[TestCase(id="a", category="reasoning", input="x", expected="y", scoring="exact")],
        )
        dataset.validate()

    def test_duplicate_ids_rejected(self):
        dataset = EvaluationDataset(
            dataset_id="test",
            version="1",
            cases=[
                TestCase(id="a", category="reasoning", input="x", expected="y"),
                TestCase(id="a", category="reasoning", input="z", expected="w"),
            ],
        )
        with self.assertRaises(ValueError):
            dataset.validate()

    def test_scoring_correctness(self):
        case = TestCase(id="a", category="reasoning", input="x", expected="Yes", scoring="exact")
        self.assertEqual(score_case(case, "yes"), (1.0, True, "scored"))
        self.assertEqual(score_case(case, "no"), (0.0, False, "scored"))

    def test_contains_scoring(self):
        case = TestCase(id="a", category="reasoning", input="x", expected="cargo test", scoring="contains")
        self.assertEqual(score_case(case, "Run cargo test now"), (1.0, True, "scored"))

    def test_missing_expected(self):
        case = TestCase(id="a", category="reasoning", input="x", expected=None, scoring="exact")
        self.assertEqual(score_case(case, "anything"), (None, None, "missing_expected"))

    def test_empty_output(self):
        case = TestCase(id="a", category="reasoning", input="x", expected="y", scoring="exact")
        self.assertEqual(score_case(case, ""), (0.0, False, "empty_output"))

    def test_manual_scoring(self):
        case = TestCase(id="a", category="reasoning", input="x", scoring="manual")
        self.assertEqual(score_case(case, "anything"), (None, None, "needs_review"))

    def test_model_execution_failure_recorded(self):
        dataset = EvaluationDataset(
            dataset_id="test",
            version="1",
            cases=[TestCase(id="a", category="reasoning", input="x", expected="y", scoring="exact")],
        )
        result = EvaluationRunner(FailingAdapter()).run(dataset)
        self.assertEqual(result.case_results[0].status, "model_error")
        self.assertIsNone(result.case_results[0].score)
        self.assertIn("RuntimeError", result.case_results[0].error)

    def test_result_export(self):
        dataset = EvaluationDataset(
            dataset_id="test",
            version="1",
            cases=[TestCase(id="a", category="reasoning", input="x", expected="ok", scoring="exact")],
        )
        result = EvaluationRunner(GoodAdapter()).run(dataset)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            EvaluationRunner(GoodAdapter()).save(result, path)
            self.assertTrue(path.exists())
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["model_name"], "good-model")

    def test_model_version_comparison(self):
        dataset = EvaluationDataset(
            dataset_id="test",
            version="1",
            cases=[TestCase(id="a", category="reasoning", input="x", expected="ok", scoring="exact")],
        )
        baseline = EvaluationRunner(GoodAdapter()).run(dataset)
        candidate_adapter = GoodAdapter()
        candidate_adapter.version = "2.0"
        candidate = EvaluationRunner(candidate_adapter).run(dataset)
        comparison = compare_evaluations(baseline, candidate)
        self.assertEqual(comparison["overall_score_delta"], 0.0)

    def test_dataset_load_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            dataset = EvaluationDataset(
                dataset_id="test",
                version="1",
                cases=[TestCase(id="a", category="reasoning", input="x", expected="ok", scoring="exact")],
            )
            dataset.save(path)
            loaded = EvaluationDataset.load(path)
            result = EvaluationRunner(GoodAdapter()).run(loaded)
            self.assertEqual(result.overall_score, 1.0)


if __name__ == "__main__":
    unittest.main()

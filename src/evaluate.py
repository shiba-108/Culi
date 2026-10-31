"""Command-line interface for the Model Culi evaluation system."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
import sys

sys.path.insert(0, sys_path)

from evaluation import EvaluationDataset, EvaluationRunner
from mock_model import MockModelAdapter


def load_adapter(spec: str, model_path: str | None = None):
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    if model_path is None:
        return adapter_class()
    return adapter_class(model_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a Model Culi evaluation")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset JSON")
    parser.add_argument("--mock", action="store_true", help="Use the mock model adapter")
    parser.add_argument("--adapter", help="Adapter import path, e.g. mock_model:MockModelAdapter")
    parser.add_argument("--model-name", help="Override model name")
    parser.add_argument("--model-version", help="Override model version")
    parser.add_argument("--model-path", help="Path passed to the adapter constructor")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--compare", help="Baseline evaluation JSON path")
    args = parser.parse_args(argv)

    if args.mock:
        adapter = MockModelAdapter()
    elif args.adapter:
        adapter = load_adapter(args.adapter, args.model_path)
    else:
        parser.error("either --mock or --adapter is required")
        return 2

    if args.model_name:
        adapter.name = args.model_name
    if args.model_version:
        adapter.version = args.model_version

    dataset = EvaluationDataset.load(args.dataset)
    result = EvaluationRunner(adapter).run(dataset)
    EvaluationRunner(adapter).save(result, args.output)

    print(f"evaluation_id={result.evaluation_id}")
    print(f"model={result.model_name} {result.model_version}")
    print(f"overall_score={result.overall_score}")
    for category, score in result.category_scores.items():
        print(f"{category}={score}")
    print(f"manual_review_count={result.manual_review_count}")
    print(f"peak_rss_bytes={result.peak_rss_bytes}")
    print(f"output={args.output}")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.output).read_text(encoding="utf-8"))
        baseline_categories = baseline.get("category_scores", {})
        candidate_categories = candidate.get("category_scores", {})
        deltas = {}
        for category in sorted(set(baseline_categories) | set(candidate_categories)):
            base_score = baseline_categories.get(category)
            candidate_score = candidate_categories.get(category)
            deltas[category] = None if base_score is None or candidate_score is None else candidate_score - base_score
        comparison = {
            "baseline_file": args.compare,
            "candidate_file": args.output,
            "baseline_overall_score": baseline.get("overall_score"),
            "candidate_overall_score": candidate.get("overall_score"),
            "overall_score_delta": (
                None
                if baseline.get("overall_score") is None or candidate.get("overall_score") is None
                else candidate.get("overall_score") - baseline.get("overall_score")
            ),
            "category_score_deltas": deltas,
        }
        comparison_path = Path(args.output).with_name(Path(args.output).stem + "_comparison.json")
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"comparison={comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

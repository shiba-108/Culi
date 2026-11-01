"""Evaluate the collaborative specialist system on a composite task dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequential_adapters import (
    SequentialRecurrentAdapter,
    SequentialMemoryAdapter,
    SequentialHierarchicalAdapter,
)
from sequential_tasks import make_dataset

BASE_TASKS = ["parity", "delayed_recall", "dependency", "multi_recall", "recall_last"]
TASK_COMPONENTS = {
    "parity": ["parity"],
    "delayed_recall": ["delayed_recall"],
    "dependency": ["dependency"],
    "multi_recall": ["multi_recall"],
    "recall_last": ["recall_last"],
    "compose_parity_delayed": ["parity", "delayed_recall"],
    "compose_dependency_multi": ["dependency", "multi_recall"],
    "compose_parity_delayed_dependency": ["parity", "delayed_recall", "dependency"],
    "compose_delayed_multi_recalllast": ["delayed_recall", "multi_recall", "recall_last"],
    "compose_all": ["parity", "delayed_recall", "dependency", "multi_recall", "recall_last"],
}

ADAPTERS = {
    "recurrent": SequentialRecurrentAdapter,
    "memory": SequentialMemoryAdapter,
    "compact_memory": SequentialMemoryAdapter,
    "hierarchical": SequentialHierarchicalAdapter,
}


def _model_kind(model: str) -> str:
    return "memory" if model == "compact_memory" else model


def _specialist_path(specialist_dir: Path, model: str, base_task: str, seed: int) -> Path:
    return specialist_dir / f"m10_base_{model}_{base_task}_seed{seed}.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate collaborative specialists")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True, choices=["recurrent", "memory", "compact_memory", "hierarchical"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--specialist-dir", default="models")
    parser.add_argument("--mode", default="collaborative", choices=["collaborative", "no_shared_memory", "all_specialists", "no_collaboration"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    specialist_dir = ROOT / args.specialist_dir
    components = TASK_COMPONENTS[args.task]
    adapter_class = ADAPTERS[args.model]
    specialists = {}
    for base_task in BASE_TASKS:
        specialists[base_task] = adapter_class(_specialist_path(specialist_dir, args.model, base_task, args.seed))

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    cases = dataset["cases"]

    shared_cache: dict[str, list[int]] = {}
    activated_counts = []
    communication_ops = 0
    correct = 0
    total = 0
    start = time.perf_counter()

    for case in cases:
        bits = [float(x) for x in case["input"].split()]
        prompt = case["input"]
        expected = int(case["expected"])

        if args.mode == "all_specialists":
            activated = list(BASE_TASKS)
        elif args.mode == "no_collaboration":
            activated = [components[0]]
        else:
            activated = list(components)

        # Shared memory cache.
        cache_key = prompt
        if args.mode != "no_shared_memory" and cache_key in shared_cache:
            base_predictions = shared_cache[cache_key]
        else:
            base_predictions = {}
            for base_task in activated:
                base_predictions[base_task] = int(specialists[base_task].generate(prompt))
            if args.mode != "no_shared_memory":
                shared_cache[cache_key] = base_predictions

        # Combine.
        selected_predictions = [base_predictions[task] for task in components if task in base_predictions]
        if args.mode == "no_collaboration" or not selected_predictions:
            prediction = selected_predictions[0] if selected_predictions else 0
        else:
            prediction = 0
            for value in selected_predictions:
                prediction ^= value

        activated_counts.append(len(activated))
        communication_ops += len(activated) + 1
        if prediction == expected:
            correct += 1
        total += 1

    runtime = time.perf_counter() - start
    import resource

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    overall_score = correct / total if total else 0.0
    result = {
        "evaluation_id": f"eval-collab-{int(time.time())}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": f"collaborative-{args.model}",
        "model_version": f"{args.mode}-seed-{args.seed}",
        "dataset_id": dataset.get("dataset_id", "unknown"),
        "dataset_version": dataset.get("version", "unknown"),
        "case_results": [],
        "category_scores": {"reasoning": overall_score},
        "overall_score": overall_score,
        "total_runtime_seconds": runtime,
        "peak_rss_bytes": peak_rss,
        "manual_review_count": 0,
        "activated_specialists_mean": (sum(activated_counts) / len(activated_counts)) if activated_counts else 0.0,
        "communication_ops_total": communication_ops,
        "routing_accuracy": 1.0,
        "mode": args.mode,
        "task": args.task,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"evaluation_id={result['evaluation_id']}")
    print(f"model=collaborative-{args.model} {result['model_version']}")
    print(f"overall_score={result['overall_score']}")
    print(f"manual_review_count=0")
    print(f"peak_rss_bytes={result['peak_rss_bytes']}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the Milestone 9 cross-task generalization experiment."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequential_tasks import make_dataset, make_examples, save_dataset
from experiment_runner import run_experiment
from train_sequential import build_model

TASKS = ["parity", "delayed_recall", "dependency", "multi_recall", "recall_last"]
MODELS = ["recurrent", "memory", "compact_memory", "hierarchical"]
SEEDS = [1, 2, 3]
TRAIN_LENGTHS = [4, 6, 8]
TRAIN_COUNT = 80
EPOCHS = 30
TEST_COUNT = 50
TEST_LENGTHS = [4, 6, 8, 10, 12, 16]
LEARNING_RATES = {
    "recurrent": 0.1,
    "memory": 0.1,
    "compact_memory": 0.1,
    "hierarchical": 0.1,
}
HIDDEN = {
    "recurrent": 4,
    "memory": 4,
    "compact_memory": 2,
    "hierarchical": 4,
}
ADAPTERS = {
    "recurrent": "sequential_adapters:SequentialRecurrentAdapter",
    "memory": "sequential_adapters:SequentialMemoryAdapter",
    "compact_memory": "sequential_adapters:SequentialMemoryAdapter",
    "hierarchical": "sequential_adapters:SequentialHierarchicalAdapter",
}


def _parse(stdout: str, key: str) -> str | None:
    import re

    match = re.search(rf"^{re.escape(key)}=(.+)$", stdout, re.MULTILINE)
    return match.group(1).strip() if match else None


def _train_model_kind(model: str) -> str:
    if model == "compact_memory":
        return "memory"
    return model


def main() -> int:
    for task in TASKS:
        for length in TEST_LENGTHS:
            examples = make_examples(task, TEST_COUNT, length, seed=999)
            dataset = make_dataset(task, examples, dataset_id=f"seq_{task}_len{length}_v1")
            save_dataset(ROOT / "evaluations" / f"seq_{task}_len{length}_v1.json", dataset)

    results = []
    started = time.perf_counter()

    # Trained specialists: one model per source task.
    for model in MODELS:
        for source_task in TASKS:
            for seed in SEEDS:
                print(f"training {model} on {source_task} seed={seed}", flush=True)
                model_path = ROOT / "models" / f"m9_{model}_{source_task}_seed{seed}.json"
                train_command = [
                    "python3", "src/train_sequential.py",
                    "--model", _train_model_kind(model),
                    "--task", source_task,
                    "--train-lengths", ",".join(str(x) for x in TRAIN_LENGTHS),
                    "--train-count", str(TRAIN_COUNT),
                    "--epochs", str(EPOCHS),
                    "--learning-rate", str(LEARNING_RATES[model]),
                    "--hidden-size", str(HIDDEN[model]),
                    "--seed", str(seed),
                    "--output", str(model_path.relative_to(ROOT)),
                ]
                train_result = run_experiment(train_command, output_dir="experiments/results")
                record = {
                    "condition": "trained",
                    "model": model,
                    "source_task": source_task,
                    "seed": seed,
                    "training_status": train_result.status,
                    "training_exit_code": train_result.exit_code,
                    "training_runtime_seconds": train_result.runtime_seconds,
                    "training_peak_rss_bytes": train_result.peak_rss_bytes,
                    "parameter_count": int(_parse(train_result.stdout, "parameter_count")) if _parse(train_result.stdout, "parameter_count") else None,
                    "initial_loss": float(_parse(train_result.stdout, "initial_loss")) if _parse(train_result.stdout, "initial_loss") else None,
                    "final_loss": float(_parse(train_result.stdout, "final_loss")) if _parse(train_result.stdout, "final_loss") else None,
                    "parameters_changed": _parse(train_result.stdout, "parameters_changed") == "True",
                    "model_path": str(model_path),
                    "evaluations": {},
                }
                for target_task in TASKS:
                    record["evaluations"][target_task] = {}
                    for test_length in TEST_LENGTHS:
                        dataset_path = f"evaluations/seq_{target_task}_len{test_length}_v1.json"
                        eval_output = ROOT / "experiments" / "results" / f"m9_eval_{model}_{source_task}_to_{target_task}_seed{seed}_len{test_length}.json"
                        eval_command = [
                            "python3", "src/evaluate.py",
                            "--dataset", dataset_path,
                            "--adapter", ADAPTERS[model],
                            "--model-path", str(model_path),
                            "--model-name", model,
                            "--model-version", f"seed-{seed}",
                            "--output", str(eval_output),
                        ]
                        eval_result = run_experiment(eval_command, output_dir="experiments/results")
                        stored_eval = json.loads(eval_output.read_text(encoding="utf-8"))
                        record["evaluations"][target_task][str(test_length)] = {
                            "status": eval_result.status,
                            "exit_code": eval_result.exit_code,
                            "evaluation_runtime_seconds": stored_eval["total_runtime_seconds"],
                            "evaluation_peak_rss_bytes": stored_eval["peak_rss_bytes"],
                            "overall_score": stored_eval["overall_score"],
                            "evaluation_id": stored_eval["evaluation_id"],
                        }
                results.append(record)

    # Untrained controls: one initialized model per model/seed.
    for model in MODELS:
        for seed in SEEDS:
            print(f"creating untrained control {model} seed={seed}", flush=True)
            model_path = ROOT / "models" / f"m9_{model}_untrained_seed{seed}.json"
            model_obj = build_model(_train_model_kind(model), TRAIN_LENGTHS[0], HIDDEN[model], LEARNING_RATES[model], seed)
            model_obj.save(model_path)
            record = {
                "condition": "untrained",
                "model": model,
                "source_task": None,
                "seed": seed,
                "training_status": "not_trained",
                "training_exit_code": 0,
                "training_runtime_seconds": 0.0,
                "training_peak_rss_bytes": None,
                "parameter_count": model_obj.parameter_count,
                "initial_loss": None,
                "final_loss": None,
                "parameters_changed": False,
                "model_path": str(model_path),
                "evaluations": {},
            }
            for target_task in TASKS:
                record["evaluations"][target_task] = {}
                for test_length in TEST_LENGTHS:
                    dataset_path = f"evaluations/seq_{target_task}_len{test_length}_v1.json"
                    eval_output = ROOT / "experiments" / "results" / f"m9_eval_{model}_untrained_to_{target_task}_seed{seed}_len{test_length}.json"
                    eval_command = [
                        "python3", "src/evaluate.py",
                        "--dataset", dataset_path,
                        "--adapter", ADAPTERS[model],
                        "--model-path", str(model_path),
                        "--model-name", model,
                        "--model-version", f"untrained-seed-{seed}",
                        "--output", str(eval_output),
                    ]
                    eval_result = run_experiment(eval_command, output_dir="experiments/results")
                    stored_eval = json.loads(eval_output.read_text(encoding="utf-8"))
                    record["evaluations"][target_task][str(test_length)] = {
                        "status": eval_result.status,
                        "exit_code": eval_result.exit_code,
                        "evaluation_runtime_seconds": stored_eval["total_runtime_seconds"],
                        "evaluation_peak_rss_bytes": stored_eval["peak_rss_bytes"],
                        "overall_score": stored_eval["overall_score"],
                        "evaluation_id": stored_eval["evaluation_id"],
                    }
            results.append(record)

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": TASKS,
        "models": MODELS,
        "seeds": SEEDS,
        "train_lengths": TRAIN_LENGTHS,
        "train_count": TRAIN_COUNT,
        "epochs": EPOCHS,
        "test_lengths": TEST_LENGTHS,
        "conditions": ["trained", "untrained"],
        "results": results,
        "wall_time_seconds": time.perf_counter() - started,
    }
    output_path = ROOT / "experiments" / "results" / "cross_task_generalization_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_file={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

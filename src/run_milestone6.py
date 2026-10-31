"""Run the Milestone 6 sequential memory and reasoning experiment."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sequential_tasks import make_dataset, make_examples, save_dataset
from experiment_runner import run_experiment

TASKS = ["parity", "delayed_recall", "dependency"]
MODELS = ["feedforward", "recurrent", "recurrent_bounded"]
SEEDS = [1, 2, 3]
TRAIN_LENGTH = 6
TRAIN_COUNT = 100
EPOCHS = 100
TEST_COUNT = 50
TEST_LENGTHS = [6, 8]
LEARNING_RATES = {
    "feedforward": 0.5,
    "recurrent": 0.1,
    "recurrent_bounded": 0.1,
}
ADAPTERS = {
    "feedforward": "sequential_adapters:SequentialFeedforwardAdapter",
    "recurrent": "sequential_adapters:SequentialRecurrentAdapter",
    "recurrent_bounded": "sequential_adapters:SequentialRecurrentAdapter",
}


def _parse(stdout: str, key: str) -> str | None:
    import re

    match = re.search(rf"^{re.escape(key)}=(.+)$", stdout, re.MULTILINE)
    return match.group(1).strip() if match else None


def main() -> int:
    # Generate deterministic test datasets.
    for task in TASKS:
        for length in TEST_LENGTHS:
            examples = make_examples(task, TEST_COUNT, length, seed=999)
            dataset = make_dataset(
                task, examples, dataset_id=f"seq_{task}_len{length}_v1"
            )
            save_dataset(ROOT / "evaluations" / f"seq_{task}_len{length}_v1.json", dataset)

    results = []
    started = time.perf_counter()
    for model in MODELS:
        for task in TASKS:
            for seed in SEEDS:
                print(f"running {model} {task} seed={seed}", flush=True)
                model_path = ROOT / "models" / f"seq_{model}_{task}_seed{seed}.json"
                train_command = [
                    "python3", "src/train_sequential.py",
                    "--model", model,
                    "--task", task,
                    "--train-length", str(TRAIN_LENGTH),
                    "--train-count", str(TRAIN_COUNT),
                    "--epochs", str(EPOCHS),
                    "--learning-rate", str(LEARNING_RATES[model]),
                    "--seed", str(seed),
                    "--output", str(model_path.relative_to(ROOT)),
                ]
                train_result = run_experiment(train_command, output_dir="experiments/results")
                record = {
                    "model": model,
                    "task": task,
                    "seed": seed,
                    "train_length": TRAIN_LENGTH,
                    "train_count": TRAIN_COUNT,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATES[model],
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

                for test_length in TEST_LENGTHS:
                    dataset_path = f"evaluations/seq_{task}_len{test_length}_v1.json"
                    eval_output = ROOT / "experiments" / "results" / f"seq_eval_{model}_{task}_seed{seed}_len{test_length}.json"
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
                    record["evaluations"][str(test_length)] = {
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
        "train_length": TRAIN_LENGTH,
        "train_count": TRAIN_COUNT,
        "epochs": EPOCHS,
        "test_lengths": TEST_LENGTHS,
        "results": results,
        "wall_time_seconds": time.perf_counter() - started,
    }
    output_path = ROOT / "experiments" / "results" / "sequential_memory_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_file={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

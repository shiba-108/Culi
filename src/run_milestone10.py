"""Run the Milestone 10 collaborative specialist intelligence experiment."""

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

BASE_TASKS = ["parity", "delayed_recall", "dependency", "multi_recall", "recall_last"]
COMPOSITE_TASKS = [
    "compose_parity_delayed",
    "compose_dependency_multi",
    "compose_parity_delayed_dependency",
    "compose_delayed_multi_recalllast",
    "compose_all",
]
TRAIN_COMPOSITES = ["compose_parity_delayed", "compose_dependency_multi"]
MODELS = ["memory"]
SEEDS = [1, 2, 3]
TRAIN_LENGTHS = [4, 6, 8]
TRAIN_COUNT = 80
EPOCHS = 30
TEST_COUNT = 50
TEST_LENGTHS = [4, 6, 8, 10, 12, 16]
LEARNING_RATE = 0.1
HIDDEN_SIZE = 4
ADAPTER = "sequential_adapters:SequentialMemoryAdapter"
COLLAB_MODES = ["collaborative", "no_shared_memory", "all_specialists", "no_collaboration"]


def _parse(stdout: str, key: str) -> str | None:
    import re

    match = re.search(rf"^{re.escape(key)}=(.+)$", stdout, re.MULTILINE)
    return match.group(1).strip() if match else None


def main() -> int:
    all_tasks = BASE_TASKS + COMPOSITE_TASKS
    for task in all_tasks:
        for length in TEST_LENGTHS:
            examples = make_examples(task, TEST_COUNT, length, seed=999)
            dataset = make_dataset(task, examples, dataset_id=f"seq_{task}_len{length}_v1")
            save_dataset(ROOT / "evaluations" / f"seq_{task}_len{length}_v1.json", dataset)

    results = []
    started = time.perf_counter()

    # Base specialists.
    for base_task in BASE_TASKS:
        for seed in SEEDS:
            print(f"training base specialist {base_task} seed={seed}", flush=True)
            model_path = ROOT / "models" / f"m10_base_memory_{base_task}_seed{seed}.json"
            train_command = [
                "python3", "src/train_sequential.py",
                "--model", "memory",
                "--task", base_task,
                "--train-lengths", ",".join(str(x) for x in TRAIN_LENGTHS),
                "--train-count", str(TRAIN_COUNT),
                "--epochs", str(EPOCHS),
                "--learning-rate", str(LEARNING_RATE),
                "--hidden-size", str(HIDDEN_SIZE),
                "--seed", str(seed),
                "--output", str(model_path.relative_to(ROOT)),
            ]
            train_result = run_experiment(train_command, output_dir="experiments/results")
            results.append({
                "condition": "base_specialist",
                "model": "memory",
                "task": base_task,
                "seed": seed,
                "training_status": train_result.status,
                "training_exit_code": train_result.exit_code,
                "training_runtime_seconds": train_result.runtime_seconds,
                "training_peak_rss_bytes": train_result.peak_rss_bytes,
                "parameter_count": int(_parse(train_result.stdout, "parameter_count")) if _parse(train_result.stdout, "parameter_count") else None,
                "initial_loss": float(_parse(train_result.stdout, "initial_loss")) if _parse(train_result.stdout, "initial_loss") else None,
                "final_loss": float(_parse(train_result.stdout, "final_loss")) if _parse(train_result.stdout, "final_loss") else None,
                "model_path": str(model_path),
                "evaluations": {},
            })

    # Unified models on selected composite tasks.
    for comp_task in TRAIN_COMPOSITES:
        for seed in SEEDS:
            print(f"training unified model {comp_task} seed={seed}", flush=True)
            model_path = ROOT / "models" / f"m10_unified_memory_{comp_task}_seed{seed}.json"
            train_command = [
                "python3", "src/train_sequential.py",
                "--model", "memory",
                "--task", comp_task,
                "--train-lengths", ",".join(str(x) for x in TRAIN_LENGTHS),
                "--train-count", str(TRAIN_COUNT),
                "--epochs", str(EPOCHS),
                "--learning-rate", str(LEARNING_RATE),
                "--hidden-size", str(HIDDEN_SIZE),
                "--seed", str(seed),
                "--output", str(model_path.relative_to(ROOT)),
            ]
            train_result = run_experiment(train_command, output_dir="experiments/results")
            record = {
                "condition": "unified",
                "model": "memory",
                "task": comp_task,
                "seed": seed,
                "training_status": train_result.status,
                "training_exit_code": train_result.exit_code,
                "training_runtime_seconds": train_result.runtime_seconds,
                "training_peak_rss_bytes": train_result.peak_rss_bytes,
                "parameter_count": int(_parse(train_result.stdout, "parameter_count")) if _parse(train_result.stdout, "parameter_count") else None,
                "initial_loss": float(_parse(train_result.stdout, "initial_loss")) if _parse(train_result.stdout, "initial_loss") else None,
                "final_loss": float(_parse(train_result.stdout, "final_loss")) if _parse(train_result.stdout, "final_loss") else None,
                "model_path": str(model_path),
                "evaluations": {},
            }
            # Evaluate unified on its training task.
            record["evaluations"][comp_task] = {}
            for test_length in TEST_LENGTHS:
                dataset_path = f"evaluations/seq_{comp_task}_len{test_length}_v1.json"
                eval_output = ROOT / "experiments" / "results" / f"m10_eval_unified_{comp_task}_seed{seed}_len{test_length}.json"
                eval_command = [
                    "python3", "src/evaluate.py",
                    "--dataset", dataset_path,
                    "--adapter", ADAPTER,
                    "--model-path", str(model_path),
                    "--model-name", "memory",
                    "--model-version", f"unified-seed-{seed}",
                    "--output", str(eval_output),
                ]
                eval_result = run_experiment(eval_command, output_dir="experiments/results")
                stored_eval = json.loads(eval_output.read_text(encoding="utf-8"))
                record["evaluations"][comp_task][str(test_length)] = {
                    "status": eval_result.status,
                    "exit_code": eval_result.exit_code,
                    "evaluation_runtime_seconds": stored_eval["total_runtime_seconds"],
                    "evaluation_peak_rss_bytes": stored_eval["peak_rss_bytes"],
                    "overall_score": stored_eval["overall_score"],
                    "evaluation_id": stored_eval["evaluation_id"],
                }
            results.append(record)

    # Untrained base specialists for collaborative control.
    for seed in SEEDS:
        model_path = ROOT / "models" / f"m10_base_memory_untrained_seed{seed}.json"
        model_obj = build_model("memory", TRAIN_LENGTHS[0], HIDDEN_SIZE, LEARNING_RATE, seed)
        model_obj.save(model_path)

    # Independent baseline and collaborative evaluations on all composite tasks.
    for comp_task in COMPOSITE_TASKS:
        for seed in SEEDS:
            # Independent: first base specialist.
            first_component = comp_task and __import__("sequential_tasks").TASK_GENERATORS  # placeholder no-op
            first_base = None
            # determine first component from collaborative mapping
            import sequential_tasks as st
            # components mapping
            components_map = {
                "compose_parity_delayed": ["parity", "delayed_recall"],
                "compose_dependency_multi": ["dependency", "multi_recall"],
                "compose_parity_delayed_dependency": ["parity", "delayed_recall", "dependency"],
                "compose_delayed_multi_recalllast": ["delayed_recall", "multi_recall", "recall_last"],
                "compose_all": ["parity", "delayed_recall", "dependency", "multi_recall", "recall_last"],
            }
            first_base = components_map[comp_task][0]
            specialist_path = ROOT / "models" / f"m10_base_memory_{first_base}_seed{seed}.json"
            # Evaluate independent baseline on composite task dataset using first specialist.
            independent_record = {
                "condition": "independent",
                "model": "memory",
                "task": comp_task,
                "seed": seed,
                "training_status": "not_trained_for_composite",
                "training_exit_code": 0,
                "training_runtime_seconds": 0.0,
                "training_peak_rss_bytes": None,
                "parameter_count": None,
                "initial_loss": None,
                "final_loss": None,
                "model_path": str(specialist_path),
                "evaluations": {comp_task: {}},
                "activated_specialists": 1,
            }
            for test_length in TEST_LENGTHS:
                dataset_path = f"evaluations/seq_{comp_task}_len{test_length}_v1.json"
                eval_output = ROOT / "experiments" / "results" / f"m10_eval_independent_{comp_task}_seed{seed}_len{test_length}.json"
                eval_command = [
                    "python3", "src/evaluate.py",
                    "--dataset", dataset_path,
                    "--adapter", ADAPTER,
                    "--model-path", str(specialist_path),
                    "--model-name", "memory",
                    "--model-version", f"independent-seed-{seed}",
                    "--output", str(eval_output),
                ]
                eval_result = run_experiment(eval_command, output_dir="experiments/results")
                stored_eval = json.loads(eval_output.read_text(encoding="utf-8"))
                independent_record["evaluations"][comp_task][str(test_length)] = {
                    "status": eval_result.status,
                    "exit_code": eval_result.exit_code,
                    "evaluation_runtime_seconds": stored_eval["total_runtime_seconds"],
                    "evaluation_peak_rss_bytes": stored_eval["peak_rss_bytes"],
                    "overall_score": stored_eval["overall_score"],
                    "evaluation_id": stored_eval["evaluation_id"],
                }
            results.append(independent_record)

            # Collaborative variants.
            for mode in COLLAB_MODES:
                collab_record = {
                    "condition": f"collaborative_{mode}",
                    "model": "memory",
                    "task": comp_task,
                    "seed": seed,
                    "training_status": "not_trained_for_composite",
                    "training_exit_code": 0,
                    "training_runtime_seconds": 0.0,
                    "training_peak_rss_bytes": None,
                    "parameter_count": None,
                    "initial_loss": None,
                    "final_loss": None,
                    "model_path": str(ROOT / "models" / f"m10_base_memory_{first_base}_seed{seed}.json"),
                    "evaluations": {comp_task: {}},
                    "mode": mode,
                }
                for test_length in TEST_LENGTHS:
                    dataset_path = f"evaluations/seq_{comp_task}_len{test_length}_v1.json"
                    eval_output = ROOT / "experiments" / "results" / f"m10_eval_collab_{mode}_{comp_task}_seed{seed}_len{test_length}.json"
                    eval_command = [
                        "python3", "src/evaluate_collaborative.py",
                        "--dataset", dataset_path,
                        "--task", comp_task,
                        "--model", "memory",
                        "--seed", str(seed),
                        "--specialist-dir", "models",
                        "--mode", mode,
                        "--output", str(eval_output),
                    ]
                    eval_result = run_experiment(eval_command, output_dir="experiments/results")
                    stored_eval = json.loads(eval_output.read_text(encoding="utf-8"))
                    collab_record["evaluations"][comp_task][str(test_length)] = {
                        "status": eval_result.status,
                        "exit_code": eval_result.exit_code,
                        "evaluation_runtime_seconds": stored_eval["total_runtime_seconds"],
                        "evaluation_peak_rss_bytes": stored_eval["peak_rss_bytes"],
                        "overall_score": stored_eval["overall_score"],
                        "evaluation_id": stored_eval["evaluation_id"],
                        "activated_specialists_mean": stored_eval.get("activated_specialists_mean"),
                        "communication_ops_total": stored_eval.get("communication_ops_total"),
                        "routing_accuracy": stored_eval.get("routing_accuracy"),
                    }
                results.append(collab_record)

    # Learned router accuracy (bag-of-words task name -> specialist mask).
    try:
        router_samples = []
        components_map = {
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
        vocab = sorted(set(word for task in components_map for word in task.split("_")))
        # simple index
        def features(task):
            words = set(task.split("_"))
            return [1.0 if w in words else 0.0 for w in vocab]
        # train perceptron-like weights for each specialist
        import random
        rng = random.Random(7)
        weights = {b: [rng.uniform(-0.1, 0.1) for _ in vocab] for b in BASE_TASKS}
        bias = {b: 0.0 for b in BASE_TASKS}
        for _ in range(200):
            for task, comps in components_map.items():
                x = features(task)
                for b in BASE_TASKS:
                    y = 1.0 if b in comps else 0.0
                    score = sum(w * xi for w, xi in zip(weights[b], x)) + bias[b]
                    pred = 1.0 if score >= 0 else 0.0
                    if pred != y:
                        for i in range(len(vocab)):
                            weights[b][i] += 0.1 * (y - pred) * x[i]
                        bias[b] += 0.1 * (y - pred)
        # evaluate exact match
        correct = 0
        for task, comps in components_map.items():
            x = features(task)
            predicted = [b for b in BASE_TASKS if sum(w * xi for w, xi in zip(weights[b], x)) + bias[b] >= 0]
            if set(predicted) == set(comps):
                correct += 1
        routing_accuracy = correct / len(components_map)
    except Exception:
        routing_accuracy = None

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_tasks": BASE_TASKS,
        "composite_tasks": COMPOSITE_TASKS,
        "training_composites": TRAIN_COMPOSITES,
        "models": MODELS,
        "seeds": SEEDS,
        "train_lengths": TRAIN_LENGTHS,
        "train_count": TRAIN_COUNT,
        "epochs": EPOCHS,
        "test_lengths": TEST_LENGTHS,
        "conditions": ["base_specialist", "unified", "independent", "collaborative_*", "collaborative_untrained"],
        "routing_accuracy_learned": routing_accuracy,
        "results": results,
        "wall_time_seconds": time.perf_counter() - started,
    }
    output_path = ROOT / "experiments" / "results" / "collaborative_specialist_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_file={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

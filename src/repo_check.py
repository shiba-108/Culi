"""Basic repository and documentation consistency checks for Model Culi."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    ".clinerules/project-rules.md",
    "docs/PROJECT_MASTER.md",
    "docs/ARCHITECTURE.md",
    "docs/TECHNICAL_CONSTRAINTS.md",
    "docs/DECISIONS.md",
    "docs/AGENT_RULES.md",
    "docs/EXPERIMENT_LOG.md",
    "docs/RESEARCH_ROADMAP.md",
    "src/memory_profile.py",
    "src/evaluation.py",
    "src/mock_model.py",
    "src/evaluate.py",
    "src/recurrent_model.py",
    "src/train_recurrent.py",
    "src/recurrent_adapter.py",
    "tests/test_recurrent_model.py",
    "tests/test_prototype_recurrent_adapter.py",
    "src/compare_models.py",
    "src/memory_model.py",
    "src/hierarchical_model.py",
    "src/memory_adapter.py",
    "src/hierarchical_adapter.py",
    "src/neural_model.py",
    "src/train_prototype.py",
    "src/train_sequential.py",
    "src/sequential_tasks.py",
    "src/sequential_adapters.py",
    "src/run_milestone6.py",
    "src/memory_adapter.py",
    "src/hierarchical_adapter.py",
    "src/prototype_adapter.py",
    "src/experiment_runner.py",
    "tests/test_neural_model.py",
    "tests/test_prototype_adapter.py",
    "evaluations/model_culi_prototype_xor_v1.json",
    "tests/test_memory_profile.py",
    "tests/test_experiment_runner.py",
    "tests/test_evaluation.py",
    "src/run_milestone7.py",
    "src/run_milestone8.py",
    "tests/test_memory_model.py",
    "tests/test_hierarchical_model.py",
    "evaluations/model_culi_eval_v1.json",
    "experiments/results/selective_memory_summary.json",
]


def check_required_paths() -> list[str]:
    failures = []
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).exists():
            failures.append(f"missing required path: {relative}")
    return failures


def check_text_contains(path: str, needle: str) -> bool:
    return needle in (ROOT / path).read_text(encoding="utf-8")


def run_checks() -> list[str]:
    failures = check_required_paths()
    checks = [
        ("README.md", "4 GB"),
        ("docs/DECISIONS.md", "D-0006"),
        ("docs/AGENT_RULES.md", "Mandatory Documentation Synchronization"),
        (".clinerules/project-rules.md", "Mandatory Documentation Synchronization"),
        ("PROJECT_HANDOVER.md", "Mandatory Documentation Synchronization"),
    ]
    for path, needle in checks:
        if not check_text_contains(path, needle):
            failures.append(f"missing expected text in {path}: {needle!r}")
    failures.extend(check_milestone8_summary())
    return failures


def check_milestone8_summary() -> list[str]:
    failures = []
    summary_path = ROOT / "experiments" / "results" / "selective_memory_summary.json"
    if not summary_path.exists():
        return [f"missing required path: experiments/results/selective_memory_summary.json"]
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid JSON in selective_memory_summary.json: {exc}"]

    required_keys = [
        "timestamp_utc",
        "tasks",
        "models",
        "seeds",
        "train_lengths",
        "train_count",
        "epochs",
        "test_lengths",
        "results",
        "wall_time_seconds",
    ]
    for key in required_keys:
        if key not in data:
            failures.append(f"selective_memory_summary.json missing key: {key}")

    models = data.get("models", [])
    tasks = data.get("tasks", [])
    seeds = data.get("seeds", [])
    test_lengths = data.get("test_lengths", [])
    results = data.get("results", [])

    expected_count = len(models) * len(tasks) * len(seeds)
    if len(results) != expected_count:
        failures.append(
            f"selective_memory_summary.json result count {len(results)} != expected {expected_count}"
        )

    for record in results:
        for field in ("model", "task", "seed", "training_status", "training_exit_code", "model_path", "evaluations"):
            if field not in record:
                failures.append(f"M8 record missing field {field!r}: {record.get('model')}/{record.get('task')}/{record.get('seed')}")
                continue
        if record.get("training_status") != "completed":
            failures.append(f"M8 training not completed: {record.get('model')}/{record.get('task')}/seed{record.get('seed')}")
        if record.get("training_exit_code") != 0:
            failures.append(f"M8 training nonzero exit: {record.get('model')}/{record.get('task')}/seed{record.get('seed')}")
        model_path = ROOT / record.get("model_path", "")
        if not model_path.exists():
            failures.append(f"M8 model artifact missing: {record.get('model_path')}")
        evaluations = record.get("evaluations", {})
        for length in test_lengths:
            key = str(length)
            if key not in evaluations:
                failures.append(f"M8 missing evaluation len{length}: {record.get('model')}/{record.get('task')}/seed{record.get('seed')}")
                continue
            evaluation = evaluations[key]
            if evaluation.get("status") != "completed":
                failures.append(f"M8 eval not completed: {record.get('model')}/{record.get('task')}/seed{record.get('seed')}/len{length}")
            if evaluation.get("exit_code") != 0:
                failures.append(f"M8 eval nonzero exit: {record.get('model')}/{record.get('task')}/seed{record.get('seed')}/len{length}")
            eval_path = ROOT / "experiments" / "results" / f"m8_eval_{record['model']}_{record['task']}_seed{record['seed']}_len{length}.json"
            if not eval_path.exists():
                failures.append(f"M8 eval artifact missing: {eval_path.relative_to(ROOT)}")
    return failures


def main() -> int:
    failures = run_checks()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: repository and documentation consistency checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

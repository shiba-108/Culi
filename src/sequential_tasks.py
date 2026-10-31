"""Synthetic sequential tasks for Model Culi Milestone 6."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def generate_parity_example(rng: random.Random, length: int) -> tuple[list[int], int]:
    bits = [rng.randint(0, 1) for _ in range(length)]
    return bits, sum(bits) % 2


def generate_delayed_recall_example(rng: random.Random, length: int) -> tuple[list[int], int]:
    bits = [rng.randint(0, 1) for _ in range(length)]
    return bits, bits[0]


def generate_dependency_example(rng: random.Random, length: int) -> tuple[list[int], int]:
    bits = [rng.randint(0, 1) for _ in range(length)]
    return bits, bits[0] ^ bits[-1]

def generate_multi_recall_example(rng: random.Random, length: int) -> tuple[list[int], int]:
    bits = [rng.randint(0, 1) for _ in range(length)]
    middle = length // 2
    return bits, bits[0] ^ bits[middle]


TASK_GENERATORS = {
    "parity": generate_parity_example,
    "delayed_recall": generate_delayed_recall_example,
    "dependency": generate_dependency_example,
    "multi_recall": generate_multi_recall_example,
}


def make_examples(task: str, count: int, length: int, seed: int) -> list[tuple[list[int], int]]:
    if task not in TASK_GENERATORS:
        raise ValueError(f"unknown task: {task}")
    rng = random.Random(seed)
    return [TASK_GENERATORS[task](rng, length) for _ in range(count)]


def make_dataset(task: str, examples: list[tuple[list[int], int]], dataset_id: str, version: str = "1.0") -> dict[str, Any]:
    cases = []
    for index, (bits, label) in enumerate(examples):
        cases.append(
            {
                "id": f"{task}-{index:04d}",
                "category": "reasoning",
                "input": " ".join(str(bit) for bit in bits),
                "expected": str(label),
                "scoring": "exact",
                "metadata": {
                    "task": task,
                    "length": len(bits),
                    "synthetic": True,
                },
            }
        )
    return {"dataset_id": dataset_id, "version": version, "cases": cases}


def save_dataset(path: str | Path, dataset: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

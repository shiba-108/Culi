"""Controlled comparison of experimental neural organizations."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_runner import run_experiment

SEEDS = [1, 2, 3]
EPOCHS = 5000


def _parse(stdout: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}=(.+)$", stdout, re.MULTILINE)
    return match.group(1).strip() if match else None


def _train(model_kind: str, seed: int) -> dict:
    if model_kind == "feedforward":
        output = ROOT / "models" / f"prototype_xor_seed{seed}.json"
        command = [
            "python3", "src/train_prototype.py",
            "--epochs", str(EPOCHS),
            "--seed", str(seed),
            "--output", str(output.relative_to(ROOT)),
        ]
        dataset = "evaluations/model_culi_prototype_xor_v1.json"
        adapter = "prototype_adapter:PrototypeModelAdapter"
    else:
        output = ROOT / "models" / f"prototype_recurrent_xor_seed{seed}.json"
        command = [
            "python3", "src/train_recurrent.py",
            "--epochs", str(EPOCHS),
            "--seed", str(seed),
            "--output", str(output.relative_to(ROOT)),
        ]
        dataset = "evaluations/model_culi_prototype_xor_v1.json"
        adapter = "recurrent_adapter:RecurrentModelAdapter"

    result = run_experiment(command, output_dir="experiments/results")
    parsed = {
        "model_kind": model_kind,
        "seed": seed,
        "training_status": result.status,
        "training_exit_code": result.exit_code,
        "training_runtime_seconds": result.runtime_seconds,
        "training_peak_rss_bytes": result.peak_rss_bytes,
        "parameter_count": int(_parse(result.stdout, "parameter_count")) if _parse(result.stdout, "parameter_count") else None,
        "initial_loss": float(_parse(result.stdout, "initial_loss")) if _parse(result.stdout, "initial_loss") else None,
        "final_loss": float(_parse(result.stdout, "final_loss")) if _parse(result.stdout, "final_loss") else None,
        "parameters_changed": _parse(result.stdout, "parameters_changed") == "True",
        "model_path": str(output),
    }

    eval_output = ROOT / "experiments" / "results" / f"comparison_{model_kind}_seed{seed}.json"
    eval_command = [
        "python3", "src/evaluate.py",
        "--dataset", dataset,
        "--adapter", adapter,
        "--model-path", str(output),
        "--output", str(eval_output),
    ]
    eval_result = run_experiment(eval_command, output_dir="experiments/results")
    stored_eval = json.loads(eval_output.read_text(encoding="utf-8"))
    parsed.update(
        {
            "evaluation_status": eval_result.status,
            "evaluation_exit_code": eval_result.exit_code,
            "inference_runtime_seconds": stored_eval["total_runtime_seconds"],
            "inference_peak_rss_bytes": stored_eval["peak_rss_bytes"],
            "overall_score": stored_eval["overall_score"],
            "evaluation_id": stored_eval["evaluation_id"],
        }
    )
    return parsed


def main() -> int:
    started = time.perf_counter()
    results = []
    for model_kind in ("feedforward", "recurrent"):
        for seed in SEEDS:
            print(f"running {model_kind} seed={seed}", flush=True)
            results.append(_train(model_kind, seed))

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "results": results,
        "wall_time_seconds": time.perf_counter() - started,
    }
    output_path = ROOT / "experiments" / "results" / "model_comparison_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_file={output_path}")

    # Print a compact comparison.
    for model_kind in ("feedforward", "recurrent"):
        subset = [item for item in results if item["model_kind"] == model_kind]
        print(f"\n{model_kind}:")
        print(f"  parameters: {subset[0]['parameter_count']}")
        print(f"  final_loss_mean: {sum(item['final_loss'] for item in subset) / len(subset):.6f}")
        print(f"  overall_score_mean: {sum(item['overall_score'] for item in subset) / len(subset):.6f}")
        print(f"  training_runtime_mean: {sum(item['training_runtime_seconds'] for item in subset) / len(subset):.6f}")
        print(f"  inference_runtime_mean: {sum(item['inference_runtime_seconds'] for item in subset) / len(subset):.6f}")
        print(f"  training_peak_rss_max: {max(item['training_peak_rss_bytes'] for item in subset)}")
        print(f"  inference_peak_rss_max: {max(item['inference_peak_rss_bytes'] for item in subset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

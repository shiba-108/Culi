"""Train a selected model on a synthetic sequential task."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_model import NeuralNetwork
from recurrent_model import RecurrentNetwork
from memory_model import GatedMemoryNetwork
from hierarchical_model import HierarchicalStateNetwork
from sequential_tasks import make_examples


def build_model(model_kind: str, train_length: int, hidden_size: int, learning_rate: float, seed: int):
    if model_kind == "feedforward":
        return NeuralNetwork(layer_sizes=[train_length, 16, 1], learning_rate=learning_rate, seed=seed)
    if model_kind == "recurrent":
        return RecurrentNetwork(input_size=1, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate, seed=seed)
    if model_kind == "recurrent_bounded":
        return RecurrentNetwork(input_size=1, hidden_size=2, output_size=1, learning_rate=learning_rate, seed=seed)
    if model_kind == "memory":
        return GatedMemoryNetwork(input_size=1, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate, seed=seed)
    if model_kind == "hierarchical":
        return HierarchicalStateNetwork(input_size=1, state_size=hidden_size, output_size=1, learning_rate=learning_rate, seed=seed)
    raise ValueError(f"unknown model kind: {model_kind}")


def train_model(model, model_kind: str, samples: list[tuple[list, int]], epochs: int):
    if model_kind == "feedforward":
        feedforward_samples = [([float(bit) for bit in bits], [float(label)]) for bits, label in samples]
        return model.train(feedforward_samples, epochs=epochs)
    recurrent_samples = [([[float(bit)] for bit in bits], [float(label)]) for bits, label in samples]
    if model_kind in ("recurrent", "recurrent_bounded", "memory", "hierarchical"):
        return model.train(recurrent_samples, epochs=epochs)
    raise ValueError(f"unknown model kind: {model_kind}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train a model on a synthetic sequential task")
    parser.add_argument("--model", required=True, choices=["feedforward", "recurrent", "recurrent_bounded", "memory", "hierarchical"])
    parser.add_argument("--task", required=True, choices=["parity", "delayed_recall", "dependency", "multi_recall", "recall_last"])
    parser.add_argument("--train-length", type=int, default=6)
    parser.add_argument("--train-lengths", default=None, help="Comma-separated training lengths, e.g. 4,6,8")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.train_lengths:
        lengths = [int(value) for value in args.train_lengths.split(",") if value]
        per_length = max(1, args.train_count // len(lengths))
        samples = []
        for length in lengths:
            samples.extend(make_examples(args.task, per_length, length, args.seed + length))
    else:
        samples = make_examples(args.task, args.train_count, args.train_length, args.seed)
    model = build_model(args.model, args.train_length, args.hidden_size, args.learning_rate, args.seed)
    before_signature = model.parameter_signature()
    start = time.perf_counter()
    result = train_model(model, args.model, samples, args.epochs)
    runtime = time.perf_counter() - start
    after_signature = model.parameter_signature()

    output_path = ROOT / args.output
    model.save(output_path)

    import resource

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(f"model={args.model}")
    print(f"task={args.task}")
    print(f"train_length={args.train_length}")
    print(f"train_count={len(samples)}")
    print(f"train_lengths={sorted(set(len(bits) for bits, _ in samples))}")
    print(f"epochs={result.epochs}")
    print(f"parameter_count={result.parameter_count}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")
    print(f"parameters_changed={before_signature != after_signature}")
    print(f"runtime_seconds={runtime:.6f}")
    print(f"peak_rss_bytes={peak_rss}")
    print(f"model_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train the experimental recurrent prototype on a small controlled dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recurrent_model import RecurrentNetwork

XOR_SAMPLES = [
    ([[0.0], [0.0]], [0.0]),
    ([[0.0], [1.0]], [1.0]),
    ([[1.0], [0.0]], [1.0]),
    ([[1.0], [1.0]], [0.0]),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the experimental recurrent prototype")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="models/prototype_recurrent_xor.json")
    args = parser.parse_args(argv)

    model = RecurrentNetwork(
        input_size=1,
        hidden_size=args.hidden_size,
        output_size=1,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    before_signature = model.parameter_signature()
    start = time.perf_counter()
    result = model.train(XOR_SAMPLES, epochs=args.epochs)
    runtime = time.perf_counter() - start
    after_signature = model.parameter_signature()

    output_path = ROOT / args.output
    model.save(output_path)

    import resource

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(f"epochs={result.epochs}")
    print(f"samples={result.samples}")
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

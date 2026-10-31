"""A small experimental recurrent/state-based neural prototype.

This model processes inputs as a sequence and maintains a hidden state. It is
an experimental candidate, not a final architecture.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _tanh(value: float) -> float:
    return math.tanh(value)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass
class TrainingResult:
    initial_loss: float
    final_loss: float
    epochs: int
    samples: int
    parameter_count: int


class RecurrentNetwork:
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 4,
        output_size: int = 1,
        learning_rate: float = 0.1,
        seed: int = 0,
    ):
        if input_size < 1 or hidden_size < 1 or output_size < 1:
            raise ValueError("sizes must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)

        scale = 1.0 / math.sqrt(hidden_size)
        self.w_xh = [[self.rng.uniform(-scale, scale) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w_hh = [[self.rng.uniform(-scale, scale) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b_h = [0.0 for _ in range(hidden_size)]
        self.w_hy = [[self.rng.uniform(-scale, scale) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b_y = [0.0 for _ in range(output_size)]

    @property
    def parameter_count(self) -> int:
        total = 0
        total += sum(len(row) for row in self.w_xh)
        total += sum(len(row) for row in self.w_hh)
        total += len(self.b_h)
        total += sum(len(row) for row in self.w_hy)
        total += len(self.b_y)
        return total

    def _forward_sequence(self, sequence: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[float]]:
        hidden_states = [[0.0 for _ in range(self.hidden_size)]]
        preactivations = []
        for time_step in sequence:
            if len(time_step) != self.input_size:
                raise ValueError(f"expected {self.input_size} inputs, received {len(time_step)}")
            z = [
                sum(self.w_xh[hidden_index][input_index] * time_step[input_index]
                    for input_index in range(self.input_size))
                + sum(self.w_hh[hidden_index][previous_index] * hidden_states[-1][previous_index]
                      for previous_index in range(self.hidden_size))
                + self.b_h[hidden_index]
                for hidden_index in range(self.hidden_size)
            ]
            preactivations.append(z)
            hidden = [_tanh(value) for value in z]
            hidden_states.append(hidden)
        output_preactivation = [
            sum(self.w_hy[output_index][hidden_index] * hidden_states[-1][hidden_index]
                for hidden_index in range(self.hidden_size))
            + self.b_y[output_index]
            for output_index in range(self.output_size)
        ]
        output = [_sigmoid(value) for value in output_preactivation]
        return hidden_states, preactivations, output

    def predict_sequence(self, sequence: list[list[float]]) -> list[float]:
        _, _, output = self._forward_sequence(sequence)
        return output

    def train(
        self,
        samples: list[tuple[list[list[float]], list[float]]],
        epochs: int,
    ) -> TrainingResult:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if not samples:
            raise ValueError("samples must not be empty")

        initial_predictions = [self.predict_sequence(sequence) for sequence, _ in samples]
        initial_loss = sum(
            sum((prediction[i] - target[i]) ** 2 for i in range(len(target))) / len(target)
            for prediction, (_, target) in zip(initial_predictions, samples)
        ) / len(samples)

        for _ in range(epochs):
            for sequence, target in samples:
                hidden_states, preactivations, output = self._forward_sequence(sequence)
                time_steps = len(sequence)

                d_output = [2.0 * (output[i] - target[i]) * output[i] * (1.0 - output[i]) for i in range(self.output_size)]

                grad_w_hy = [[d_output[output_index] * hidden_states[-1][hidden_index]
                              for hidden_index in range(self.hidden_size)]
                             for output_index in range(self.output_size)]
                grad_b_y = [d_output[output_index] for output_index in range(self.output_size)]

                grad_w_xh = [[0.0 for _ in range(self.input_size)] for _ in range(self.hidden_size)]
                grad_w_hh = [[0.0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
                grad_b_h = [0.0 for _ in range(self.hidden_size)]

                delta_hidden = [0.0 for _ in range(self.hidden_size)]
                for output_index in range(self.output_size):
                    for hidden_index in range(self.hidden_size):
                        delta_hidden[hidden_index] += d_output[output_index] * self.w_hy[output_index][hidden_index]

                for time_index in range(time_steps - 1, -1, -1):
                    z = preactivations[time_index]
                    tanh_derivative = [1.0 - hidden_states[time_index + 1][h] ** 2 for h in range(self.hidden_size)]
                    delta_z = [delta_hidden[h] * tanh_derivative[h] for h in range(self.hidden_size)]
                    previous_hidden = hidden_states[time_index]
                    inputs = sequence[time_index]
                    for hidden_index in range(self.hidden_size):
                        for input_index in range(self.input_size):
                            grad_w_xh[hidden_index][input_index] += delta_z[hidden_index] * inputs[input_index]
                        for previous_index in range(self.hidden_size):
                            grad_w_hh[hidden_index][previous_index] += delta_z[hidden_index] * previous_hidden[previous_index]
                        grad_b_h[hidden_index] += delta_z[hidden_index]

                    if time_index > 0:
                        next_delta_hidden = [0.0 for _ in range(self.hidden_size)]
                        for hidden_index in range(self.hidden_size):
                            for previous_index in range(self.hidden_size):
                                next_delta_hidden[previous_index] += delta_z[hidden_index] * self.w_hh[hidden_index][previous_index]
                        delta_hidden = next_delta_hidden

                for output_index in range(self.output_size):
                    for hidden_index in range(self.hidden_size):
                        self.w_hy[output_index][hidden_index] -= self.learning_rate * grad_w_hy[output_index][hidden_index]
                    self.b_y[output_index] -= self.learning_rate * grad_b_y[output_index]

                for hidden_index in range(self.hidden_size):
                    for input_index in range(self.input_size):
                        self.w_xh[hidden_index][input_index] -= self.learning_rate * grad_w_xh[hidden_index][input_index]
                    for previous_index in range(self.hidden_size):
                        self.w_hh[hidden_index][previous_index] -= self.learning_rate * grad_w_hh[hidden_index][previous_index]
                    self.b_h[hidden_index] -= self.learning_rate * grad_b_h[hidden_index]

        final_predictions = [self.predict_sequence(sequence) for sequence, _ in samples]
        final_loss = sum(
            sum((prediction[i] - target[i]) ** 2 for i in range(len(target))) / len(target)
            for prediction, (_, target) in zip(final_predictions, samples)
        ) / len(samples)

        return TrainingResult(
            initial_loss=initial_loss,
            final_loss=final_loss,
            epochs=epochs,
            samples=len(samples),
            parameter_count=self.parameter_count,
        )

    def parameter_signature(self) -> str:
        values = []
        for matrix in (self.w_xh, self.w_hh, self.w_hy):
            for row in matrix:
                values.extend(row)
        values.extend(self.b_h)
        values.extend(self.b_y)
        return ",".join(repr(value) for value in values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "w_xh": self.w_xh,
            "w_hh": self.w_hh,
            "b_h": self.b_h,
            "w_hy": self.w_hy,
            "b_y": self.b_y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecurrentNetwork":
        network = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            output_size=data["output_size"],
            learning_rate=data["learning_rate"],
            seed=0,
        )
        network.w_xh = data["w_xh"]
        network.w_hh = data["w_hh"]
        network.b_h = data["b_h"]
        network.w_hy = data["w_hy"]
        network.b_y = data["b_y"]
        return network

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RecurrentNetwork":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

"""A small experimental feedforward neural network prototype.

This is an experimental candidate, not a final architecture. It uses only the
Python standard library and manual gradients so that training and parameter
changes can be observed in a controlled way.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _sigmoid_derivative(value: float) -> float:
    sigmoid_value = _sigmoid(value)
    return sigmoid_value * (1.0 - sigmoid_value)


@dataclass
class TrainingResult:
    initial_loss: float
    final_loss: float
    epochs: int
    samples: int
    parameter_count: int


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.1,
        seed: int = 0,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")
        if any(size < 1 for size in layer_sizes):
            raise ValueError("layer sizes must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.layer_sizes = list(layer_sizes)
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        for input_size, output_size in zip(layer_sizes, layer_sizes[1:]):
            scale = 1.0 / math.sqrt(input_size)
            weights = [
                [self.rng.uniform(-scale, scale) for _ in range(input_size)]
                for _ in range(output_size)
            ]
            biases = [0.0 for _ in range(output_size)]
            self.weights.append(weights)
            self.biases.append(biases)

    @property
    def parameter_count(self) -> int:
        total = 0
        for weights, biases in zip(self.weights, self.biases):
            total += sum(len(row) for row in weights) + len(biases)
        return total

    def _forward(self, inputs: list[float]) -> tuple[list[list[float]], list[list[float]]]:
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError(
                f"expected {self.layer_sizes[0]} inputs, received {len(inputs)}"
            )
        activations = [list(inputs)]
        preactivations: list[list[float]] = []
        for weights, biases in zip(self.weights, self.biases):
            z = [
                sum(weights[output_index][input_index] * activations[-1][input_index]
                    for input_index in range(len(activations[-1])))
                + biases[output_index]
                for output_index in range(len(weights))
            ]
            preactivations.append(z)
            activations.append([_sigmoid(value) for value in z])
        return activations, preactivations

    def predict(self, inputs: list[float]) -> list[float]:
        activations, _ = self._forward(inputs)
        return activations[-1]

    def _mean_squared_error(self, prediction: list[float], target: list[float]) -> float:
        if len(prediction) != len(target):
            raise ValueError("prediction and target sizes do not match")
        return sum((prediction[i] - target[i]) ** 2 for i in range(len(prediction))) / len(prediction)

    def train(
        self,
        samples: list[tuple[list[float], list[float]]],
        epochs: int,
        ) -> TrainingResult:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if not samples:
            raise ValueError("samples must not be empty")

        initial_predictions = [self.predict(inputs) for inputs, _ in samples]
        initial_loss = sum(
            self._mean_squared_error(prediction, target)
            for prediction, (_, target) in zip(initial_predictions, samples)
        ) / len(samples)

        for _ in range(epochs):
            for inputs, target in samples:
                activations, preactivations = self._forward(inputs)
                output = activations[-1]
                deltas = [
                    (output[i] - target[i]) * _sigmoid_derivative(preactivations[-1][i])
                    for i in range(len(output))
                ]

                weight_gradients = [None] * len(self.weights)
                bias_gradients = [None] * len(self.biases)
                layer_deltas = [None] * len(self.weights)
                layer_deltas[-1] = deltas

                for layer_index in range(len(self.weights) - 1, -1, -1):
                    current_deltas = layer_deltas[layer_index]
                    previous_activation = activations[layer_index]
                    gradients = [
                        [current_deltas[output_index] * previous_activation[input_index]
                         for input_index in range(len(previous_activation))]
                        for output_index in range(len(current_deltas))
                    ]
                    biases = [current_deltas[output_index] for output_index in range(len(current_deltas))]
                    weight_gradients[layer_index] = gradients
                    bias_gradients[layer_index] = biases

                    if layer_index > 0:
                        previous_deltas = [
                            sum(
                                current_deltas[output_index]
                                * self.weights[layer_index][output_index][input_index]
                                for output_index in range(len(current_deltas))
                            )
                            * _sigmoid_derivative(preactivations[layer_index - 1][input_index])
                            for input_index in range(len(activations[layer_index]))
                        ]
                        layer_deltas[layer_index - 1] = previous_deltas

                for layer_index in range(len(self.weights)):
                    for output_index in range(len(self.weights[layer_index])):
                        for input_index in range(len(self.weights[layer_index][output_index])):
                            self.weights[layer_index][output_index][input_index] -= (
                                self.learning_rate
                                * weight_gradients[layer_index][output_index][input_index]
                            )
                        self.biases[layer_index][output_index] -= (
                            self.learning_rate * bias_gradients[layer_index][output_index]
                        )

        final_predictions = [self.predict(inputs) for inputs, _ in samples]
        final_loss = sum(
            self._mean_squared_error(prediction, target)
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
        """Return a deterministic signature of all parameters."""
        values = []
        for weights, biases in zip(self.weights, self.biases):
            for row in weights:
                values.extend(row)
            values.extend(biases)
        return ",".join(repr(value) for value in values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuralNetwork":
        network = cls(
            layer_sizes=data["layer_sizes"],
            learning_rate=data["learning_rate"],
            seed=0,
        )
        network.weights = data["weights"]
        network.biases = data["biases"]
        return network

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

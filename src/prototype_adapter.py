"""Evaluation adapter for the experimental neural prototype."""

from __future__ import annotations

from pathlib import Path

from neural_model import NeuralNetwork


class PrototypeModelAdapter:
    name = "prototype-xor-model"
    version = "0.0.1-experimental"

    def __init__(self, model_path: str | Path):
        self.model = NeuralNetwork.load(model_path)

    def generate(self, prompt: str) -> str:
        parts = prompt.split()
        if len(parts) != 2:
            return ""
        try:
            inputs = [float(parts[0]), float(parts[1])]
        except ValueError:
            return ""
        prediction = self.model.predict(inputs)[0]
        return "1" if prediction >= 0.5 else "0"

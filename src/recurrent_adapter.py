"""Evaluation adapter for the experimental recurrent prototype."""

from __future__ import annotations

from pathlib import Path

from recurrent_model import RecurrentNetwork


class RecurrentModelAdapter:
    name = "prototype-recurrent-model"
    version = "0.0.1-experimental"

    def __init__(self, model_path: str | Path):
        self.model = RecurrentNetwork.load(model_path)

    def generate(self, prompt: str) -> str:
        parts = prompt.split()
        if len(parts) != 2:
            return ""
        try:
            sequence = [[float(parts[0])], [float(parts[1])]]
        except ValueError:
            return ""
        prediction = self.model.predict_sequence(sequence)[0]
        return "1" if prediction >= 0.5 else "0"

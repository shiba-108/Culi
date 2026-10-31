"""Evaluation adapter for the experimental hierarchical prototype."""

from __future__ import annotations

from pathlib import Path

from hierarchical_model import HierarchicalStateNetwork


class HierarchicalModelAdapter:
    name = "prototype-hierarchical-model"
    version = "0.0.1-experimental"

    def __init__(self, model_path: str | Path):
        self.model = HierarchicalStateNetwork.load(model_path)

    def generate(self, prompt: str) -> str:
        parts = prompt.split()
        try:
            sequence = [[float(part)] for part in parts]
        except ValueError:
            return ""
        if not sequence:
            return ""
        prediction = self.model.predict_sequence(sequence)[0]
        return "1" if prediction >= 0.5 else "0"

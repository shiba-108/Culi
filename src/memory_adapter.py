"""Evaluation adapter for the experimental gated-memory prototype."""

from __future__ import annotations

from pathlib import Path

from memory_model import GatedMemoryNetwork


class MemoryModelAdapter:
    name = "prototype-memory-model"
    version = "0.0.1-experimental"

    def __init__(self, model_path: str | Path):
        self.model = GatedMemoryNetwork.load(model_path)

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

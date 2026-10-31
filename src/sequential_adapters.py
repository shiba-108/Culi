"""Evaluation adapters for sequential-task prototypes."""

from __future__ import annotations

from pathlib import Path

from neural_model import NeuralNetwork
from recurrent_model import RecurrentNetwork


class SequentialFeedforwardAdapter:
    def __init__(self, model_path: str | Path, train_length: int = 6):
        self.model = NeuralNetwork.load(model_path)
        self.train_length = train_length
        self.name = "sequential-feedforward-model"
        self.version = "0.0.1-experimental"

    def generate(self, prompt: str) -> str:
        parts = prompt.split()
        try:
            bits = [float(part) for part in parts]
        except ValueError:
            return ""
        if len(bits) < self.train_length:
            bits = bits + [0.0] * (self.train_length - len(bits))
        elif len(bits) > self.train_length:
            bits = bits[-self.train_length:]
        prediction = self.model.predict(bits)[0]
        return "1" if prediction >= 0.5 else "0"


class SequentialRecurrentAdapter:
    def __init__(self, model_path: str | Path):
        self.model = RecurrentNetwork.load(model_path)
        self.name = "sequential-recurrent-model"
        self.version = "0.0.1-experimental"

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


class SequentialBoundedAdapter(SequentialRecurrentAdapter):
    name = "sequential-bounded-recurrent-model"


class SequentialMemoryAdapter:
    def __init__(self, model_path: str | Path):
        from memory_model import GatedMemoryNetwork

        self.model = GatedMemoryNetwork.load(model_path)
        self.name = "sequential-memory-model"
        self.version = "0.0.1-experimental"

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


class SequentialHierarchicalAdapter:
    def __init__(self, model_path: str | Path):
        from hierarchical_model import HierarchicalStateNetwork

        self.model = HierarchicalStateNetwork.load(model_path)
        self.name = "sequential-hierarchical-model"
        self.version = "0.0.1-experimental"

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

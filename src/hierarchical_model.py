"""A small experimental hierarchical-state neural prototype."""

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


def _tanh(value: float) -> float:
    return math.tanh(value)


@dataclass
class TrainingResult:
    initial_loss: float
    final_loss: float
    epochs: int
    samples: int
    parameter_count: int


class HierarchicalStateNetwork:
    def __init__(self, input_size: int = 1, state_size: int = 4, output_size: int = 1, learning_rate: float = 0.1, seed: int = 0):
        if input_size < 1 or state_size < 1 or output_size < 1:
            raise ValueError("sizes must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)
        s = 1.0 / math.sqrt(state_size)
        self.w_xh = [[self.rng.uniform(-s, s) for _ in range(input_size)] for _ in range(state_size)]
        self.w_hh = [[self.rng.uniform(-s, s) for _ in range(state_size)] for _ in range(state_size)]
        self.w_hs = [[self.rng.uniform(-s, s) for _ in range(state_size)] for _ in range(state_size)]
        self.b_h = [0.0 for _ in range(state_size)]
        self.w_xg = [[self.rng.uniform(-s, s) for _ in range(input_size)] for _ in range(state_size)]
        self.w_hg = [[self.rng.uniform(-s, s) for _ in range(state_size)] for _ in range(state_size)]
        self.w_sg = [[self.rng.uniform(-s, s) for _ in range(state_size)] for _ in range(state_size)]
        self.b_g = [0.0 for _ in range(state_size)]
        self.w_oh = [[self.rng.uniform(-s, s) for _ in range(state_size)] for _ in range(output_size)]
        self.w_os = [[self.rng.uniform(-s, s) for _ in range(state_size)] for _ in range(output_size)]
        self.b_o = [0.0 for _ in range(output_size)]

    @property
    def parameter_count(self) -> int:
        matrices = [self.w_xh, self.w_hh, self.w_hs, self.w_xg, self.w_hg, self.w_sg, self.w_oh, self.w_os]
        return sum(len(m) * len(m[0]) for m in matrices) + len(self.b_h) + len(self.b_g) + len(self.b_o)

    def _forward(self, sequence):
        h = [0.0 for _ in range(self.state_size)]
        s = [0.0 for _ in range(self.state_size)]
        states = []
        for x in sequence:
            z_h = [sum(self.w_xh[i][j] * x[j] for j in range(self.input_size)) + sum(self.w_hh[i][j] * h[j] for j in range(self.state_size)) + sum(self.w_hs[i][j] * s[j] for j in range(self.state_size)) + self.b_h[i] for i in range(self.state_size)]
            h_new = [_tanh(v) for v in z_h]
            z_g = [sum(self.w_xg[i][j] * x[j] for j in range(self.input_size)) + sum(self.w_hg[i][j] * h[j] for j in range(self.state_size)) + sum(self.w_sg[i][j] * s[j] for j in range(self.state_size)) + self.b_g[i] for i in range(self.state_size)]
            g = [_sigmoid(v) for v in z_g]
            s_new = [(1.0 - g[i]) * s[i] + g[i] * h_new[i] for i in range(self.state_size)]
            states.append((list(x), list(h), list(s), z_h, h_new, z_g, g, s_new))
            h, s = h_new, s_new
        z_o = [sum(self.w_oh[o][i] * h[i] for i in range(self.state_size)) + sum(self.w_os[o][i] * s[i] for i in range(self.state_size)) + self.b_o[o] for o in range(self.output_size)]
        output = [_sigmoid(v) for v in z_o]
        return output, states

    def predict_sequence(self, sequence):
        output, _ = self._forward(sequence)
        return output

    def train(self, samples, epochs):
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if not samples:
            raise ValueError("samples must not be empty")
        initial_predictions = [self.predict_sequence(seq) for seq, _ in samples]
        initial_loss = sum(sum((p[i]-t[i])**2 for i in range(len(t)))/len(t) for p, (_, t) in zip(initial_predictions, samples)) / len(samples)
        H = self.state_size
        I = self.input_size
        O = self.output_size
        for _ in range(epochs):
            for sequence, target in samples:
                output, states = self._forward(sequence)
                T = len(sequence)
                d_o = [2.0*(output[o]-target[o])*output[o]*(1-output[o]) for o in range(O)]
                # gradients
                gW_xh = [[0.0 for _ in range(I)] for _ in range(H)]
                gW_hh = [[0.0 for _ in range(H)] for _ in range(H)]
                gW_hs = [[0.0 for _ in range(H)] for _ in range(H)]
                gb_h = [0.0 for _ in range(H)]
                gW_xg = [[0.0 for _ in range(I)] for _ in range(H)]
                gW_hg = [[0.0 for _ in range(H)] for _ in range(H)]
                gW_sg = [[0.0 for _ in range(H)] for _ in range(H)]
                gb_g = [0.0 for _ in range(H)]
                gW_oh = [[0.0 for _ in range(H)] for _ in range(O)]
                gW_os = [[0.0 for _ in range(H)] for _ in range(O)]
                gb_o = [0.0 for _ in range(O)]

                for o in range(O):
                    for i in range(H):
                        gW_oh[o][i] += d_o[o]*states[-1][4][i]
                        gW_os[o][i] += d_o[o]*states[-1][7][i]
                    gb_o[o] += d_o[o]

                grad_h = [sum(self.w_oh[o][i]*d_o[o] for o in range(O)) for i in range(H)]
                grad_s = [sum(self.w_os[o][i]*d_o[o] for o in range(O)) for i in range(H)]

                for t in range(T-1, -1, -1):
                    x, h_prev, s_prev, z_h, h, z_g, g, s = states[t]
                    # grad_h total includes from s_t dependency
                    grad_h_total = [grad_h[i] + grad_s[i]*g[i] for i in range(H)]
                    dz_h = [grad_h_total[i]*(1-h[i]**2) for i in range(H)]
                    # gate gradient
                    grad_g = [grad_s[i]*(h[i]-s_prev[i]) for i in range(H)]
                    dz_g = [grad_g[i]*g[i]*(1-g[i]) for i in range(H)]
                    # accumulate z_h params
                    for i in range(H):
                        for j in range(I):
                            gW_xh[i][j] += dz_h[i]*x[j]
                        for j in range(H):
                            gW_hh[i][j] += dz_h[i]*h_prev[j]
                            gW_hs[i][j] += dz_h[i]*s_prev[j]
                        gb_h[i] += dz_h[i]
                    # accumulate gate params
                    for i in range(H):
                        for j in range(I):
                            gW_xg[i][j] += dz_g[i]*x[j]
                        for j in range(H):
                            gW_hg[i][j] += dz_g[i]*h_prev[j]
                            gW_sg[i][j] += dz_g[i]*s_prev[j]
                        gb_g[i] += dz_g[i]
                    # propagate to previous states
                    grad_h_prev = [0.0 for _ in range(H)]
                    grad_s_prev = [0.0 for _ in range(H)]
                    for i in range(H):
                        for j in range(H):
                            grad_h_prev[j] += dz_h[i]*self.w_hh[i][j]
                            grad_h_prev[j] += dz_g[i]*self.w_hg[i][j]
                            grad_s_prev[j] += dz_h[i]*self.w_hs[i][j]
                            grad_s_prev[j] += dz_g[i]*self.w_sg[i][j]
                    # s_t = (1-g)*s_prev + g*h, so s_prev also gets (1-g)*grad_s[t]
                    for i in range(H):
                        grad_s_prev[i] += grad_s[i]*(1-g[i])
                    grad_h = grad_h_prev
                    grad_s = grad_s_prev

                # update
                for i in range(H):
                    for j in range(I):
                        self.w_xh[i][j] -= self.learning_rate*gW_xh[i][j]
                        self.w_xg[i][j] -= self.learning_rate*gW_xg[i][j]
                    for j in range(H):
                        self.w_hh[i][j] -= self.learning_rate*gW_hh[i][j]
                        self.w_hs[i][j] -= self.learning_rate*gW_hs[i][j]
                        self.w_hg[i][j] -= self.learning_rate*gW_hg[i][j]
                        self.w_sg[i][j] -= self.learning_rate*gW_sg[i][j]
                    self.b_h[i] -= self.learning_rate*gb_h[i]
                    self.b_g[i] -= self.learning_rate*gb_g[i]
                for o in range(O):
                    for i in range(H):
                        self.w_oh[o][i] -= self.learning_rate*gW_oh[o][i]
                        self.w_os[o][i] -= self.learning_rate*gW_os[o][i]
                    self.b_o[o] -= self.learning_rate*gb_o[o]

        final_predictions = [self.predict_sequence(seq) for seq, _ in samples]
        final_loss = sum(sum((p[i]-t[i])**2 for i in range(len(t)))/len(t) for p, (_, t) in zip(final_predictions, samples)) / len(samples)
        return TrainingResult(initial_loss=initial_loss, final_loss=final_loss, epochs=epochs, samples=len(samples), parameter_count=self.parameter_count)

    def parameter_signature(self) -> str:
        vals = []
        for m in (self.w_xh, self.w_hh, self.w_hs, self.w_xg, self.w_hg, self.w_sg, self.w_oh, self.w_os):
            for row in m:
                vals.extend(row)
        vals.extend(self.b_h + self.b_g + self.b_o)
        return ",".join(repr(v) for v in vals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "state_size": self.state_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "w_xh": self.w_xh, "w_hh": self.w_hh, "w_hs": self.w_hs, "b_h": self.b_h,
            "w_xg": self.w_xg, "w_hg": self.w_hg, "w_sg": self.w_sg, "b_g": self.b_g,
            "w_oh": self.w_oh, "w_os": self.w_os, "b_o": self.b_o,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalStateNetwork":
        net = cls(data["input_size"], data["state_size"], data["output_size"], data["learning_rate"], 0)
        for key in ("w_xh", "w_hh", "w_hs", "b_h", "w_xg", "w_hg", "w_sg", "b_g", "w_oh", "w_os", "b_o"):
            setattr(net, key, data[key])
        return net

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalStateNetwork":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

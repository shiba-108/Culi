"""A small experimental gated-memory neural prototype."""

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


class GatedMemoryNetwork:
    def __init__(self, input_size: int = 1, hidden_size: int = 4, output_size: int = 1, learning_rate: float = 0.1, seed: int = 0):
        if input_size < 1 or hidden_size < 1 or output_size < 1:
            raise ValueError("sizes must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.rng = random.Random(seed)
        s = 1.0 / math.sqrt(hidden_size)
        self.w_xh = [[self.rng.uniform(-s, s) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w_hh = [[self.rng.uniform(-s, s) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b_h = [0.0 for _ in range(hidden_size)]
        self.w_xc = [[self.rng.uniform(-s, s) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w_hc = [[self.rng.uniform(-s, s) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b_c = [0.0 for _ in range(hidden_size)]
        self.w_xg = [[self.rng.uniform(-s, s) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w_hg = [[self.rng.uniform(-s, s) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b_g = [0.0 for _ in range(hidden_size)]
        self.w_mh = [[self.rng.uniform(-s, s) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.w_oh = [[self.rng.uniform(-s, s) for _ in range(hidden_size)] for _ in range(output_size)]
        self.w_om = [[self.rng.uniform(-s, s) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b_o = [0.0 for _ in range(output_size)]

    @property
    def parameter_count(self) -> int:
        matrices = [self.w_xh, self.w_hh, self.w_xc, self.w_hc, self.w_xg, self.w_hg, self.w_mh, self.w_oh, self.w_om]
        return sum(len(m) * len(m[0]) for m in matrices) + len(self.b_h) + len(self.b_c) + len(self.b_g) + len(self.b_o)

    def _forward(self, sequence):
        h = [0.0 for _ in range(self.hidden_size)]
        m = [0.0 for _ in range(self.hidden_size)]
        states = []
        for x in sequence:
            z_c = [sum(self.w_xc[i][j] * x[j] for j in range(self.input_size)) + sum(self.w_hc[i][j] * h[j] for j in range(self.hidden_size)) + self.b_c[i] for i in range(self.hidden_size)]
            c = [_tanh(v) for v in z_c]
            z_g = [sum(self.w_xg[i][j] * x[j] for j in range(self.input_size)) + sum(self.w_hg[i][j] * h[j] for j in range(self.hidden_size)) + self.b_g[i] for i in range(self.hidden_size)]
            g = [_sigmoid(v) for v in z_g]
            m_new = [(1.0 - g[i]) * m[i] + g[i] * c[i] for i in range(self.hidden_size)]
            z_h = [sum(self.w_xh[i][j] * x[j] for j in range(self.input_size)) + sum(self.w_hh[i][j] * h[j] for j in range(self.hidden_size)) + sum(self.w_mh[i][j] * m_new[j] for j in range(self.hidden_size)) + self.b_h[i] for i in range(self.hidden_size)]
            h_new = [_tanh(v) for v in z_h]
            states.append((x, list(h), list(m), z_c, c, z_g, g, z_h, h_new, m_new))
            h, m = h_new, m_new
        z_o = [sum(self.w_oh[o][i] * h[i] for i in range(self.hidden_size)) + sum(self.w_om[o][i] * m[i] for i in range(self.hidden_size)) + self.b_o[o] for o in range(self.output_size)]
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

        H = self.hidden_size
        O = self.output_size
        I = self.input_size
        for _ in range(epochs):
            for sequence, target in samples:
                output, states = self._forward(sequence)
                T = len(sequence)
                # output gradients
                d_o = [2.0*(output[o]-target[o])*output[o]*(1-output[o]) for o in range(O)]
                # accumulate gradients
                gW_xh = [[0.0 for _ in range(I)] for _ in range(H)]
                gW_hh = [[0.0 for _ in range(H)] for _ in range(H)]
                gb_h = [0.0 for _ in range(H)]
                gW_xc = [[0.0 for _ in range(I)] for _ in range(H)]
                gW_hc = [[0.0 for _ in range(H)] for _ in range(H)]
                gb_c = [0.0 for _ in range(H)]
                gW_xg = [[0.0 for _ in range(I)] for _ in range(H)]
                gW_hg = [[0.0 for _ in range(H)] for _ in range(H)]
                gb_g = [0.0 for _ in range(H)]
                gW_mh = [[0.0 for _ in range(H)] for _ in range(H)]
                gW_oh = [[0.0 for _ in range(H)] for _ in range(O)]
                gW_om = [[0.0 for _ in range(H)] for _ in range(O)]
                gb_o = [0.0 for _ in range(O)]

                for o in range(O):
                    for i in range(H):
                        gW_oh[o][i] += d_o[o]*states[-1][8][i]
                        gW_om[o][i] += d_o[o]*states[-1][9][i]
                    gb_o[o] += d_o[o]

                grad_h = [sum(self.w_oh[o][i]*d_o[o] for o in range(O)) for i in range(H)]
                grad_m = [sum(self.w_om[o][i]*d_o[o] for o in range(O)) for i in range(H)]

                for t in range(T-1, -1, -1):
                    _, h_prev, m_prev, z_c, c, z_g, g, z_h, h, m = states[t]
                    # gradient wrt z_h
                    dz_h = [grad_h[i]*(1-h[i]**2) for i in range(H)]
                    # gradient wrt m (through h and output future already in grad_m)
                    # m = (1-g)*m_prev + g*c
                    # gradient to g
                    dg = [grad_m[i]*(c[i]-m_prev[i]) for i in range(H)]
                    # gradient to c
                    dc = [grad_m[i]*g[i] for i in range(H)]
                    # gradient to m_prev
                    grad_m_prev = [grad_m[i]*(1-g[i]) for i in range(H)]
                    # gradient to z_c
                    dz_c = [dc[i]*(1-c[i]**2) for i in range(H)]
                    # gradient to z_g
                    dz_g = [dg[i]*g[i]*(1-g[i]) for i in range(H)]
                    # gradient to z_h inputs
                    # z_h = W_xh x + W_hh h_prev + W_mh m + b_h
                    x = states[t][0]
                    for i in range(H):
                        for j in range(I):
                            gW_xh[i][j] += dz_h[i]*x[j]
                        for j in range(H):
                            gW_hh[i][j] += dz_h[i]*h_prev[j]
                            gW_mh[i][j] += dz_h[i]*m[i]
                        gb_h[i] += dz_h[i]
                    # z_c grads
                    for i in range(H):
                        for j in range(I):
                            gW_xc[i][j] += dz_c[i]*x[j]
                        for j in range(H):
                            gW_hc[i][j] += dz_c[i]*h_prev[j]
                        gb_c[i] += dz_c[i]
                    # z_g grads
                    for i in range(H):
                        for j in range(I):
                            gW_xg[i][j] += dz_g[i]*x[j]
                        for j in range(H):
                            gW_hg[i][j] += dz_g[i]*h_prev[j]
                        gb_g[i] += dz_g[i]
                    # propagate gradients to previous h
                    grad_h_prev = [0.0 for _ in range(H)]
                    for i in range(H):
                        for j in range(I):
                            pass
                        for j in range(H):
                            grad_h_prev[j] += dz_h[i]*self.w_hh[i][j]
                            grad_h_prev[j] += dz_c[i]*self.w_hc[i][j]
                            grad_h_prev[j] += dz_g[i]*self.w_hg[i][j]
                    # W_mh gradient to m already accounted; m_t depends on m_prev via (1-g)
                    grad_m_prev_total = grad_m_prev
                    grad_h = grad_h_prev
                    grad_m = grad_m_prev_total

                # update params
                for i in range(H):
                    for j in range(I):
                        self.w_xh[i][j] -= self.learning_rate*gW_xh[i][j]
                        self.w_xc[i][j] -= self.learning_rate*gW_xc[i][j]
                        self.w_xg[i][j] -= self.learning_rate*gW_xg[i][j]
                    for j in range(H):
                        self.w_hh[i][j] -= self.learning_rate*gW_hh[i][j]
                        self.w_hc[i][j] -= self.learning_rate*gW_hc[i][j]
                        self.w_hg[i][j] -= self.learning_rate*gW_hg[i][j]
                        self.w_mh[i][j] -= self.learning_rate*gW_mh[i][j]
                    self.b_h[i] -= self.learning_rate*gb_h[i]
                    self.b_c[i] -= self.learning_rate*gb_c[i]
                    self.b_g[i] -= self.learning_rate*gb_g[i]
                for o in range(O):
                    for i in range(H):
                        self.w_oh[o][i] -= self.learning_rate*gW_oh[o][i]
                        self.w_om[o][i] -= self.learning_rate*gW_om[o][i]
                    self.b_o[o] -= self.learning_rate*gb_o[o]

        final_predictions = [self.predict_sequence(seq) for seq, _ in samples]
        final_loss = sum(sum((p[i]-t[i])**2 for i in range(len(t)))/len(t) for p, (_, t) in zip(final_predictions, samples)) / len(samples)
        return TrainingResult(initial_loss=initial_loss, final_loss=final_loss, epochs=epochs, samples=len(samples), parameter_count=self.parameter_count)

    def parameter_signature(self) -> str:
        vals = []
        for m in (self.w_xh, self.w_hh, self.w_xc, self.w_hc, self.w_xg, self.w_hg, self.w_mh, self.w_oh, self.w_om):
            for row in m:
                vals.extend(row)
        vals.extend(self.b_h + self.b_c + self.b_g + self.b_o)
        return ",".join(repr(v) for v in vals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "w_xh": self.w_xh, "w_hh": self.w_hh, "b_h": self.b_h,
            "w_xc": self.w_xc, "w_hc": self.w_hc, "b_c": self.b_c,
            "w_xg": self.w_xg, "w_hg": self.w_hg, "b_g": self.b_g,
            "w_mh": self.w_mh,
            "w_oh": self.w_oh, "w_om": self.w_om, "b_o": self.b_o,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GatedMemoryNetwork":
        net = cls(data["input_size"], data["hidden_size"], data["output_size"], data["learning_rate"], 0)
        for key in ("w_xh", "w_hh", "b_h", "w_xc", "w_hc", "b_c", "w_xg", "w_hg", "b_g", "w_mh", "w_oh", "w_om", "b_o"):
            setattr(net, key, data[key])
        return net

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GatedMemoryNetwork":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

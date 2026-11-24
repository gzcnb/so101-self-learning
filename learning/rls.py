"""Online joint-tracking compensation extracted from the physical prototype."""

import math
import numpy as np

TICK_TO_RAD = 2 * math.pi / 4096
AERIAL_HOME = np.array([2099.0, 1750.0, 2250.0, 1700.0])

class CerebellumRLS:
    """Position-space orthogonal 5-D physics-basis RLS feedforward compensator."""

    def __init__(self, zeros_arr: np.ndarray, mode: str = "rls", lam: float = 0.9985, p0: float = 0.8,
                 max_ff: float = 85.0, eta: float = 0.55, delay_steps: int = 2):
        self.zeros = zeros_arr.astype(float)
        self.mode = mode  # "none", "bias", "rls"
        self.lam = lam
        self.p0 = p0
        self.max_ff = max_ff
        self.eta = eta
        self.delay_steps = delay_steps
        self.dim = 1 if mode == "bias" else 5
        self.W = np.zeros((4, self.dim), dtype=float)
        self.P = np.array([np.eye(self.dim) * p0 for _ in range(4)], dtype=float)
        self.ff_smooth = np.zeros(4, dtype=float)
        self.history = []
        self.step_count = 0

    def features(self, q_ref: np.ndarray, v_ref: np.ndarray, a_ref: np.ndarray):
        if self.mode == "bias":
            return [np.array([1.0], dtype=float) for _ in range(4)]
        dr = (q_ref - AERIAL_HOME) * TICK_TO_RAD
        vr = v_ref * TICK_TO_RAD
        phis = []
        for j in range(4):
            phis.append(np.array([
                1.0,
                3.0 * dr[1],
                3.0 * dr[2],
                vr[j],
                math.tanh(vr[j] / 0.15),
            ], dtype=float))
        return phis

    def reset_runtime_buffers(self):
        self.ff_smooth[:] = 0.0
        self.history.clear()
        self.step_count = 0

    def predict(self, q_ref: np.ndarray, v_ref: np.ndarray, a_ref: np.ndarray):
        if self.mode == "none":
            return np.zeros(4, dtype=float), None
        phis = self.features(q_ref, v_ref, a_ref)
        raw_ff = np.array([float(self.W[j] @ phis[j]) for j in range(4)], dtype=float)
        raw_ff = np.clip(raw_ff, -self.max_ff, self.max_ff)
        delta = np.clip(raw_ff - self.ff_smooth, -4.0, 4.0)
        self.ff_smooth = 0.75 * self.ff_smooth + 0.25 * (self.ff_smooth + delta)
        self.ff_smooth = np.clip(self.ff_smooth, -self.max_ff, self.max_ff)
        return self.ff_smooth.copy(), phis

    def update(self, phis, ff_applied: np.ndarray, q_ref: np.ndarray, q_meas: np.ndarray):
        if self.mode == "none" or phis is None:
            return
        self.history.append((phis, ff_applied.copy()))
        if len(self.history) <= self.delay_steps:
            return
        old_phis, old_ff = self.history.pop(0)
        self.step_count += 1
        if self.step_count % 2 != 0:
            return
        err = np.clip(q_ref - q_meas, -120.0, 120.0)
        y = np.clip(old_ff + self.eta * err, -self.max_ff, self.max_ff)
        for j in range(4):
            phi = old_phis[j]
            Pj = self.P[j]
            Pphi = Pj @ phi
            denom = self.lam + float(phi @ Pphi)
            k = Pphi / denom
            innov = y[j] - float(self.W[j] @ phi)
            self.W[j] += k * innov
            self.W[j] *= 0.9997
            self.W[j] = np.clip(self.W[j], -80.0, 80.0)
            P_next = (Pj - np.outer(k, Pphi)) / self.lam
            tr = float(np.trace(P_next))
            max_tr = self.p0 * self.dim * 1.25
            if tr > max_tr:
                P_next *= (max_tr / tr)
            self.P[j] = 0.5 * (P_next + P_next.T)

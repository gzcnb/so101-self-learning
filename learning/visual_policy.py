"""Visual process encoder, recurrent joint policy, and servo guard from the experimental branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


ARM4_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
ALL6_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Shared safe aerial bounds in raw STS3215 ticks (well clear of desk and blocks; ID4 protected)
AERIAL_SAFE_LIMITS_TICKS: Dict[str, Tuple[float, float]] = {
    "shoulder_pan": (1720.0, 2460.0),
    "shoulder_lift": (1280.0, 2180.0),
    "elbow_flex": (1860.0, 2560.0),
    "wrist_flex": (1560.0, 1840.0),
    "wrist_roll": (600.0, 1500.0),
    "gripper": (2050.0, 2560.0),
}

TICK_TO_RAD = 2.0 * np.pi / 4096.0
RAD_TO_TICK = 4096.0 / (2.0 * np.pi)


class S2VisualProcessEncoder(nn.Module):
    """Maps a 3-frame RGB process clip (I_start, I_mid, I_end) [B, 9, 64, 64] -> z in R^24.

    Trained strictly end-to-end with S1 (no privileged mode labels or 3D coordinates).
    Because I_start and I_end are identical across counterfactual prompts, all variance
    in z must arise purely from the visual passage shown in I_mid.
    """

    def __init__(self, z_dim: int = 24) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.conv = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, z_dim),
            nn.Tanh(),
        )

    def forward(self, clip_9ch: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(clip_9ch))


class S1ClosedLoopPolicyGRU(nn.Module):
    """Closed-loop recurrent motor policy operating purely on real proprioception + z.

    Inputs per 33.3 Hz tick:
      - q_meas:  (B, T, 4) measured joint positions (rad) for ARM4
      - dq_filt: (B, T, 4) causal filtered joint velocities (rad/s) for ARM4
      - z:       (B, z_dim) frozen visual process intent from S2

    Output per tick:
      - delta_q: (B, T, 4) bounded closed-loop joint lead/offset (rad) so
                 q_target = q_meas + delta_q.
    """

    def __init__(
        self,
        q_center_rad: np.ndarray,
        z_dim: int = 24,
        hidden_dim: int = 128,
        max_lead_rad: float = 0.28,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.max_lead_rad = float(max_lead_rad)
        self.register_buffer("q_center", torch.as_tensor(q_center_rad, dtype=torch.float32).view(1, 1, 4))

        in_dim = 4 + 4 + z_dim
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
            nn.Tanh(),
        )

    def forward(
        self,
        q_meas: torch.Tensor,
        dq_filt: torch.Tensor,
        z: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, t_len, _ = q_meas.shape
        q_norm = (q_meas - self.q_center) / 0.8
        dq_norm = torch.clamp(dq_filt / 1.5, -2.0, 2.0)
        z_exp = z.unsqueeze(1).expand(bsz, t_len, self.z_dim) if z.ndim == 2 else z
        x = torch.cat([q_norm, dq_norm, z_exp], dim=-1)
        h_in = self.in_proj(x)
        gru_out, hn = self.gru(h_in, h0)
        delta_q = self.max_lead_rad * self.out_head(gru_out)
        return delta_q, hn


@dataclass
class S0RealConfig:
    dt: float = 0.030  # 33.33 Hz
    soft_margin_rad: float = 0.035  # ~23 ticks inside AERIAL_SAFE_LIMITS_TICKS
    max_vel_rad_s: float = 1.65  # smooth hardware slew cap (~0.0495 rad / 32 ticks per 30ms step)
    stall_err_rad: float = 0.22  # ~143 ticks sustained lag -> spinal stall backoff
    stall_vel_rad_s: float = 0.16
    stall_persist_steps: int = 5  # ~150 ms
    stall_release_err_rad: float = 0.10
    backoff_alpha: float = 0.45
    backoff_max_residual_rad: float = 0.04


class S0ServoGuardReal:
    """Shared hardware-deployable S0 safety guard (identical across all test conditions)."""

    def __init__(self, low_rad: np.ndarray, high_rad: np.ndarray, cfg: Optional[S0RealConfig] = None) -> None:
        self.cfg = cfg or S0RealConfig()
        self.low_rad = np.asarray(low_rad, dtype=np.float64).copy()
        self.high_rad = np.asarray(high_rad, dtype=np.float64).copy()
        self.soft_low = self.low_rad + self.cfg.soft_margin_rad
        self.soft_high = self.high_rad - self.cfg.soft_margin_rad
        self.n = len(self.low_rad)
        self.q_cmd_prev: Optional[np.ndarray] = None
        self.stall_counters = np.zeros(self.n, dtype=np.int32)
        self.stall_active = np.zeros(self.n, dtype=bool)
        self.total_limit_clips = 0
        self.total_slew_clips = 0
        self.total_stall_steps = 0

    def reset(self, q_init_rad: np.ndarray) -> None:
        self.q_cmd_prev = np.clip(np.asarray(q_init_rad, dtype=np.float64), self.soft_low, self.soft_high).copy()
        self.stall_counters.fill(0)
        self.stall_active.fill(False)
        self.total_limit_clips = 0
        self.total_slew_clips = 0
        self.total_stall_steps = 0

    def filter(
        self,
        q_target_rad: np.ndarray,
        q_meas_rad: np.ndarray,
        dq_filt_rad_s: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        q_target = np.asarray(q_target_rad, dtype=np.float64)
        q_meas = np.asarray(q_meas_rad, dtype=np.float64)
        dq_filt = np.asarray(dq_filt_rad_s, dtype=np.float64)
        if self.q_cmd_prev is None:
            self.reset(q_meas)
        assert self.q_cmd_prev is not None

        # 1. Soft joint limits
        q_clipped = np.clip(q_target, self.soft_low, self.soft_high)
        limit_clipped = np.abs(q_clipped - q_target) > 1e-9
        self.total_limit_clips += int(np.sum(limit_clipped))

        # 2. Tracking-error stall detection
        prev_err = self.q_cmd_prev - q_meas
        abs_prev_err = np.abs(prev_err)
        abs_dq = np.abs(dq_filt)
        for j in range(self.n):
            if abs_prev_err[j] > self.cfg.stall_err_rad and abs_dq[j] < self.cfg.stall_vel_rad_s:
                self.stall_counters[j] += 1
                if self.stall_counters[j] >= self.cfg.stall_persist_steps:
                    self.stall_active[j] = True
            else:
                self.stall_counters[j] = max(0, self.stall_counters[j] - 1)
                if self.stall_active[j]:
                    req_err = abs(q_clipped[j] - q_meas[j])
                    if req_err < self.cfg.stall_release_err_rad and abs_prev_err[j] < self.cfg.stall_err_rad:
                        self.stall_active[j] = False

        # 3. Velocity slew-rate cap vs stall backoff
        max_step = self.cfg.max_vel_rad_s * self.cfg.dt
        delta = q_clipped - self.q_cmd_prev
        slew_clipped = np.abs(delta) > max_step
        self.total_slew_clips += int(np.sum(slew_clipped & (~self.stall_active)))
        q_safe = self.q_cmd_prev + np.clip(delta, -max_step, max_step)

        if np.any(self.stall_active):
            self.total_stall_steps += int(np.sum(self.stall_active))
            for j in range(self.n):
                if self.stall_active[j]:
                    pulled = self.q_cmd_prev[j] + self.cfg.backoff_alpha * (q_meas[j] - self.q_cmd_prev[j])
                    resid = np.clip(pulled - q_meas[j], -self.cfg.backoff_max_residual_rad, self.cfg.backoff_max_residual_rad)
                    q_safe[j] = np.clip(q_meas[j] + resid, self.soft_low[j], self.soft_high[j])

        q_safe = np.clip(q_safe, self.soft_low, self.soft_high)
        self.q_cmd_prev = q_safe.copy()
        info = {
            "limit_clipped": bool(np.any(limit_clipped)),
            "slew_clipped": bool(np.any(slew_clipped)),
            "stall_active": self.stall_active.tolist(),
        }
        return q_safe, info

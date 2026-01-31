# project/core/system.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple
import numpy as np


class Schedule(Protocol):
    def action(self, t: int) -> np.ndarray: ...


@dataclass
class RolloutResult:
    losses: List[float]
    states: Optional[List[np.ndarray]] = None  # optional: list of z_t snapshots


class AsyncSystem:
    """
    Base class for executable deterministic transition system:
      z_{t+1} = F(z_t, a_t)
      L_t = ell(z_t)
    """
    def step(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def instant_loss(self, z: np.ndarray) -> float:
        raise NotImplementedError

    def init_state(self, seed: int = 0) -> np.ndarray:
        raise NotImplementedError

    def rollout(self, schedule: Schedule, T: int, z0: Optional[np.ndarray] = None, record_states: bool = False) -> RolloutResult:
        if z0 is None:
            z = self.init_state()
        else:
            z = np.array(z0, copy=True)

        losses: List[float] = []
        states: Optional[List[np.ndarray]] = [] if record_states else None

        for t in range(T):
            losses.append(float(self.instant_loss(z)))
            if record_states:
                states.append(np.array(z, copy=True))
            a = schedule.action(t)
            z = self.step(z, a)

        return RolloutResult(losses=losses, states=states)

    def eval_JT(self, schedule: Schedule, T: int, z0: Optional[np.ndarray] = None) -> float:
        rr = self.rollout(schedule, T, z0=z0, record_states=False)
        return float(np.sum(rr.losses))

    def get_Q(self) -> np.ndarray:
        n = self.objective.H.shape[0]
        D = self.D
        m = (D + 1) * n
        Q = np.zeros((m, m), dtype=float)
        H = np.asarray(self.objective.H, dtype=float)
        Q[:n, :n] = 0.5 * (H + H.T)
        return Q


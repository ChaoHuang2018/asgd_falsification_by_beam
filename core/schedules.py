# project/core/schedules.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class StaticSchedule:
    d: np.ndarray  # (K,)

    def action(self, t: int) -> np.ndarray:
        return self.d


@dataclass(frozen=True)
class WordSchedule:
    word: List[np.ndarray]  # list of (K,) actions

    def action(self, t: int) -> np.ndarray:
        L = len(self.word)
        if L <= 0:
            raise ValueError("WordSchedule: empty word.")
        return self.word[t % L]

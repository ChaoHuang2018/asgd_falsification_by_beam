# project/core/results.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ScheduleResult:
    branch: str  # "uniform" | "static" | "bb"
    JT: float
    schedule: Any
    losses: List[float]
    meta: Dict[str, Any]
    states: Optional[List[np.ndarray]] = None

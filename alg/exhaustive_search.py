# alg/exhaustive_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExhaustiveSearchParams:
    """
    Parameters for brute-force exhaustive search over action sequences.
    """
    # Safety guard: stop after visiting this many nodes (partial search)
    max_nodes: Optional[int] = None

    # If True, enumerate actions by histograms (equivalence classes) to remove permutation duplicates.
    # This is exact for ASGD semantics where only counts of delays matter.
    use_histograms: bool = True

    # Print progress every this many visited nodes
    progress_every: int = 0


def _enumerate_action_set(action_space: Any, use_histograms: bool) -> List[np.ndarray]:
    """
    Return a list of canonical delay-vectors d (shape (K,)) representing the action set.
    """
    if use_histograms:
        if not hasattr(action_space, "enumerate_histograms") or not hasattr(action_space, "histogram_to_delay"):
            raise RuntimeError("use_histograms=True requires ActionSpace.enumerate_histograms() and histogram_to_delay().")
        Ms = action_space.enumerate_histograms()
        actions = [action_space.histogram_to_delay(m) for m in Ms]
        return actions

    # full enumeration of delay vectors (may contain permutation duplicates)
    if not hasattr(action_space, "enumerate_actions"):
        raise RuntimeError("ActionSpace.enumerate_actions() is required.")
    return list(action_space.enumerate_actions())


def exhaustive_sequence_search(
    system: Any,
    action_space: Any,
    T: int,
    params: Optional[ExhaustiveSearchParams] = None,
    z0 = None, 
) -> Dict[str, Any]:
    """
    Brute-force exhaustive search:
      maximize JT = sum_{t=0}^{T-1} instant_loss(z_t)
    over all sequences of length T, where each action is a delay vector d with sum(d)<=B and each entry in [0,D].

    Returns:
      {
        "best_word": List[np.ndarray]  # length T, each (K,)
        "best_JT": float,
        "visited_nodes": int,
        "num_actions": int,
        "params": {...},
        "complete": bool,  # whether we finished without hitting max_nodes
      }
    """
    if params is None:
        params = ExhaustiveSearchParams()
    
    if z0 is None:
        z0 = system.init_state()
    else:
        z0 = np.asarray(z0, dtype=float)

    T = int(T)
    if T < 0:
        raise ValueError("T must be >= 0")

    actions = _enumerate_action_set(action_space, use_histograms=bool(params.use_histograms))
    num_actions = len(actions)
    if num_actions == 0:
        return {
            "best_word": [],
            "best_JT": float("-inf"),
            "visited_nodes": 0,
            "num_actions": 0,
            "params": params.__dict__,
            "complete": True,
        }

    # init
    # z0 = np.asarray(system.init_state(), dtype=float)
    best_JT = float("-inf")
    best_word: List[np.ndarray] = []

    visited = 0
    complete = True

    # preallocate a mutable word buffer for speed
    word_buf: List[Optional[np.ndarray]] = [None] * T

    def rec(t: int, z: np.ndarray, JT_so_far: float) -> None:
        nonlocal visited, best_JT, best_word, complete

        if params.max_nodes is not None and visited >= int(params.max_nodes):
            complete = False
            return

        if t == T:
            # reached length T, update best
            if JT_so_far > best_JT:
                best_JT = float(JT_so_far)
                best_word = [np.array(word_buf[i], copy=True) for i in range(T)]  # deep copy
            return

        # stage cost at current state (area semantics)
        JT_next = JT_so_far + float(system.instant_loss(z))

        # expand one action step
        for a in actions:
            if params.max_nodes is not None and visited >= int(params.max_nodes):
                complete = False
                return

            visited += 1
            if params.progress_every and (visited % int(params.progress_every) == 0):
                print(f"[exhaustive] visited={visited}  best_JT={best_JT:.6g}  depth={t}/{T}")

            word_buf[t] = a
            z_next = system.step(z, a)
            rec(t + 1, z_next, JT_next)

    rec(0, z0, 0.0)

    return {
        "best_word": best_word,
        "best_JT": float(best_JT),
        "visited_nodes": int(visited),
        "num_actions": int(num_actions),
        "params": params.__dict__,
        "complete": bool(complete),
    }

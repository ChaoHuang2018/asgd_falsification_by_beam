# project/alg/bb_search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from core.action_space import ActionSpace
from core.system import AsyncSystem
from core.schedules import WordSchedule


@dataclass
class BBSearchParams:
    L_max: int = 5
    beam_width: int = 10
    max_enum_actions: int = 50_000  # threshold on (D+1)^K heuristic
    candidates_per_expand: int = 50
    seed: int = 0


def bb_search(system: AsyncSystem, action_space: ActionSpace, T: int, params: BBSearchParams):
    rng = np.random.default_rng(params.seed)

    # Candidate action set for expansions (may be enumerated or sampled)
    candidates = action_space.candidate_actions(
        max_enum=params.max_enum_actions,
        rng=rng,
        n_sample=params.candidates_per_expand
    )

    beam: List[Tuple[List[np.ndarray], float]] = [([], -np.inf)]
    best_word: List[np.ndarray] = []
    best_JT = -np.inf
    best_losses = []

    for L in range(1, params.L_max + 1):
        new_items: List[Tuple[List[np.ndarray], float, List[float]]] = []

        # Expand each beam item
        for word, _score in beam:
            for a in candidates:
                w2 = word + [a]
                sched = WordSchedule(word=w2)
                rr = system.rollout(sched, T, record_states=False)
                JT = float(np.sum(rr.losses))
                new_items.append((w2, JT, rr.losses))

                if JT > best_JT:
                    best_JT = JT
                    best_word = w2
                    best_losses = rr.losses

        # keep top beam_width
        new_items.sort(key=lambda x: x[1], reverse=True)
        beam = [(w, s) for (w, s, _losses) in new_items[:params.beam_width]]

    return {
        "best_word": best_word,
        "best_JT": best_JT,
        "losses": best_losses,
        "meta": {
            "L_max": params.L_max,
            "beam_width": params.beam_width,
            "candidates_per_expand": params.candidates_per_expand,
            "seed": params.seed
        }
    }

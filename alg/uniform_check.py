# alg/uniform_check.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.schedules import StaticSchedule


def _as_np_int_vec(d: Any) -> np.ndarray:
    return np.asarray(d, dtype=int).reshape(-1)


def _canonicalize_delay(d: Any) -> Tuple[int, ...]:
    v = _as_np_int_vec(d)
    return tuple(sorted(int(x) for x in v))


def _all_actions(action_space: Any) -> List[np.ndarray]:
    for name in ("all_actions", "enumerate_all", "enumerate_actions", "actions"):
        if hasattr(action_space, name):
            actions = getattr(action_space, name)()
            return [np.asarray(a, dtype=int) for a in actions]
    raise AttributeError(
        "action_space must provide one of: all_actions()/enumerate_all()/enumerate_actions()/actions()"
    )


def _balanced(action_space: Any, s: int) -> np.ndarray:
    for name in ("balanced", "balanced_delay", "make_balanced"):
        if hasattr(action_space, name):
            return _as_np_int_vec(getattr(action_space, name)(s))
    raise AttributeError("action_space must provide balanced(s) (or balanced_delay/make_balanced).")


def _eval_JT(system: Any, d: np.ndarray, T: int) -> float:
    sch = StaticSchedule(d)
    if hasattr(system, "eval_JT"):
        return float(system.eval_JT(sch, T))
    if hasattr(system, "rollout"):
        out = system.rollout(sch, T)
        if isinstance(out, dict) and "losses" in out:
            return float(np.sum(out["losses"]))
        if isinstance(out, tuple) and len(out) >= 2:
            return float(np.sum(out[1]))
    raise AttributeError("system must provide eval_JT(schedule,T) or rollout(schedule,T)->losses.")


@dataclass
class UniformCheckParams:
    # Numerical tolerance for comparisons
    tol: float = 1e-12

    # U1: permutation invariance check
    check_permutation_invariance: bool = True
    perm_test_samples: int = 25
    perm_test_vectors_per_layer: int = 3
    rng_seed: int = 0

    # U2: Robin-Hood neighbor monotonicity (discrete Schur-concavity)
    check_robin_hood_monotone: bool = True

    # U3: layer maximum monotonicity
    check_layer_max_monotone: bool = True

    # Extra: require balanced(s) is exactly the layer maximizer (strong but decidable; aligns with U-area proof)
    check_balanced_is_layer_max: bool = True


def check_uniform_worst(
    system: Any,
    action_space: Any,
    T: int,
    B: int,
    params: Optional[UniformCheckParams] = None,
) -> Dict[str, Any]:
    """
    Theorem U-area (static-class uniform worst) sufficient condition check:

      U1) permutation invariance
      U2) Robin-Hood monotonicity within each fixed-sum layer
      U3) layer maximum monotonicity in s

    Returns:
      pass: bool
      witness_d: balanced(B) if pass else None
      JT_witness: float if pass else None
      diagnostics: reason/details on failure
      layer_max: list of (s, M_T(s), argmax_d)
    """
    if params is None:
        params = UniformCheckParams()

    actions = _all_actions(action_space)
    # canonicalize to unique (permutation-invariant) representatives
    uniq: Dict[Tuple[int, ...], np.ndarray] = {}
    for a in actions:
        uniq[_canonicalize_delay(a)] = _as_np_int_vec(a)
    actions = list(uniq.values())

    if len(actions) == 0:
        return {
            "pass": False,
            "witness_d": None,
            "JT_witness": None,
            "diagnostics": {"failed": True, "reason": "no_actions", "details": None},
            "layer_max": [],
        }

    max_sum = min(B, max(int(np.sum(a)) for a in actions))
    layers: Dict[int, List[np.ndarray]] = {s: [] for s in range(max_sum + 1)}
    for a in actions:
        s = int(np.sum(a))
        if 0 <= s <= max_sum:
            layers[s].append(a)

    diagnostics: Dict[str, Any] = {
        "failed": False,
        "reason": None,
        "details": None,
        "U1": params.check_permutation_invariance,
        "U2": params.check_robin_hood_monotone,
        "U3": params.check_layer_max_monotone,
    }

    JT_cache: Dict[Tuple[int, ...], float] = {}

    def JT_of(d: np.ndarray) -> float:
        key = _canonicalize_delay(d)
        if key not in JT_cache:
            JT_cache[key] = _eval_JT(system, np.asarray(key, dtype=int), T)
        return JT_cache[key]

    # -------- U1: permutation invariance --------
    if params.check_permutation_invariance:
        rng = np.random.default_rng(params.rng_seed)
        for s in range(max_sum + 1):
            if len(layers[s]) == 0:
                continue
            reps: List[np.ndarray] = []
            reps.append(_balanced(action_space, s))
            if params.perm_test_vectors_per_layer > 1:
                idxs = rng.choice(
                    len(layers[s]),
                    size=min(params.perm_test_vectors_per_layer - 1, len(layers[s])),
                    replace=False,
                )
                reps.extend([layers[s][int(i)] for i in idxs])

            for d in reps:
                d = _as_np_int_vec(d)
                base = _eval_JT(system, d, T)
                K = d.shape[0]
                for _ in range(params.perm_test_samples):
                    perm = rng.permutation(K)
                    dp = d[perm]
                    val = _eval_JT(system, dp, T)
                    if abs(val - base) > params.tol:
                        diagnostics.update(
                            {
                                "failed": True,
                                "reason": "U1_permutation_invariance_failed",
                                "details": {"s": s, "d": d.tolist(), "perm_d": dp.tolist(), "J": base, "J_perm": val},
                            }
                        )
                        return {
                            "pass": False,
                            "witness_d": None,
                            "JT_witness": None,
                            "diagnostics": diagnostics,
                            "layer_max": [],
                        }

    # compute layer maxima & (optional) check balanced is layer max
    layer_max: List[Tuple[int, float, List[int]]] = []
    M: Dict[int, float] = {}
    argmax_d: Dict[int, np.ndarray] = {}

    for s in range(max_sum + 1):
        if len(layers[s]) == 0:
            M[s] = -np.inf
            argmax_d[s] = None
            continue
        bestJ = -np.inf
        bestd = None
        for d in layers[s]:
            j = JT_of(d)
            if j > bestJ:
                bestJ, bestd = j, d
        M[s] = float(bestJ)
        argmax_d[s] = _as_np_int_vec(bestd)
        layer_max.append((s, float(bestJ), argmax_d[s].tolist()))

        if params.check_balanced_is_layer_max:
            db = _balanced(action_space, s)
            jb = JT_of(db)
            if jb + params.tol < bestJ:
                diagnostics.update(
                    {
                        "failed": True,
                        "reason": "U_layer_balanced_not_max",
                        "details": {
                            "s": s,
                            "balanced": list(_canonicalize_delay(db)),
                            "J_balanced": float(jb),
                            "best_d": list(_canonicalize_delay(bestd)),
                            "J_best": float(bestJ),
                        },
                    }
                )
                return {
                    "pass": False,
                    "witness_d": None,
                    "JT_witness": None,
                    "diagnostics": diagnostics,
                    "layer_max": layer_max,
                }

    # -------- U2: Robin-Hood monotonicity within each fixed-sum layer --------
    if params.check_robin_hood_monotone:
        for s in range(max_sum + 1):
            if len(layers[s]) == 0:
                continue
            for d in layers[s]:
                d = _as_np_int_vec(d)
                Jd = JT_of(d)
                K = d.shape[0]
                for i in range(K):
                    for j in range(K):
                        if d[i] >= d[j] + 2:
                            dn = d.copy()
                            dn[i] -= 1
                            dn[j] += 1
                            dn = np.asarray(_canonicalize_delay(dn), dtype=int)
                            Jn = JT_of(dn)
                            if Jn + params.tol < Jd:
                                diagnostics.update(
                                    {
                                        "failed": True,
                                        "reason": "U2_robin_hood_violation",
                                        "details": {
                                            "s": s,
                                            "d": list(_canonicalize_delay(d)),
                                            "dn": list(_canonicalize_delay(dn)),
                                            "J": float(Jd),
                                            "Jn": float(Jn),
                                        },
                                    }
                                )
                                return {
                                    "pass": False,
                                    "witness_d": None,
                                    "JT_witness": None,
                                    "diagnostics": diagnostics,
                                    "layer_max": layer_max,
                                }

    # -------- U3: layer maximum monotonicity in s --------
    if params.check_layer_max_monotone:
        for s in range(max_sum):
            if M[s] == -np.inf or M[s + 1] == -np.inf:
                continue
            if M[s + 1] + params.tol < M[s]:
                diagnostics.update(
                    {
                        "failed": True,
                        "reason": "U3_layer_max_not_monotone",
                        "details": {"s": s, "M_s": float(M[s]), "M_s1": float(M[s + 1])},
                    }
                )
                return {
                    "pass": False,
                    "witness_d": None,
                    "JT_witness": None,
                    "diagnostics": diagnostics,
                    "layer_max": layer_max,
                }

    # PASS: balanced(B) is worst static action under the U-area conditions (within static class)
    dB = _balanced(action_space, max_sum)
    JT_w = _eval_JT(system, dB, T)
    return {
        "pass": True,
        "witness_d": _as_np_int_vec(dB),
        "JT_witness": float(JT_w),
        "diagnostics": diagnostics,
        "layer_max": layer_max,
    }

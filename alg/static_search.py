# alg/static_search.py
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


def _get_Q(system: Any, remodeling: Any, A_dim: int, action_space: Any) -> Optional[np.ndarray]:
    """
    Try to obtain Q representing the one-step area term: l_t = s_t^T Q s_t.
    Priority:
      1) system.get_Q() or system.Q
      2) remodeling.get_Q() or remodeling.Q
      3) fallback heuristic: if system has H (quadratic), and action_space has D and system has n,
         build Q = diag(H, 0, ..., 0) matching "only current x" area.
    """
    for obj in (system, remodeling):
        if obj is None:
            continue
        if hasattr(obj, "get_Q"):
            Q = np.asarray(obj.get_Q(), dtype=float)
            return Q
        if hasattr(obj, "Q"):
            Q = np.asarray(getattr(obj, "Q"), dtype=float)
            return Q

    # heuristic fallback: "area = current x block only"
    n = getattr(system, "n", None)
    D = getattr(action_space, "D", None)
    if n is None or D is None:
        return None
    n = int(n)
    D = int(D)
    m = (D + 1) * n
    if m != int(A_dim):
        return None

    H = getattr(system, "H", None)
    if H is None:
        H = np.eye(n, dtype=float)
    else:
        H = np.asarray(H, dtype=float)
    Q = np.zeros((m, m), dtype=float)
    Q[:n, :n] = 0.5 * (H + H.T)
    return Q


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _psd(M: np.ndarray, tol: float) -> bool:
    w = np.linalg.eigvalsh(_sym(M))
    return bool(np.min(w) >= -tol)


def _psd_on_subspace(Delta: np.ndarray, U: np.ndarray, tol: float) -> Tuple[bool, float]:
    """
    Check Delta ⪰ 0 restricted to subspace span(U):
      forall s in span(U), s^T Delta s >= 0  <=>  U^T Delta U ⪰ 0.

    Returns: (ok, min_eig_projected)
    """
    G = _sym(U.T @ _sym(Delta) @ U)
    w = np.linalg.eigvalsh(G)
    minw = float(np.min(w)) if w.size > 0 else 0.0
    return (minw >= -tol), minw


@dataclass
class StaticSearchParams:
    tol: float = 1e-12
    psd_tol: float = 1e-10

    # Optional subsampling for S-area-T check (still a sufficient-condition check on the subset)
    max_actions_for_S_area: Optional[int] = None
    rng_seed: int = 0

    # NEW: reachable subspace basis in lifted state space (m x r).
    # If provided, all PSD checks are performed on the projected matrix U^T Δ U.
    subspace_basis: Optional[np.ndarray] = None
    subspace_name: Optional[str] = None


def check_S_area_static_worst_dp(
    system: Any,
    remodeling: Any,
    action_space: Any,
    T: int,
    params: Optional[StaticSearchParams] = None,
) -> Dict[str, Any]:
    """
    S-area-T (finite-horizon) sufficient condition check for:
      "global worst schedule (over all sequences) is achieved by a static action a*"

    If params.subspace_basis is provided, the certificate is restricted to the reachable
    subspace span(U) (strictly sufficient for initial-set ⊂ span(U) under invariance).

    Certificate (full-space):
      For candidate a*, define P_T=0 and P_t = Q + A(a*)^T P_{t+1} A(a*) (static recursion).
      If for all t and all actions a:
          A(a*)^T P_{t+1} A(a*) - A(a)^T P_{t+1} A(a) ⪰ 0
      then DP-max is always attained by a* for all states, hence static a* is globally worst.

    Certificate (subspace):
      Same, but only requires Δ ⪰ 0 on span(U), i.e., U^T Δ U ⪰ 0.
    """
    if params is None:
        params = StaticSearchParams()

    if remodeling is None or not hasattr(remodeling, "get_A"):
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "no_remodeling_get_A", "details": None},
        }

    actions = _all_actions(action_space)
    uniq: Dict[Tuple[int, ...], np.ndarray] = {}
    for a in actions:
        uniq[_canonicalize_delay(a)] = _as_np_int_vec(a)
    actions = list(uniq.values())

    if len(actions) == 0:
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "no_actions", "details": None},
        }

    if params.max_actions_for_S_area is not None and len(actions) > params.max_actions_for_S_area:
        rng = np.random.default_rng(params.rng_seed)
        idx = rng.choice(len(actions), size=int(params.max_actions_for_S_area), replace=False)
        actions = [actions[int(i)] for i in idx]

    # Precompute all A(a)
    A_list: List[np.ndarray] = []
    for d in actions:
        A = np.asarray(remodeling.get_A(d), dtype=float)
        A_list.append(A)

    m = int(A_list[0].shape[0])
    for A in A_list:
        if A.shape != (m, m):
            return {
                "pass": False,
                "witness_d": None,
                "diagnostics": {"failed": True, "reason": "A_shape_inconsistent", "details": None},
            }

    Q = _get_Q(system, remodeling, m, action_space)
    if Q is None:
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {
                "failed": True,
                "reason": "no_Q_available",
                "details": "Provide system.get_Q()/system.Q or remodeling.get_Q()/remodeling.Q.",
            },
        }
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (m, m):
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "Q_shape_mismatch", "details": {"Q_shape": list(Q.shape), "A_dim": m}},
        }
    Q = _sym(Q)

    U = None
    if params.subspace_basis is not None:
        U = np.asarray(params.subspace_basis, dtype=float)
        if U.ndim != 2 or U.shape[0] != m:
            return {
                "pass": False,
                "witness_d": None,
                "diagnostics": {
                    "failed": True,
                    "reason": "subspace_basis_shape_mismatch",
                    "details": {"U_shape": list(U.shape), "A_dim": m},
                },
            }

    # Heuristic order over candidates: larger trace(A^T Q A) first
    scores = np.array([np.trace(A.T @ Q @ A) for A in A_list], dtype=float)
    order = list(np.argsort(-scores))

    for idx_star in order:
        A_star = A_list[int(idx_star)]

        # static recursion P_{t} along A_star: P_T=0, P_t = Q + A*^T P_{t+1} A*
        P_next = np.zeros((m, m), dtype=float)  # P_{t+1}
        certified = True
        counterexample = None

        for t in range(T - 1, -1, -1):
            M_star = _sym(A_star.T @ P_next @ A_star)

            for j, A in enumerate(A_list):
                M = _sym(A.T @ P_next @ A)

                Delta = _sym(M_star - M)

                if U is None:
                    ok = _psd(Delta, params.psd_tol)
                    min_eig = float(np.min(np.linalg.eigvalsh(_sym(Delta))))
                else:
                    ok, min_eig = _psd_on_subspace(Delta, U, params.psd_tol)

                if not ok:
                    certified = False
                    counterexample = {
                        "t": t,
                        "a_star": actions[int(idx_star)].tolist(),
                        "a_bad": actions[int(j)].tolist(),
                        "min_eig_diff": float(min_eig),
                        "check_space": "subspace" if U is not None else "full",
                        "subspace_name": params.subspace_name,
                        "subspace_dim": int(U.shape[1]) if U is not None else None,
                    }
                    break

            if not certified:
                break

            P_next = _sym(Q + A_star.T @ P_next @ A_star)

        if certified:
            return {
                "pass": True,
                "witness_d": actions[int(idx_star)],
                "diagnostics": {
                    "failed": False,
                    "reason": None,
                    "details": {
                        "certificate": (
                            "S-area-T DP certificate on reachable subspace: same a* maximizes x^T A^T P_{t+1} A x "
                            "for all t and all x in span(U)"
                            if U is not None
                            else "S-area-T DP certificate: same a* maximizes x^T A^T P_{t+1} A x for all t and all x"
                        ),
                        "T": T,
                        "num_actions_checked": len(actions),
                        "candidate_score_trace_ATQA": float(scores[int(idx_star)]),
                        "check_space": "subspace" if U is not None else "full",
                        "subspace_name": params.subspace_name,
                        "subspace_dim": int(U.shape[1]) if U is not None else None,
                    },
                },
            }

    return {
        "pass": False,
        "witness_d": None,
        "diagnostics": {
            "failed": True,
            "reason": "S_area_T_no_static_global_maximizer_found",
            "details": {
                "T": T,
                "num_actions_checked": len(actions),
                "check_space": "subspace" if U is not None else "full",
                "subspace_name": params.subspace_name,
                "subspace_dim": int(U.shape[1]) if U is not None else None,
            },
        },
    }

# --- structured S-check for ASGD (histogram + eigenmode decomposition) ---

def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def _psd(M: np.ndarray, tol: float) -> bool:
    w = np.linalg.eigvalsh(_sym(M))
    return bool(np.min(w) >= -tol)

def _A_lambda_from_hist(m: np.ndarray, lam: float, D: int, eta: float, K: int) -> np.ndarray:
    """
    Per-eigenmode reduced matrix A_lam(m) in R^{(D+1)x(D+1)} for scalar lambda:
      top row: [1 - alpha*m0*lam, -alpha*m1*lam, ..., -alpha*mD*lam]
      subdiag: shift
    """
    alpha = float(eta) / float(K)
    m = np.asarray(m, dtype=int).reshape(-1)
    A = np.zeros((D + 1, D + 1), dtype=float)
    # top row
    A[0, 0] = 1.0 - alpha * float(m[0]) * float(lam)
    for j in range(1, D + 1):
        A[0, j] = -alpha * float(m[j]) * float(lam)
    # shift
    for j in range(1, D + 1):
        A[j, j - 1] = 1.0
    return A

def check_S_area_static_worst_dp_structured(
    system: Any,
    remodeling: Any,
    action_space: Any,
    T: int,
    params: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Structured S-area-T certificate for ASGD:
      - Enumerate actions by histogram m (exact equivalence classes; no adversary restriction).
      - Decompose along eigenmodes of H: n independent (D+1)-dim systems.
      - Check the standard S-area-T DP dominator condition on each eigenmode.
      - A candidate m* passes only if it passes for ALL eigenvalues.

    Returns dict compatible with existing S-check:
      {"pass": bool, "witness_d": delay_vector, "diagnostics": {...}}
    """
    # tolerances
    psd_tol = getattr(params, "psd_tol", 1e-10) if params is not None else 1e-10

    # get H
    H = None
    if hasattr(system, "objective") and hasattr(system.objective, "H"):
        H = np.asarray(system.objective.H, dtype=float)
    elif hasattr(system, "H"):
        H = np.asarray(system.H, dtype=float)
    elif hasattr(remodeling, "H"):
        H = np.asarray(remodeling.H, dtype=float)
    if H is None:
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "no_H_found", "details": None},
        }

    # eigenvalues (H SPD assumed in this stage)
    lambdas = np.linalg.eigvalsh(_sym(H))
    n = int(lambdas.shape[0])

    K = int(action_space.K)
    D = int(action_space.D)
    eta = float(getattr(remodeling, "eta", getattr(system, "eta", None)))
    if eta is None:
        eta = float(system.eta)

    # enumerate histogram actions (exact equivalence classes)
    if not hasattr(action_space, "enumerate_histograms"):
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "ActionSpace_missing_enumerate_histograms", "details": None},
        }
    actions_m = action_space.enumerate_histograms()
    if len(actions_m) == 0:
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "no_hist_actions", "details": None},
        }

    # heuristic candidate ordering: score(m)=sum_i trace(A^T Q A) on reduced system
    # Q_lam = diag([lam,0,...,0])
    scores = []
    for m in actions_m:
        sc = 0.0
        for lam in lambdas:
            A = _A_lambda_from_hist(m, float(lam), D=D, eta=eta, K=K)
            # trace(A^T Q A) = lam * || first-row of A ||^2 because Q selects state[0]
            sc += float(lam) * float(np.sum(A[0, :] ** 2))
        scores.append(sc)
    order = list(np.argsort(-np.asarray(scores, dtype=float)))

    # Try each candidate m*
    for idx_star in order:
        m_star = np.asarray(actions_m[int(idx_star)], dtype=int)

        ok_all_modes = True
        counterexample = None

        # Precompute A_star for each lambda
        A_star_list = [
            _A_lambda_from_hist(m_star, float(lam), D=D, eta=eta, K=K) for lam in lambdas
        ]

        # For each eigenmode, run DP recursion and check dominance vs all other actions
        for mode_i, lam in enumerate(lambdas):
            A_star = A_star_list[mode_i]
            Q = np.zeros((D + 1, D + 1), dtype=float)
            Q[0, 0] = float(lam)

            P_next = np.zeros((D + 1, D + 1), dtype=float)

            certified_mode = True
            for t in range(T - 1, -1, -1):
                M_star = _sym(A_star.T @ P_next @ A_star)

                for j, m in enumerate(actions_m):
                    A = _A_lambda_from_hist(m, float(lam), D=D, eta=eta, K=K)
                    M = _sym(A.T @ P_next @ A)
                    Delta = _sym(M_star - M)
                    if not _psd(Delta, psd_tol):
                        certified_mode = False
                        min_eig = float(np.min(np.linalg.eigvalsh(Delta)))
                        counterexample = {
                            "mode": int(mode_i),
                            "lambda": float(lam),
                            "t": int(t),
                            "m_star": m_star.tolist(),
                            "m_bad": np.asarray(m, dtype=int).tolist(),
                            "min_eig_diff": float(min_eig),
                        }
                        break

                if not certified_mode:
                    break

                P_next = _sym(Q + A_star.T @ P_next @ A_star)

            if not certified_mode:
                ok_all_modes = False
                break

        if ok_all_modes:
            # return a canonical delay-vector witness (expand histogram)
            if hasattr(action_space, "histogram_to_delay"):
                d = action_space.histogram_to_delay(m_star)
            else:
                # fallback: expand by hand
                d = []
                for j, cnt in enumerate(m_star.tolist()):
                    d.extend([j] * int(cnt))
                d = np.asarray(d, dtype=int)

            return {
                "pass": True,
                "witness_d": d,
                "diagnostics": {
                    "failed": False,
                    "reason": None,
                    "details": {
                        "certificate": "Structured S-area-T: histogram actions + eigenmode (H) decomposition",
                        "T": int(T),
                        "num_hist_actions_checked": int(len(actions_m)),
                        "candidate_score": float(scores[int(idx_star)]),
                        "counterexample": None,
                    },
                },
            }

    return {
        "pass": False,
        "witness_d": None,
        "diagnostics": {
            "failed": True,
            "reason": "S_area_T_no_static_global_maximizer_found_structured",
            "details": {
                "T": int(T),
                "num_hist_actions_checked": int(len(actions_m)),
                "note": "No histogram action m* dominates all others across all eigenmodes of H.",
            },
        },
    }



def static_worst_search(
    system: Any,
    action_space: Any,
    T: int,
) -> Dict[str, Any]:
    """Brute-force worst among static actions by area J_T(d). Always returns a witness."""
    actions = _all_actions(action_space)
    uniq: Dict[Tuple[int, ...], np.ndarray] = {}
    for a in actions:
        uniq[_canonicalize_delay(a)] = _as_np_int_vec(a)
    actions = list(uniq.values())

    bestJ = -np.inf
    bestd = None
    for d in actions:
        j = _eval_JT(system, d, T)
        if j > bestJ:
            bestJ = j
            bestd = d

    return {"best_d": _as_np_int_vec(bestd), "best_JT": float(bestJ)}


def check_not_static_worst_dp(
    system: Any,
    remodeling: Any,
    action_space: Any,
    params: Optional[StaticSearchParams] = None,
) -> Dict[str, Any]:
    """
    Strong sufficient condition that global 'static worst' is IMPOSSIBLE (for any horizon T>=2),
    under the area objective l_t = s_t^T Q s_t.

    If params.subspace_basis is provided, this becomes a subspace-aware certificate:
      - We look for a dominator on span(U) (i.e., U^T (M_star - M) U ⪰ 0 for all competitors).
      - If no dominator exists on span(U), then static-worst is impossible under the restricted initial-set game.
    """
    if params is None:
        params = StaticSearchParams()

    if remodeling is None or not hasattr(remodeling, "get_A"):
        return {
            "not_static_certified": False,
            "diagnostics": {"failed": True, "reason": "no_remodeling_get_A", "details": None},
            "dominators": [],
        }

    actions = _all_actions(action_space)
    uniq: Dict[Tuple[int, ...], np.ndarray] = {}
    for a in actions:
        uniq[_canonicalize_delay(a)] = _as_np_int_vec(a)
    actions = list(uniq.values())
    if len(actions) == 0:
        return {
            "not_static_certified": False,
            "diagnostics": {"failed": True, "reason": "no_actions", "details": None},
            "dominators": [],
        }

    A_list: List[np.ndarray] = []
    for d in actions:
        A = np.asarray(remodeling.get_A(d), dtype=float)
        A_list.append(A)

    m = int(A_list[0].shape[0])
    for A in A_list:
        if A.shape != (m, m):
            return {
                "not_static_certified": False,
                "diagnostics": {"failed": True, "reason": "A_shape_inconsistent", "details": None},
                "dominators": [],
            }

    Q = _get_Q(system, remodeling, m, action_space)
    if Q is None:
        return {
            "not_static_certified": False,
            "diagnostics": {"failed": True, "reason": "no_Q_available", "details": "Provide system.get_Q()/system.Q or remodeling.get_Q()/remodeling.Q."},
            "dominators": [],
        }
    Q = _sym(np.asarray(Q, dtype=float))
    if Q.shape != (m, m):
        return {
            "not_static_certified": False,
            "diagnostics": {"failed": True, "reason": "Q_shape_mismatch", "details": {"Q_shape": list(Q.shape), "A_dim": m}},
            "dominators": [],
        }

    U = None
    if params.subspace_basis is not None:
        U = np.asarray(params.subspace_basis, dtype=float)
        if U.ndim != 2 or U.shape[0] != m:
            return {
                "not_static_certified": False,
                "diagnostics": {
                    "failed": True,
                    "reason": "subspace_basis_shape_mismatch",
                    "details": {"U_shape": list(U.shape), "A_dim": m},
                },
                "dominators": [],
            }

    # compute M_a = A^T Q A
    M_list = [_sym(A.T @ Q @ A) for A in A_list]

    dominators: List[int] = []
    best_witness = None  # (idx_star, idx_bad, min_eig_diff)

    for i, M_star in enumerate(M_list):
        ok = True
        worst_min_eig = +np.inf
        worst_j = None

        for j, M in enumerate(M_list):
            if i == j:
                continue
            diff = _sym(M_star - M)

            if U is None:
                w = np.linalg.eigvalsh(diff)
                minw = float(np.min(w))
            else:
                _, minw = _psd_on_subspace(diff, U, params.psd_tol)

            if minw < -params.psd_tol:
                ok = False
                if minw < worst_min_eig:
                    worst_min_eig = minw
                    worst_j = j

        if ok:
            dominators.append(i)
        else:
            if worst_j is not None:
                if best_witness is None or worst_min_eig < best_witness[2]:
                    best_witness = (i, worst_j, worst_min_eig)

    if len(dominators) == 0:
        details = {
            "certificate": (
                "No PSD-dominator among M_a = A(a)^T Q A(a) on the reachable subspace span(U). "
                "Hence 2-step DP optimal action depends on state; static-worst is impossible under this initial-set game."
                if U is not None
                else
                "No PSD-dominator among M_a = A(a)^T Q A(a). Hence 2-step DP optimal action depends on state; global static-worst is impossible for any T>=2."
            ),
            "num_actions_checked": len(actions),
            "check_space": "subspace" if U is not None else "full",
            "subspace_name": params.subspace_name,
            "subspace_dim": int(U.shape[1]) if U is not None else None,
        }
        if best_witness is not None:
            i, j, min_eig = best_witness
            details["witness_pair"] = {
                "a_candidate": actions[int(i)].tolist(),
                "a_competitor": actions[int(j)].tolist(),
                "min_eig(M_candidate - M_competitor)": float(min_eig),
            }
        return {
            "not_static_certified": True,
            "diagnostics": {"failed": False, "reason": None, "details": details},
            "dominators": [],
        }

    return {
        "not_static_certified": False,
        "diagnostics": {
            "failed": False,
            "reason": "dominator_exists_at_2step",
            "details": {
                "note": (
                    "At the 2-step level, some action PSD-dominates all others on span(U). "
                    "This does NOT prove static-worst, but prevents this particular non-static certificate."
                    if U is not None
                    else
                    "At the 2-step level, some action PSD-dominates all others. This does NOT prove global static-worst, but prevents this particular non-static certificate."
                ),
                "dominators": [actions[i].tolist() for i in dominators],
                "num_actions_checked": len(actions),
                "check_space": "subspace" if U is not None else "full",
                "subspace_name": params.subspace_name,
                "subspace_dim": int(U.shape[1]) if U is not None else None,
            },
        },
        "dominators": [actions[i] for i in dominators],
    }


def _M_lambda_from_hist(m: np.ndarray, lam: float, D: int, eta: float, K: int) -> np.ndarray:
    """
    Reduced 2-step DP matrix component:
    M(m) = A(m)^T Q A(m), with Q = diag([lam, 0,...,0])  (size (D+1)x(D+1))
    """
    A = _A_lambda_from_hist(m, lam, D=D, eta=eta, K=K)
    Q = np.zeros((D + 1, D + 1), dtype=float)
    Q[0, 0] = float(lam)
    return _sym(A.T @ Q @ A)

def _min_eig(M: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(_sym(M))))

def check_not_static_worst_dp_structured(
    system: Any,
    remodeling: Any,
    action_space: Any,
    params: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Structured Not-S certificate (2-step DP obstruction) for ASGD:

    Full-space statement you were using:
    Let M_a = A(a)^T Q A(a). If there is NO PSD-dominator a* s.t. M_{a*} ⪰ M_a for all a,
    then the 2-step DP optimal action depends on state direction => global static-worst is impossible for any T>=2.

    Structured version:
    - Enumerate actions by histogram m (exact equivalence classes; no adversary restriction).
    - Decompose along eigenmodes of H: M(m) becomes block-diagonal across eigenvalues.
    - A dominator must dominate in *every* eigenmode.
    - If no histogram dominates all others across all eigenmodes, certify Not-S.

    Returns dict compatible with your existing return shape:
    {
        "not_static_certified": bool,
        "diagnostics": {...},
        "dominators": [...]
    }
    """
    psd_tol = getattr(params, "psd_tol", 1e-10) if params is not None else 1e-10

    # get H
    H = None
    if hasattr(system, "objective") and hasattr(system.objective, "H"):
        H = np.asarray(system.objective.H, dtype=float)
    elif hasattr(system, "H"):
        H = np.asarray(system.H, dtype=float)
    elif hasattr(remodeling, "H"):
        H = np.asarray(remodeling.H, dtype=float)

    if H is None:
        return {
            "not_static_certified": False,
            "diagnostics": {"failed": True, "reason": "no_H_found", "details": None},
            "dominators": [],
        }

    lambdas = np.linalg.eigvalsh(_sym(H))

    if not hasattr(action_space, "enumerate_histograms"):
        return {
            "not_static_certified": False,
            "diagnostics": {"failed": True, "reason": "ActionSpace_missing_enumerate_histograms", "details": None},
            "dominators": [],
        }

    actions_m = action_space.enumerate_histograms()
    if len(actions_m) == 0:
        return {
            "not_static_certified": False,
            "diagnostics": {"failed": True, "reason": "no_hist_actions", "details": None},
            "dominators": [],
        }

    K = int(action_space.K)
    D = int(action_space.D)
    eta = float(getattr(remodeling, "eta", getattr(system, "eta", None)))
    if eta is None:
        eta = float(system.eta)

    # Precompute M_lambda(m) for all m and all eigenmodes
    # M_mode[i][j] = M^{(i)}(m_j)  where i over lambdas, j over actions
    M_mode: List[List[np.ndarray]] = []
    for lam in lambdas:
        Mi = []
        for m in actions_m:
            Mi.append(_M_lambda_from_hist(m, float(lam), D=D, eta=eta, K=K))
        M_mode.append(Mi)

    dominators: List[int] = []
    best_witness = None  # (idx_candidate, idx_competitor, mode_i, min_eig_diff)

    # A candidate dominates if for every competitor, for every mode, diff ⪰ 0
    for i in range(len(actions_m)):
        ok = True
        worst = None  # (min_eig, j, mode_i)
        for j in range(len(actions_m)):
            if i == j:
                continue

            # Check all eigenmodes
            for mode_i in range(len(lambdas)):
                diff = _sym(M_mode[mode_i][i] - M_mode[mode_i][j])
                mine = _min_eig(diff)
                if mine < -psd_tol:
                    ok = False
                    if (worst is None) or (mine < worst[0]):
                        worst = (mine, j, mode_i)
                    break  # fail fast on this competitor
            if not ok:
                break

        if ok:
            dominators.append(i)
        else:
            if worst is not None:
                mine, j_bad, mode_i = worst
                if best_witness is None or mine < best_witness[3]:
                    best_witness = (i, j_bad, mode_i, mine)

    if len(dominators) == 0:
        details: Dict[str, Any] = {
            "certificate": (
                "No PSD-dominator among M_m = A(m)^T Q A(m) across all eigenmodes of H "
                "(histogram action space). Hence 2-step DP optimal action depends on state; "
                "global static-worst is impossible for any T>=2."
            ),
            "num_hist_actions_checked": int(len(actions_m)),
            "num_eigenmodes": int(len(lambdas)),
        }
        if best_witness is not None:
            i, j, mode_i, min_eig = best_witness
            details["witness_pair"] = {
                "m_candidate": np.asarray(actions_m[int(i)], dtype=int).tolist(),
                "m_competitor": np.asarray(actions_m[int(j)], dtype=int).tolist(),
                "mode_index": int(mode_i),
                "lambda": float(lambdas[int(mode_i)]),
                "min_eig(M_candidate - M_competitor)": float(min_eig),
            }
            # also provide delay-vector form for readability if helper exists
            if hasattr(action_space, "histogram_to_delay"):
                details["witness_pair"]["a_candidate_delay"] = action_space.histogram_to_delay(actions_m[int(i)]).tolist()
                details["witness_pair"]["a_competitor_delay"] = action_space.histogram_to_delay(actions_m[int(j)]).tolist()

        return {
            "not_static_certified": True,
            "diagnostics": {"failed": False, "reason": None, "details": details},
            "dominators": [],
        }

    # dominator exists at 2-step level (prevents this Not-S certificate)
    dom_list = [np.asarray(actions_m[int(i)], dtype=int) for i in dominators]
    dom_json = [d.tolist() for d in dom_list]
    dom_delay = None
    if hasattr(action_space, "histogram_to_delay"):
        dom_delay = [action_space.histogram_to_delay(actions_m[int(i)]).tolist() for i in dominators]

    return {
        "not_static_certified": False,
        "diagnostics": {
            "failed": False,
            "reason": "dominator_exists_at_2step_structured",
            "details": {
                "note": (
                    "At the 2-step level, some histogram action PSD-dominates all others across all eigenmodes of H. "
                    "This does NOT prove global static-worst, but prevents this particular non-static certificate."
                ),
                "dominators_hist": dom_json,
                "dominators_delay": dom_delay,
                "num_hist_actions_checked": int(len(actions_m)),
                "num_eigenmodes": int(len(lambdas)),
            },
        },
        "dominators": dom_list,
    }

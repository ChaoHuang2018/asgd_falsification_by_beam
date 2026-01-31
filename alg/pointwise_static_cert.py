# alg/pointwise_static_cert.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from core.schedules import StaticSchedule
from alg.static_search import static_worst_search, _A_lambda_from_hist  # reuse existing helper


@dataclass
class PointwiseStaticCertParams:
    """
    Pointwise (single x0) sufficient certifier parameters.
    """
    # Numerical tolerance for comparing candidate vs adversary upper bound
    margin: float = 1e-12

    # Use continuous polytope vertices (safe upper bound). Recommended True.
    relax_vertices: bool = True

    # Cache rho_lambda across calls (if you run many samples)
    cache_rho: bool = True


def _get_H(system: Any, remodeling: Any) -> np.ndarray:
    if hasattr(system, "objective") and hasattr(system.objective, "H"):
        return np.asarray(system.objective.H, dtype=float)
    if hasattr(system, "H"):
        return np.asarray(system.H, dtype=float)
    if remodeling is not None and hasattr(remodeling, "H"):
        return np.asarray(remodeling.H, dtype=float)
    raise ValueError("Cannot find H on system.objective.H / system.H / remodeling.H")


def _delay_to_hist(d: np.ndarray, D: int) -> np.ndarray:
    d = np.asarray(d, dtype=int).reshape(-1)
    m = np.zeros(D + 1, dtype=int)
    for x in d:
        if x < 0 or x > D:
            raise ValueError(f"delay entry out of range: {x} not in [0,{D}]")
        m[int(x)] += 1
    return m


def _project_init_to_modes(z0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    z0: (D+1, n)
    U: (n, n) eigenvectors of H
    Return y0: (D+1, n) where y0[j] = U^T z0[j]
    """
    return (U.T @ z0.T).T  # (D+1, n)


def _mode_step_from_hist(z_mode: np.ndarray, m: np.ndarray, lam: float, eta: float, K: int) -> np.ndarray:
    """
    z_mode: (D+1,) for a single eigenmode
    m: histogram (D+1,) possibly float if relax_vertices=True
    """
    alpha = float(eta) / float(K)
    # new x = x0 - alpha*lam * sum_j m_j * x_j
    s = float(np.dot(m, z_mode))
    x_next = float(z_mode[0]) - alpha * float(lam) * s
    z_next = np.empty_like(z_mode, dtype=float)
    z_next[0] = x_next
    z_next[1:] = z_mode[:-1]
    return z_next


def _geom_sum(rho2: float, k: int) -> float:
    # sum_{i=0}^{k-1} rho^(2i)
    if k <= 0:
        return 0.0
    if abs(rho2 - 1.0) < 1e-15:
        return float(k)
    return float((rho2**k - 1.0) / (rho2 - 1.0))


def certify_pointwise_static_worst_upperbound(
    system: Any,
    remodeling: Any,
    action_space: Any,
    T: int,
    params: Optional[PointwiseStaticCertParams] = None,
) -> Dict[str, Any]:
    """
    Sufficient pointwise certificate:

    Let x0 be the fixed init produced by system.init_state().
    Let d* be the empirically worst STATIC action for this x0 (computed exactly over static set).

    Define an upper bound H_k(.) on the remaining worst-case cost from a given state,
    constructed via per-eigenmode spectral-norm bounds rho_lambda over the continuous histogram polytope.

    At each time t along the rollout under d*, we check:
        H_{T-t-1}( next_state(d*) ) >= max_{m in Vert(M)} H_{T-t-1}( next_state(m) )
    where M is the histogram polytope and Vert(M) its vertices.

    If all t pass -> certify that (for this x0) worst schedule is achieved by a static action (d*).
    """
    if params is None:
        params = PointwiseStaticCertParams()

    H = _get_H(system, remodeling)
    H = 0.5 * (H + H.T)
    lambdas, U = np.linalg.eigh(H)
    n = int(lambdas.shape[0])

    K = int(action_space.K)
    D = int(action_space.D)
    eta = float(getattr(remodeling, "eta", getattr(system, "eta", None)))

    # fixed point init
    z0 = np.asarray(system.init_state(), dtype=float)  # (D+1, n)
    if z0.shape != (D + 1, n):
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "init_state_shape_mismatch", "details": {"z0_shape": list(z0.shape)}},
        }

    # best static for this x0 (baseline candidate)
    Sbest = static_worst_search(system, action_space, T)
    d_star = np.asarray(Sbest["best_d"], dtype=int)
    m_star = _delay_to_hist(d_star, D=D)

    # vertices (continuous polytope default for safety)
    if not hasattr(action_space, "enumerate_histogram_vertices"):
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "ActionSpace_missing_enumerate_histogram_vertices", "details": None},
        }
    V = action_space.enumerate_histogram_vertices(relax=params.relax_vertices)
    if len(V) == 0:
        return {
            "pass": False,
            "witness_d": None,
            "diagnostics": {"failed": True, "reason": "no_vertices", "details": None},
        }

    # --- rho_lambda cache (optional) ---
    cache_key = None
    if params.cache_rho:
        cache_key = ("rho_lambda_cache", int(K), int(D), float(eta), tuple(np.round(lambdas, 12).tolist()), bool(params.relax_vertices))
        if hasattr(system, "_pointwise_cache") and cache_key in system._pointwise_cache:
            rho_lams = system._pointwise_cache[cache_key]
        else:
            rho_lams = None
    else:
        rho_lams = None

    if rho_lams is None:
        rho_lams = np.zeros(n, dtype=float)
        for i, lam in enumerate(lambdas):
            # upper bound: max_{v in vertices} ||A_lam(v)||_2
            best = 0.0
            for v in V:
                # use existing helper; it expects integer hist, but formula is linear in m,
                # so we can safely build the matrix ourselves when v is float:
                if np.issubdtype(np.asarray(v).dtype, np.integer):
                    A = _A_lambda_from_hist(np.asarray(v, dtype=int), float(lam), D=D, eta=eta, K=K)
                else:
                    # build A for float histogram
                    alpha = float(eta) / float(K)
                    A = np.zeros((D + 1, D + 1), dtype=float)
                    A[0, 0] = 1.0 - alpha * float(v[0]) * float(lam)
                    for j in range(1, D + 1):
                        A[0, j] = -alpha * float(v[j]) * float(lam)
                    for j in range(1, D + 1):
                        A[j, j - 1] = 1.0
                smax = float(np.linalg.svd(A, compute_uv=False)[0])
                if smax > best:
                    best = smax
            rho_lams[i] = best

        if params.cache_rho:
            if not hasattr(system, "_pointwise_cache"):
                system._pointwise_cache = {}
            system._pointwise_cache[cache_key] = rho_lams

    # precompute geom sums per mode and k
    # G[i][k] = sum_{r=0}^{k-1} rho_i^{2r}
    max_k = T
    G = np.zeros((n, max_k + 1), dtype=float)
    for i in range(n):
        rho2 = float(rho_lams[i] ** 2)
        for k in range(max_k + 1):
            G[i, k] = _geom_sum(rho2, k)

    # mode-space init histories: y0[j] = U^T z0[j]
    y0 = _project_init_to_modes(z0, U)  # (D+1, n)

    # store per-mode history vectors z_mode[i] of length D+1
    z_modes = np.asarray([y0[:, i] for i in range(n)], dtype=float)  # (n, D+1)

    z_path = np.zeros((T + 1, n, D + 1), dtype=float)
    z_path[0] = z_modes
    m_star_f = m_star.astype(float)
    for t in range(T):
        for i, lam in enumerate(lambdas):
            z_path[t + 1, i] = _mode_step_from_hist(z_path[t, i], m_star_f, float(lam), eta=eta, K=K)

    stage_loss = np.zeros(T, dtype=float)
    for t in range(T):
        # x_t^{(i)} is the first coordinate of the mode history vector
        xt = z_path[t, :, 0]  # (n,)
        stage_loss[t] = float(np.dot(lambdas, xt * xt))

    tail_true = np.zeros(T + 1, dtype=float)
    for t in range(T - 1, -1, -1):
        tail_true[t] = tail_true[t + 1] + stage_loss[t]

    # rollout along candidate static m_star, and check at each t
    worst_vertex_at_fail = None
    fail_t = None
    gap = None

    for t in range(T):
        k_remain = T - t
        # current candidate state on the static trajectory
        z_curr = z_path[t]          # (n, D+1)
        z_next_star = z_path[t + 1] # (n, D+1)

        # upper bound value at next state (k_remain-1 steps left)
        k_tail = k_remain - 1
        # candidate uses TRUE remaining tail (exact, deterministic under static m_star)
        # stage loss at time t is action-independent, so we compare tails starting at t+1.
        val_star = float(tail_true[t + 1])
        
        # adversary upper bound: max over vertices
        val_adv = -1e300
        arg_adv = None
        for v in V:
            v = np.asarray(v, dtype=float).reshape(-1)
            z_next_v = np.zeros_like(z_curr)
            for i, lam in enumerate(lambdas):
                z_next_v[i] = _mode_step_from_hist(z_curr[i], v, float(lam), eta=eta, K=K)
            vv = 0.0
            for i, lam in enumerate(lambdas):
                vv += float(lam) * float(np.dot(z_next_v[i], z_next_v[i])) * float(G[i, k_tail])
            if vv > val_adv:
                val_adv = vv
                arg_adv = v

        if val_star + params.margin < val_adv:
            fail_t = t
            worst_vertex_at_fail = arg_adv
            gap = float(val_adv - val_star)
            break

    if fail_t is None:
        return {
            "pass": True,
            "witness_d": d_star,
            "diagnostics": {
                "failed": False,
                "reason": None,
                "details": {
                    "certificate": "pointwise_static_worst_upperbound (vertices + eigenmodes)",
                    "T": int(T),
                    "relax_vertices": bool(params.relax_vertices),
                    "rho_lams": [float(x) for x in rho_lams.tolist()],
                    "best_static_JT": float(Sbest.get("best_JT", np.nan)),
                },
            },
        }

    return {
        "pass": False,
        "witness_d": None,
        "diagnostics": {
            "failed": True,
            "reason": "pointwise_certificate_failed",
            "details": {
                "t": int(fail_t),
                "gap_adv_minus_star": float(gap),
                "adv_vertex_m": None if worst_vertex_at_fail is None else np.asarray(worst_vertex_at_fail).tolist(),
                "note": "This is a sufficient test; FAIL means UNKNOWN (not a counterexample).",
            },
        },
    }

# alg/beam_certificate.py

import numpy as np
from typing import Dict, List, Tuple, Any

from alg.bound_search import _A_lambda_from_hist  

def _block_diag(mats: List[np.ndarray]) -> np.ndarray:
    n = sum(M.shape[0] for M in mats)
    out = np.zeros((n, n), dtype=float)
    i = 0
    for M in mats:
        d = M.shape[0]
        out[i:i+d, i:i+d] = M
        i += d
    return out

def _psd_check(A: np.ndarray, tol: float = 1e-10) -> bool:
    # symmetric part for safety
    A = 0.5 * (A + A.T)
    lam_min = np.linalg.eigvalsh(A).min()
    return lam_min >= -tol

def build_F_of_m(lambdas: np.ndarray, D: int, eta: float, K: int, m: np.ndarray) -> np.ndarray:
    # Mode-wise lifted A_lambda(m), then block diagonal across modes
    mats = []
    for lam in lambdas:
        A = _A_lambda_from_hist(m, lam=float(lam), D=D, eta=float(eta), K=int(K))
        mats.append(A.astype(float))
    return _block_diag(mats)

def build_Rk(lambdas: np.ndarray, G: np.ndarray, D: int, k: int) -> np.ndarray:
    # ub_tail_geom(z,k)=sum_i lambdas[i]*||z_i||^2 * G[i,k]
    # => block diag with (lambdas[i]*G[i,k]) * I_{D+1}
    blocks = []
    for i, lam in enumerate(lambdas):
        w = float(lam) * float(G[i, k])
        blocks.append(w * np.eye(D+1))
    return _block_diag(blocks)

def verify_prior_bellman_sufficient_condition(
    action_space: Any,
    lambdas: np.ndarray,     # eigenvalues of PSD H in spectral basis
    G: np.ndarray,           # precomputed geometric sums using rho_lams
    D: int,
    K: int,
    B: int,
    eta: float,
    T: int,
    tol_psd: float = 1e-10,
    tol_eq: float = 1e-9,
) -> Dict[str, Any]:
    """
    Returns a report dict:
      - pass_C1, pass_C2, overall_pass
      - required_candidates_per_expand (=2 if pass)
      - endpoint actions m_min, m_max
      - diagnostics on failures
    """

    actions = [np.asarray(m, dtype=int) for m in action_space.enumerate_histograms()]
    # sanity: filter to constraints (optional if action_space already guarantees)
    def s_of(m): return int(np.dot(np.arange(D+1, dtype=int), m))

    # Group actions by s
    by_s: Dict[int, List[np.ndarray]] = {}
    for m in actions:
        s = s_of(m)
        if s <= B and m.sum() == K:
            by_s.setdefault(s, []).append(m)

    # ---- C1: for each k, all actions with same s yield same M_{m,k} = F^T Rk F
    C1_fail = []
    Ms_cache: Dict[Tuple[int, int], np.ndarray] = {}  # (s,k) -> representative M

    for k in range(0, T+1):
        Rk = build_Rk(lambdas, G, D, k)
        for s, ms in by_s.items():
            if len(ms) <= 1:
                continue
            M_ref = None
            for idx, m in enumerate(ms):
                Fm = build_F_of_m(lambdas, D, eta, K, m)
                Mm = Fm.T @ Rk @ Fm
                if M_ref is None:
                    M_ref = Mm
                else:
                    if np.linalg.norm(Mm - M_ref, ord='fro') > tol_eq:
                        C1_fail.append({"k": k, "s": s, "m0": ms[0].tolist(), "m_bad": m.tolist()})
                        break
            if M_ref is not None:
                Ms_cache[(s, k)] = M_ref
            if C1_fail:
                break
        if C1_fail:
            break

    pass_C1 = (len(C1_fail) == 0)

    # ---- C2: PSD discrete concavity in s for each k
    C2_fail = []
    if pass_C1:
        for k in range(0, T+1):
            # only check s that exist
            s_list = sorted(by_s.keys())
            # we need consecutive s to talk about second difference; restrict to 0..B
            for s in range(0, B-1):
                if (s, k) in Ms_cache and (s+1, k) in Ms_cache and (s+2, k) in Ms_cache:
                    Sd = 2*Ms_cache[(s+1, k)] - Ms_cache[(s, k)] - Ms_cache[(s+2, k)]
                    if not _psd_check(Sd, tol=tol_psd):
                        C2_fail.append({"k": k, "s": s})
                        break
            if C2_fail:
                break
    pass_C2 = (len(C2_fail) == 0) if pass_C1 else False

    # endpoint actions (purely from K,D,B)
    m_min = np.zeros(D+1, dtype=int)
    m_min[0] = K

    # maximize s subject to <=B and j<=D
    s_max = min(B, K*D)
    a = s_max // K
    b = min(a + 1, D)
    r = s_max - a*K
    if b > D:
        # should not happen with b=min(a+1,D), but keep safe
        b = D
    m_max = np.zeros(D+1, dtype=int)
    if r == 0 or a == b:
        # exactly fits at a
        m_max[a] = K
    else:
        # mix a and b
        m_max[b] = r
        m_max[a] = K - r

    # check m_min/m_max are indeed in action set (optional)
    m_min_ok = any(np.all(m == m_min) for m in actions)
    m_max_ok = any(np.all(m == m_max) for m in actions)

    overall_pass = pass_C1 and pass_C2 and m_min_ok and m_max_ok

    return {
        "pass_C1_score_depends_only_on_s": pass_C1,
        "pass_C2_discrete_concavity_in_s": pass_C2,
        "m_min": m_min.tolist(),
        "m_max": m_max.tolist(),
        "m_min_in_action_space": m_min_ok,
        "m_max_in_action_space": m_max_ok,
        "required_candidates_per_expand": 2 if overall_pass else None,
        "overall_pass": overall_pass,
        "C1_fail": C1_fail[:3],
        "C2_fail": C2_fail[:3],
    }

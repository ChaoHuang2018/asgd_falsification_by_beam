
"""
beam_certificate_subspace.py

New prior (a-priori) certificate based on **candidate-set dominance on a reachable subspace**.

Motivation:
- The previous "same-s equivalence + concavity" certificate is too strong for D>1,B>1 at practical eta.
- Here we certify beam-optimality by proving that, on the reachable subspace C_T generated from (z0, K,D,B,eta,T),
  the maximizer(s) of the beam scoring quadratic form are contained in a small candidate set S_k (often extremes),
  because all other actions are PSD-dominated on C_T.

This is still **prior**:
- It uses only (H, z0, eta, K, D, B, T) and the known action set.
- It does NOT use any beam rollout, optimal schedule, or posterior state trajectory.

What it guarantees (lookahead_steps=0, no prefilter):
- At remaining horizon k, beam compares actions by the tail upper bound ub_tail_geom(z_next, k-1),
  which is a quadratic form z^T M_{m,k-1} z on the lifted-mode state z (stacked across modes).
- If for each k there exists a set S_k such that every action m outside S_k is PSD-dominated by some action in S_k
  on C_T, then the argmax over all actions equals the argmax over S_k for all reachable states.
- Therefore you may safely restrict per-node expansion to S_k without ever discarding a potentially optimal action
  under the beam score. This yields a *checkable* sufficient condition tying candidates_per_expand to |S_k|.

NOTE:
- This certificate is about **beam's scoring objective** (prefix + ub_tail_geom with lookahead=0).
  If you run beam exactly with that scoring and keep at least |S_k| candidates per expansion at depth k,
  beam will not lose the best-scoring successor at that depth.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

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


def _orth(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for the columns of A via QR; drops near-zero columns."""
    if A.size == 0:
        return A
    Q, R = np.linalg.qr(A, mode="reduced")
    # keep columns with non-trivial diagonal in R
    diag = np.abs(np.diag(R))
    keep = diag > tol
    return Q[:, keep] if keep.ndim == 1 else Q


def _project_residual_norm(v: np.ndarray, U: np.ndarray) -> float:
    """||v - UU^T v||_2 when U has orthonormal columns (or U empty)."""
    if U.size == 0:
        return float(np.linalg.norm(v))
    return float(np.linalg.norm(v - U @ (U.T @ v)))


def _psd_check(A: np.ndarray, tol: float = 1e-10) -> bool:
    A = 0.5 * (A + A.T)
    w = np.linalg.eigvalsh(A)
    return float(w.min()) >= -tol

def _flatten_z_modes(z_modes: np.ndarray) -> np.ndarray:
    # z_modes shape: (n_modes, D+1) or equivalent; flatten to 1D
    return np.asarray(z_modes, dtype=float).reshape(-1)

def _prefix_nondominated_filter(nodes, U_pref: np.ndarray, psd_tol: float):
    """
    nodes: list of tuples (ub, z_modes, prefix_cost, word)
    Keep nodes that are not strictly PSD-dominated in projected outer-product space,
    with prefix_cost as an additional monotone coordinate.
    """
    if len(nodes) <= 1:
        return nodes

    def psd(A: np.ndarray) -> bool:
        A = 0.5 * (A + A.T)
        w = np.linalg.eigvalsh(A)
        return float(w.min()) >= -float(psd_tol)

    # Precompute projected Grams
    proj = []
    for ub, z_modes, prefix_cost, word in nodes:
        z = _flatten_z_modes(z_modes)
        yz = U_pref.T @ z
        G = np.outer(yz, yz)  # U^T (z z^T) U
        proj.append((prefix_cost, G))

    keep = []
    for i in range(len(nodes)):
        ci, Gi = proj[i]
        dominated = False
        for j in range(len(nodes)):
            if i == j:
                continue
            cj, Gj = proj[j]
            if cj < ci:
                continue
            # strict dominance: (Gj-Gi) PSD and NOT (Gi-Gj) PSD
            if psd(Gj - Gi) and (not psd(Gi - Gj)):
                dominated = True
                break
        if not dominated:
            keep.append(nodes[i])
    return keep


def _reachable_subspace_basis(
    ubm: Any,
    actions: List[np.ndarray],
    z0_modes: np.ndarray,
    depth: int,
    tol_new_vec: float = 1e-9,
) -> np.ndarray:
    """
    Build C_depth = span{ F(w) z0 : |w|<=depth } in the *stacked* lifted-mode space.
    This is prior and finite-horizon: for horizon T, choose depth=T-1.
    """
    n, D1 = z0_modes.shape
    N = n * D1

    def flatten(zm: np.ndarray) -> np.ndarray:
        return zm.reshape(-1)

    def unflatten(v: np.ndarray) -> np.ndarray:
        return v.reshape(n, D1)

    # basis as orthonormal columns U
    U = np.zeros((N, 0), dtype=float)
    frontier = [flatten(z0_modes)]
    # add z0
    v0 = frontier[0]
    if np.linalg.norm(v0) == 0:
        return U
    U = _orth(v0[:, None])

    # BFS-style closure up to given depth
    for _d in range(depth):
        new_frontier: List[np.ndarray] = []
        for v in frontier:
            zm = unflatten(v)
            for m in actions:
                nxt = ubm.step(zm, m)
                vn = flatten(nxt)
                if _project_residual_norm(vn, U) > tol_new_vec:
                    # augment basis
                    U = _orth(np.concatenate([U, vn[:, None]], axis=1))
                    new_frontier.append(vn)
        frontier = new_frontier
        if not frontier:
            break
    return U


def _Rk_from_lams_G(lambdas: np.ndarray, G: np.ndarray, D: int, k: int) -> np.ndarray:
    blocks = []
    for i, lam in enumerate(lambdas):
        w = float(lam) * float(G[i, k])
        blocks.append(w * np.eye(D + 1))
    return _block_diag(blocks)


def _F_of_m(lambdas: np.ndarray, D: int, eta: float, K: int, m: np.ndarray) -> np.ndarray:
    mats = []
    for lam in lambdas:
        mats.append(_A_lambda_from_hist(m, lam=float(lam), D=D, eta=float(eta), K=int(K)))
    return _block_diag(mats)


def _Mm_k(lambdas: np.ndarray, G: np.ndarray, D: int, eta: float, K: int, m: np.ndarray, k: int) -> np.ndarray:
    """
    Matrix for quadratic tail score at remaining steps k (i.e., ub_tail_geom(z,k) = z^T R_k z),
    and one-step lookahead compares z_next = F(m) z, so score = z^T (F^T R_k F) z.
    """
    Rk = _Rk_from_lams_G(lambdas, G, D, k)
    Fm = _F_of_m(lambdas, D, eta, K, m)
    return Fm.T @ Rk @ Fm

def _Q_stage_matrix_from_lambdas(lambdas: np.ndarray, D: int) -> np.ndarray:
    """
    Stage cost ell(z)=x^T H x in the stacked *mode* lifted state.
    In spectral(mode) coordinates, H becomes diag(lambdas) and ell only depends on the
    current slice (j=0) of each mode.

    For each mode r, the local (D+1)x(D+1) block has lam at (0,0) and zeros elsewhere.
    Full Q is block-diagonal over modes.
    """
    blocks = []
    for lam in np.asarray(lambdas, dtype=float):
        B = np.zeros((D + 1, D + 1), dtype=float)
        B[0, 0] = float(lam)
        blocks.append(B)
    return _block_diag(blocks)


def _nullspace_basis_psd(M: np.ndarray, zero_tol: float) -> np.ndarray:
    """
    Return an orthonormal basis for the (approximate) nullspace of symmetric PSD matrix M,
    i.e., eigenvectors with eigenvalue <= zero_tol.
    Output shape: (r, q) where q is nullity estimate.
    """
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    idx = np.where(w <= float(zero_tol))[0]
    if idx.size == 0:
        return np.zeros((M.shape[0], 0), dtype=float)
    return V[:, idx]


def _span_rank(A: np.ndarray, svd_tol: float) -> int:
    """
    Rank of column span of A using SVD threshold.
    """
    if A.size == 0:
        return 0
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return int(np.sum(s > float(svd_tol)))

def _rollout_true_JT(
    ubm: Any,
    z0_modes: np.ndarray,          # (n, D+1) or your internal shape
    hist_sequence: np.ndarray,     # (T, D+1)
) -> float:
    """
    Exact rollout under given histogram schedule, returning true JT = sum_t x_t^T H x_t.
    Must be consistent with ubm.step and ubm's internal basis.
    """
    z = z0_modes
    JT = 0.0
    for t in range(hist_sequence.shape[0]):
        # stage cost at current state
        JT += float(ubm.loss(z))
        # advance
        z = ubm.step(z, hist_sequence[t])
    return float(JT)

def _stage_cost_from_modes(z_modes: np.ndarray, lambdas: np.ndarray) -> float:
    # z_modes: (n, D+1) where column 0 is current x_t in spectral coords
    x0 = z_modes[:, 0]
    return float(np.sum(lambdas * (x0 * x0)))

@dataclass
class SubspaceCandidateCertificate:
    proxy_overall_pass: bool
    proxy_action_cert_pass: bool

    subspace_dim: int
    depth: int
    max_residual_added: float
    psd_tol: float
    # For each k, candidate set indices (into actions list)
    S_by_k: Dict[int, List[int]]
    required_candidates_per_expand: int
    required_beam_width_outer: int
    bucket_actions: List[np.ndarray]

    diagnostics: Dict[str, Any]

    # Backward-compatible aliases (avoid breaking older scripts)
    @property
    def overall_pass(self) -> bool:
        return bool(self.proxy_overall_pass)

    @property
    def action_cert_pass(self) -> bool:
        return bool(self.proxy_action_cert_pass)


def verify_prior_subspace_candidate_certificate(
    ubm: Any,
    action_space: Any,
    z0: np.ndarray,
    T: int,
    *,
    depth: Optional[int] = None,
    tol_new_vec: float = 1e-9,
    psd_tol: float = 1e-10,
    max_S: Optional[int] = None,
) -> SubspaceCandidateCertificate:
    """
    Prior certificate:
      1) Build C = reachable subspace up to 'depth' (default T-1) in stacked lifted-mode space.
      2) For each remaining k in [0..T], compute projected matrices P_{m,k} = U^T M_{m,k} U.
      3) Build S_k as the set of PSD-maximal actions (Pareto maxima) under order:
            m dominates m'  iff  P_{m,k} - P_{m',k} is PSD.
         Then any dominated action can never be the argmax for any z in C.
      4) If max_k |S_k| is not too large (or <= max_S if provided), return pass and the required candidates_per_expand.

    NOTE: This cert is about *not discarding* potentially best-scoring successors (under ub_tail_geom score).
          It does not require same-s equivalence and works for D>1,B>1.

    Returns:
      - S_by_k and required_candidates_per_expand = max_k |S_k|.
    """
    depth = (T - 1) if depth is None else int(depth)

    # enumerate all feasible histograms
    actions = [np.asarray(m, dtype=int) for m in action_space.enumerate_histograms()]

    # B' buckets: integer vertices (small, prior, geometry-aligned)
    if not hasattr(action_space, "enumerate_histogram_vertices"):
        raise ValueError("ActionSpace missing enumerate_histogram_vertices(relax=...) for B' buckets.")
    bucket_actions = [np.asarray(v, dtype=int).reshape(-1) for v in action_space.enumerate_histogram_vertices(relax=False)]
    # de-dup in case
    uniq = {}
    for m in bucket_actions:
        uniq[tuple(m.tolist())] = m
    bucket_actions = list(uniq.values())

    action_index = {tuple(a.tolist()): i for i, a in enumerate(actions)}
    bucket_indices = []
    for b in bucket_actions:
        key = tuple(np.asarray(b, dtype=int).reshape(-1).tolist())
        if key not in action_index:
            raise ValueError("bucket action not found in enumerate_histograms() action list")
        bucket_indices.append(action_index[key])
    bucket_indices = sorted(set(bucket_indices))



    # initialize z0 to modes
    z0_modes = ubm.init_z_modes(z0)  # (n, D+1)
    # build reachable subspace basis U
    U = _reachable_subspace_basis(ubm, actions, z0_modes, depth=depth, tol_new_vec=tol_new_vec)
    sub_dim = int(U.shape[1])
    diags_U = U
    diags: Dict[str, Any] = {"num_actions": len(actions), "U": diags_U}
    if sub_dim == 0:
        # z0=0 or numerically zero: trivial
        return SubspaceCandidateCertificate(
            overall_pass=True,
            action_cert_pass=True,
            subspace_dim=0,
            depth=depth,
            max_residual_added=0.0,
            psd_tol=psd_tol,
            S_by_k={k: bucket_indices for k in range(T)},
            required_candidates_per_expand=0,
            required_beam_width_outer=len(bucket_actions),
            bucket_actions=bucket_actions,
            diagnostics={"note": "z0 is (numerically) zero; trivial certificate.", "num_actions": len(actions)},
        )

    # Precompute projected matrices for each k and action
    lambdas = np.asarray(ubm.lambdas, dtype=float)
    G = np.asarray(ubm.G, dtype=float)
    D = int(ubm.D)
    eta = float(ubm.eta)
    K = int(ubm.K)

    def _validate_candidate_hist(candidate_hist_in: np.ndarray) -> np.ndarray:
        ch = np.asarray(candidate_hist_in, dtype=int)
        if ch.ndim != 2:
            raise ValueError(f"candidate_hist must be 2D (T, D+1). Got shape={ch.shape}")
        if ch.shape[0] != int(T):
            raise ValueError(f"candidate_hist length mismatch: expected T={T}, got {ch.shape[0]}")
        if ch.shape[1] != int(D + 1):
            raise ValueError(f"candidate_hist must have D+1={D+1} columns (histogram). Got {ch.shape[1]}")

        # Check each row is a valid histogram from enumerate_histograms()
        action_set = set(tuple(a.tolist()) for a in actions)
        for t in range(int(T)):
            mt = tuple(ch[t].reshape(-1).tolist())
            if mt not in action_set:
                raise ValueError(
                    f"candidate_hist[t] is not a valid histogram from action_space.enumerate_histograms(): "
                    f"t={t}, m={list(mt)}. "
                    f"Possible cause: you passed delay-vectors (length K) instead of histograms (length D+1)."
                )
        return ch


    S_by_k: Dict[int, List[int]] = {}
    Sprime_by_k: Dict[int, List[int]] = {}
    worst_S = 0

    # NOTE: beam uses ub_tail(z_next, k_remain-1), so k is a "tail length" in [0, T-1]
    for k in range(0, T):
        # k steps remaining in ub_tail_geom; comparing z_next uses M_{m,k}
        Pm: List[np.ndarray] = []
        for m in actions:
            Mm = _Mm_k(lambdas, G, D, eta, K, m, k)
            Pm.append(U.T @ Mm @ U)

        nA = len(actions)

        # ---------- (1) build equivalence classes under PSD-bidominance ----------
        # i ~ j iff (Pj-Pi) PSD and (Pi-Pj) PSD  (=> equal on subspace up to tol)
        rep = list(range(nA))

        def find(x):
            while rep[x] != x:
                rep[x] = rep[rep[x]]
                x = rep[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                rep[rb] = ra

        for i in range(nA):
            for j in range(i + 1, nA):
                if _psd_check(Pm[j] - Pm[i], tol=psd_tol) and _psd_check(Pm[i] - Pm[j], tol=psd_tol):
                    union(i, j)

        # representatives of equivalence classes
        classes: Dict[int, List[int]] = {}
        for i in range(nA):
            r = find(i)
            classes.setdefault(r, []).append(i)

        reps = sorted(classes.keys())  # choose one representative index per class

        # ---------- (2) strict PSD dominance between representatives ----------
        # j strictly dominates i if (Pj-Pi) PSD and NOT (Pi-Pj) PSD
        maximal_reps: List[int] = []
        for ii in reps:
            dominated = False
            for jj in reps:
                if ii == jj:
                    continue
                if _psd_check(Pm[jj] - Pm[ii], tol=psd_tol) and (not _psd_check(Pm[ii] - Pm[jj], tol=psd_tol)):
                    dominated = True
                    break
            if not dominated:
                maximal_reps.append(ii)

        # map back: keep one representative per maximal equivalence class
        maximal = maximal_reps

        S_by_k[k] = maximal

        Sprime_by_k[k] = sorted(set(S_by_k[k]).union(bucket_indices))

        worst_S = max(worst_S, len(Sprime_by_k[k]))


        if max_S is not None and worst_S > int(max_S):
            return SubspaceCandidateCertificate(
                proxy_overall_pass=False,
                proxy_action_cert_pass=False,
                subspace_dim=sub_dim,
                depth=depth,
                max_residual_added=float("nan"),
                psd_tol=psd_tol,
                S_by_k=Sprime_by_k,
                required_candidates_per_expand=worst_S,
                required_beam_width_outer=int(len(bucket_actions)),
                bucket_actions=bucket_actions,
                diagnostics={
                    "fail_reason": "candidate_set_too_large",
                    "max_S": int(max_S),
                    "worst_S_so_far": int(worst_S),
                    "k": int(k),
                    "num_actions": int(len(actions)),
                },
            )

        
    
    required_beam_width_outer = len(bucket_actions)

    action_cert_pass = True

    overall_pass = bool(action_cert_pass)

    return SubspaceCandidateCertificate(
        proxy_overall_pass=overall_pass,
        proxy_action_cert_pass=action_cert_pass,
        subspace_dim=sub_dim,
        depth=depth,
        max_residual_added=float("nan"),
        psd_tol=psd_tol,
        S_by_k=Sprime_by_k,
        required_candidates_per_expand=max(len(v) for v in Sprime_by_k.values()),
        required_beam_width_outer=int(required_beam_width_outer),
        bucket_actions=bucket_actions,
        diagnostics=diags,
    )

def build_cprime_buckets_by_projected_operator(
    *,
    ubm: Any,
    action_space: Any,
    U: np.ndarray,
    delta: float = 1e-6,
    max_buckets: Optional[int] = None,
    restrict_to_action_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    C' buckets: group actions m by identical projected operator on reachable subspace.

    Define stacked lifted-mode state vector z in R^{n*(D+1)}.
    For action m, the one-step linear operator is F(m) such that z_{t+1} = F(m) z_t.
    On subspace span(U), if U^T F(m) U == U^T F(m') U, then for ANY z in span(U),
    successors are identical in projected coordinates, hence future TRUE costs are identical
    (since stage loss depends only on state, not on action label).

    IMPORTANT (v4 strong soundness mode):
      - If restrict_to_action_indices is None, we are in ALL-action scope.
        In this mode, we must NOT cap buckets, and we must NOT use delta-quantization
        (which could merge non-identical operators).
        We instead use an EXACT float-bit hash of A_U = U^T F(m) U.
        This makes "same bucket => identical A_U (in float bits)" a strict, checkable statement.

      - If restrict_to_action_indices is not None, we keep the old (quantized) behavior
        for speed (NOT a strong certificate).
    """
    # enumerate all feasible histograms in the SAME order as in bound_search
    actions = [np.asarray(m, dtype=int).reshape(-1) for m in action_space.enumerate_histograms()]

    U = np.asarray(U, dtype=float)
    dim_z, r = U.shape
    if r == 0:
        return {
            "bucket_id_by_action": {},
            "bucket_actions": [],
            "all_bucket_ids": [],
            "scope": "all" if restrict_to_action_indices is None else "restricted",
            "report": {"num_buckets": 0, "note": "rank(U)=0"},
        }

    n = int(ubm.lambdas.shape[0])
    D = int(ubm.D)

    def vec_to_modes(v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=float).reshape(n, D + 1)

    def modes_to_vec(zm: np.ndarray) -> np.ndarray:
        return np.asarray(zm, dtype=float).reshape(-1)

    U_cols_modes = [vec_to_modes(U[:, j]) for j in range(r)]

    full_scope = (restrict_to_action_indices is None)
    if full_scope:
        idxs = list(range(len(actions)))
        scope = "all"
    else:
        idxs = list(restrict_to_action_indices)
        scope = "restricted"

    # buckets: key -> list[action_index]
    buckets: Dict[Tuple[int, ...], List[int]] = {}

    # helper: compute A_U = U^T F(m) U without forming F(m)
    def compute_A_U(m: np.ndarray) -> np.ndarray:
        Y = np.zeros((dim_z, r), dtype=float)
        for j in range(r):
            z_next = ubm.step(U_cols_modes[j], m)  # (n, D+1)
            Y[:, j] = modes_to_vec(z_next)
        A_U = U.T @ Y  # (r, r)
        return np.asarray(A_U, dtype=np.float64)

    # --- key function ---
    if full_scope:
        # STRONG (sound) mode: exact float-bit hash (no delta)
        def make_key(A_U: np.ndarray) -> Tuple[int, ...]:
            flat = np.asarray(A_U, dtype=np.float64).reshape(-1)
            bits = flat.view(np.int64)
            return tuple(int(x) for x in bits.tolist())

        hash_mode = "exact_float_bits"
        # in full scope, we MUST NOT cap buckets; ignore max_buckets even if provided
        cap_applied = False
    else:
        # FAST mode: delta-quantized key (NOT a strong certificate)
        def make_key(A_U: np.ndarray) -> Tuple[int, ...]:
            q = np.round((A_U.reshape(-1) / float(delta))).astype(int)
            return tuple(int(x) for x in q.tolist())

        hash_mode = "quantized_delta"
        # optional cap number of buckets in restricted mode
        cap_applied = False

    # build buckets
    for ai in idxs:
        m = actions[ai]
        A_U = compute_A_U(m)
        key = make_key(A_U)
        buckets.setdefault(key, []).append(int(ai))

    keys = list(buckets.keys())

    # In restricted mode only, we allow capping (heuristic).
    # In full scope, capping would destroy completeness and thus soundness, so we do NOT cap.
    if (not full_scope) and (max_buckets is not None) and (len(keys) > int(max_buckets)):
        keys.sort(key=lambda k: len(buckets[k]), reverse=True)
        keys = keys[: int(max_buckets)]
        buckets = {k: buckets[k] for k in keys}
        cap_applied = True

    # produce mapping action->bucket id, and representative action per bucket
    bucket_id_by_action: Dict[Tuple[int, ...], int] = {}
    bucket_actions: List[np.ndarray] = []

    for bid, key in enumerate(keys):
        members = buckets[key]
        rep_ai = int(min(members))
        rep_m = np.asarray(actions[rep_ai], dtype=int).reshape(-1)
        bucket_actions.append(rep_m.copy())
        for ai in members:
            mm = tuple(np.asarray(actions[ai], dtype=int).tolist())
            bucket_id_by_action[mm] = int(bid)

    sizes = [len(buckets[k]) for k in keys]
    report = {
        "scope": scope,
        "hash_mode": hash_mode,
        "num_buckets": int(len(keys)),
        "min_bucket_size": int(min(sizes) if sizes else 0),
        "max_bucket_size": int(max(sizes) if sizes else 0),
        "avg_bucket_size": float(np.mean(sizes) if sizes else 0.0),
        "delta": float(delta),
        "restricted_num_actions": int(len(idxs)),
        "max_buckets_arg": (None if max_buckets is None else int(max_buckets)),
        "cap_applied": bool(cap_applied),
        "note": (
            "FULL scope: exact hash, no cap (strong)."
            if full_scope
            else "RESTRICTED scope: quantized hash; cap may apply (not strong)."
        ),
    }

    return {
        "bucket_id_by_action": bucket_id_by_action,
        "bucket_actions": bucket_actions,
        "all_bucket_ids": list(range(int(len(keys)))),  # used by main for ALL-scope S1b required set
        "scope": scope,
        "report": report,
    }

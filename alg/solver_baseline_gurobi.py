# alg/solver_baseline_gurobi.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import os

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import Model, GRB, QuadExpr
except ImportError as e:
    raise ImportError("Please install gurobipy (Gurobi Python API).") from e

# -----------------------------------------------------------------------------
# Bounds for MIQCP stability
#
# Legacy bound path (USE_SOUND_BOUNDS=False) uses a spectral-norm-based rho_max
# and then turns it into entry-wise bounds. This can be UNSOUND due to norm
# conversion (||.||_2 -> ||.||_inf).
#
# Sound bound path (USE_SOUND_BOUNDS=True) uses the induced infinity norm
# rho_inf = max ||A||_inf of the lifted companion operator. Then
# ||state_t||_inf <= rho_inf^t * ||state_0||_inf, which implies entry-wise bounds.
# This is theoretically sound (may be looser).
# -----------------------------------------------------------------------------

def _load_hist_sequence_from_json(path: str) -> np.ndarray:
    import json, numpy as np
    with open(path, "r") as f:
        data = json.load(f)
    if "hist_sequence" not in data:
        raise ValueError(f"{path} missing key 'hist_sequence'")
    m = np.asarray(data["hist_sequence"], dtype=float)
    return m



@dataclass
class SolverBaselineParams:
    # solver controls
    time_limit: float = 60.0
    mip_gap: float = 1e-6
    threads: int = 0               # 0 = solver default
    nonconvex: int = 2             # Gurobi: enable nonconvex QCP/QP handling
    output_flag: int = 1

    # modeling variants for ablation
    basis: str = "original"        # {"original","spectral"}
    bound_mode: str = "global"     # {"global","time_indexed","per_mode_time_indexed"}
    m_encoding: str = "hist_miqcp"   # {"hist_miqcp","onehot_bigm"}


    # conservative bounds for state variables
    bound_scale: float = 1.0       # multiply computed bounds by this factor


def _compute_rho_max_relaxed(H: np.ndarray, action_space, eta: float, K: int, D: int) -> float:
    """
    Sound rho_max upper bound for the lifted (D+1)-history companion matrix A(m,lambda).
    We maximize ||A|| over relaxed histogram vertices and eigen-modes.
    """
    lambdas = np.linalg.eigvalsh(H)
    alpha = float(eta) / float(K)

    V = action_space.enumerate_histogram_vertices(relax=True)
    V = [np.asarray(v, dtype=float).reshape(-1) for v in V]
    if not V:
        raise ValueError("No relaxed histogram vertices; check ActionSpace and constraints.")

    rho_max = 1.0  # IMPORTANT: shift rows imply at least 1 in general
    for lam in lambdas:
        lam = float(lam)
        best = 1.0
        for m in V:
            m = np.asarray(m, dtype=float).reshape(-1)
            if m.shape[0] != D + 1:
                raise ValueError(f"Histogram vertex length mismatch: got {m.shape[0]} expected {D+1}.")

            c0 = 1.0 - alpha * m[0] * lam
            cj = -alpha * lam * m[1:]
            c = np.concatenate(([c0], cj), axis=0)  # (D+1,)

            # companion-like lifted matrix
            A = np.zeros((D + 1, D + 1), dtype=float)
            A[0, :] = c
            if D > 0:
                A[1:, :-1] = np.eye(D, dtype=float)

            # spectral norm of A (exact for small D)
            val = float(np.linalg.norm(A, 2))
            if val > best:
                best = val

        rho_max = max(rho_max, best)

    # extra safety: never allow < 1 in downstream bounds to avoid accidental shrinking
    return float(max(1.0, rho_max))

def _compute_rho_max_inf_relaxed(H: np.ndarray, action_space, eta: float, K: int, D: int, *, basis: str) -> float:
    """
    A *sound* upper bound on the induced infinity norm growth of the lifted dynamics.

    We want entry-wise bounds on the decision variables x[t,j,i]. Using ||.||_inf is
    the correct norm: if ||z_t||_inf <= M, then every coordinate is within [-M, M].

    - basis="spectral": the dynamics decouple per eigen-mode. For each eigenvalue λ
      and relaxed histogram vertex m, we build the (D+1)x(D+1) companion-like matrix A
      and take ||A||_inf (max absolute row-sum). Then rho_inf is the max over (λ, m).
      This is sound for the lifted per-mode history stack.

    - basis="original": use a simple block-matrix row-sum bound. With alpha=eta/K and
      sum_j m_j = K, the top block row has row-sum <= 1 + eta * ||H||_inf, and all
      shift rows have row-sum 1. Hence rho_inf <= max(1, 1 + eta*||H||_inf).
      This is sound but can be looser.
    """
    alpha = float(eta) / float(K)
    basis = str(basis).lower()

    if basis == "original":
        H_inf = float(np.linalg.norm(H, np.inf))
        rho = 1.0 + abs(float(eta)) * H_inf
        return float(max(1.0, rho))

    # spectral (default)
    lambdas = np.linalg.eigvalsh(H)
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)

    V = action_space.enumerate_histogram_vertices(relax=True)
    V = [np.asarray(v, dtype=float).reshape(-1) for v in V]
    if not V:
        raise ValueError("No relaxed histogram vertices; check ActionSpace and constraints.")

    rho_inf = 1.0
    for lam in lambdas:
        lam = float(lam)
        best = 1.0
        for m in V:
            m = np.asarray(m, dtype=float).reshape(-1)
            if m.shape[0] != D + 1:
                raise ValueError(f"Histogram vertex length mismatch: got {m.shape[0]} expected {D+1}.")

            c0 = 1.0 - alpha * m[0] * lam
            cj = -alpha * lam * m[1:]
            c = np.concatenate(([c0], cj), axis=0)

            A = np.zeros((D + 1, D + 1), dtype=float)
            A[0, :] = c
            if D > 0:
                A[1:, :-1] = np.eye(D, dtype=float)

            # induced infinity norm = max absolute row sum
            val = float(np.max(np.sum(np.abs(A), axis=1)))
            if val > best:
                best = val

        rho_inf = max(rho_inf, best)

    return float(max(1.0, rho_inf))


def _make_state_bounds(
    T: int,
    D: int,
    n: int,
    max_abs0: float,
    abs0_vec: Optional[np.ndarray],
    rho_max: float,
    params: SolverBaselineParams,
) -> Tuple[float, Optional[Dict[Tuple[int, int, int], float]], Optional[Dict[Tuple[int, int, int], float]]]:
    """
    Return (M_global, lb_dict, ub_dict). If lb_dict/ub_dict are None => use global bounds.

    - global: |x[t,j,i]| <= M_global where M_global ~ max_abs0 * rho_max^T
    - time_indexed: |x[t,j,i]| <= max_abs0 * rho_max^{max(0, t-j)}  (sound & tighter)
    - per_mode_time_indexed: |x[t,j,i]| <= abs0_vec[i] * rho_max^{max(0, t-j)}  (sound & typically tighter)
    """
    max_abs0 = float(max(1e-12, max_abs0))
    rho_max = float(max(0.0, rho_max))

    # baseline global bound (kept for ablation compatibility)
    M_global = params.bound_scale * max_abs0 * (rho_max ** max(1, T))
    if (not np.isfinite(M_global)) or M_global <= 0:
        M_global = params.bound_scale * max(1.0, max_abs0)

    if params.bound_mode == "global":
        return float(M_global), None, None

    if params.bound_mode not in {"time_indexed", "per_mode_time_indexed"}:
        raise ValueError(
            f"Unknown bound_mode={params.bound_mode}. Use 'global', 'time_indexed', or 'per_mode_time_indexed'."
        )

    # time-indexed bounds (scalar max_abs0) OR per-mode time-indexed bounds (vector abs0_vec)
    if params.bound_mode == "per_mode_time_indexed":
        if abs0_vec is None:
            raise ValueError("per_mode_time_indexed requires abs0_vec (per-dimension initial amplitude).")
        abs0_vec = np.asarray(abs0_vec, dtype=float).reshape(-1)
        if abs0_vec.shape[0] != n:
            raise ValueError(f"abs0_vec length mismatch: got {abs0_vec.shape[0]} expected {n}.")
        abs0_vec = np.maximum(abs0_vec, 1e-12)

    lb: Dict[Tuple[int, int, int], float] = {}
    ub: Dict[Tuple[int, int, int], float] = {}
    for t in range(T + 1):
        for j in range(D + 1):
            exp = max(0, t - j)
            if params.bound_mode == "time_indexed":
                # Mtj = params.bound_scale * max_abs0 * (rho_max ** exp)
                grow = max(1.0, rho_max ** exp)   # envelope
                Mtj = params.bound_scale * max_abs0 * grow
                if (not np.isfinite(Mtj)) or Mtj <= 0:
                    Mtj = params.bound_scale * max(1.0, max_abs0)
                for i in range(n):
                    key = (t, j, i)
                    lb[key] = -float(Mtj)
                    ub[key] = float(Mtj)
            else:
                # per-mode/per-coordinate bounds: Mtj_i uses abs0_vec[i]
                for i in range(n):
                    # Mtj_i = params.bound_scale * float(abs0_vec[i]) * (rho_max ** exp)
                    grow = max(1.0, rho_max ** exp)   # envelope
                    Mtj_i = params.bound_scale * float(abs0_vec[i]) * grow

                    if (not np.isfinite(Mtj_i)) or Mtj_i <= 0:
                        Mtj_i = params.bound_scale * max(1.0, float(abs0_vec[i]))
                    key = (t, j, i)
                    lb[key] = -float(Mtj_i)
                    ub[key] = float(Mtj_i)            

    return float(M_global), lb, ub


def solve_worst_sequence_miqcp_gurobi(
    system,
    action_space,
    z0: np.ndarray,               # shape (D+1, n), history-stack initial
    T: int,
    params: Optional[SolverBaselineParams] = None,
    warm_start_m_hist: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Solve the exact worst-case sequence for ASGD area objective using Gurobi MIQCP (nonconvex).

    Baseline formulation (params.basis="original"):
      - Decision variables:
          m[t,j] integer histogram counts (j=0..D) per step t=0..T-1
          x[t,j,i] continuous history-stack states (t=0..T, j=0..D, i=0..n-1)
      - Dynamics:
          x[t+1,0] = x[t,0] - (eta/K) * sum_j m[t,j] * H x[t,j]
          x[t+1,j] = x[t,j-1]  for j>=1
      - Objective:
          maximize sum_{t=0..T-1} x[t,0]^T H x[t,0]

    Improvement 1 (params.basis="spectral"):
      Use H = Q diag(lambda) Q^T. Let u = Q^T x, so the system decouples by modes:
          u[t+1,0,r] = u[t,0,r] - (eta/K) * lambda_r * sum_j m[t,j] * u[t,j,r]
          u[t+1,j,r] = u[t,j-1,r]
      Objective becomes: sum_t sum_r lambda_r * u[t,0,r]^2.
      This is an equivalent MIQCP with far fewer quadratic couplings.

    Improvement 2 (params.bound_mode="time_indexed"):
      Replace the single global bound M ~ rho^T by sound time-indexed bounds:
          |state[t,j,*]| <= max_abs0 * rho^{max(0,t-j)}
      (applies in either basis, for clean ablations).

    Returns a dict with status/objective/solution and trajectories.
    """
    if params is None:
        params = SolverBaselineParams()

    H = np.asarray(system.objective.H, dtype=float)
    H = 0.5 * (H + H.T)
    # --- H spectral stats (for analysis / logging) ---
    evals = np.linalg.eigvalsh(H)
    lam_min = float(np.min(evals))
    lam_max = float(np.max(evals))
    cond_H = float(lam_max / lam_min) if lam_min > 0 else float("inf")


    n = int(H.shape[0])
    D = int(system.D)
    K = int(system.K)
    eta = float(system.eta)
    alpha = eta / float(K)

    if z0.shape != (D + 1, n):
        raise ValueError(f"z0 must be (D+1,n)={(D+1,n)}, got {z0.shape}")

    # --- basis handling (ablation-friendly) ---
    basis = str(params.basis).lower().strip()
    if basis not in {"original", "spectral"}:
        raise ValueError(f"Unknown basis={params.basis}. Use 'original' or 'spectral'.")

    Q = None
    lambdas = None
    if basis == "spectral":
        # H assumed symmetric in the project; eigh gives orthonormal Q
        lambdas, Q = np.linalg.eigh(H)
        lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
        Q = np.asarray(Q, dtype=float)
        # transform initial history-stack: u0[j,:] = Q^T x0[j,:]
        # (row-vector convention) u_row = x_row @ Q
        state0 = z0 @ Q  # (D+1,n)
        max_abs0 = float(np.max(np.abs(state0)))
    else:
        state0 = z0
        max_abs0 = float(np.max(np.abs(z0)))

    # # --- conservative bound(s) for state variables (needed for nonconvex MIQCP stability) ---
    # if USE_SOUND_BOUNDS:
    #     # Sound: rho_inf is induced infinity norm growth, consistent with entry-wise bounds
    #     rho_max = _compute_rho_max_inf_relaxed(H, action_space, eta=eta, K=K, D=D, basis=params.basis)
    # else:
    #     # Legacy: spectral-norm bound (can be UNSOUND when converted to entry-wise bounds)
    #     rho_max = _compute_rho_max_relaxed(H, action_space, eta=eta, K=K, D=D)

    rho_max = _compute_rho_max_inf_relaxed(H, action_space, eta=eta, K=K, D=D, basis=params.basis)

    abs0_vec = np.max(np.abs(state0), axis=0)  # shape (n,)
    M_global, lb_dict, ub_dict = _make_state_bounds(
        T=T, D=D, n=n, max_abs0=max_abs0, abs0_vec=abs0_vec, rho_max=rho_max, params=params
    )

    # if USE_CERTIFIED_GLOBAL_L2_BOUND and params.bound_mode == "global":
    #     # certified (sound) global bound based on L2 growth
    #     evals = np.linalg.eigvalsh(H)
    #     lam_max = float(np.max(evals))
    #     rho_max = max(1.0, 1.0 + abs(eta) * lam_max)

    #     s0 = float(np.linalg.norm(state0.reshape(-1), 2))   # history-stack L2
    #     # reuse existing machinery: just feed max_abs0 = s0
    #     max_abs0 = max(1e-12, s0)

    #     abs0_vec = None  # unused in global
    #     M_global, lb_dict, ub_dict = _make_state_bounds(
    #         T=T, D=D, n=n, max_abs0=max_abs0, abs0_vec=abs0_vec, rho_max=rho_max, params=params
    #     )
    # else:
    #     # legacy path (may be unsound for entry-wise bounds)
    #     rho_max = _compute_rho_max_relaxed(H, action_space, eta=eta, K=K, D=D)
    #     abs0_vec = np.max(np.abs(state0), axis=0)
    #     M_global, lb_dict, ub_dict = _make_state_bounds(
    #         T=T, D=D, n=n, max_abs0=max_abs0, abs0_vec=abs0_vec, rho_max=rho_max, params=params
    #     )


    model = gp.Model("asgd_worst_miqcp")
    model.Params.OutputFlag = int(params.output_flag)
    model.Params.NonConvex = int(params.nonconvex)
    model.Params.TimeLimit = float(params.time_limit)
    model.Params.MIPGap = float(params.mip_gap)
    if int(params.threads) > 0:
        model.Params.Threads = int(params.threads)

    # Histogram decision variables
    m_encoding = str(getattr(params, "m_encoding", "hist_miqcp")).lower().strip()
    if m_encoding not in {"hist_miqcp", "onehot_bigm"}:
        raise ValueError(f"Unknown m_encoding={m_encoding}")
    
    # ---------------- DEBUG: fix schedule ----------------
    DEBUG_FIX_SCHEDULE_JSON = None  # e.g. "./schedules/beam_best_word.json"
    # -----------------------------------------------------


    # If using onehot_bigM, we still keep m[t,j] for output, but link it to y.
    m = model.addVars(T, D + 1, vtype=GRB.INTEGER, lb=0, ub=K, name="m")

    y = None
    w = None
    if m_encoding == "onehot_bigm":
        if basis != "spectral":
            raise ValueError("onehot_bigM is currently implemented only for basis='spectral' (cleanest & safest).")

        # one-hot worker choices
        y = model.addVars(T, K, D + 1, vtype=GRB.BINARY, name="y")

        # each worker picks exactly one delay
        for t in range(T):
            for k in range(K):
                model.addConstr(gp.quicksum(y[t, k, j] for j in range(D + 1)) == 1, name=f"onehot[{t},{k}]")

        # link histogram counts: m[t,j] = sum_k y[t,k,j]
        for t in range(T):
            for j in range(D + 1):
                model.addConstr(m[t, j] == gp.quicksum(y[t, k, j] for k in range(K)), name=f"m_link[{t},{j}]")

        for t in range(T):
            for k in range(K-1):
                lhs = gp.quicksum(j * y[t, k, j] for j in range(D+1))
                rhs = gp.quicksum(j * y[t, k+1, j] for j in range(D+1))
                model.addConstr(lhs >= rhs, name=f"symbreak[{t},{k}]")


        # Big-M linearization variables: w[t,k,j,r] = y[t,k,j] * x[t,j,r]
        w = model.addVars(
            T, K, D + 1, n,
            vtype=GRB.CONTINUOUS,
            lb=-M_global,
            ub= M_global,
            name="w"
        )

    # State variables (x or u depending on basis)
    if lb_dict is None:
        x = model.addVars(T + 1, D + 1, n, vtype=GRB.CONTINUOUS, lb=-M_global, ub=M_global, name="x")
    else:
        x = model.addVars(T + 1, D + 1, n, vtype=GRB.CONTINUOUS, lb=lb_dict, ub=ub_dict, name="x")

    def _rollout_states_for_start(
        *,
        basis: str,
        state0: np.ndarray,     # (D+1, n) already in the solver's basis (x or u)
        H: np.ndarray,
        lambdas: Optional[np.ndarray],
        alpha: float,
        m_hist: np.ndarray,     # (T, D+1) int
    ) -> np.ndarray:
        # returns traj: (T+1, D+1, n) in solver basis
        T = m_hist.shape[0]
        D = state0.shape[0] - 1
        n = state0.shape[1]
        xtraj = np.zeros((T+1, D+1, n), dtype=float)
        xtraj[0] = state0

        for t in range(T):
            # shift
            for j in range(D, 0, -1):
                xtraj[t+1, j] = xtraj[t, j-1]

            if basis == "original":
                # x[t+1,0] = x[t,0] - alpha * sum_j m[t,j] * (H x[t,j])
                x_next0 = xtraj[t, 0].copy()
                for j in range(D+1):
                    x_next0 -= alpha * float(m_hist[t, j]) * (H @ xtraj[t, j])
                xtraj[t+1, 0] = x_next0
            else:
                # spectral: x is u, dynamics decoupled: u[t+1,0,r] = u[t,0,r] - alpha*lam_r*sum_j m[t,j]*u[t,j,r]
                assert lambdas is not None
                u_next0 = xtraj[t, 0].copy()
                for j in range(D+1):
                    u_next0 -= alpha * float(m_hist[t, j]) * (lambdas * xtraj[t, j])
                xtraj[t+1, 0] = u_next0

        return xtraj

    if warm_start_m_hist is not None:
        m0 = np.asarray(warm_start_m_hist, dtype=float)
        if m0.shape != (T, D+1):
            raise ValueError(...)
        # 1) m start
        for t in range(T):
            for j in range(D+1):
                m[t,j].Start = float(m0[t,j])

        # 2) x start (basis-aware)
        m_int = np.asarray(warm_start_m_hist, dtype=int)
        traj = _rollout_states_for_start(
            basis=basis,
            state0=state0,     # note: already transformed if spectral (you already compute state0 = z0@Q) :contentReference[oaicite:5]{index=5}
            H=H,
            lambdas=lambdas,
            alpha=alpha,
            m_hist=m_int,
        )
        for tt in range(T+1):
            for j in range(D+1):
                for i in range(n):
                    x[tt,j,i].Start = float(traj[tt,j,i])


    if m_encoding == "onehot_bigm":
        # helper to fetch bounds for x[t,j,r]
        def _bounds(t: int, j: int, r: int) -> Tuple[float, float]:
            if lb_dict is None:
                return -float(M_global), float(M_global)
            return float(lb_dict[(t, j, r)]), float(ub_dict[(t, j, r)])

        for t in range(T):
            for k in range(K):
                for j in range(D + 1):
                    for r in range(n):
                        L, U = _bounds(t, j, r)
                        # standard big-M linearization for w = y * x
                        model.addConstr(w[t, k, j, r] <= U * y[t, k, j], name=f"bm1[{t},{k},{j},{r}]")
                        model.addConstr(w[t, k, j, r] >= L * y[t, k, j], name=f"bm2[{t},{k},{j},{r}]")
                        model.addConstr(w[t, k, j, r] <= x[t, j, r] - L * (1 - y[t, k, j]), name=f"bm3[{t},{k},{j},{r}]")
                        model.addConstr(w[t, k, j, r] >= x[t, j, r] - U * (1 - y[t, k, j]), name=f"bm4[{t},{k},{j},{r}]")


    # Initial conditions: fix x[0,j,i] = state0[j,i]
    for j in range(D + 1):
        for i in range(n):
            model.addConstr(x[0, j, i] == float(state0[j, i]), name=f"init[{j},{i}]")

    if DEBUG_FIX_SCHEDULE_JSON is not None:
        m_fix = _load_hist_sequence_from_json(DEBUG_FIX_SCHEDULE_JSON)
        if m_fix.shape != (T, D + 1):
            raise ValueError(f"Fixed schedule shape {m_fix.shape} != {(T, D+1)}")
        # IMPORTANT: fix with equality constraints (do NOT rely on .Start)
        for t in range(T):
            for j in range(D + 1):
                model.addConstr(m[t, j] == float(m_fix[t, j]), name=f"fix_m[{t},{j}]")
        print("[DEBUG] Fixed m to schedule from", DEBUG_FIX_SCHEDULE_JSON)


    # Per-step feasibility for histogram: sum m = K and sum j*m <= B
    B = int(action_space.B)
    for t in range(T):
        model.addConstr(gp.quicksum(m[t, j] for j in range(D + 1)) == K, name=f"hist_sumK[{t}]")
        model.addConstr(gp.quicksum(j * m[t, j] for j in range(D + 1)) <= B, name=f"hist_budget[{t}]")

    # Shift constraints: x[t+1,j] = x[t,j-1] for j>=1
    for t in range(T):
        for j in range(1, D + 1):
            for i in range(n):
                model.addConstr(x[t + 1, j, i] == x[t, j - 1, i], name=f"shift[{t},{j},{i}]")

    # Main update
    if basis == "original":
        # x[t+1,0,i] = x[t,0,i] - alpha * sum_j m[t,j] * sum_k H[i,k]*x[t,j,k]
        for t in range(T):
            for i in range(n):
                qexpr = gp.QuadExpr()
                qexpr += x[t, 0, i]
                for j in range(D + 1):
                    # inner linear expr: (H x[t,j])_i
                    inner = gp.LinExpr()
                    for k in range(n):
                        hij = float(H[i, k])
                        if hij != 0.0:
                            inner += hij * x[t, j, k]
                    qexpr += (-alpha) * m[t, j] * inner
                model.addQConstr(x[t + 1, 0, i] == qexpr, name=f"main[{t},{i}]")
    else:
        # spectral / modal form: x is actually u
        assert lambdas is not None
        for t in range(T):
            for r in range(n):
                lam = float(lambdas[r])
                
                if m_encoding == "hist_miqcp":
                    qexpr = gp.QuadExpr()
                    qexpr += x[t, 0, r]
                    for j in range(D + 1):
                        qexpr += (-alpha * lam) * m[t, j] * x[t, j, r]
                    model.addQConstr(x[t + 1, 0, r] == qexpr, name=f"main[{t},{r}]")

                else:
                    # onehot_bigM: linear dynamics using w
                    expr = gp.LinExpr()
                    expr += x[t, 0, r]
                    expr += (-alpha * lam) * gp.quicksum(w[t, k, j, r] for k in range(K) for j in range(D + 1))
                    model.addConstr(x[t + 1, 0, r] == expr, name=f"main_lin[{t},{r}]")

    # Objective
    if basis == "original":
        obj = gp.QuadExpr()
        for t in range(T):
            for i in range(n):
                for k in range(n):
                    hij = float(H[i, k])
                    if hij != 0.0:
                        obj += hij * x[t, 0, i] * x[t, 0, k]
        model.setObjective(obj, GRB.MAXIMIZE)
    else:
        obj = gp.QuadExpr()
        for t in range(T):
            for r in range(n):
                lam = float(lambdas[r])
                if lam != 0.0:
                    obj += lam * x[t, 0, r] * x[t, 0, r]
        model.setObjective(obj, GRB.MAXIMIZE)

    # Determinism / numeric stability
    model.Params.Threads = 1
    model.Params.Seed = 0
    model.Params.NumericFocus = 3

    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol  = 1e-9
    model.Params.IntFeasTol     = 1e-9


    # model.update()
    # print("NumQConstrs =", model.NumQConstrs)
    # print("NumConstrs  =", model.NumConstrs)
    # print("NumQNZs     =", model.NumQNZs)     
    print("DEBUG basis=", basis, "m_encoding=", m_encoding, "bound_mode=", params.bound_mode)
    model.update()
    print("DEBUG NumQConstrs=", model.NumQConstrs, "NumConstrs=", model.NumConstrs, "NumQNZs=", model.NumQNZs)



    model.optimize()

    def _safe_get_float(obj, attr):
        try:
            v = getattr(obj, attr)
            return float(v)
        except Exception:
            return None

    def _safe_get_int(obj, attr):
        try:
            v = getattr(obj, attr)
            return int(v)
        except Exception:
            return None

    out: Dict[str, Any] = {
        "status": int(model.Status),
        "obj": None,
        "m_hist": None,
        "x_traj": None,    # in original basis
        "u_traj": None,    # in spectral basis (if used)
        "runtime": float(model.Runtime),
        "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
        "M_global": float(M_global),
        "rho_max": float(rho_max),
        "basis": basis,
        "bound_mode": str(params.bound_mode),
        "m_encoding": m_encoding,

        # structure stats
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond_H": cond_H,

        # solver stats (even if no feasible solution)
        "sol_count": _safe_get_int(model, "SolCount"),
        "nodecount": _safe_get_float(model, "NodeCount"),
        "simplex_iters": _safe_get_float(model, "IterCount"),
        "best_bound": _safe_get_float(model, "ObjBound"),
        "work": _safe_get_float(model, "Work"),

        # model size
        "num_vars": _safe_get_int(model, "NumVars"),
        "num_constrs": _safe_get_int(model, "NumConstrs"),
        "num_qconstrs": _safe_get_int(model, "NumQConstrs"),
        "num_bin_vars": _safe_get_int(model, "NumBinVars"),
        "num_int_vars": _safe_get_int(model, "NumIntVars"),
    }

    print("[DEBUG] Status   =", model.Status)
    if model.Status == GRB.OPTIMAL:
        print("[DEBUG] ObjVal   =", model.ObjVal)
        print("[DEBUG] ObjBound =", model.ObjBound)
        print("[DEBUG] Gap      =", model.MIPGap)
        print("[DEBUG] SolCount =", model.SolCount)


    if model.Status == GRB.INFEASIBLE:
        try:
            model.computeIIS()
            iis_path = "asgd_iis.ilp"
            model.write(iis_path)
            out["iis_path"] = iis_path
        except Exception as e:
            out["iis_path"] = None
            out["iis_error"] = str(e)
        return out

    if model.SolCount == 0:
        return out

    out["obj"] = float(model.ObjVal)

    # Extract histogram sequence
    m_hist = np.zeros((T, D + 1), dtype=int)
    for t in range(T):
        for j in range(D + 1):
            m_hist[t, j] = int(round(m[t, j].X))
    out["m_hist"] = m_hist

    # Extract trajectory
    traj = np.zeros((T + 1, D + 1, n), dtype=float)
    for t in range(T + 1):
        for j in range(D + 1):
            for i in range(n):
                traj[t, j, i] = float(x[t, j, i].X)

    if basis == "original":
        out["x_traj"] = traj
    else:
        out["u_traj"] = traj
        # map back to original basis for convenience: x = u @ Q^T
        assert Q is not None
        out["x_traj"] = traj @ Q.T

    # ---------------------------------------------------------------------
    # DEBUG: dump MIQCP-optimal schedule and verify objective consistency
    #
    # This checks whether the MIQCP objective matches the rollout JT under the
    # same schedule (m_hist) + same z0 + same basis.
    # ---------------------------------------------------------------------
    DEBUG_DUMP_AND_VERIFY_MIQCP_OPT = False  # <-- flip manually if needed
    DEBUG_DUMP_PATH = "./schedules/gurobi_opt.json"
    if DEBUG_DUMP_AND_VERIFY_MIQCP_OPT and out.get("m_hist", None) is not None and model.SolCount > 0:
        # ensure directory exists
        dump_dir = os.path.dirname(DEBUG_DUMP_PATH)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

        payload = {
            "meta": {
                "source": "miqcp_opt",
                "basis": str(basis),
                "m_encoding": str(m_encoding),
                "bound_mode": str(params.bound_mode),
                "K": int(K),
                "D": int(D),
                "T": int(T),
                "eta": float(eta),
            },
            "hist_sequence": out["m_hist"].tolist(),
        }
        with open(DEBUG_DUMP_PATH, "w") as f:
            json.dump(payload, f, indent=2)

        try:
            JT_model, JT_rollout, diff = verify_fixed_schedule_gurobi(
                system, json_path=DEBUG_DUMP_PATH, basis=basis, z0=z0, action_space=action_space, bounds_params=params, check_miqcp_bounds=True,
            )
            out["debug_verify_JT_model"] = float(JT_model)
            out["debug_verify_JT_rollout"] = float(JT_rollout)
            out["debug_obj_minus_JT_rollout"] = (float(out["obj"]) - float(JT_rollout)) if out["obj"] is not None else None
            print(
                f"[DEBUG] verify(gurobi_opt) JT_rollout={JT_rollout:.12g} "
                f"JT_model={JT_model:.12g} obj={out['obj']:.12g} "
                f"obj-JT={out['obj']-JT_rollout:+.3e}"
            )
        except Exception as e:
            out["debug_verify_error"] = str(e)
            print("[DEBUG] verify(gurobi_opt) failed:", e)

    # ---------------- DEBUG: constraint residuals ----------------
    if DEBUG_DUMP_AND_VERIFY_MIQCP_OPT:
        traj = out["x_traj"]            # (T+1, D+1, n) in the chosen basis
        m_hist = out["m_hist"].astype(float)  # (T, D+1)
        alpha = eta / float(K)

        # compute JT from MIQCP trajectory (same formula as objective)
        JT_from_traj = 0.0
        if basis == "spectral":
            # lambdas must exist
            for t in range(T):
                JT_from_traj += float(np.sum(lambdas * (traj[t, 0, :] ** 2)))
        else:
            for t in range(T):
                xt = traj[t, 0, :]
                JT_from_traj += float(xt @ (H @ xt))

        # constraint residuals
        shift_res = 0.0
        main_res = 0.0

        for t in range(T):
            # shift: x[t+1,j] = x[t,j-1]
            if D > 0:
                r = traj[t + 1, 1:, :] - traj[t, :-1, :]
                shift_res = max(shift_res, float(np.max(np.abs(r))))

            # main update:
            if basis == "spectral":
                # x is u
                v = traj[t, 0, :]
                grad_sum = np.zeros(n, dtype=float)
                for j in range(D + 1):
                    mj = m_hist[t, j]
                    if mj != 0:
                        grad_sum += mj * (lambdas * traj[t, j, :])
                v_next = v - alpha * grad_sum
                r = traj[t + 1, 0, :] - v_next
                main_res = max(main_res, float(np.max(np.abs(r))))
            else:
                v = traj[t, 0, :]
                grad_sum = np.zeros(n, dtype=float)
                for j in range(D + 1):
                    mj = m_hist[t, j]
                    if mj != 0:
                        grad_sum += mj * (H @ traj[t, j, :])
                v_next = v - alpha * grad_sum
                r = traj[t + 1, 0, :] - v_next
                main_res = max(main_res, float(np.max(np.abs(r))))

        out["debug_JT_from_traj"] = float(JT_from_traj)
        out["debug_shift_res_inf"] = float(shift_res)
        out["debug_main_res_inf"] = float(main_res)

        print(f"[DEBUG] JT_from_traj={JT_from_traj:.12g}  obj={out['obj']:.12g}  (obj-JTtraj)={out['obj']-JT_from_traj:+.3e}")
        print(f"[DEBUG] shift_res_inf={shift_res:.3e}  main_res_inf={main_res:.3e}")
    # -------------------------------------------------------------


    return out

def verify_fixed_schedule_gurobi(
    system,
    json_path: str,
    basis: str = "spectral",
    z0=None,
    action_space=None,
    bounds_params: "SolverBaselineParams|None" = None,
    check_miqcp_bounds: bool = True,
):
    """
    Fixed-schedule verification using Gurobi WITHOUT quadratic objective:
      - Build a linear feasibility model of lifted dynamics with fixed m_hist from JSON.
      - Solve feasibility (objective=0).
      - Compute J_T from solved trajectory in Python.
      - Compare with direct Python rollout J_T.

    Returns: (JT_model, JT_rollout, diff) where diff = JT_model - JT_rollout
    """
    import json
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB

    with open(json_path, "r") as f:
        data = json.load(f)
    if "hist_sequence" not in data:
        raise ValueError(f"{json_path} missing key 'hist_sequence'")
    m_hist = np.asarray(data["hist_sequence"], dtype=float)  # (T, D+1)
    T = int(m_hist.shape[0])

    # --- system params (match your MIQCP solver conventions) ---
    H = np.asarray(system.objective.H, dtype=float)
    H = 0.5 * (H + H.T)
    n = int(H.shape[0])
    D = int(system.D)
    K = int(system.K)
    eta = float(system.eta)
    alpha = eta / float(K)

    # --- init state: accept either x0 (n) or lifted z0 ((D+1)*n) ---
    # --- init state: MUST match the experiment instance ---
    if z0 is None:
        state0 = np.asarray(system.init_state(), dtype=float).ravel()
        if state0.size == (D + 1) * n:
            z0 = state0.reshape(D + 1, n)
        elif state0.size == n:
            x0 = state0.reshape(1, n)
            z0 = np.repeat(x0, repeats=(D + 1), axis=0)
        else:
            raise ValueError(f"init_state size {state0.size} unexpected; expected {n} or {(D+1)*n}")
    else:
        z0 = np.asarray(z0, dtype=float)
        if z0.ndim == 1:
            if z0.size == (D + 1) * n:
                z0 = z0.reshape(D + 1, n)
            elif z0.size == n:
                z0 = np.repeat(z0.reshape(1, n), repeats=(D + 1), axis=0)
            else:
                raise ValueError(f"passed z0 size {z0.size} unexpected; expected {n} or {(D+1)*n}")
        if z0.shape != (D + 1, n):
            raise ValueError(f"passed z0 has shape {z0.shape}, expected {(D+1,n)}")

    # --- spectral basis option ---
    basis = str(basis).lower().strip()
    if basis not in {"original", "spectral"}:
        raise ValueError(f"basis must be 'original' or 'spectral', got {basis}")

    if basis == "spectral":
        lambdas, Q = np.linalg.eigh(H)
        lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
        Q = np.asarray(Q, dtype=float)
        # row-vector convention: u = x @ Q
        z0_work = z0 @ Q
    else:
        lambdas, Q = None, None
        z0_work = z0

    # ---------- optional: MIQCP bounds check prep ----------
    if bounds_params is None:
        bounds_params = SolverBaselineParams(basis=basis, bound_mode="global", bound_scale=1.0)

    # In verifier, the "state" we evolve is exactly z0_work (basis-aligned history stack)
    state0 = z0_work
    max_abs0_state0 = float(np.max(np.abs(state0)))

    # ---------- Python rollout (ground truth under same semantics) ----------
    z = z0_work.copy()  # (D+1,n), z[j]=x_{t-j}
    JT_rollout = 0.0

    traj_roll = np.zeros((T + 1, D + 1, n), dtype=float)
    traj_roll[0] = z.copy()

    for t in range(T):
        v = z[0]
        if basis == "spectral":
            JT_rollout += float(np.sum(lambdas * (v ** 2)))
        else:
            JT_rollout += float(v @ (H @ v))

        grad_sum = np.zeros(n, dtype=float)
        if basis == "spectral":
            for j in range(D + 1):
                mj = float(m_hist[t, j])
                if mj == 0.0:
                    continue
                grad_sum += mj * (lambdas * z[j])
        else:
            for j in range(D + 1):
                mj = float(m_hist[t, j])
                if mj == 0.0:
                    continue
                grad_sum += mj * (H @ z[j])

        v_next = v - alpha * grad_sum
        z[1:] = z[:-1]
        z[0] = v_next
        traj_roll[t + 1] = z.copy()

    # ---------- (NEW) Check whether this feasible rollout violates MIQCP bounds ----------
    if check_miqcp_bounds:
        if action_space is None:
            raise ValueError("check_miqcp_bounds=True requires action_space (to match MIQCP bound computation).")

        
        rho_max = _compute_rho_max_inf_relaxed(H, action_space, eta=eta, K=K, D=D)
        max_abs0_local = max(1e-12, max_abs0_state0)
        abs0_vec = np.max(np.abs(state0), axis=0)

        M_global_local, lb_dict_local, ub_dict_local = _make_state_bounds(
            T=T, D=D, n=n, max_abs0=max_abs0_local, abs0_vec=abs0_vec, rho_max=rho_max, params=bounds_params
        )

        def get_bounds(t, j, i):
            if lb_dict_local is None:
                return -float(M_global_local), float(M_global_local)
            key = (t, j, i)
            return float(lb_dict_local[key]), float(ub_dict_local[key])

        max_violation = 0.0
        worst = None
        for t in range(T + 1):
            for j in range(D + 1):
                for i in range(n):
                    v = float(traj_roll[t, j, i])
                    lo, hi = get_bounds(t, j, i)
                    if v < lo:
                        viol = lo - v
                        if viol > max_violation:
                            max_violation = viol
                            worst = ("LOW", t, j, i, v, lo, hi)
                    elif v > hi:
                        viol = v - hi
                        if viol > max_violation:
                            max_violation = viol
                            worst = ("HIGH", t, j, i, v, lo, hi)

        print(f"[bound_check] basis={basis} bound_mode={bounds_params.bound_mode} "
              f"M_global={M_global_local:.6g} rho_max={rho_max:.6g} max_abs0_local={max_abs0_local:.6g}")

        if max_violation > 1e-9:
            print("[bound_check] FAIL: rollout violates MIQCP bounds.")
            print("  max_violation =", max_violation)
            print("  worst         =", worst)
        else:
            print("[bound_check] OK: rollout stays within MIQCP bounds.")


    # ---------- Gurobi feasibility model (linear constraints only) ----------
    model = gp.Model("verify_fixed_schedule_feas")
    model.Params.OutputFlag = 0

    # variables: x[t,j,i], t=0..T, j=0..D
    x = model.addVars(T + 1, D + 1, n, lb=-GRB.INFINITY, name="x")

    # init
    for j in range(D + 1):
        for i in range(n):
            model.addConstr(x[0, j, i] == float(z0_work[j, i]), name=f"init[{j},{i}]")

    # shift
    for t in range(T):
        for j in range(1, D + 1):
            for i in range(n):
                model.addConstr(x[t + 1, j, i] == x[t, j - 1, i], name=f"shift[{t},{j},{i}]")

    # main update
    for t in range(T):
        for i in range(n):
            if basis == "spectral":
                lam = float(lambdas[i])
                # x[t+1,0,i] = x[t,0,i] - alpha * lam * sum_j m[t,j] * x[t,j,i]
                rhs = x[t, 0, i] - alpha * lam * gp.quicksum(float(m_hist[t, j]) * x[t, j, i] for j in range(D + 1))
                model.addConstr(x[t + 1, 0, i] == rhs, name=f"main[{t},{i}]")
            else:
                # x[t+1,0,i] = x[t,0,i] - alpha * sum_j m[t,j] * (H x[t,j])_i
                inner = gp.LinExpr()
                for j in range(D + 1):
                    mj = float(m_hist[t, j])
                    if mj == 0.0:
                        continue
                    inner += mj * gp.quicksum(float(H[i, k]) * x[t, j, k] for k in range(n))
                model.addConstr(x[t + 1, 0, i] == x[t, 0, i] - alpha * inner, name=f"main[{t},{i}]")

    # feasibility
    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        # infeasible schedule under MIQCP dynamics => smoking gun!
        return None, float(JT_rollout), None

    # compute JT from solved trajectory (Python)
    JT_model = 0.0
    for t in range(T):
        if basis == "spectral":
            for r in range(n):
                lam = float(lambdas[r])
                v = float(x[t, 0, r].X)
                JT_model += lam * v * v
        else:
            xt = np.array([float(x[t, 0, i].X) for i in range(n)], dtype=float)
            JT_model += float(xt @ (H @ xt))

    diff = float(JT_model - JT_rollout)
    return float(JT_model), float(JT_rollout), diff

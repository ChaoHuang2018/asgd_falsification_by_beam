# alg/bound_search.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import heapq
import numpy as np
import time
import json, os
import time

# -----------------------------
# Params / stats
# -----------------------------
@dataclass
class BoundParams:
    """
    Parameters for a *sound* upper bound UB_tail(z, k) on remaining k-step cost.
    We follow the same logic as pointwise_static_cert:
      - decompose by eigenmodes of H
      - compute rho_lambda = max_{v in vertices} ||A_lambda(v)||_2
      - bound future lifted-state norms by rho^t
      - bound remaining sum of losses by sum_i lambda_i * ||z_mode||^2 * G_i(k)
    """
    relax_vertices: bool = True      # use continuous vertices (sound UB)
    cache_rho: bool = True           # cache rho_lams on system._bound_cache
    margin: float = 1e-12            # pruning/cover check tolerance

    lookahead_steps: int = 0

    # whether lookahead max uses exact histogram action set (tighter) instead of relaxed vertices
    lookahead_use_exact_actions: bool = True

@dataclass
class SearchParams:
    mode: str = "heuristic"          # "exact" | "cover_check" | "heuristic"
    T: int = 20
    seed: int = 0

    # heuristic-only
    beam_width: int = 50
    candidates_per_expand: int = 50

    # cover_check-only
    cover_beam_width: int = 200
    cover_candidates_per_expand: int = 200

    # pruning knobs
    enable_pruning: bool = True
    prune_eps: float = 0.0

    # bound builder (FIX: use default_factory)
    bound: BoundParams = field(default_factory=BoundParams)

    # --- static heuristic policy ---
    static_policy: str = "random"     # "random" | "single_max" | "average"
    static_seed: int = 0              # RNG seed used ONLY for random/static sampling

    dump_word_dir: str = ""

    # B-route: prefix nondominance filter
    enable_prefix_nondominance: bool = False
    prefix_nd_psd_tol: float = 1e-10
    # optional: a precomputed lifted-subspace basis U_pref of shape (dim_z, r)
    prefix_nd_U: Optional[np.ndarray] = None

    # B'-route: outer-bucket coverage truncation
    enable_bucket_coverage: bool = False
    # list of histogram actions (np.ndarray shape (D+1,)) used as buckets
    bucket_actions: Optional[List[np.ndarray]] = None

    # Which bucket IDs drive the truncation rule:
    #   - "bprime": use bucket_actions (vertex anchors) -> bid
    #   - "cprime": use cprime_bucket_id_by_action -> cbid
    bucket_coverage_mode: str = "bprime"

    # If True, do not truncate per-node expansions by candidates_per_expand
    # (useful for structurally complete runs driven by C' buckets).
    bucket_coverage_expand_all: bool = False


    # Certified action set per remaining tail length k (k = k_remain-1), indices into `actions`
    enable_certified_action_set: bool = False
    certified_action_indices_by_k: Optional[Dict[int, List[int]]] = None

    # Mapping: action histogram tuple(ints) -> cprime bucket id
    cprime_bucket_id_by_action: Optional[Dict[Tuple[int, ...], int]] = None

    # Only used for posterior consistency check; does NOT affect search.
    cprime_required_bucket_ids_by_k: Optional[Dict[int, List[int]]] = None

    # Posterior leakage bound (only meaningful when enable_certified_action_set=True):
    #   U1: one-step competitor UB for actions outside S_k
    #   U2: truncation UB = best UB among dropped nodes at beam_width cut
    #   Global_UB_run = max(U1, U2) is a sound global upper bound if ub_tail is sound.
    enable_leakage_ub: bool = False

    time_limit_sec: Optional[float] = None

def _get_H(system: Any, remodeling: Any) -> np.ndarray:
    if hasattr(system, "objective") and hasattr(system.objective, "H"):
        return np.asarray(system.objective.H, dtype=float)
    if hasattr(system, "H"):
        return np.asarray(system.H, dtype=float)
    if remodeling is not None and hasattr(remodeling, "H"):
        return np.asarray(remodeling.H, dtype=float)
    raise ValueError("Cannot find H on system.objective.H / system.H / remodeling.H")


def _get_eta(system: Any, remodeling: Any) -> float:
    eta = getattr(remodeling, "eta", None)
    if eta is None:
        eta = getattr(system, "eta", None)
    if eta is None:
        raise ValueError("Cannot find eta on remodeling.eta or system.eta")
    return float(eta)


def _project_init_to_modes(z0: np.ndarray, U: np.ndarray) -> np.ndarray:
    # z0: (D+1, n), U: (n,n)
    return (U.T @ z0.T).T  # (D+1, n)


def _A_lambda_from_hist(m: np.ndarray, lam: float, D: int, eta: float, K: int) -> np.ndarray:
    """
    Build the (D+1)x(D+1) lifted matrix for a single eigenmode (lambda=lam) and histogram m.
    m may be float if relax_vertices=True.
    """
    m = np.asarray(m, dtype=float).reshape(-1)
    alpha = float(eta) / float(K)
    A = np.zeros((D + 1, D + 1), dtype=float)
    A[0, 0] = 1.0 - alpha * float(m[0]) * float(lam)
    for j in range(1, D + 1):
        A[0, j] = -alpha * float(m[j]) * float(lam)
    for j in range(1, D + 1):
        A[j, j - 1] = 1.0
    return A


def _mode_step_from_hist(z_mode: np.ndarray, m: np.ndarray, lam: float, eta: float, K: int) -> np.ndarray:
    """
    z_mode: (D+1,) for a single eigenmode
    m: histogram (D+1,), may be float (for scoring), but for search actions we use integer.
    """
    alpha = float(eta) / float(K)
    s = float(np.dot(np.asarray(m, dtype=float), z_mode))
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

def _dump_best_word_json(out_dir: str, prefix: str, word: list):
    """Dump best schedule (histogram sequence) to JSON"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_best_word.json")
    data = {
        "prefix": prefix,
        "num_steps": len(word),
        "hist_sequence": [m.tolist() for m in word],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[dump] saved schedule to {path}")


class UpperBoundModel:
    """
    Holds (lambdas, U, rho_lams, G) and provides:
      - step(z_modes, m) -> z_modes_next
      - loss(z_modes) -> x^T H x
      - ub_tail(z_modes, k) -> sound upper bound on remaining k-step sum of losses
    """
    def __init__(self, system: Any, remodeling: Any, action_space: Any, T_max: int, params: BoundParams):
        H = _get_H(system, remodeling)
        H = 0.5 * (H + H.T)
        lambdas, U = np.linalg.eigh(H)
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.U = np.asarray(U, dtype=float)

        self.K = int(action_space.K)
        self.D = int(action_space.D)
        self.eta = _get_eta(system, remodeling)

        if not hasattr(action_space, "enumerate_histogram_vertices"):
            raise ValueError("ActionSpace missing enumerate_histogram_vertices(relax=...)")

        # ---------- sets for (i) rho_x precompute and (ii) lookahead maximization ----------
        # V_rho: vertices for rho_x (sound UB backbone). Use relaxed vertices (superset) => sound.
        if not hasattr(action_space, "enumerate_histogram_vertices"):
            raise ValueError("ActionSpace missing enumerate_histogram_vertices(relax=...)")

        V_rho = action_space.enumerate_histogram_vertices(relax=params.relax_vertices)
        if len(V_rho) == 0:
            raise ValueError("No histogram vertices found; check K,D,B")
        V_rho = [np.asarray(v, dtype=float).reshape(-1) for v in V_rho]  # list of (D+1,)

        # Uset: candidates for lookahead max. Choose exact action set for tighter UB if available.
        self.lookahead_steps = int(getattr(params, "lookahead_steps", 0))
        use_exact = bool(getattr(params, "lookahead_use_exact_actions", True))

        if use_exact:
            if not hasattr(action_space, "enumerate_histograms"):
                raise ValueError("ActionSpace missing enumerate_histograms() required for exact lookahead.")
            A_exact = action_space.enumerate_histograms()
            Uset = [np.asarray(m, dtype=float).reshape(-1) for m in A_exact]  # list of (D+1,)
        else:
            Uset = list(V_rho)

        self.lookahead_actions = Uset
        self._rho_vertices = V_rho  # keep for rho_x computation

        if len(self._rho_vertices) == 0:
            raise ValueError("No histogram vertices found; check K,D,B")

        # rho cache
        rho_lams = None
        cache_key = None
        if params.cache_rho:
            cache_key = (
                "rho_lambda_cache",
                int(self.K),
                int(self.D),
                float(self.eta),
                tuple(np.round(self.lambdas, 12).tolist()),
                bool(params.relax_vertices),
            )
            if hasattr(system, "_bound_cache") and cache_key in system._bound_cache:
                rho_lams = system._bound_cache[cache_key]

        if rho_lams is None:
            rho_lams = np.zeros_like(self.lambdas, dtype=float)
            for i, lam in enumerate(self.lambdas):
                lam = float(lam)
                best = 0.0
                for v in self._rho_vertices:
                    A = _A_lambda_from_hist(v, lam=lam, D=self.D, eta=self.eta, K=self.K)
                    val = float(np.linalg.norm(A, ord=2))  # spectral norm
                    if val > best:
                        best = val
                rho_lams[i] = best

            if params.cache_rho:
                if not hasattr(system, "_bound_cache"):
                    system._bound_cache = {}
                system._bound_cache[cache_key] = rho_lams

        self.rho_lams = np.asarray(rho_lams, dtype=float)
        # rho_x cache: rho_x_lambda = max_{v in vertices} ||c_lambda(v)||_2
        # where x_{t+1} = c_lambda(m)^T z_t  (single eigenmode)
        # rho_x_lams = None
        # cache_key = None
        # if params.cache_rho:
        #     cache_key = (
        #         "rho_x_lambda_cache",
        #         int(self.K),
        #         int(self.D),
        #         float(self.eta),
        #         tuple(np.round(self.lambdas, 12).tolist()),
        #         bool(params.relax_vertices),
        #     )
        #     if hasattr(system, "_bound_cache") and cache_key in system._bound_cache:
        #         rho_x_lams = system._bound_cache[cache_key]

        # if rho_x_lams is None:
        #     rho_x_lams = np.zeros_like(self.lambdas, dtype=float)
        #     alpha = float(self.eta) / float(self.K)

        #     for i, lam in enumerate(self.lambdas):
        #         lam = float(lam)
        #         best = 0.0
        #         for v in self._rho_vertices:
        #             m = np.asarray(v, dtype=float).reshape(-1)  # (D+1,)
        #             # c = [1 - alpha*m0*lam, -alpha*m1*lam, ..., -alpha*mD*lam]
        #             c0 = 1.0 - alpha * float(m[0]) * lam
        #             cj = -alpha * lam * m[1:]
        #             c = np.concatenate(([c0], cj), axis=0)
        #             val = float(np.linalg.norm(c, ord=2))
        #             if val > best:
        #                 best = val
        #         rho_x_lams[i] = best

        #     if params.cache_rho:
        #         if not hasattr(system, "_bound_cache"):
        #             system._bound_cache = {}
        #         system._bound_cache[cache_key] = rho_x_lams

        # self.rho_x_lams = np.asarray(rho_x_lams, dtype=float)

        # precompute G[i,k] for k up to T_max using rho_x
        self.G = np.zeros((self.lambdas.shape[0], T_max + 1), dtype=float)
        for i in range(self.lambdas.shape[0]):
            rho2 = float(self.rho_lams[i] ** 2)
            for k in range(T_max + 1):
                self.G[i, k] = _geom_sum(rho2, k)

        self.lookahead_steps = int(params.lookahead_steps)


    def init_z_modes(self, z0: np.ndarray) -> np.ndarray:
        # z0: (D+1, n)
        y0 = _project_init_to_modes(np.asarray(z0, dtype=float), self.U)  # (D+1, n)
        # store as (n, D+1)
        return np.asarray([y0[:, i] for i in range(y0.shape[1])], dtype=float)

    def loss(self, z_modes: np.ndarray) -> float:
        # stage loss = x^T H x = sum_i lambda_i * (x_i)^2 where x_i = z_modes[i,0]
        x = z_modes[:, 0]
        return float(np.sum(self.lambdas * (x * x)))

    def step(self, z_modes: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = np.empty_like(z_modes, dtype=float)
        for i, lam in enumerate(self.lambdas):
            out[i] = _mode_step_from_hist(z_modes[i], m, float(lam), eta=self.eta, K=self.K)
        return out

    def ub_tail_geom(self, z_modes: np.ndarray, k: int) -> float:
        """
        Sound UB on remaining k-step cost from current state.
        Uses: cost_t <= sum_i lambda_i * ||z_mode_i||^2
        and ||z_{t+r}|| <= rho^r ||z_t||.
        """
        if k <= 0:
            return 0.0
        # sum_i lambda_i * ||z_i||^2 * G[i,k]
        s = 0.0
        for i, lam in enumerate(self.lambdas):
            zi2 = float(np.dot(z_modes[i], z_modes[i]))
            s += float(lam) * zi2 * float(self.G[i, k])
        return float(s)
    
    def ub_tail_lookahead(self, z_modes: np.ndarray, k: int, s: int) -> float:
        """
        Sound upper bound:
        UB(z,k,s) = loss(z) + max_{m in lookahead_actions} UB(step(z,m), k-1, s-1)
        and when s==0: fallback to geom bound.
        """
        if k <= 0:
            return 0.0

        loss0 = self.loss(z_modes)
        if k == 1:
            return float(loss0)

        if s <= 0:
            # geom already includes loss0 + future; here we want tail from z_modes including loss0
            return float(self.ub_tail_geom(z_modes, k))

        best = -1e300
        for m in self.lookahead_actions:
            z1 = self.step(z_modes, m)
            val = self.ub_tail_lookahead(z1, k - 1, s - 1)
            if val > best:
                best = val

        return float(loss0 + best)


    def ub_tail(self, z_modes: np.ndarray, k: int) -> float:
        if k <= 0:
            return 0.0
        s = int(getattr(self, "lookahead_steps", 0))
        if s <= 0:
            return self.ub_tail_geom(z_modes, k)
        return self.ub_tail_lookahead(z_modes, k, s)



# -----------------------------
# Helpers: action sets
# -----------------------------
def _enumerate_hist_actions(action_space: Any) -> List[np.ndarray]:
    if not hasattr(action_space, "enumerate_histograms"):
        raise ValueError("ActionSpace missing enumerate_histograms()")
    A = [np.asarray(m, dtype=int) for m in action_space.enumerate_histograms()]
    return A


def _integer_vertex_actions(action_space: Any) -> List[np.ndarray]:
    """
    A small "subclass" action set for cover_check: integer vertices only.
    NOTE: not a sound upper bound; used only for LB_sub generation.
    """
    if not hasattr(action_space, "enumerate_histogram_vertices"):
        raise ValueError("ActionSpace missing enumerate_histogram_vertices(relax=...)")
    V_int = action_space.enumerate_histogram_vertices(relax=False)
    out = []
    for v in V_int:
        m = np.asarray(v, dtype=int).reshape(-1)
        if int(np.sum(m)) != int(action_space.K):
            continue
        if int(np.dot(np.arange(action_space.D + 1), m)) > int(action_space.B):
            continue
        out.append(m)
    # de-dup
    uniq = {}
    for m in out:
        uniq[tuple(m.tolist())] = m
    return list(uniq.values())

def _delays_to_hist(d: List[int], D: int) -> np.ndarray:
    m = np.zeros(D + 1, dtype=int)
    for x in d:
        m[int(x)] += 1
    return m

def _sample_static_hist_random(K: int, D: int, B: int, rng: np.random.Generator) -> np.ndarray:
    # sequentially sample each worker delay, respecting remaining budget
    rem = int(B)
    d = []
    for i in range(K):
        hi = min(D, rem)
        di = int(rng.integers(0, hi + 1))
        d.append(di)
        rem -= di
    return _delays_to_hist(d, D)

def _sample_static_hist_single_max(K: int, D: int, B: int, rng: np.random.Generator) -> np.ndarray:
    # one worker uses delay D (or min(D,B) if B < D), others random with remaining budget
    d0 = min(int(D), int(B))
    rem = int(B) - d0
    d = [d0]
    for i in range(K - 1):
        hi = min(D, rem)
        di = int(rng.integers(0, hi + 1))
        d.append(di)
        rem -= di
    return _delays_to_hist(d, D)

def _sample_static_hist_average(K: int, D: int, B: int) -> np.ndarray:
    # try to use budget B as evenly as possible, but cap each delay <= D
    total = min(int(B), int(K) * int(D))
    base = total // K
    rem = total - base * K
    d = [base + 1] * rem + [base] * (K - rem)
    # base should already be <= D due to total<=K*D, but keep safe
    d = [min(int(D), int(x)) for x in d]
    # if cap reduced sum (rare), it's still valid since "not necessarily use full B"
    return _delays_to_hist(d, D)

def _run_static_policy(
    ubm: UpperBoundModel,
    z0_modes: np.ndarray,
    T: int,
    m_static: np.ndarray,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    z = np.asarray(z0_modes, dtype=float)
    JT = 0.0
    for _ in range(T):
        JT += ubm.loss(z)
        z = ubm.step(z, m_static)

    word = [np.asarray(m_static, dtype=int).copy() for _ in range(T)]
    return {
        "mode": "static",
        "best_word": word,
        "best_JT": float(JT),
        "visited_nodes": int(T),
        "generated_nodes": 0,
        "pushed_nodes": 0,
        "pruned_prefixes": 0,
        "pruned_nodes_est": 0.0,
        "pruned_leaves_est": 0.0,
        "total_nodes_full": float(T + 1),
        "coverage_est": 0.0,
        "complete": False,
        "early_stop": False,
        "num_actions": None,
        "runtime_sec": float(time.perf_counter() - t0),
    }

# -----------------------------
# Core search engines
# -----------------------------
@dataclass
class PruningStats:
    A: int
    T: int

    visited: int = 0
    generated: int = 0
    pushed: int = 0

    # local (node-wise) pruning counts
    local_pruned_prefixes: int = 0
    local_pruned_nodes_est: float = 0.0
    local_pruned_leaves_est: float = 0.0

    # optional global discard (best-first style); DFS mode通常为0
    global_discarded_after_optimality: int = 0

    early_stop: bool = False

    # depth-wise stats (length T+1)
    expanded_by_depth: list = field(default_factory=list)
    pruned_by_depth: list = field(default_factory=list)
    generated_by_depth: list = field(default_factory=list)

    def __post_init__(self):
        # depth index = current depth (0..T)
        if not self.expanded_by_depth:
            self.expanded_by_depth = [0] * (self.T + 1)
        if not self.pruned_by_depth:
            self.pruned_by_depth = [0] * (self.T + 1)
        if not self.generated_by_depth:
            self.generated_by_depth = [0] * (self.T + 1)

    def total_nodes_full(self) -> float:
        if self.A <= 1:
            return float(self.T + 1)
        return float((self.A ** (self.T + 1) - 1) / (self.A - 1))

    def _subtree_est(self, remaining_depth: int):
        if remaining_depth < 0:
            remaining_depth = 0
        leaves = float(self.A ** remaining_depth)
        if self.A <= 1:
            nodes = float(remaining_depth + 1)
        else:
            nodes = float((self.A ** (remaining_depth + 1) - 1) / (self.A - 1))
        return nodes, leaves

    def add_local_prune(self, depth: int) -> None:
        self.local_pruned_prefixes += 1
        if 0 <= depth <= self.T:
            self.pruned_by_depth[depth] += 1
        remaining = int(self.T - depth)
        nodes, leaves = self._subtree_est(remaining)
        self.local_pruned_nodes_est += nodes
        self.local_pruned_leaves_est += leaves



def _run_exact(
    ubm: UpperBoundModel,
    actions: List[np.ndarray],
    z0_modes: np.ndarray,
    T: int,
    best_LB_init: float,
    params: SearchParams,
    incumbent_word: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    A = len(actions)
    stats = PruningStats(A=A, T=T)

    best_JT = float(best_LB_init)
    best_word: Optional[List[np.ndarray]] = list(incumbent_word) if incumbent_word is not None else None

    # node: (-ub_total, counter, depth, prefix_J, z_modes, word)
    # counter avoids heap comparing numpy arrays when priorities tie
    root_ub = ubm.ub_tail(z0_modes, T)
    heap: List[Tuple[float, int, int, float, np.ndarray, List[np.ndarray]]] = []
    counter = 0
    heapq.heappush(heap, (-float(root_ub), counter, 0, 0.0, z0_modes, []))


    eps = float(params.prune_eps)

    while heap:
        neg_ub, _, depth, prefix_J, z_modes, word = heapq.heappop(heap)
        stats.visited += 1
        ub_total = -float(neg_ub)

        # best-first global termination:
        # if the current *global max UB* cannot beat incumbent LB, then all remaining heap nodes cannot.
        if ub_total <= best_JT + params.bound.margin + eps:
            stats.early_stop = True
            # current popped node + everything still in heap are discarded due to global UB<=LB proof
            stats.global_discarded_after_optimality += 1 + len(heap)
            break


        # prune prefix if even its UB can't beat incumbent
        if params.enable_pruning and ub_total <= best_JT + params.bound.margin + eps:
            stats.add_local_prune(depth)         # pruning current prefix at depth
            continue

        if depth == T:
            # complete, prefix_J is JT
            if prefix_J > best_JT:
                best_JT = float(prefix_J)
                best_word = list(word)
            continue

        # expand all actions
        stage_loss = ubm.loss(z_modes)  # loss at time depth
        for m in actions:
            stats.generated += 1
            z_next = ubm.step(z_modes, m)
            new_prefix = float(prefix_J + stage_loss)
            child_ub = float(new_prefix + ubm.ub_tail(z_next, T - depth - 1))

            if params.enable_pruning and child_ub <= best_JT + params.bound.margin + eps:
                stats.add_local_prune(depth + 1)     # pruning child prefix at depth+1
                continue
            
            counter += 1
            heapq.heappush(heap, (-child_ub, counter, depth + 1, new_prefix, z_next, word + [m]))
            stats.pushed += 1

    return {
        "mode": "exact",
        "best_word": best_word,
        "best_JT": float(best_JT) if best_word is not None else None,
        "visited_nodes": int(stats.visited),
        "generated_nodes": int(stats.generated),
        "pushed_nodes": int(stats.pushed),
        "pruned_prefixes": int(stats.local_pruned_prefixes),
        "pruned_nodes_est": float(stats.local_pruned_nodes_est),
        "pruned_leaves_est": float(stats.local_pruned_leaves_est),
        "coverage_est": float(stats.local_pruned_nodes_est / max(1.0, stats.total_nodes_full())),
        "global_discarded_after_optimality": int(stats.global_discarded_after_optimality),
        "total_nodes_full": float(stats.total_nodes_full()),
        "complete": True,
        "early_stop": bool(stats.early_stop),
        "num_actions": int(A),
        "runtime_sec": float(time.perf_counter() - t0),
    }

def _run_exact_dfs(
    ubm: UpperBoundModel,
    actions: List[np.ndarray],
    z0_modes: np.ndarray,
    T: int,
    best_LB_init: float,
    params: SearchParams,
    incumbent_word: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Exact (complete) search with sound UB pruning, using DFS order.
    - Still sound / complete: explores all nodes not pruned by UB.
    - DFS helps reach leaves quickly (good incumbent earlier).
    - Children are expanded in descending child_ub order.
    - After sorting, we break when child_ub <= best_JT and count the rest as locally pruned children.
      (Note: we still computed child_ub for sorting; this does NOT reduce generated_nodes.)
    """
    t0 = time.perf_counter()

    A = len(actions)
    stats = PruningStats(A=A, T=T)

    best_JT = float(best_LB_init)
    best_word: Optional[List[np.ndarray]] = list(incumbent_word) if incumbent_word is not None else None

    eps = float(params.prune_eps)

    # stack items: (depth, prefix_J, z_modes, word)
    stack: List[Tuple[int, float, np.ndarray, List[np.ndarray]]] = [(0, 0.0, z0_modes, [])]

    while stack:
        depth, prefix_J, z_modes, word = stack.pop()
        stats.visited += 1
        if 0 <= depth <= T:
            stats.expanded_by_depth[depth] += 1

        # Prefix-level pruning (sound)
        ub_total = float(prefix_J + ubm.ub_tail(z_modes, T - depth))
        if params.enable_pruning and ub_total <= best_JT + params.bound.margin + eps:
            stats.add_local_prune(depth)
            continue

        # leaf
        if depth == T:
            if prefix_J > best_JT:
                best_JT = float(prefix_J)
                best_word = list(word)
            continue

        stage_loss = ubm.loss(z_modes)
        new_prefix = float(prefix_J + stage_loss)

        # Evaluate all children UBs (needed for exact ordering)
        children: List[Tuple[float, np.ndarray, np.ndarray]] = []
        # store: (child_ub, z_next, m)

        for m in actions:
            stats.generated += 1
            if 0 <= depth + 1 <= T:
                stats.generated_by_depth[depth + 1] += 1

            z_next = ubm.step(z_modes, m)
            child_ub = float(new_prefix + ubm.ub_tail(z_next, T - depth - 1))
            children.append((child_ub, z_next, m))

        # Sort children by UB descending (explore promising first)
        children.sort(key=lambda x: x[0], reverse=True)

        # Push children with child_ub > best_JT; once <=, remaining are also <= (due to sorting)
        # Count remaining as local prunes (depth+1) for reporting.
        pushed_now = 0
        for idx, (child_ub, z_next, m) in enumerate(children):
            if params.enable_pruning and child_ub <= best_JT + params.bound.margin + eps:
                # remaining children also prunable
                remaining = len(children) - idx
                for _ in range(remaining):
                    stats.add_local_prune(depth + 1)
                break

            # keep
            stack.append((depth + 1, new_prefix, z_next, word + [m]))
            pushed_now += 1

        stats.pushed += pushed_now

    return {
        "mode": "exact",
        "best_word": best_word,
        "best_JT": float(best_JT) if best_word is not None else None,
        "visited_nodes": int(stats.visited),
        "generated_nodes": int(stats.generated),
        "pushed_nodes": int(stats.pushed),
        # keep existing output keys expected by your printer
        "pruned_prefixes": int(stats.local_pruned_prefixes),
        "pruned_nodes_est": float(stats.local_pruned_nodes_est),
        "pruned_leaves_est": float(stats.local_pruned_leaves_est),
        "total_nodes_full": float(stats.total_nodes_full()),
        "coverage_est": float(stats.local_pruned_nodes_est / max(1.0, stats.total_nodes_full())),
        "complete": True,
        "early_stop": False,
        "global_discarded_after_optimality": int(stats.global_discarded_after_optimality),
        "num_actions": int(A),
        # depth-wise
        "expanded_by_depth": stats.expanded_by_depth,
        "pruned_by_depth": stats.pruned_by_depth,
        "generated_by_depth": stats.generated_by_depth,
        "runtime_sec": float(time.perf_counter() - t0),
    }

def _flatten_z_modes(z_modes: np.ndarray) -> np.ndarray:
    # z_modes: (n_modes, D+1) where each entry is scalar in modal coords
    # In your code, z_modes[i] is a (D+1,) vector per mode i.
    # Flatten to (n_modes*(D+1),)
    return np.asarray(z_modes, dtype=float).reshape(-1)

def _prefix_nondominated_filter(
    nodes,
    U_pref: np.ndarray,
    psd_tol: float,
):
    """
    Accept nodes in either shape:
      - (ub, z_next_modes, prefix_cost, word)                      [legacy]
      - (ub, bid, z_next_modes, prefix_cost, word)                 [beam v2]
      - (ub, bid, cbid, z_next_modes, prefix_cost, word)           [beam + C']
    Returns the same node tuples, filtered.
    """
    if len(nodes) <= 1:
        return nodes

    def psd(A: np.ndarray) -> bool:
        A = 0.5 * (A + A.T)
        w = np.linalg.eigvalsh(A)
        return float(w.min()) >= -float(psd_tol)

    # Extract (prefix_cost, projected_gram) for each node
    proj = []
    for item in nodes:
        # last 3 fields are always (z_modes, prefix_cost, word) in our beam tuples
        z_modes = item[-3]
        prefix_cost = float(item[-2])

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


def _beam_deadline(params: SearchParams, t_start: float) -> Optional[float]:
    tl = getattr(params, "time_limit_sec", None)
    if tl is None:
        return None
    try:
        tl = float(tl)
    except Exception:
        return None
    if tl <= 0:
        return t_start  # immediate timeout
    return t_start + tl


def _complete_greedy_to_leaf(
    ubm: UpperBoundModel,
    actions: List[np.ndarray],
    z_modes: np.ndarray,
    prefix_J: float,
    word: List[np.ndarray],
    depth: int,
    T: int,
    params: SearchParams,
    allowed_idxs_depth0: Optional[List[int]] = None,
) -> Tuple[float, List[np.ndarray], np.ndarray]:
    """
    Make a feasible full-length schedule by greedy completion from a prefix.
    Returns (JT, word_full, m_hist_full).
    Uses the same per-depth certified restriction if enabled.
    """
    z = np.asarray(z_modes, dtype=float)
    J = float(prefix_J)
    w = list(word)

    A = len(actions)
    for d in range(depth, T):
        k_remain = T - d
        k_tail = k_remain - 1

        # Determine allowed actions at this depth (if certified set is enabled)
        allowed = None
        if params.enable_certified_action_set and params.certified_action_indices_by_k is not None:
            allowed = params.certified_action_indices_by_k.get(int(k_tail), None)

        if allowed is None:
            iter_actions = enumerate(actions)
        else:
            iter_actions = ((i, actions[int(i)]) for i in allowed)

        stage_loss = ubm.loss(z)
        new_prefix = float(J + stage_loss)

        # Greedy pick by child_ub (prefix + ub_tail(next))
        best_child_ub = -1e300
        best_m = None
        best_z1 = None
        for i, m in iter_actions:
            z1 = ubm.step(z, m)
            child_ub = float(new_prefix + ubm.ub_tail(z1, k_remain - 1))
            if child_ub > best_child_ub:
                best_child_ub = child_ub
                best_m = m
                best_z1 = z1

        if best_m is None:
            # Should not happen; fallback to first action
            best_m = actions[0]
            best_z1 = ubm.step(z, best_m)
            new_prefix = float(J + ubm.loss(z))

        # Commit one step
        J = float(new_prefix)
        w.append(np.asarray(best_m, dtype=int).copy())
        z = np.asarray(best_z1, dtype=float)

    m_hist = np.stack([np.asarray(m, dtype=int).reshape(-1) for m in w], axis=0) if len(w) > 0 else None
    return float(J), w, m_hist


def _run_heuristic_beam(
    ubm: UpperBoundModel,
    actions: List[np.ndarray],
    z0_modes: np.ndarray,
    T: int,
    best_LB_init: float,
    beam_width: int,
    candidates_per_expand: int,
    params: SearchParams,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    A = len(actions)
    stats = PruningStats(A=A, T=T)

    best_JT = float(best_LB_init)
    best_word: Optional[List[np.ndarray]] = None

    eps = float(params.prune_eps)

    # beam elements: (z_modes, prefix_J, word)
    beam: List[Tuple[np.ndarray, float, List[np.ndarray]]] = [(z0_modes, 0.0, [])]

    bucket_map = None
    if params.enable_bucket_coverage and (params.bucket_coverage_mode == "bprime") and (params.bucket_actions is not None):
        bucket_map = {tuple(np.asarray(m, dtype=int).tolist()): i for i, m in enumerate(params.bucket_actions)}

    cprime_map = None
    if getattr(params, "cprime_bucket_id_by_action", None) is not None:
        cprime_map = params.cprime_bucket_id_by_action

    # --- S1b posterior consistency check (per-depth) ---
    req_cprime_by_k = None
    if getattr(params, "cprime_required_bucket_ids_by_k", None) is not None:
        req_cprime_by_k = params.cprime_required_bucket_ids_by_k

    s1b_rows = []  # list of dicts per depth
    s1b_all_pass = True
    s1b_applicable = (cprime_map is not None) and (req_cprime_by_k is not None)

    # --- Leakage UB (posterior quantitative certificate) ---
    leakage_applicable = bool(getattr(params, "enable_leakage_ub", False)) and \
        bool(params.enable_certified_action_set) and (params.certified_action_indices_by_k is not None)

    # --- Leakage UB (quantitative gap): sound ONLY when per-node expansion is "expand-all".
    # We compute:
    #   U1: best one-step competitor outside S_k (only when S_k restriction is enabled)
    #   U2: best dropped node at beam-width cut (works for both STRONG and Level III),
    #       but only sound if we did not truncate per-node expansions by candidates_per_expand.
    leakage_enabled = bool(getattr(params, "enable_leakage_ub", False))
    leakage_sound = True  # will be set False if we ever do per-node truncation at any depth

    leakage_U1_max = -1e300
    leakage_U1_best = None  # dict

    leakage_U2_max = -1e300
    leakage_U2_best = None  # dict

    leakage_by_depth = []  # list of dicts

    deadline = _beam_deadline(params, t0)
    anytime_timeout = False
    anytime_depth = None


    for depth in range(T):
        if deadline is not None and time.perf_counter() >= deadline:
            anytime_timeout = True
            anytime_depth = int(depth)
            break


        next_nodes = []
        # (score_ub, bucket_id, z_next, new_prefix, new_word)
        # Depth-level bookkeeping
        k_remain_depth = T - depth
        k_tail_depth = k_remain_depth - 1  # key used by certified_action_indices_by_k

        allowed_idxs_depth = None
        if params.enable_certified_action_set and params.certified_action_indices_by_k is not None:
            allowed_idxs_depth = params.certified_action_indices_by_k.get(k_tail_depth, None)

        # Determine whether this depth uses per-node expand-all.
        # Sound leakage UB is only meaningful under expand-all.
        per_node_expand_all = False
        if params.enable_certified_action_set and (allowed_idxs_depth is not None):
            per_node_expand_all = True
        elif params.enable_bucket_coverage and bool(getattr(params, "bucket_coverage_expand_all", False)):
            per_node_expand_all = True

        # Outside indices for U1 (only meaningful when certified action set S_k is active)
        outside_idxs_depth = None
        if leakage_enabled and params.enable_certified_action_set and (allowed_idxs_depth is not None):
            allowed_set = set(int(i) for i in allowed_idxs_depth)
            outside_idxs_depth = [i for i in range(A) if i not in allowed_set]

        # per-depth bests
        depth_U1_best = -1e300
        depth_U1_meta = None

        depth_U2_best = -1e300
        depth_U2_meta = None

        # Track nodes dropped by bucket-coverage pre-filter (must be included in U2 for soundness)
        cov_drop_best = -1e300
        cov_drop_meta = None


        for z_modes, prefix_J, word in beam:
            stats.visited += 1
            k_remain = k_remain_depth
            ub_total = float(prefix_J + ubm.ub_tail(z_modes, k_remain))

            k_tail = k_tail_depth  # child tail length (matches ub_tail(z_next, k_remain-1))


            if params.enable_pruning and ub_total <= best_JT + params.bound.margin + eps:
                stats.add_local_prune(depth)
                continue

            stage_loss = ubm.loss(z_modes)

            # If we can't afford scoring all actions, sample or take a subset.
            # Here: score all and take top candidates_per_expand by child UB (still ok when A small).
            new_prefix = float(prefix_J + stage_loss)

            # --- U1: Restriction leakage UB (one-step outside S_k) ---
            if leakage_enabled and (outside_idxs_depth is not None) and (len(outside_idxs_depth) > 0):
                node_best_out = -1e300
                node_best_i = None
                node_best_m = None

                for j in outside_idxs_depth:
                    m_out = actions[j]
                    z_out = ubm.step(z_modes, m_out)
                    out_ub = float(new_prefix + ubm.ub_tail(z_out, k_remain - 1))
                    if out_ub > node_best_out:
                        node_best_out = out_ub
                        node_best_i = int(j)
                        node_best_m = np.asarray(m_out, dtype=int).copy()

                if node_best_i is not None and node_best_out > depth_U1_best:
                    depth_U1_best = float(node_best_out)
                    depth_U1_meta = {
                        "depth": int(depth),
                        "k_tail": int(k_tail),
                        "parent_prefix_len": int(len(word)),
                        "parent_prefix_J_plus_stage": float(new_prefix),
                        "action_index": int(node_best_i),
                        "action_m": node_best_m.tolist(),
                        "ub": float(node_best_out),
                    }


            if allowed_idxs_depth is None:
                iter_actions = enumerate(actions)
            else:
                iter_actions = ((i, actions[i]) for i in allowed_idxs_depth)

            rescored: List[Tuple[float, np.ndarray, np.ndarray]] = []
            for i, m in iter_actions:
                z_next = ubm.step(z_modes, m)
                child_ub = float(new_prefix + ubm.ub_tail(z_next, k_remain - 1))
                rescored.append((child_ub, z_next, m))


            rescored.sort(key=lambda x: x[0], reverse=True)

            if per_node_expand_all:
                top = rescored
            else:
                cpe_eff = max(1, int(candidates_per_expand))
                top = rescored[:cpe_eff]

                # --- U1(b): leakage due to *per-node* action truncation within the iterated action set ---
                # We already computed and sorted rescored over iter_actions, so the best unexpanded action
                # is rescored[cpe_eff] (if it exists). This must be counted to make U1 sound.
                if leakage_enabled and len(rescored) > cpe_eff:
                    missed_ub, _, missed_m = rescored[cpe_eff]
                    if float(missed_ub) > float(depth_U1_best):
                        depth_U1_best = float(missed_ub)
                        depth_U1_meta = {
                            "depth": int(depth),
                            "k_tail": int(k_tail),
                            "parent_prefix_len": int(len(word)),
                            "parent_prefix_J_plus_stage": float(new_prefix),
                            "type": "unexpanded_within_iter_actions",
                            "action_index": None,
                            "action_m": np.asarray(missed_m, dtype=int).tolist(),
                            "ub": float(missed_ub),
                        }




            for child_ub, z_next, m in top:
                stats.generated += 1
                if params.enable_pruning and child_ub <= best_JT + params.bound.margin + eps:
                    stats.add_local_prune(depth + 1)
                    continue
                
                bid = -1
                if bucket_map is not None:
                    bid = bucket_map.get(tuple(np.asarray(m, dtype=int).tolist()), -1)

                cbid = -1
                if cprime_map is not None:
                    cbid = cprime_map.get(tuple(np.asarray(m, dtype=int).tolist()), -1)

                # item = (ub, bid, cbid, z_next, new_prefix, new_word)
                next_nodes.append((child_ub, bid, cbid, z_next, new_prefix, word + [m]))


                stats.pushed += 1

        if anytime_timeout:
            break

        if not next_nodes:
            break

        # ---------------- prefix safe filtering ----------------
        if params.enable_prefix_nondominance and (params.prefix_nd_U is not None):
            # Safe shrink by certificate-style nondominance (do NOT rely on UB-only truncation)
            next_nodes = _prefix_nondominated_filter(
                next_nodes,
                U_pref=params.prefix_nd_U,
                psd_tol=float(params.prefix_nd_psd_tol),
            )

        # ---------------- bucket coverage truncation ----------------
        # This modifies the definition of G_keep (the kept frontier) but does not affect UB soundness.
        if params.enable_bucket_coverage:
            mode = str(getattr(params, "bucket_coverage_mode", "bprime"))
            use_bprime = (mode == "bprime") and (bucket_map is not None)
            use_cprime = (mode == "cprime") and (cprime_map is not None)

            if use_bprime or use_cprime:
                # Keep a snapshot BEFORE coverage filtering so we can account for all dropped nodes (U2).
                pre_cov_nodes = list(next_nodes)

                # keep the best node per bucket (bucket_id>=0), by UB
                best_per_bucket = {}
                for item in next_nodes:
                    ub, bid, cbid, z_next, new_prefix, new_word = item
                    bucket_id = (bid if use_bprime else cbid)
                    if bucket_id >= 0:
                        prev = best_per_bucket.get(bucket_id, None)
                        if (prev is None) or (ub > prev[0]):
                            best_per_bucket[bucket_id] = item

                covered = list(best_per_bucket.values())
                selected_set = set(id(x) for x in covered)
                rest = [x for x in next_nodes if id(x) not in selected_set]
                rest.sort(key=lambda x: x[0], reverse=True)

                covered.sort(key=lambda x: x[0], reverse=True)
                next_nodes = list(covered)

                if len(next_nodes) < int(beam_width):
                    need = int(beam_width) - len(next_nodes)
                    next_nodes.extend(rest[:need])

                # --- U2(a): leakage due to coverage filtering ---
                if leakage_enabled:
                    kept_ids = set(id(x) for x in next_nodes)
                    dropped_cov = [x for x in pre_cov_nodes if id(x) not in kept_ids]
                    if len(dropped_cov) > 0:
                        dropped_cov.sort(key=lambda x: x[0], reverse=True)
                        dropped = dropped_cov[0]
                        dropped_ub, dropped_bid, dropped_cbid, _, dropped_new_prefix, dropped_word = dropped

                        dropped_last_m = None
                        if isinstance(dropped_word, list) and len(dropped_word) > 0:
                            try:
                                dropped_last_m = np.asarray(dropped_word[-1], dtype=int).tolist()
                            except Exception:
                                dropped_last_m = None

                        cov_drop_best = float(dropped_ub)
                        cov_drop_meta = {
                            "depth": int(depth),
                            "k_tail": int(k_tail_depth),
                            "type": "coverage_drop",
                            "dropped_prefix_len": int(len(dropped_word)) if isinstance(dropped_word, list) else None,
                            "dropped_prefix_J_plus_stage": float(dropped_new_prefix),
                            "dropped_action_last_m": dropped_last_m,
                            "dropped_bid": int(dropped_bid) if dropped_bid is not None else None,
                            "dropped_cbid": int(dropped_cbid) if dropped_cbid is not None else None,
                            "ub": float(dropped_ub),
                        }


        next_nodes.sort(key=lambda x: x[0], reverse=True)

        # --- U2: Truncation leakage UB (best dropped node at beam-width cut) ---
        bw = max(1, int(beam_width))

        # --- U2(b): leakage due to final beam-width truncation (after coverage, after sorting) ---
        bw_drop_best = -1e300
        bw_drop_meta = None
        if leakage_enabled and (len(next_nodes) > bw):
            dropped = next_nodes[bw]  # best among those cut by beam width (sorted desc)
            dropped_ub, dropped_bid, dropped_cbid, _, dropped_new_prefix, dropped_word = dropped

            dropped_last_m = None
            if isinstance(dropped_word, list) and len(dropped_word) > 0:
                try:
                    dropped_last_m = np.asarray(dropped_word[-1], dtype=int).tolist()
                except Exception:
                    dropped_last_m = None

            bw_drop_best = float(dropped_ub)
            bw_drop_meta = {
                "depth": int(depth),
                "k_tail": int(k_tail_depth),
                "type": "beamwidth_drop",
                "dropped_rank": int(bw + 1),  # 1-based
                "dropped_prefix_len": int(len(dropped_word)) if isinstance(dropped_word, list) else None,
                "dropped_prefix_J_plus_stage": float(dropped_new_prefix),
                "dropped_action_last_m": dropped_last_m,
                "dropped_bid": int(dropped_bid) if dropped_bid is not None else None,
                "dropped_cbid": int(dropped_cbid) if dropped_cbid is not None else None,
                "ub": float(dropped_ub),
            }

        # Depth U2 is the worst among ALL dropped-by-truncation mechanisms at this depth
        if leakage_enabled:
            # start from coverage drop if present
            depth_U2_best = float(cov_drop_best) if cov_drop_meta is not None else -1e300
            depth_U2_meta = dict(cov_drop_meta) if cov_drop_meta is not None else None

            # compare with beam-width cut
            if bw_drop_meta is not None and float(bw_drop_best) > float(depth_U2_best):
                depth_U2_best = float(bw_drop_best)
                depth_U2_meta = dict(bw_drop_meta)


        # keep top beam_width
        next_nodes = next_nodes[:bw]
        beam = [(z_next, new_prefix, new_word) for _, _, _, z_next, new_prefix, new_word in next_nodes]


        # --- S1b: after FINAL truncation at this depth, check C' bucket coverage for S_k ---
        if s1b_applicable:
            # k_tail matches the key used by certified_action_indices_by_k (k_tail = k_remain-1)
            k_remain = T - depth
            k_tail = k_remain - 1

            req_list = req_cprime_by_k.get(k_tail, [])
            req_set = set(int(x) for x in req_list if int(x) >= 0)

            kept_cbids = set(int(item[2]) for item in next_nodes if int(item[2]) >= 0)

            covered = len(req_set & kept_cbids)
            total = len(req_set)
            row_pass = (covered == total)

            s1b_rows.append({
                "depth": int(depth),
                "k_tail": int(k_tail),
                "required_buckets": int(total),
                "covered_buckets": int(covered),
                "pass": bool(row_pass),
            })

            if not row_pass:
                s1b_all_pass = False

        # finalize leakage UB for this depth
        if leakage_enabled and per_node_expand_all:
            # update global U1
            if depth_U1_meta is not None and float(depth_U1_best) > float(leakage_U1_max):
                leakage_U1_max = float(depth_U1_best)
                leakage_U1_best = dict(depth_U1_meta)

            # update global U2
            if depth_U2_meta is not None and float(depth_U2_best) > float(leakage_U2_max):
                leakage_U2_max = float(depth_U2_best)
                leakage_U2_best = dict(depth_U2_meta)

            leakage_by_depth.append({
                "depth": int(depth),
                "k_tail": int(k_tail_depth),
                "U1_best_outside_Sk": (float(depth_U1_best) if depth_U1_meta is not None else None),
                "U2_best_dropped": (float(depth_U2_best) if depth_U2_meta is not None else None),
                "U1_meta": depth_U1_meta,
                "U2_meta": depth_U2_meta,
            })


    # finalize: among beam nodes at depth==T, prefix_J is JT
    if len(beam) > 0 and len(beam[0][2]) == T:
        for z_modes, prefix_J, word in beam:
            if prefix_J > best_JT:
                best_JT = float(prefix_J)
                best_word = list(word)
    
    best_m_hist = None
    if best_word is not None:
        best_m_hist = np.stack([np.asarray(m, dtype=int).reshape(-1) for m in best_word], axis=0)  # (T, D+1)

    # ---------------- anytime finalize ----------------
    # If timeout happened before reaching depth T, build a feasible incumbent by greedy completion.
    if anytime_timeout:
        # Choose a good prefix node to complete:
        # Prefer the node with the largest current UB estimate (prefix_J + ub_tail).
        best_idx = 0
        best_pref_ub = -1e300
        cur_depth = int(anytime_depth) if anytime_depth is not None else 0

        for i, (z_modes, prefix_J, word) in enumerate(beam):
            k_remain = T - len(word)
            ub_est = float(prefix_J + ubm.ub_tail(z_modes, k_remain))
            if ub_est > best_pref_ub:
                best_pref_ub = ub_est
                best_idx = i

        z_sel, J_sel, w_sel = beam[best_idx]
        JT_full, w_full, m_hist_full = _complete_greedy_to_leaf(
            ubm=ubm,
            actions=actions,
            z_modes=z_sel,
            prefix_J=float(J_sel),
            word=list(w_sel),
            depth=len(w_sel),
            T=T,
            params=params,
        )

        best_JT = float(JT_full)
        best_word = list(w_full)
        best_m_hist = m_hist_full

    leakage_ub_run = None
    if leakage_enabled:
        vals = []
        if leakage_U1_best is not None:
            vals.append(float(leakage_U1_max))
        if leakage_U2_best is not None:
            vals.append(float(leakage_U2_max))
        if len(vals) > 0:
            leakage_ub_run = float(max(vals))

    return {
        "mode": "heuristic",
        "best_word": best_word,
        "best_m_hist": best_m_hist,
        "best_JT": float(best_JT) if best_word is not None else None,
        "visited_nodes": int(stats.visited),
        "generated_nodes": int(stats.generated),
        "pushed_nodes": int(stats.pushed),
        "pruned_prefixes": int(stats.local_pruned_prefixes),
        "pruned_nodes_est": float(stats.local_pruned_nodes_est),
        "pruned_leaves_est": float(stats.local_pruned_leaves_est),
        "total_nodes_full": float(stats.total_nodes_full()),
        "coverage_est": float(stats.local_pruned_nodes_est / max(1.0, stats.total_nodes_full())),
        "global_discarded_after_optimality": 0,
        "complete": False,
        "num_actions": int(A),
        "beam_width": int(beam_width),
        "candidates_per_expand": int(candidates_per_expand),
        "runtime_sec": float(time.perf_counter() - t0),
        # Posterior consistency certificate (S1b)
        "consistency_s1b_applicable": bool(s1b_applicable),
        "consistency_s1b_pass": (bool(s1b_all_pass) if s1b_applicable else None),
        "consistency_s1b_by_depth": s1b_rows,
        # Leakage UB (posterior quantitative certificate) for conditional mode
        "leakage_ub_applicable": bool(leakage_enabled and (leakage_ub_run is not None)),
        "leakage_ub_global_max": leakage_ub_run,
        "leakage_U1_restriction_max": (float(leakage_U1_max) if leakage_U1_best is not None else None),
        "leakage_U1_restriction_best": leakage_U1_best,
        "leakage_U2_truncation_max": (float(leakage_U2_max) if leakage_U2_best is not None else None),
        "leakage_U2_truncation_best": leakage_U2_best,
        # "leakage_ub_global_max": (
        #     float(max(leakage_U1_max if leakage_U1_best is not None else -1e300,
        #               leakage_U2_max if leakage_U2_best is not None else -1e300))
        #     if (leakage_U1_best is not None or leakage_U2_best is not None) else None
        # ),
        "leakage_ub_by_depth": leakage_by_depth,
        "anytime_timeout": bool(anytime_timeout),
        "anytime_timeout_depth": int(anytime_depth) if anytime_depth is not None else None,
    }


def run_bound_search(
    system: Any,
    remodeling: Any,
    action_space: Any,
    z0: Optional[np.ndarray],
    params: SearchParams,
) -> Dict[str, Any]:
    """
    Unified entry:
      - exact:    best-first full expansion over histogram actions + sound UB pruning
      - cover_check: UB_all vs LB_sub (vertex subclass) + optional stats
      - heuristic: beam search over histogram actions + sound UB pruning + stats
    """
    t_total0 = time.perf_counter()

    T = int(params.T)

    # init z0 (history stack) and mode-space state
    if z0 is None:
        z0 = np.asarray(system.init_state(), dtype=float)
    ubm = UpperBoundModel(system, remodeling, action_space, T_max=T, params=params.bound)
    z0_modes = ubm.init_z_modes(z0)

    # Full action set = exact histogram actions (exact quotient, not restricting adversary)
    actions_full = _enumerate_hist_actions(action_space)

    # Always compute UB_all (useful for logging/cover)
    UB_all = float(ubm.ub_tail(z0_modes, T))

    # A simple initial LB: 0 (you can pass in something better if you want)
    best_LB_init = 0.0

    t_cover0 = time.perf_counter()
    if params.mode == "cover_check":
        sub_actions = _integer_vertex_actions(action_space)
        # LB_sub via beam on subclass (cheap: |sub_actions| is small)
        sub_res = _run_heuristic_beam(
            ubm=ubm,
            actions=sub_actions,
            z0_modes=z0_modes,
            T=T,
            best_LB_init=best_LB_init,
            beam_width=max(1, int(params.cover_beam_width)),
            candidates_per_expand=max(1, int(params.cover_candidates_per_expand)),
            params=params,
        )
        LB_sub = float(sub_res["best_JT"]) if sub_res.get("best_JT", None) is not None else -1e300
        gap = float(UB_all - LB_sub)

        return {
            "mode": "cover_check",
            "UB_all": float(UB_all),
            "LB_sub": float(LB_sub),
            "gap": float(gap),
            "cover_pass": bool(UB_all <= LB_sub + params.bound.margin),
            "subclass": "integer_vertices",
            "sub_details": sub_res,
            "num_actions_full": int(len(actions_full)),
            "num_actions_sub": int(len(sub_actions)),
            "runtime_sec": float(time.perf_counter() - t_cover0),
            "runtime_sub_sec": float(sub_res.get("runtime_sec", 0.0)),
        }

    if params.mode == "exact":
        # ---- seed incumbent via a quick heuristic run (feasible LB + explicit sequence) ----
        # IMPORTANT: disable pruning in seed run to avoid pruning everything before any LB exists.
        seed_params = SearchParams(
            mode="heuristic",
            T=params.T,
            seed=params.seed,
            beam_width=min(len(actions_full), max(50, params.beam_width)),
            candidates_per_expand=min(len(actions_full), max(50, params.candidates_per_expand)),
            enable_pruning=False,          # <-- force OFF for seeding
            prune_eps=params.prune_eps,
            bound=params.bound,
            cover_beam_width=params.cover_beam_width,
            cover_candidates_per_expand=params.cover_candidates_per_expand,
        )

        t_seed0 = time.perf_counter()
        seed = _run_heuristic_beam(
            ubm=ubm,
            actions=actions_full,
            z0_modes=z0_modes,
            T=T,
            best_LB_init=0.0,
            beam_width=seed_params.beam_width,
            candidates_per_expand=seed_params.candidates_per_expand,
            params=seed_params,
        )
        seed_runtime = float(time.perf_counter() - t_seed0)

        seed_LB = float(seed["best_JT"]) if seed.get("best_JT", None) is not None else 0.0
        seed_word = seed.get("best_word", None)

        # If seed already matches (or exceeds) global UB, we can certify optimal immediately.
        if seed_word is not None and UB_all <= seed_LB + params.bound.margin:
            return {
                "mode": "exact",
                "best_word": seed_word,
                "best_JT": seed_LB,
                "visited_nodes": 0,
                "generated_nodes": 0,
                "pushed_nodes": 0,
                "pruned_prefixes": 0,
                "pruned_nodes_est": 0.0,
                "pruned_leaves_est": 0.0,
                "total_nodes_full": float(PruningStats(A=len(actions_full), T=T).total_nodes_full()),
                "coverage_est": 0.0,
                "complete": True,
                "early_stop": True,
                "num_actions": int(len(actions_full)),
                "UB_all": float(UB_all),
                "seed_LB": seed_LB,
                "seed_certified_optimal": True,
                "runtime_seed_sec": seed_runtime,
                "runtime_sec_total": float(time.perf_counter() - t_total0),
            }

        # ---- now run exact with seeded incumbent (LB + word) ----
        res = _run_exact_dfs(
            ubm=ubm,
            actions=actions_full,
            z0_modes=z0_modes,
            T=T,
            best_LB_init=seed_LB,
            params=params,
            incumbent_word=seed_word,
        )

        res["seed_LB"] = seed_LB
        res["seed_certified_optimal"] = False
        res["UB_all"] = float(UB_all)
        res["runtime_seed_sec"] = seed_runtime
        res["runtime_sec_total"] = float(time.perf_counter() - t_total0)

        # Dump best schedule if requested
        dump_dir = getattr(params, "dump_word_dir", None)
        if dump_dir and "best_word" in res and res["best_word"] is not None:
            prefix = f"{params.mode}_{params.seed}_T{params.T}"
            _dump_best_word_json(dump_dir, prefix, res["best_word"])

        return res

    if params.mode in ("heuristic", "beam"):
        res = _run_heuristic_beam(
            ubm=ubm,
            actions=actions_full,
            z0_modes=z0_modes,
            T=T,
            best_LB_init=best_LB_init,
            beam_width=max(1, int(params.beam_width)),
            candidates_per_expand=max(1, int(params.candidates_per_expand)),
            params=params,
        )
        res["runtime_sec_total"] = float(time.perf_counter() - t_total0)
        res["UB_all"] = float(UB_all)

        # Dump best schedule if requested
        dump_dir = getattr(params, "dump_word_dir", None)
        if dump_dir and "best_word" in res and res["best_word"] is not None:
            prefix = f"{params.mode}_{params.seed}_T{params.T}"
            _dump_best_word_json(dump_dir, prefix, res["best_word"])

        return res

    if params.mode == "static":
        rng = np.random.default_rng(int(getattr(params, "static_seed", 0)))
        pol = str(getattr(params, "static_policy", "random"))

        if pol == "random":
            m_static = _sample_static_hist_random(ubm.K, ubm.D, int(action_space.B), rng)
        elif pol == "single_max":
            m_static = _sample_static_hist_single_max(ubm.K, ubm.D, int(action_space.B), rng)
        elif pol == "average":
            m_static = _sample_static_hist_average(ubm.K, ubm.D, int(action_space.B))
        else:
            raise ValueError(f"Unknown static_policy: {pol}")

        res = _run_static_policy(ubm=ubm, z0_modes=z0_modes, T=T, m_static=m_static)
        res["static_policy"] = pol
        res["runtime_sec_total"] = float(time.perf_counter() - t_total0)
        res["UB_all"] = float(UB_all)

        # Dump best schedule if requested
        dump_dir = getattr(params, "dump_word_dir", None)
        if dump_dir and "best_word" in res and res["best_word"] is not None:
            prefix = f"{params.mode}_{params.seed}_T{params.T}"
            _dump_best_word_json(dump_dir, prefix, res["best_word"])

        return res


    raise ValueError(f"Unknown mode: {params.mode}")


# Optional: pretty-print helper to convert histogram word to delay-vectors for display
def word_hist_to_delay(action_space: Any, word: List[np.ndarray]) -> List[np.ndarray]:
    if not hasattr(action_space, "histogram_to_delay"):
        # fallback: expand manually
        out = []
        for m in word:
            d = []
            for j, cnt in enumerate(np.asarray(m, dtype=int).tolist()):
                d.extend([j] * int(cnt))
            out.append(np.asarray(d, dtype=int))
        return out
    return [np.asarray(action_space.histogram_to_delay(m), dtype=int) for m in word]


def _as_history_stack(z0: np.ndarray, D: int, n: int) -> np.ndarray:
    """
    Normalize z0 into shape (D+1, n) where row j is x_{t-j} at current t.
    Accepts:
      - (D+1, n)
      - flat ((D+1)*n,)
      - (n,) : treated as cold start => repeat x0 across history
    """
    z0 = np.asarray(z0, dtype=float)
    if z0.ndim == 2:
        assert z0.shape == (D + 1, n), f"z0 must be {(D+1,n)} but got {z0.shape}"
        return z0.copy()
    if z0.ndim == 1:
        if z0.size == (D + 1) * n:
            return z0.reshape(D + 1, n).copy()
        if z0.size == n:
            x0 = z0.reshape(1, n)
            return np.repeat(x0, repeats=(D + 1), axis=0).copy()
    raise ValueError(f"Unsupported z0 shape {z0.shape}, expected (D+1,n) or flat or (n,)")


def verify_fixed_schedule_no_gurobi(system, remodeling, action_space, z0, json_path: str,
                                    relax_vertices: bool, atol: float = 1e-9, rtol: float = 1e-9):
    """
    Fixed-schedule cross-check WITHOUT gurobi:
      - x-space rollout using H (ground truth)
      - mode-space rollout using UpperBoundModel (spectral semantics)
    """

    with open(json_path, "r") as f:
        data = json.load(f)
    m_hist = np.asarray(data["hist_sequence"], dtype=float)  # (T, D+1)
    T = int(m_hist.shape[0])

    H = _get_H(system, remodeling)
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    D = int(action_space.D)
    K = int(action_space.K)
    eta = float(_get_eta(system, remodeling))
    alpha = eta / float(K)

    z0 = np.asarray(z0, dtype=float)
    if z0.ndim == 1:
        z0 = z0.reshape(D + 1, n)
    assert z0.shape == (D + 1, n), f"z0 must be {(D+1,n)} but got {z0.shape}"

    # UpperBoundModel in the SAME basis as used by beam (remodeling already carries that)
    bp = BoundParams(
        relax_vertices=bool(relax_vertices),
        cache_rho=False,
        margin=0.0,
        lookahead_steps=0,
        lookahead_use_exact_actions=True,
    )
    ubm = UpperBoundModel(system, remodeling, action_space, T_max=T, params=bp)
    z_modes = ubm.init_z_modes(z0)

    # x-space rollout with history stack
    z_x = z0.copy()  # z_x[j] = x_{t-j}

    JT_x = 0.0
    JT_modes = 0.0
    max_stage_abs = 0.0

    for t in range(T):
        x_t = z_x[0]
        lx = float(x_t @ (H @ x_t))
        lm = float(ubm.loss(z_modes))

        JT_x += lx
        JT_modes += lm
        max_stage_abs = max(max_stage_abs, abs(lx - lm))

        # x_{t+1}
        grad_sum = np.zeros((n,), dtype=float)
        for j in range(D + 1):
            mj = float(m_hist[t, j])
            if mj == 0.0:
                continue
            grad_sum += mj * (H @ z_x[j])
        x_next = x_t - alpha * grad_sum

        # shift
        z_x[1:] = z_x[:-1]
        z_x[0] = x_next

        # step in mode-space
        z_modes = ubm.step(z_modes, m_hist[t])

    abs_diff = abs(JT_x - JT_modes)
    rel_diff = abs_diff / max(1.0, abs(JT_x))

    ok = (np.isclose(JT_x, JT_modes, atol=atol, rtol=rtol) and max_stage_abs <= (atol + rtol * max(1.0, abs(JT_x))))

    print("[verify_no_gurobi] OK" if ok else "[verify_no_gurobi] MISMATCH")
    print(f"  JT_x       = {JT_x:.12e}")
    print(f"  JT_modes   = {JT_modes:.12e}")
    print(f"  abs_diff   = {abs_diff:.3e} (rel {rel_diff:.3e})")
    print(f"  max_stage  = {max_stage_abs:.3e}")

    return {
        "ok": bool(ok),
        "JT_x": float(JT_x),
        "JT_modes": float(JT_modes),
        "abs_diff": float(abs_diff),
        "rel_diff": float(rel_diff),
        "max_stage_abs": float(max_stage_abs),
    }
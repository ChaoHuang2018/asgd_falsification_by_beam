# test_bound_search.py
from __future__ import annotations

import argparse
from typing import Optional
import numpy as np
import json

import time
import os

from core.action_space import ActionSpace
from examples.quadratic_asgd import QuadraticASGDSystem, QuadraticObjective

# New solver
from alg.bound_search import (
    SearchParams,
    BoundParams,
    run_bound_search,
    word_hist_to_delay,
    verify_fixed_schedule_no_gurobi
)


def make_spd_H(n: int, condition: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.linspace(1.0, condition, n)
    H = Q @ np.diag(eigs) @ Q.T
    H = 0.5 * (H + H.T)
    return H


def _hist_from_delay(d: np.ndarray, D: int) -> np.ndarray:
    d = np.asarray(d, dtype=int).reshape(-1)
    m = np.zeros(D + 1, dtype=int)
    for x in d:
        m[int(x)] += 1
    return m


def sample_init_state_ball_fallback(rng: np.random.Generator, D: int, n: int, radius: float) -> np.ndarray:
    # uniform in L2-ball in R^{(D+1)*n}, then reshape to (D+1,n)
    dim = int((D + 1) * n)
    v = rng.normal(size=(dim,))
    nv = float(np.linalg.norm(v))
    if nv < 1e-15:
        v[0] = 1.0
        nv = 1.0
    v = v / nv
    r = float(radius) * float(rng.random() ** (1.0 / dim))
    return (v * r).reshape(D + 1, n)


def print_best_sequence_from_bound_search(
    out: dict,
    action_space: ActionSpace,
    K: int,
    D: int,
    B: int,
    max_lines: int = 200,
) -> None:
    best_word = out.get("best_word", None)
    if not best_word:
        print("[best_sequence] <empty or not found>")
        return

    # best_word is a list of histograms m_t; convert to delay vectors d_t for printing
    d_word = word_hist_to_delay(action_space, best_word)
    T = len(d_word)

    print(f"=== Best sequence (length T={T}) ===")
    print("best_JT:", out.get("best_JT", None))
    print("num_actions:", out.get("num_actions", None),
          "visited_nodes:", out.get("visited_nodes", None),
          "complete:", out.get("complete", None),
          "early_stop:", out.get("early_stop", None))
    
    if "expanded_by_depth" in out:
        print("expanded_by_depth:", out["expanded_by_depth"])
    if "pruned_by_depth" in out:
        print("pruned_by_depth  :", out["pruned_by_depth"])


    print("Format: t | delay-vector d_t | sum(d_t) | histogram m_t")
    for t, d in enumerate(d_word):
        if t >= max_lines:
            print(f"... (truncated after {max_lines} steps)")
            break
        d = np.asarray(d, dtype=int).reshape(-1)
        s = int(d.sum())
        m = _hist_from_delay(d, D)
        ok = ("OK" if (np.all((0 <= d) & (d <= D)) and s <= B) else "INVALID")
        print(f"{t:3d} | {d.tolist()} | sum={s:2d}/{B:2d} | m={m.tolist()} | {ok}")


def print_pruning_stats(out: dict) -> None:
    print("=== Pruning / search stats ===")
    for k in [
        "runtime_sec",
        "runtime_seed_sec",
        "runtime_sub_sec",
        "runtime_sec_total",
        "visited_nodes",
        "generated_nodes",
        "pushed_nodes",
        "pruned_prefixes",
        "pruned_nodes_est",
        "pruned_leaves_est",
        "total_nodes_full",
        "coverage_est",
        "UB_all",
    ]:
        if k in out:
            print(f"{k:18s}: {out[k]}")


def build_H(args) -> np.ndarray:
    # Keep consistent with main_v2: if you later want to reuse init_subspace semantics,
    # you can extend this here similarly. For now, just SPD.
    return make_spd_H(args.n, args.cond, args.seed)


def main():
    parser = argparse.ArgumentParser("Test bound_search (exact / cover_check / heuristic)")

    # System
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--cond", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--B", type=int, default=6)
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    # Initial state for this test
    parser.add_argument("--x0_radius", type=float, default=1.0, help="L2-ball radius for z0 sampling")
    parser.add_argument("--x0_seed", type=int, default=0, help="RNG seed for z0 sampling")
    parser.add_argument("--use_ball_init", action="store_true", help="sample z0 from L2 ball (else use system.init_state())")

    # Search modes
    parser.add_argument("--search_mode", type=str, default="heuristic",
                    choices=[
                        "exact", "cover_check",
                        "beam",
                        "static_random", "static_single_max", "static_average",
                        "heuristic",   
                        "gurobi",
                    ])
    
    parser.add_argument("--policy_seed", type=int, default=-1,
                    help="RNG seed for static_random/static_single_max. If <0, use seed*1000 + x0_seed.")

    # ---- gurobi ablation knobs (new) ----
    parser.add_argument("--gurobi_basis", type=str, default="original",
                        choices=["original", "spectral"],
                        help="MIQCP remodeling basis: original vs spectral(decoupled)")
    parser.add_argument("--gurobi_bound_mode", type=str, default="global",
                        choices=["global", "time_indexed", "per_mode_time_indexed"],
                        help="State bounds: global M vs time-indexed M(t,j,*)")
    parser.add_argument("--gurobi_m_encoding", type=str, default="hist_miqcp",
                        choices=["hist_miqcp", "onehot_bigm"],
                        help="Use MIQCP or BigM")
    parser.add_argument("--gurobi_bound_scale", type=float, default=2.0,
                        help="Conservative multiplier for computed bounds")
    parser.add_argument("--gurobi_time_limit", type=float, default=60.0)
    parser.add_argument("--gurobi_gap", type=float, default=1e-8)
    parser.add_argument("--gurobi_output", type=int, default=1,
                        help="Gurobi OutputFlag (0 silent, 1 verbose)")

    # heuristic params
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--cand", type=int, default=50)

    # cover_check subclass LB params
    parser.add_argument("--cover_beam", type=int, default=200)
    parser.add_argument("--cover_cand", type=int, default=200)

    # bound / pruning
    parser.add_argument("--no_pruning", action="store_true")
    parser.add_argument("--prune_eps", type=float, default=0.0)
    parser.add_argument("--bound_relax_vertices", action="store_true",
                        help="use continuous polytope vertices (sound UB, usually recommended)")
    parser.add_argument("--no_cache_rho", action="store_true")
    parser.add_argument("--bound_margin", type=float, default=1e-12)

    parser.add_argument("--dump_word_dir", type=str, default=None,
                        help="If set, dump the best schedule (histogram sequence) to this directory in JSON format.")
    parser.add_argument("--verify_fixed_with_gurobi", action="store_true",
                        help="Verify the dumped schedule with Gurobi fixed-schedule QP to check consistency.")


    args = parser.parse_args()

    # Build system
    H = build_H(args)
    obj = QuadraticObjective(H=H)
    system = QuadraticASGDSystem(
        objective=obj,
        K=args.K,
        D=args.D,
        eta=args.eta,
        seed=args.seed,
        init_subspace_mode="none",
        init_subspace_dim=0,
    )
    remodeling = system.make_remodeling()
    action_space = ActionSpace(K=args.K, D=args.D, B=args.B)

    # Choose z0
    rng = np.random.default_rng(args.x0_seed)
    if args.use_ball_init:
        if hasattr(system, "sample_init_state_ball"):
            z0 = system.sample_init_state_ball(rng, radius=float(args.x0_radius))
        else:
            z0 = sample_init_state_ball_fallback(rng, D=args.D, n=args.n, radius=float(args.x0_radius))
        print(f"[init] z0 sampled from L2-ball: radius={args.x0_radius} x0_seed={args.x0_seed} ||z0||={float(np.linalg.norm(z0)):.6f}")
    else:
        z0 = np.asarray(system.init_state(), dtype=float)
        print(f"[init] z0 = system.init_state() ||z0||={float(np.linalg.norm(z0)):.6f}")

    if args.search_mode == "gurobi":
        # Lazy import so the rest of the code runs without gurobipy installed.
        from alg.solver_baseline_gurobi import (
            SolverBaselineParams,
            solve_worst_sequence_miqcp_gurobi,
        )

        gp = SolverBaselineParams(
            time_limit=float(args.gurobi_time_limit),
            mip_gap=float(args.gurobi_gap),
            output_flag=int(args.gurobi_output),
            bound_scale=float(args.gurobi_bound_scale),
            basis=str(args.gurobi_basis),
            bound_mode=str(args.gurobi_bound_mode),
            m_encoding=str(args.gurobi_m_encoding),
        )
        res = solve_worst_sequence_miqcp_gurobi(
            system=system,
            action_space=action_space,
            z0=z0,
            T=int(args.T),
            params=gp,
        )

        print("=== gurobi baseline result ===")
        keys = [
            "basis", "bound_mode", "m_encoding",
            "status", "sol_count",
            "obj", "mip_gap", "best_bound",
            "runtime", "nodecount", "simplex_iters", "work",
            "M_global", "rho_max",
            "lambda_min", "lambda_max", "cond_H",
            "num_vars", "num_constrs", "num_qconstrs", "num_bin_vars", "num_int_vars",
        ]
        for k in keys:
            print(f"{k:12s}: {res.get(k)}")

        m_hist = res.get("m_hist", None)
        if m_hist is None:
            print("[gurobi] no solution found")
            return

        # Print sequence (convert histogram->delay vector)
        print(f"=== Best sequence (length T={m_hist.shape[0]}) ===")
        print("Format: t | delay-vector d_t | sum(d_t) | histogram m_t")
        for t in range(m_hist.shape[0]):
            m = np.asarray(m_hist[t], dtype=int).reshape(-1)
            d = np.asarray(action_space.histogram_to_delay(m), dtype=int).reshape(-1)
            s = int(d.sum())
            ok = ("OK" if (np.all((0 <= d) & (d <= args.D)) and s <= args.B) else "INVALID")
            print(f"{t:3d} | {d.tolist()} | sum={s:2d}/{args.B:2d} | m={m.tolist()} | {ok}")
        return

    
    # Build params
    bp = BoundParams(
        relax_vertices=bool(args.bound_relax_vertices),
        cache_rho=not bool(args.no_cache_rho),
        margin=float(args.bound_margin),
        lookahead_steps=0,
        lookahead_use_exact_actions=True,
    )

    mode = args.search_mode
    static_policy = "random"
    static_seed = args.policy_seed if args.policy_seed >= 0 else (args.seed * 1000 + args.x0_seed)

    if mode == "beam":
        mode2 = "beam"
    elif mode.startswith("static_"):
        mode2 = "static"
        if mode == "static_random":
            static_policy = "random"
        elif mode == "static_single_max":
            static_policy = "single_max"
        elif mode == "static_average":
            static_policy = "average"
        else:
            raise ValueError("unknown static mode")
    else:
        mode2 = mode  # exact/cover_check/heuristic/gurobi

    sp = SearchParams(
        mode=mode2,
        T=args.T,
        seed=args.seed,
        beam_width=int(args.beam),
        candidates_per_expand=int(args.cand),
        cover_beam_width=int(args.cover_beam),
        cover_candidates_per_expand=int(args.cover_cand),
        enable_pruning=not bool(args.no_pruning),
        prune_eps=float(args.prune_eps),
        bound=bp,
        static_policy=static_policy,
        static_seed=int(static_seed),
        dump_word_dir = args.dump_word_dir
    )


    out = run_bound_search(system, remodeling, action_space, z0=z0, params=sp)

    print("=== bound_search result ===")
    print("mode:", out.get("mode"))
    if out.get("mode") == "cover_check":
        print("UB_all:", out.get("UB_all"))
        print("LB_sub:", out.get("LB_sub"))
        print("gap   :", out.get("gap"))
        print("cover_pass:", out.get("cover_pass"))
        print("num_actions_full:", out.get("num_actions_full"),
              "num_actions_sub:", out.get("num_actions_sub"))
        print("runtime_sec:", out.get("runtime_sec"))
        print("runtime_sub_sec:", out.get("runtime_sub_sec"))
        # optional: print the subclass best sequence
        sub = out.get("sub_details", {}) or {}
        if sub.get("best_word", None):
            print("=== subclass (integer_vertices) best sequence ===")
            print_best_sequence_from_bound_search(
                sub, action_space=action_space, K=args.K, D=args.D, B=args.B
            )
        return

    # exact / heuristic
    print_pruning_stats(out)
    print_best_sequence_from_bound_search(out, action_space, K=args.K, D=args.D, B=args.B)

    dump_path = None
    if args.dump_word_dir:
        os.makedirs(args.dump_word_dir, exist_ok=True)

        bw = out.get("best_word", None)
        if bw is None:
            print("[dump] best_word is None, cannot dump.")
        else:
            # make a deterministic filename for THIS run
            prefix = (
                f"{args.search_mode}"
                f"_n{args.n}_K{args.K}_D{args.D}_B{args.B}"
                f"_T{args.T}_eta{args.eta}_cond{args.cond}"
                f"_seed{args.seed}_x0{args.x0_seed}"
            )
            dump_path = os.path.join(args.dump_word_dir, f"{prefix}_best_word.json")

            payload = {"hist_sequence": [np.asarray(m, dtype=int).tolist() for m in bw]}
            with open(dump_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[dump] saved schedule to {dump_path}")

    if args.verify_fixed_with_gurobi:
        if dump_path is None:
            print("[verify] need --dump_word_dir to verify.")
        else:
            # Try gurobi-based verifier first (if available), fallback to no-gurobi.
            try:
                from alg.solver_baseline_gurobi import SolverBaselineParams, verify_fixed_schedule_gurobi
                bp = SolverBaselineParams(
                    basis=str(args.gurobi_basis),
                    bound_mode=str(args.gurobi_bound_mode),
                    bound_scale=float(args.gurobi_bound_scale),
                    m_encoding=str(args.gurobi_m_encoding),
                )
                JT_model, JT_rollout, diff = verify_fixed_schedule_gurobi(
                    system,
                    json_path=dump_path,
                    z0=z0,
                    basis=str(args.gurobi_basis),
                    action_space=action_space,
                    bounds_params=bp,
                    check_miqcp_bounds=True,
                )
                print("[verify_gurobi] done")
                print(f"  JT_model   = {JT_model}")
                print(f"  JT_rollout = {JT_rollout}")
                print(f"  diff       = {diff}")
            except Exception as e:
                print(f"[verify_gurobi] failed: {type(e).__name__}: {e}")
                print("[verify_gurobi] falling back to no-gurobi cross-check...")
                verify_fixed_schedule_no_gurobi(
                    system=system,
                    remodeling=remodeling,
                    action_space=action_space,
                    z0=z0,
                    json_path=dump_path,
                    relax_vertices=bool(args.bound_relax_vertices),
                )    



if __name__ == "__main__":
    main()

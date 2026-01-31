#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated experiment runner for ASGD worst-case schedule falsification.

- Runs python -m test_bound_search with a showcase grid of (n,K,D,B,T,eta,cond,seed) and 3 x0 seeds.
- Collects results into CSV: obj, status, gap, runtime, node count, etc.
- Stores full stdout/stderr logs per run under logs/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


# -------------------------
# Parsing utilities
# -------------------------

_RE_KV_LINE = re.compile(r"^\s*([A-Za-z0-9_\-\(\)\/]+)\s*:\s*(.+?)\s*$")
_RE_GUROBI_EXPLORED = re.compile(r"Explored\s+(\d+)\s+nodes\s+\((\d+)\s+simplex iterations\)")
_RE_GUROBI_BESTBOUND = re.compile(
    r"Best objective\s+([-\d.eE+]+)\s*,\s*best bound\s+([-\d.eE+]+)\s*,\s*gap\s+([-\d.eE+]+)"
)

def _safe_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        return float(s)
    except Exception:
        return None

def _safe_int(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        return int(float(s))
    except Exception:
        return None

def _safe_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y")



def parse_test_bound_search_stdout(stdout: str) -> Dict[str, Any]:
    """
    Parse stdout from python -m test_bound_search.
    Works for both gurobi branch and exact/heuristic branches.

    Returns a dict with best-effort fields:
      status, obj, gap, runtime,
      nodecount, simplex_iters, best_bound (if present),
      plus any pruning stats key-values if printed.
    """
    out: Dict[str, Any] = {}

    # Parse the "=== gurobi baseline result ===" style block
    m = re.search(r"status\s*:\s*(\d+)", stdout)
    if m:
        out["status"] = int(m.group(1))

    m = re.search(r"obj\(JT\)\s*:\s*([-\d.eE+]+)", stdout)
    if m:
        out["obj"] = _safe_float(m.group(1))

    m = re.search(r"gap\s*:\s*([-\d.eE+]+)", stdout)
    if m:
        out["gap"] = _safe_float(m.group(1))

    m = re.search(r"runtime\s*:\s*([-\d.eE+]+)", stdout)
    if m:
        out["runtime"] = _safe_float(m.group(1))

    # Parse Gurobi "Explored X nodes (Y simplex iterations)"
    m = _RE_GUROBI_EXPLORED.search(stdout)
    if m:
        out["nodecount"] = _safe_int(m.group(1))
        out["simplex_iters"] = _safe_int(m.group(2))

    # Parse Gurobi "Best objective, best bound, gap" line (for TIME_LIMIT cases too)
    m = _RE_GUROBI_BESTBOUND.search(stdout)
    if m:
        out["best_objective_line_obj"] = _safe_float(m.group(1))
        out["best_bound"] = _safe_float(m.group(2))
        out["best_objective_line_gap"] = _safe_float(m.group(3))

    # Parse pruning stats key: value lines under "=== Pruning / search stats ==="
    # We just capture numeric-ish values for known keys.
    for line in stdout.splitlines():
        mm = _RE_KV_LINE.match(line)
        if not mm:
            continue
        k = mm.group(1).strip()
        v = mm.group(2).strip()
        # Try numeric
        fv = _safe_float(v)
        if fv is not None:
            out[k] = fv
        else:
            out[k] = v

    # ---- Exact / bound_search parsing (your output format) ----
    # Example:
    # === bound_search result ===
    # mode: exact
    # ...
    # best_JT: 6.677894522456574
    # ... complete: True early_stop: False
    m = re.search(r"best_JT:\s*([-\d.eE+]+)", stdout)
    if m and out.get("obj") is None:
        out["obj"] = _safe_float(m.group(1))

    m = re.search(r"complete:\s*(True|False)", stdout)
    if m:
        out["complete"] = (m.group(1) == "True")
        # Map to status-like semantics for CSV convenience:
        # complete True -> treat as OPTIMAL (2), else TIME_LIMIT-like (9)
        if out.get("status") is None:
            out["status"] = 2 if out["complete"] else 9
        if out.get("gap") is None and out["complete"]:
            out["gap"] = 0.0

    # Prefer exact runtime_sec_total if present
    m = re.search(r"runtime_sec_total\s*:\s*([-\d.eE+]+)", stdout)
    if m and out.get("runtime") is None:
        out["runtime"] = _safe_float(m.group(1))

    # Backward/forward compatibility: gurobi prints mip_gap in new format
    if "gap" not in out and "mip_gap" in out:
        out["gap"] = out["mip_gap"]


    return out


# -------------------------
# CLI detection
# -------------------------

def get_help_text() -> str:
    cmd = [sys.executable, "-m", "test_bound_search", "--help"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception:
        return ""


def filter_supported_args(args: List[str], help_text: str) -> List[str]:
    """
    Remove CLI flags not present in help text to avoid unknown-arg crashes.
    """
    if not help_text:
        # If we can't detect help, do not filter (fail fast).
        return args

    supported = set()
    # crude but robust: collect tokens like "--foo"
    for tok in re.findall(r"(--[A-Za-z0-9_\-]+)", help_text):
        supported.add(tok)

    filtered: List[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if a not in supported:
                # drop flag and its value if it looks like flag takes a value
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    i += 2
                else:
                    i += 1
                continue
        filtered.append(a)
        i += 1
    return filtered


# -------------------------
# Experiment grid
# -------------------------

@dataclass(frozen=True)
class Case:
    n: int
    K: int
    D: int
    B: int
    T: int
    eta: float
    cond: float
    H_seed: int  # maps to --seed in test_bound_search (also used by system), consistent with your current file


@dataclass(frozen=True)
class Method:
    name: str
    search_mode: str  # "gurobi" or "exact"
    extra_args: Tuple[str, ...]  # additional CLI args


def make_showcase_grid() -> List[Case]:
    """
    '展示性最好' grid:
      - Small-T region for exact vs solver comparison
      - Larger-T + scaling only for solver
    """
    grid: List[Case] = []

    # Base configuration (your current best-known setting)
    base = dict(n=4, K=4, D=3, B=6, eta=1e-4, cond=10.0, H_seed=0)

    # (A) Small-T exact comparison (main-table friendly)
    for T in [4, 6]:
        grid.append(Case(**base, T=T))

    # (B) T scaling under solver (show scalability; exact will timeout so we don't run it here)
    for T in [10, 20, 40]:
        grid.append(Case(**base, T=T))

    # (C) n scaling (fix T=20; keep K,D,B)
    for n in [4, 8, 16]:
        grid.append(Case(n=n, K=4, D=3, B=6, T=20, eta=1e-4, cond=10.0, H_seed=0))

    # (D) K/D scaling (choose budget B=K*D/2, integer)
    kd_list = [(4, 3), (8, 3), (8, 5)]
    for (K, D) in kd_list:
        B = int((K * D) / 2)
        grid.append(Case(n=8, K=K, D=D, B=B, T=20, eta=1e-4, cond=10.0, H_seed=0))

    # (E) eta sweep (optional robustness)
    for eta in [5e-5, 1e-4, 2e-4]:
        grid.append(Case(n=8, K=4, D=3, B=6, T=20, eta=eta, cond=10.0, H_seed=0))

    # (F) objective conditioning sweep
    for cond in [1.0, 10.0, 100.0]:
        grid.append(Case(n=8, K=4, D=3, B=6, T=20, eta=1e-4, cond=cond, H_seed=0))

    # De-duplicate while preserving order
    seen = set()
    uniq: List[Case] = []
    for c in grid:
        key = (c.n, c.K, c.D, c.B, c.T, c.eta, c.cond, c.H_seed)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def make_methods(include_onehot: bool = True) -> List[Method]:
    """
    Methods for ablation:
      - solver baseline: original + hist_miqcp
      - +spectral: spectral + hist_miqcp
      - +onehot_bigM: spectral + onehot_bigm (symbreak is baked in onehot in your codebase)
      - exact: schedule-space exact search
    """
    methods: List[Method] = []

    # Gurobi ref (no spectral)
    methods.append(Method(
        name="gurobi_ref_original_hist",
        search_mode="gurobi",
        extra_args=(
            "--gurobi_basis", "original",
            "--gurobi_bound_mode", "global",
            "--gurobi_m_encoding", "hist_miqcp",
        )
    ))

    # + spectral
    methods.append(Method(
        name="gurobi_spectral_hist",
        search_mode="gurobi",
        extra_args=(
            "--gurobi_basis", "spectral",
            "--gurobi_bound_mode", "global",
            "--gurobi_m_encoding", "hist_miqcp",
        )
    ))

    if include_onehot:
        # + onehot_bigM (symbreak is implicit/baked-in)
        methods.append(Method(
            name="gurobi_spectral_onehot_bigm",
            search_mode="gurobi",
            extra_args=(
                "--gurobi_basis", "spectral",
                "--gurobi_bound_mode", "global",
                "--gurobi_m_encoding", "onehot_bigm",
            )
        ))

    # trivial exact baseline
    methods.append(Method(
        name="exact_tree_search",
        search_mode="exact",
        extra_args=tuple()
    ))

    # --- static heuristic baselines ---
    methods.append(Method(
        name="static_random",
        search_mode="static_random",
        extra_args=tuple(),
    ))
    methods.append(Method(
        name="static_single_max",
        search_mode="static_single_max",
        extra_args=tuple(),
    ))
    methods.append(Method(
        name="static_average",
        search_mode="static_average",
        extra_args=tuple(),
    ))

    # --- beam heuristic baseline (same implementation as bound_search heuristic, but clearer label) ---
    methods.append(Method(
        name="beam_search",
        search_mode="beam",
        extra_args=tuple(),
    ))


    return methods



# -------------------------
# Runner
# -------------------------

def run_one(cmd: List[str], timeout_sec: int) -> Tuple[int, str, str, float]:
    """
    Run a command with timeout. Returns (returncode, stdout, stderr, wall_time_sec).
    """
    t0 = dt.datetime.now()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
        t1 = dt.datetime.now()
        return p.returncode, (p.stdout or ""), (p.stderr or ""), (t1 - t0).total_seconds()
    except subprocess.TimeoutExpired as e:
        t1 = dt.datetime.now()
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + "\n[TIMEOUT]"
        return 124, stdout, stderr, (t1 - t0).total_seconds()

def _case_key_from_row(r):
    return (
        _safe_int(r.get("n")),
        _safe_int(r.get("K")),
        _safe_int(r.get("D")),
        _safe_int(r.get("B")),
        _safe_int(r.get("T")),
        _safe_float(r.get("eta")),
        _safe_float(r.get("cond")),
        _safe_int(r.get("H_seed")),
        _safe_int(r.get("x0_seed")),
        _safe_bool(r.get("use_ball_init")),
        _safe_float(r.get("x0_radius")),
        _safe_bool(r.get("bound_relax_vertices")),
    )

def _method_key_from_row(r):
    return (r.get("method"),) + _case_key_from_row(r)


def load_existing_results(path: str):
    done_keys = set()
    ub_map = {}   # case_key -> tightest upper bound (min best_bound)

    if not os.path.exists(path):
        return done_keys, ub_map

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            mk = _method_key_from_row(r)
            done_keys.add(mk)

            # Identify gurobi rows robustly:
            # - In your CSV, "search_mode" should be "gurobi" for all gurobi runs
            # - Some people also encode method name like "gurobi_*"
            sm = str(r.get("search_mode") or "").strip().lower()
            method = str(r.get("method") or "").strip().lower()
            is_gurobi = (sm == "gurobi") or method.startswith("gurobi")

            if not is_gurobi:
                continue

            case_key = _case_key_from_row(r)

            bb = _safe_float(r.get("best_bound"))
            if bb is None:
                # fallback: if best_bound missing but we have obj and mip_gap, approximate UB
                obj = _safe_float(r.get("obj"))
                mg = _safe_float(r.get("mip_gap"))  # or r.get("gap") if you store that
                if obj is not None and mg is not None:
                    # This is an approximation; prefer real best_bound when available
                    bb = obj * (1.0 + mg)

            if bb is None:
                continue

            prev = ub_map.get(case_key, None)
            # for maximization, smaller UB is tighter
            if prev is None or bb < prev:
                ub_map[case_key] = bb

    return done_keys, ub_map


def main():
    ap = argparse.ArgumentParser("Run experiment grid and collect CSV results")
    ap.add_argument("--out_csv", type=str, default="results_grid.csv")
    ap.add_argument("--logs_dir", type=str, default="logs")
    ap.add_argument("--timeout_gurobi", type=int, default=900, help="wall-clock timeout per gurobi run (sec)")
    ap.add_argument("--timeout_exact", type=int, default=900, help="wall-clock timeout per exact run (sec)")
    ap.add_argument("--x0_seeds", type=str, default="0,1,2", help="comma-separated x0 seeds")
    ap.add_argument("--use_ball_init", action="store_true", default=True)
    ap.add_argument("--x0_radius", type=float, default=1.0)
    ap.add_argument("--bound_relax_vertices", action="store_true", default=True)
    ap.add_argument("--include_onehot", action="store_true", default=True)
    ap.add_argument("--only_small_exact", action="store_true", default=True,
                    help="run exact only on small-T cases (T<=6) to keep grid manageable")
    ap.add_argument("--conda_env", type=str, default="", help="run commands under `conda run -n <env>`")


    args = ap.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    x0_seeds = [int(s.strip()) for s in args.x0_seeds.split(",") if s.strip() != ""]

    help_text = get_help_text()

    grid = make_showcase_grid()
    methods = make_methods(include_onehot=bool(args.include_onehot))

    done_keys, ub_map = load_existing_results(args.out_csv)

    # Pre-compute total number of jobs for progress display
    total_jobs = 0
    done_jobs = 0
    for c in grid:
        for x0_seed in x0_seeds:
            for mm in methods:
                if mm.search_mode == "exact" and args.only_small_exact and c.T > 6:
                    continue
                key = (
                    mm.name,
                    c.n, c.K, c.D, c.B, c.T, float(c.eta), float(c.cond),
                    c.H_seed, x0_seed,
                    bool(args.use_ball_init), float(args.x0_radius), bool(args.bound_relax_vertices),
                )
                if key in done_keys:
                    done_jobs = done_jobs + 1
                total_jobs += 1

    # CSV columns
    fieldnames = [
        "timestamp",
        "case_id",
        "method",
        "search_mode",
        "n","K","D","B","T","eta","cond","H_seed",
        "x0_seed","use_ball_init","x0_radius","bound_relax_vertices",
        "returncode",
        "status","is_optimal",
        "obj","gap","runtime_reported",
        "nodecount","simplex_iters","best_bound",
        "mip_gap",          # 
        "sol_count",        # 
        "work",             # 
        "M_global","rho_max",# 
        "lambda_min","lambda_max","cond_H",  # 
        "num_vars","num_constrs","num_qconstrs","num_bin_vars","num_int_vars",  #
        "visited_nodes","generated_nodes","UB_all",
        "wall_time_sec",
        "cmd",
        "log_path",
    ]

    file_exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a" if file_exists else "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        case_id = 0
        for c in grid:
            case_id += 1
            for x0_seed in x0_seeds:
                for m in methods:
                    # Optionally skip exact beyond small T
                    if m.search_mode == "exact" and args.only_small_exact and c.T > 6:
                        continue
                    
                    key = (
                        m.name,
                        c.n, c.K, c.D, c.B, c.T, float(c.eta), float(c.cond),
                        c.H_seed, x0_seed,
                        bool(args.use_ball_init), float(args.x0_radius), bool(args.bound_relax_vertices),
                    )
                    if key in done_keys:
                        # done_jobs += 1
                        # pct = 100.0 * done_jobs / max(1, total_jobs + len(done_keys))
                        # pct = 100.0 * done_jobs / max(1, total_jobs)
                        # print(f"[SKIP] ({pct:6.2f}%) already done: {m.name} n={c.n} ... x0={x0_seed}")
                        continue

                    # prefix = []
                    # if args.conda_env.strip():
                    #     prefix = ["conda", "run", "-n", args.conda_env.strip()]

                    base_cmd = [
                        sys.executable, "-m", "test_bound_search",
                        "--search_mode", m.search_mode,
                        "--n", str(c.n),
                        "--K", str(c.K),
                        "--D", str(c.D),
                        "--B", str(c.B),
                        "--eta", str(c.eta),
                        "--T", str(c.T),
                        "--cond", str(c.cond),
                        "--seed", str(c.H_seed),
                        "--x0_seed", str(x0_seed),
                        "--x0_radius", str(args.x0_radius),
                    ]
                    if args.use_ball_init:
                        base_cmd.append("--use_ball_init")
                    if args.bound_relax_vertices:
                        base_cmd.append("--bound_relax_vertices")
                    # Ensure internal Gurobi TimeLimit matches the outer timeout budget
                    if m.search_mode == "gurobi":
                        base_cmd += ["--gurobi_time_limit", str(args.timeout_gurobi)]

                    full_cmd = base_cmd + list(m.extra_args)
                    full_cmd = filter_supported_args(full_cmd, help_text)

                    if m.search_mode == "gurobi":
                        timeout_sec = int(args.timeout_gurobi) + 120  # for logging
                    else:
                        timeout_sec = args.timeout_exact


                    # Run
                    rc, stdout, stderr, wall = run_one(full_cmd, timeout_sec=timeout_sec)

                    # Save logs
                    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_name = f"case{case_id:03d}_{m.name}_n{c.n}_K{c.K}_D{c.D}_B{c.B}_T{c.T}_eta{c.eta}_cond{c.cond}_Hseed{c.H_seed}_x0seed{x0_seed}_{stamp}.log"
                    log_path = os.path.join(args.logs_dir, log_name)
                    with open(log_path, "w", encoding="utf-8") as flog:
                        flog.write("CMD:\n")
                        flog.write(" ".join(full_cmd) + "\n\n")
                        flog.write("STDOUT:\n")
                        flog.write(stdout + "\n\n")
                        flog.write("STDERR:\n")
                        flog.write(stderr + "\n")

                    # Parse
                    parsed = parse_test_bound_search_stdout(stdout + "\n" + stderr)

                    status = parsed.get("status", None)
                    is_opt = (status == 2)

                    row = {
                        "timestamp": stamp,
                        "case_id": case_id,
                        "method": m.name,
                        "search_mode": m.search_mode,
                        "n": c.n, "K": c.K, "D": c.D, "B": c.B, "T": c.T,
                        "eta": c.eta, "cond": c.cond, "H_seed": c.H_seed,
                        "x0_seed": x0_seed,
                        "use_ball_init": bool(args.use_ball_init),
                        "x0_radius": float(args.x0_radius),
                        "bound_relax_vertices": bool(args.bound_relax_vertices),
                        "returncode": rc,
                        "status": status,
                        "is_optimal": is_opt,
                        "obj": parsed.get("obj", None),
                        "gap": parsed.get("gap", None),
                        "runtime_reported": parsed.get("runtime", None),
                        "nodecount": parsed.get("nodecount", None),
                        "simplex_iters": parsed.get("simplex_iters", None),
                        "best_bound": parsed.get("best_bound", None),
                        "mip_gap": parsed.get("mip_gap", None),
                        "sol_count": parsed.get("sol_count", None),
                        "work": parsed.get("work", None),
                        "M_global": parsed.get("M_global", None),
                        "rho_max": parsed.get("rho_max", None),
                        "lambda_min": parsed.get("lambda_min", None),
                        "lambda_max": parsed.get("lambda_max", None),
                        "cond_H": parsed.get("cond_H", None),
                        "num_vars": parsed.get("num_vars", None),
                        "num_constrs": parsed.get("num_constrs", None),
                        "num_qconstrs": parsed.get("num_qconstrs", None),
                        "num_bin_vars": parsed.get("num_bin_vars", None),
                        "num_int_vars": parsed.get("num_int_vars", None),
                        "visited_nodes": parsed.get("visited_nodes", None),
                        "generated_nodes": parsed.get("generated_nodes", None),
                        "UB_all": parsed.get("UB_all", None),
                        "wall_time_sec": wall,
                        "cmd": " ".join(full_cmd),
                        "log_path": log_path,
                    }

                    case_key = (
                        c.n, c.K, c.D, c.B, c.T, float(c.eta), float(c.cond),
                        c.H_seed, x0_seed,
                        bool(args.use_ball_init), float(args.x0_radius),
                        bool(args.bound_relax_vertices),
                    )

                    if m.search_mode in ("static_random", "static_single_max", "static_average", "beam"):
                        obj_h = _safe_float(row.get("obj"))
                        ub = ub_map.get(case_key, None)
                        if obj_h is not None and ub is not None:
                            row["best_bound"] = ub
                            row["gap"] = float(ub) - float(obj_h)
                        else:
                            # couldn't compute gap because gurobi UB missing
                            row["gap"] = None

                    if m.search_mode == "gurobi":
                        bb = _safe_float(parsed.get("best_bound"))
                        if bb is not None:
                            prev = ub_map.get(case_key, None)
                            if prev is None or bb < prev:
                                ub_map[case_key] = bb

                    writer.writerow(row)
                    fcsv.flush()

                    # Minimal progress print
                    # print(f"[{case_id:03d}] {m.name} n={c.n} K={c.K} D={c.D} B={c.B} T={c.T} eta={c.eta} cond={c.cond} x0={x0_seed} -> status={status}, obj={row['obj']}, gap={row['gap']}, wall={wall:.2f}s")
                    done_jobs += 1
                    pct = 100.0 * done_jobs / max(1, total_jobs)
                    rem = total_jobs - done_jobs

                    print(
                        f"[{done_jobs}/{total_jobs} | {pct:6.2f}% | rem={rem}] "
                        f"(case {case_id:03d}) {m.name} "
                        f"n={c.n} K={c.K} D={c.D} B={c.B} T={c.T} "
                        f"eta={c.eta} cond={c.cond} x0={x0_seed} "
                        f"-> rc={rc}, status={status}, obj={row['obj']}, gap={row['gap']}, wall={wall:.2f}s"
)


    print(f"\nDone. CSV saved to: {args.out_csv}")
    print(f"Logs saved under:   {args.logs_dir}/")


if __name__ == "__main__":
    main()

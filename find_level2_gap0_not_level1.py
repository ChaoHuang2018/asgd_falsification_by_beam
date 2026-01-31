#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find instances such that:
  - Level 1 (prior concavity endpoints) FAILS
  - Level 2 (FULL C' buckets, no S_k restriction, per-node expand-all, leakage UB enabled) achieves gap==0

Run:
  python find_level2_gap0_not_level1.py --max_trials 400 --seed 0

It will print two found examples in the requested template format and also save them to:
  found_level2_gap0.py
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

# ---- project imports (must run inside your repo) ----
import alg.bound_search as bs
import alg.beam_certificate as cert
import alg.beam_certificate_subspace as cert_sub
from core.action_space import ActionSpace
from core.remodeling import QuadraticRemodeling

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


@dataclass
class _FallbackQuadraticSystem:
    """Minimal system wrapper used by bound_search to access H."""
    H: np.ndarray


def _gap_from_res(res: Dict[str, Any]) -> Optional[float]:
    lb = res.get("best_JT", None)
    if lb is None:
        return None
    if res.get("leakage_ub_global_max", None) is not None:
        return float(res["leakage_ub_global_max"]) - float(lb)
    if res.get("UB_all", None) is not None:
        return float(res["UB_all"]) - float(lb)
    return None


def _build_ubm(H: np.ndarray, K: int, D: int, B: int, eta: float, T: int):
    system = _FallbackQuadraticSystem(H=H)
    action_space = ActionSpace(K=K, D=D, B=B)
    remodeling = QuadraticRemodeling(H=H, eta=eta, K=K, D=D)

    bp = bs.BoundParams(
        relax_vertices=True,
        cache_rho=False,
        margin=1e-12,
        lookahead_steps=0,
        lookahead_use_exact_actions=True,
    )
    ubm = bs.UpperBoundModel(system=system, remodeling=remodeling, action_space=action_space, T_max=T, params=bp)
    return system, action_space, remodeling, ubm


def _run_level2_full_cprime(
    system,
    action_space,
    remodeling,
    ubm: bs.UpperBoundModel,
    z0: np.ndarray,
    T: int,
    beam_width: int,
    candidates_per_expand: int,
    time_limit_sec: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Level 2: FULL C' buckets, no S_k restriction, per-node expand-all, leakage UB enabled.
    Returns (res, meta) where meta includes bucket stats.
    """
    depth = int(T) - 1

    # Build reachable basis U using subspace cert module
    rep2 = cert_sub.verify_prior_subspace_candidate_certificate(
        ubm=ubm,
        action_space=action_space,
        z0=z0,
        T=int(T),
        depth=depth,
        tol_new_vec=1e-9,
        psd_tol=1e-10,
    )
    U = None
    try:
        U = rep2.diagnostics.get("U", None)
    except Exception:
        U = None
    if U is None:
        return {"ok": False, "error": "missing_U"}, {"num_buckets": None}

    cprime = cert_sub.build_cprime_buckets_by_projected_operator(
        ubm=ubm,
        action_space=action_space,
        U=np.asarray(U, dtype=float),
        delta=1e-6,
        max_buckets=400,
        restrict_to_action_indices=None,  # FULL scope
    )
    bucket_id_by_action = cprime["bucket_id_by_action"]
    num_buckets = len(set(int(v) for v in bucket_id_by_action.values() if int(v) >= 0))

    # Run heuristic beam with FULL C' truncation + expand-all
    bp = bs.BoundParams(
        relax_vertices=True,
        cache_rho=False,
        margin=1e-12,
        lookahead_steps=0,             
        lookahead_use_exact_actions=True,
    )
    sp = bs.SearchParams(
        mode="heuristic",
        T=int(T),
        seed=0,
        beam_width=int(beam_width),
        candidates_per_expand=int(candidates_per_expand),
        enable_pruning=True,
        prune_eps=0.0,
        bound=bp,  #
    )
    sp.time_limit_sec = (float(time_limit_sec) if time_limit_sec is not None else None)

    sp.enable_bucket_coverage = True
    sp.bucket_coverage_mode = "cprime"
    sp.bucket_actions = None
    sp.bucket_coverage_expand_all = True  # critical for sound leakage UB in strong mode

    sp.enable_certified_action_set = False
    sp.certified_action_indices_by_k = None

    sp.cprime_bucket_id_by_action = bucket_id_by_action
    sp.cprime_required_bucket_ids_by_k = None

    sp.enable_leakage_ub = True

    res = bs.run_bound_search(
        system=system,
        remodeling=remodeling,
        action_space=action_space,
        z0=z0,
        params=sp,
    )
    res = dict(res)
    res["ok"] = True
    return res, {"num_buckets": num_buckets, "U_dim": (np.asarray(U).shape[1] if np.asarray(U).ndim == 2 else None)}


def _format_example_fn(name: str, inst: Dict[str, Any]) -> str:
    H = inst["H"]
    z0 = inst["z0"]
    K, D, B = inst["K"], inst["D"], inst["B"]
    eta, T = inst["eta"], inst["T"]
    bw, cpe = inst["beam_width"], inst["candidates_per_expand"]

    lines: List[str] = []
    lines.append(f"def {name}() -> Dict[str, Any]:")
    lines.append(f"    K, D, B = {K}, {D}, {B}")
    lines.append(f"    eta = {eta:.6g}")
    lines.append(f"    T = {T}")
    lines.append("")
    # H
    if np.allclose(H, np.diag(np.diag(H))):
        diag = np.diag(H)
        lines.append(f"    H = np.diag({diag.tolist()})")
    else:
        lines.append("    H = np.array(" + np.array2string(H, separator=", ") + ", dtype=float)")
    lines.append("")
    n = H.shape[0]
    lines.append(f"    n = H.shape[0]")
    lines.append(f"    z0 = np.zeros((D + 1, n), dtype=float)")
    for i in range(z0.shape[0]):
        lines.append(f"    z0[{i}] = np.array({z0[i].tolist()})")
    lines.append("")
    lines.append(f"    beam_width = {bw}")
    lines.append(f"    candidates_per_expand = {cpe}")
    lines.append("")
    lines.append("    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0,")
    lines.append("                beam_width=beam_width, candidates_per_expand=candidates_per_expand)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_trials", type=int, default=500)
    ap.add_argument("--need", type=int, default=2, help="How many instances to find")
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--time_limit_sec", type=float, default=20.0, help="Per-trial Level-2 time limit")
    ap.add_argument("--gap_eps", type=float, default=1e-12)
    ap.add_argument("--obj_cap", type=float, default=1e8, help="Reject if |best_JT| exceeds this cap")
    ap.add_argument("--out", type=str, default="found_level2_gap0.py")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    found: List[Dict[str, Any]] = []
    t_start = time.time()

    for trial in range(1, args.max_trials + 1):
        # ---- sample small-to-moderate problem sizes (avoid blow-up) ----
        eta = float(args.eta)
        T = int(args.T)

        n = int(rng.integers(2, 6))     # n in [2..5]
        K = int(rng.integers(3, 8))     # K in [3..7]
        D = int(rng.integers(2, 5))     # D in [2..4]

        # B in a "middle" range to avoid near-1D action sets (which often makes Level-1 pass)
        B_max = K * D
        B_low = max(1, int(0.30 * B_max))
        B_high = max(B_low + 1, int(0.70 * B_max))
        B = int(rng.integers(B_low, B_high + 1))

        # H diagonal with moderate spectrum (avoid objective blow-up)
        diag = rng.uniform(0.3, 3.0, size=(n,))
        diag.sort()
        H = np.diag(diag)

        # z0 with decaying history to avoid instability
        z0 = np.zeros((D + 1, n), dtype=float)
        base = rng.normal(0.0, 1.0, size=(n,))
        base = base / (np.linalg.norm(base) + 1e-12)
        amp0 = rng.uniform(0.15, 0.45)
        for t in range(D + 1):
            amp = amp0 * (0.65 ** t)
            z0[t] = amp * (base + 0.25 * rng.normal(0.0, 1.0, size=(n,)))

        # Beam settings for Level 2: give enough capacity but keep bounded
        beam_width = int(rng.integers(16, 65))  # [16..64]
        candidates_per_expand = 10**9           # irrelevant under expand-all, keep huge

        # ---- build ubm/system ----
        system, action_space, remodeling, ubm = _build_ubm(H=H, K=K, D=D, B=B, eta=eta, T=T)

        # ---- Level 1: prior certificate ----
        rep = cert.verify_prior_bellman_sufficient_condition(
            action_space=action_space,
            lambdas=ubm.lambdas,
            G=ubm.G,
            D=ubm.D,
            K=ubm.K,
            B=action_space.B,
            eta=ubm.eta,
            T=int(T),
        )
        if bool(rep.get("overall_pass", False)):
            continue  # reject: Level 1 passed

        # ---- Level 2: FULL C' buckets, expand-all ----
        res2, meta = _run_level2_full_cprime(
            system, action_space, remodeling, ubm, z0,
            T=T, beam_width=beam_width, candidates_per_expand=candidates_per_expand,
            time_limit_sec=float(args.time_limit_sec) if args.time_limit_sec is not None else None,
        )
        if not bool(res2.get("ok", False)):
            continue

        g = _gap_from_res(res2)
        if g is None or g > float(args.gap_eps):
            continue

        best_JT = float(res2.get("best_JT", 0.0))
        if abs(best_JT) > float(args.obj_cap):
            continue

        inst = dict(
            H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0,
            beam_width=beam_width, candidates_per_expand=candidates_per_expand,
            level1_overall_pass=False,
            level2_gap=float(g),
            level2_best_JT=float(best_JT),
            level2_num_buckets=meta.get("num_buckets", None),
            level2_U_dim=meta.get("U_dim", None),
        )
        found.append(inst)

        print(f"[FOUND {len(found)}/{args.need}] trial={trial} "
              f"K={K},D={D},B={B},n={n}, bw={beam_width}, gap={g:.3e}, JT={best_JT:.3e}, "
              f"buckets={inst['level2_num_buckets']}, Udim={inst['level2_U_dim']}")

        if len(found) >= int(args.need):
            break

    print(f"\nDone. tried={trial}, found={len(found)}, elapsed={time.time()-t_start:.2f}s")
    if not found:
        print("No instance found. Try increasing --max_trials or --time_limit_sec, "
              "or increasing beam_width range in the script.")
        return

    # Emit two example functions
    out_lines: List[str] = []
    out_lines.append("from typing import Dict, Any")
    out_lines.append("import numpy as np\n")
    for i, inst in enumerate(found[: int(args.need)], start=1):
        out_lines.append(_format_example_fn("_example_lvl2_cprime_gap0_not_lvl1", inst))
        out_lines.append("\n")

    text = "\n".join(out_lines)
    print("\n" + text)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment runner for ASGD falsification verification (v8 main/beam logic).

What it does (per example, per T):
  1) Runs our verification framework in a 3-stage cascade:
        (A) prior concavity certificate (if passes, run "prior-restricted" beam + leakage UB)
        (B) strong certificate (FULL C' buckets, no S_k restriction; S1b must pass)
        (C) subspace certificate (S_k restriction; then either leak-UB gap==0 or (rare) strong-all-buckets)
     Stops at the first stage that certifies true optimality (gap==0, sound).

  2) Runs Gurobi twice:
        - gurobi-calculate: no warm start
        - gurobi-verify: warm start using our beam schedule
     Time limit: 600s.

  3) Runs two ASGD baselines:
        - no delay: always histogram [K,0,...,0]
        - random delay: per-step random histogram satisfying D,B (not necessarily using full budget)

  4) Saves all results to a CSV.

Assumptions:
  - You run this inside the project repo where these imports work:
      alg.bound_search as bs
      alg.beam_certificate as cert
      alg.beam_certificate_subspace as cert_sub
      core.action_space.ActionSpace, core.remodeling.QuadraticRemodeling
      examples.quadratic_asgd.QuadraticASGDSystem, QuadraticObjective
      alg.solver_baseline_gurobi (optional; if missing, gurobi fields are left blank)
  - Default experimental defaults match your current papersetting:
      lookahead_steps=0, lookahead_use_exact_actions=True, relax_vertices=True
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import multiprocessing as mp
import traceback


# ---- project imports (expected to exist in your repo) ----
import alg.bound_search as bs
import alg.beam_certificate as cert
import alg.beam_certificate_subspace as cert_sub
from core.action_space import ActionSpace
from core.remodeling import QuadraticRemodeling
from examples.quadratic_asgd import QuadraticASGDSystem, QuadraticObjective

# examples (for H/K/D/B/eta/z0 library)
import main_verify_beam

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ----------------------------
# Utilities
# ----------------------------

def _hard_timeout_worker(q, fn, args):
    """
    Must be at module top-level to be picklable under multiprocessing 'spawn'.
    """
    try:
        out = fn(*args)
        q.put({"ok": True, "out": out})
    except Exception as e:
        q.put({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })

def _now() -> float:
    return time.perf_counter()


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _run_with_hard_timeout(fn, args: tuple, timeout_sec: float) -> Dict[str, Any]:
    """
    Run fn(*args) in a separate process; if timeout, terminate and return timeout marker.
    Uses a top-level worker so it works under 'spawn'.
    """
    timeout_sec = float(timeout_sec)

    # Prefer fork on Linux (no pickling needed for most objects), but keep spawn as fallback.
    try:
        ctx = mp.get_context("fork")
    except Exception:
        ctx = mp.get_context("spawn")

    q = ctx.Queue()
    p = ctx.Process(target=_hard_timeout_worker, args=(q, fn, args))
    p.daemon = True
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join()
        return {"ok": False, "timeout": True, "timeout_sec": timeout_sec}

    if not q.empty():
        msg = q.get()
        if msg.get("ok", False):
            return {"ok": True, "out": msg["out"]}
        return {
            "ok": False,
            "timeout": False,
            "error": msg.get("error", "unknown error"),
            "traceback": msg.get("traceback", None),
        }

    return {"ok": False, "timeout": False, "error": "worker returned no message"}


def _enumerate_actions(action_space: ActionSpace) -> List[np.ndarray]:
    return [np.asarray(m, dtype=int).reshape(-1) for m in action_space.enumerate_histograms()]


def _find_action_index(actions: List[np.ndarray], target: np.ndarray) -> Optional[int]:
    tt = tuple(np.asarray(target, dtype=int).tolist())
    for i, m in enumerate(actions):
        if tuple(m.tolist()) == tt:
            return int(i)
    return None


def _sample_hist_random(K: int, D: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Per-step random histogram: sample delays sequentially under remaining budget."""
    rem = int(B)
    d = []
    for _ in range(K):
        hi = min(D, rem)
        di = int(rng.integers(0, hi + 1))
        d.append(di)
        rem -= di
    m = np.zeros(D + 1, dtype=int)
    for di in d:
        m[int(di)] += 1
    return m


def _rollout_obj(
    ubm: bs.UpperBoundModel,
    z0_modes: np.ndarray,
    T: int,
    m_policy_fn,
) -> float:
    z = np.asarray(z0_modes, dtype=float)
    JT = 0.0
    for t in range(T):
        JT += float(ubm.loss(z))
        m = np.asarray(m_policy_fn(t), dtype=int).reshape(-1)
        z = ubm.step(z, m)
    return float(JT)


def _build_instance_with_T(example_name: str, T_override: int, seed: int = 0):
    """Clone main_verify_beam.build_instance but override T (and rebuild ubm.G up to T)."""
    inst = main_verify_beam.EXAMPLES[example_name]()
    inst = dict(inst)
    inst["T"] = int(T_override)

    H = inst["H"]
    K, D, B = int(inst["K"]), int(inst["D"]), int(inst["B"])
    eta, T = float(inst["eta"]), int(inst["T"])
    z0 = np.asarray(inst["z0"], dtype=float)

    beam_width = int(inst.get("beam_width", 10))
    candidates_per_expand = int(inst.get("candidates_per_expand", 20))

    # same as main_verify_beam
    system = main_verify_beam._FallbackQuadraticSystem(H=H)
    action_space = ActionSpace(K=K, D=D, B=B)
    remodeling = QuadraticRemodeling(H=H, eta=eta, K=K, D=D)

    # obj = QuadraticObjective(H=H)
    # system = QuadraticASGDSystem(
    #     objective=obj,
    #     K=K,
    #     D=D,
    #     eta=eta,
    #     seed=seed,
    #     init_subspace_mode="none",
    #     init_subspace_dim=0,
    # )
    # remodeling = system.make_remodeling()

    bp = bs.BoundParams(
        relax_vertices=True,
        cache_rho=False,
        margin=1e-12,
        lookahead_steps=0,
        lookahead_use_exact_actions=True,
    )
    ubm = bs.UpperBoundModel(system=system, remodeling=remodeling, action_space=action_space, T_max=T, params=bp)
    return system, action_space, remodeling, ubm, z0, beam_width, candidates_per_expand, inst


def _run_beam(
    system,
    action_space,
    remodeling,
    ubm: bs.UpperBoundModel,
    z0: np.ndarray,
    T: int,
    beam_width: int,
    candidates_per_expand: int,
    *,
    enable_bucket_coverage: bool = False,
    bucket_coverage_mode: str = "cprime",          # "cprime" or "bprime"
    bucket_coverage_expand_all: bool = False,      # force per-node expand-all (needed for strong gap-only)
    bucket_actions: Optional[List[np.ndarray]] = None,
    enable_certified_action_set: bool = False,
    certified_action_indices_by_k: Optional[Dict[int, List[int]]] = None,
    cprime_bucket_id_by_action: Optional[Dict[Tuple[int, ...], int]] = None,
    cprime_required_bucket_ids_by_k: Optional[Dict[int, List[int]]] = None,
    enable_leakage_ub: bool = False,
    time_limit_sec: Optional[float] = None,
) -> Dict[str, Any]:
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
        bound=bp,
    )
    sp.time_limit_sec = (float(time_limit_sec) if time_limit_sec is not None else None)


    sp.enable_bucket_coverage = bool(enable_bucket_coverage)
    sp.bucket_coverage_mode = str(bucket_coverage_mode)
    sp.bucket_coverage_expand_all = bool(bucket_coverage_expand_all)

    # Allow B' anchors only when requested (bucket_coverage_mode="bprime").
    if sp.enable_bucket_coverage and sp.bucket_coverage_mode == "bprime":
        sp.bucket_actions = bucket_actions
    else:
        sp.bucket_actions = None



    sp.enable_certified_action_set = bool(enable_certified_action_set)
    sp.certified_action_indices_by_k = certified_action_indices_by_k

    sp.cprime_bucket_id_by_action = cprime_bucket_id_by_action
    sp.cprime_required_bucket_ids_by_k = cprime_required_bucket_ids_by_k

    sp.enable_leakage_ub = bool(enable_leakage_ub)

    return bs.run_bound_search(
        system=system,
        remodeling=remodeling,
        action_space=action_space,
        z0=z0,
        params=sp,
    )


def _gap_from_result(res: Dict[str, Any]) -> Optional[float]:
    lb = res.get("best_JT", None)
    if lb is None:
        return None

    # If the search terminated early, rely on the root spectral UB (always sound).
    if bool(res.get("anytime_timeout", False)):
        if res.get("UB_all", None) is None:
            return None
        return float(res["UB_all"]) - float(lb)

    ub_all = res.get("UB_all", None)
    ub_leak = res.get("leakage_ub_global_max", None)

    if ub_leak is None and ub_all is None:
        return None

    # Sound UB from leakage (covers all discarded branches) plus LB (covers explored leaves)
    if ub_leak is not None:
        ub_candidate = max(float(lb), float(ub_leak))
        if ub_all is not None:
            ub = min(float(ub_all), float(ub_candidate))
        else:
            ub = float(ub_candidate)
        return float(ub) - float(lb)

    # Fallback: root spectral UB only
    return float(ub_all) - float(lb)



def _is_gap0(res: Dict[str, Any], eps: float = 1e-12) -> bool:
    g = _gap_from_result(res)
    return (g is not None) and (g <= eps)


# ----------------------------
# Verification stages
# ----------------------------
def try_prior_stage(system, action_space, remodeling, ubm, z0, T, beam_width, cpe) -> Dict[str, Any]:
    """
    Prior concavity certificate (global proxy restriction to two endpoints),
    then run beam in certified-action-set mode with S_k = {m_min, m_max} for all k_tail,
    and compute leakage UB; certify if gap==0.
    """
    t0 = _now()
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
    t_cert = _now() - t0
    # print("prior_overall_pass: ", rep.get("overall_pass"))
    if not bool(rep.get("overall_pass", False)):
        return {
            "stage": "prior_concavity",
            "cert_pass": False,
            "cert_time": float(t_cert),
            "rep": rep,
        }

    # Build S_k as the two endpoints
    actions = _enumerate_actions(action_space)
    m_min = np.asarray(rep["m_min"], dtype=int)
    m_max = np.asarray(rep["m_max"], dtype=int)
    i_min = _find_action_index(actions, m_min)
    i_max = _find_action_index(actions, m_max)
    if i_min is None or i_max is None:
        return {
            "stage": "prior_concavity",
            "cert_pass": False,
            "cert_time": float(t_cert),
            "rep": rep,
            "note": "endpoint actions not found in enumerate_histograms()",
        }

    S_by_k = {int(k_tail): [int(i_min), int(i_max)] for k_tail in range(int(T))}

    # Enable leakage UB: one-step outside endpoints + truncation dropped-node UB
    t1 = _now()
    res = _run_beam(
        system, action_space, remodeling, ubm, z0, T, beam_width, cpe,
        enable_certified_action_set=True,
        certified_action_indices_by_k=S_by_k,
        enable_leakage_ub=True,
    )
    t_beam = float(res.get("runtime_sec_total", 0.0))
    t_total = float(t_cert + t_beam)

    gap = _gap_from_result(res)
    ok = (gap is not None) and (gap <= 1e-12)

    prior_pass = True
    # print("prior_overall_pass: ", rep.get("overall_pass"))
    cpe_ok = (cpe is None) or (int(cpe) >= 2)

    # Prior is a sufficient condition: if it passes and we can expand at least 2 candidates,
    # we treat it as a true-optimal certificate (no need to rely on gap).
    true_opt_from_prior = bool(prior_pass and cpe_ok)

    return {
        "stage": "prior_concavity",
        "cert_pass": True,
        "cert_time": float(t_cert),
        "beam_time": float(t_beam),
        "total_time": float(t_total),
        "beam_res": res,
        "gap": 0,
        "true_optimal_certified": bool(true_opt_from_prior),
        "true_optimal_cert_route": ("prior_concavity_endpoints" if true_opt_from_prior else None),
        "prior_report": rep.to_dict() if hasattr(rep, "to_dict") else None,
        "prior_candidates_required": 2,
        "prior_candidates_per_expand": int(cpe) if cpe is not None else None,
        "rep": rep,
    }


def try_strong_stage(system, action_space, remodeling, ubm, z0, T, beam_width, cpe) -> Dict[str, Any]:
    """
    Strong certificate: FULL C' buckets (scope='all'), no S_k restriction.
    Certify if S1b coverage holds at every depth.
    """
    # Need subspace cert only to get a reachable basis U (for projecting operators to build C')
    depth = int(T) - 1
    t0 = _now()
    rep2 = cert_sub.verify_prior_subspace_candidate_certificate(
        ubm=ubm,
        action_space=action_space,
        z0=z0,
        T=int(T),
        depth=depth,
        tol_new_vec=1e-9,
        psd_tol=1e-10,
    )
    t_subspace = _now() - t0

    bucket_actions = getattr(rep2, "bucket_actions", None)

    U = None
    try:
        U = rep2.diagnostics.get("U", None)
    except Exception:
        U = None
    if U is None:
        return {
            "stage": "strong_all_buckets",
            "cert_pass": False,
            "cert_time": float(t_subspace),
            "note": "rep2.diagnostics['U'] missing; cannot build C' buckets",
        }

    t1 = _now()
    cprime = cert_sub.build_cprime_buckets_by_projected_operator(
        ubm=ubm,
        action_space=action_space,
        U=np.asarray(U, dtype=float),
        delta=1e-6,
        max_buckets=200,
        restrict_to_action_indices=None,  # FULL scope
    )
    t_cprime = _now() - t1

    cprime_bucket_id_by_action = cprime["bucket_id_by_action"]
    # all_bucket_ids = cprime.get("all_bucket_ids", None)
    # if all_bucket_ids is None:
    #     # try reconstruct
    #     all_bucket_ids = sorted(set(int(v) for v in cprime_bucket_id_by_action.values() if int(v) >= 0))

    # cprime_required_bucket_ids_by_k = {int(k_tail): [int(x) for x in all_bucket_ids] for k_tail in range(int(T))}

    # Run beam (unrestricted), but with posterior S1b check for full bucket coverage
    t2 = _now()
    res = _run_beam(
        system, action_space, remodeling, ubm, z0, T, beam_width, cpe,
        enable_bucket_coverage=True,
        bucket_coverage_mode="cprime",
        bucket_coverage_expand_all=True,          # critical: make UB sound in strong mode
        bucket_actions=None,                      # no B' anchors
        enable_certified_action_set=False,
        certified_action_indices_by_k=None,
        cprime_bucket_id_by_action=cprime_bucket_id_by_action,
        cprime_required_bucket_ids_by_k=None,     # no S1b coverage check
        enable_leakage_ub=True,                   # critical: compute truncation leakage U2
    )

    t_beam = float(res.get("runtime_sec_total", 0.0))
    t_total = float(t_subspace + t_cprime + t_beam)

    gap = _gap_from_result(res)
    ok = (gap is not None) and (gap <= 1e-12)

    return {
        "stage": "strong_all_buckets",
        "cert_pass": True,  # we treat "strong stage ran" as pass; optimality is decided by gap0
        "cert_time": float(t_subspace + t_cprime),
        "beam_time": float(t_beam),
        "total_time": float(t_total),
        "beam_res": res,
        "gap": gap,
        "true_optimal_certified": bool(ok),
        "true_optimal_cert_route": ("gap0_strong_all_actions" if ok else None),
        "cprime_report": cprime.get("report", None),
    }



def try_subspace_stage(system, action_space, remodeling, ubm, z0, T, beam_width, cpe, time_limit_sec: float = 600.0, use_bprime: bool = True,) -> Dict[str, Any]:
    """
    Subspace certificate (proxy optimality S_k).
    If use_bprime=True: conditional C' buckets + B' anchors truncation (Level-3 default).
    If use_bprime=False: no buckets, no truncation; use only S_k restriction + leakage UB.
    """

    t_stage0 = _now()
    deadline = t_stage0 + float(time_limit_sec)


    depth = int(T) - 1
    t0 = _now()
    rep2 = cert_sub.verify_prior_subspace_candidate_certificate(
        ubm=ubm,
        action_space=action_space,
        z0=z0,
        T=int(T),
        depth=depth,
        tol_new_vec=1e-9,
        psd_tol=1e-10,
    )
    t_subspace = _now() - t0

    S_by_k = rep2.S_by_k
    bucket_actions = getattr(rep2, "bucket_actions", None)

    # --- New ablation: subspace without B' (no buckets, no truncation) ---
    if not bool(use_bprime):
        t_after_rep2 = _now()
        deadline = t_stage0 + float(time_limit_sec)
        remaining = max(0.0, deadline - t_after_rep2)

        t_beam0 = _now()
        res = _run_beam(
            system, action_space, remodeling, ubm, z0, T, beam_width, cpe,
            enable_bucket_coverage=False,          # <-- no truncation
            enable_certified_action_set=True,      # <-- keep S_k restriction
            certified_action_indices_by_k=S_by_k,
            enable_leakage_ub=True,                # <-- still compute leakage UB
            time_limit_sec=remaining,
        )
        t_beam = float(res.get("runtime_sec_total", 0.0))
        gap = _gap_from_result(res)
        ok = (gap is not None) and (gap <= 1e-12)

        proxy_pass = bool(rep2.overall_pass) and bool(rep2.action_cert_pass)

        return {
            "stage": "subspace_certificate_no_bprime",
            "use_bprime": False,
            "cert_pass": bool(proxy_pass),
            "cert_time": float(t_subspace),        # no C' time here
            "beam_time": float(t_beam),
            "total_time": float(t_subspace + t_beam),
            "beam_res": res,
            "gap": gap,
            "true_optimal_certified": bool(ok),
            "true_optimal_cert_route": ("leak_ub_gap0" if ok else None),
            "proxy_optimality_certified": bool(proxy_pass),
            "rep2_summary": {
                "overall_pass": bool(rep2.overall_pass),
                "action_cert_pass": bool(rep2.action_cert_pass),
                "required_candidates_per_expand": int(rep2.required_candidates_per_expand),
                "required_beam_width_outer": int(rep2.required_beam_width_outer),
            },
            "anytime_timeout": bool(res.get("anytime_timeout", False)),
            "anytime_timeout_depth": res.get("anytime_timeout_depth", None),
            "anytime_timeout_sec": float(time_limit_sec) if bool(res.get("anytime_timeout", False)) else None,
        }


    # Build C' on restricted set (union of S_k)
    U = None
    try:
        U = rep2.diagnostics.get("U", None)
    except Exception:
        U = None
    if U is None:
        return {
            "stage": "subspace_certificate",
            "cert_pass": False,
            "cert_time": float(t_subspace),
            "note": "rep2.diagnostics['U'] missing; cannot build C' buckets",
        }

    idx_union = set()
    for _, idxs in S_by_k.items():
        idx_union.update(list(idxs))
    restrict = sorted(idx_union)

    t1 = _now()
    cprime = cert_sub.build_cprime_buckets_by_projected_operator(
        ubm=ubm,
        action_space=action_space,
        U=np.asarray(U, dtype=float),
        delta=1e-6,
        max_buckets=200,
        restrict_to_action_indices=restrict,
    )
    t_cprime = _now() - t1

    t_after_rep2 = _now()
    if t_after_rep2 >= deadline:
        # No time left: fall back to an unrestricted anytime beam run with remaining time=0
        res = _run_beam(
            system, action_space, remodeling, ubm, z0, T, beam_width, cpe,
            enable_bucket_coverage=False,
            enable_certified_action_set=False,
            enable_leakage_ub=False,
            time_limit_sec=0.0,
        )
        gap = _gap_from_result(res)
        return {
            "stage": "subspace_certificate",
            "use_bprime": True,
            "cert_pass": False,
            "cert_time": float(t_after_rep2 - t_stage0),
            "beam_time": float(res.get("runtime_sec_total", 0.0) or 0.0),
            "total_time": float(_now() - t_stage0),
            "beam_res": res,
            "gap": gap,
            "true_optimal_certified": False,
            "true_optimal_cert_route": None,
            "proxy_optimality_certified": False,
            "anytime_timeout": True,
            "anytime_timeout_sec": float(time_limit_sec),
            "note": "timeout before running restricted beam; returned fallback incumbent",
        }


    cprime_bucket_id_by_action = cprime["bucket_id_by_action"]

    # required bucket ids per k_tail induced by S_k (conditional)
    actions_full = _enumerate_actions(action_space)
    cprime_required_bucket_ids_by_k = {}
    for k_tail, idxs in S_by_k.items():
        req = set()
        for i in idxs:
            m = actions_full[int(i)]
            cbid = cprime_bucket_id_by_action.get(tuple(m.tolist()), -1)
            if int(cbid) >= 0:
                req.add(int(cbid))
        cprime_required_bucket_ids_by_k[int(k_tail)] = sorted(req)
    
    if _now() >= deadline:
        # same fallback idea: run restricted beam with 0 remaining (will instantly finalize incumbent)
        rem = max(0.0, deadline - _now())


    # Run restricted beam + leakage UB
    t2 = _now()
    remaining = max(0.0, deadline - _now())
    res = _run_beam(
        system, action_space, remodeling, ubm, z0, T, beam_width, cpe,
        enable_bucket_coverage=True,
        bucket_coverage_mode="bprime",              # <-- revert Level 3 to B'-anchors truncation
        bucket_coverage_expand_all=False,
        bucket_actions=bucket_actions,              # <-- use B' anchors from subspace certificate
        enable_certified_action_set=True,
        certified_action_indices_by_k=S_by_k,       # <-- restrict actions by S_k
        cprime_bucket_id_by_action=cprime_bucket_id_by_action,  # keep for leakage/diagnostics
        cprime_required_bucket_ids_by_k=None,
        enable_leakage_ub=True,
        time_limit_sec=remaining,
    )



    t_beam = float(res.get("runtime_sec_total", 0.0))
    t_total = float(t_subspace + t_cprime + t_beam)

    gap = _gap_from_result(res)
    ok = (gap is not None) and (gap <= 1e-12)

    # Label sub-route
    route = "leak_ub_gap0" if ok else None

    proxy_pass = bool(rep2.overall_pass) and bool(rep2.action_cert_pass)

    return {
        "stage": "subspace_certificate",
        "use_bprime": True,
        "cert_pass": bool(proxy_pass),
        "cert_time": float(t_subspace + t_cprime),
        "beam_time": float(t_beam),
        "total_time": float(t_total),
        "beam_res": res,
        "gap": gap,
        "true_optimal_certified": bool(ok),
        "true_optimal_cert_route": route,
        "proxy_optimality_certified": bool(proxy_pass),
        "rep2_summary": {
            "overall_pass": bool(rep2.overall_pass),
            "action_cert_pass": bool(rep2.action_cert_pass),
            "required_candidates_per_expand": int(rep2.required_candidates_per_expand),
            "required_beam_width_outer": int(rep2.required_beam_width_outer),
        },
        "cprime_report": cprime.get("report", None),
        "anytime_timeout": bool(res.get("anytime_timeout", False)),
        "anytime_timeout_depth": res.get("anytime_timeout_depth", None),
        "anytime_timeout_sec": float(time_limit_sec) if bool(res.get("anytime_timeout", False)) else None,
    }


# ----------------------------
# Gurobi baseline
# ----------------------------
def _run_gurobi(system, action_space, z0, T: int, warmstart_m_hist: Optional[np.ndarray]):
    try:
        from alg.solver_baseline_gurobi import SolverBaselineParams, solve_worst_sequence_miqcp_gurobi
    except Exception as e:
        return {
            "available": False,
            "error": f"{type(e).__name__}: {e}",
        }

    gp = SolverBaselineParams(
        time_limit=float(600),
        mip_gap=float(1e-8),
        output_flag=int(0),
        bound_scale=float(2.0),
        basis="original",
        bound_mode="global",
        m_encoding="hist_miqcp",
    )

    t0 = _now()
    if warmstart_m_hist is not None:
        out = solve_worst_sequence_miqcp_gurobi(
            system=system,
            action_space=action_space,
            z0=z0,
            T=int(T),
            params=gp,
            warm_start_m_hist=warmstart_m_hist,
        )
    else:
        out = solve_worst_sequence_miqcp_gurobi(
            system=system,
            action_space=action_space,
            z0=z0,
            T=int(T),
            params=gp,
        )
    out = dict(out)
    out["available"] = True
    out["runtime_wall"] = float(_now() - t0)

    # normalize status/gap
    status = out.get("status", None)
    obj = out.get("obj", None)
    best_bound = out.get("best_bound", None)
    mip_gap = out.get("mip_gap", None)

    out["status"] = int(status) if status is not None else None
    out["obj"] = _safe_float(obj)
    out["best_bound"] = _safe_float(best_bound)
    out["mip_gap"] = _safe_float(mip_gap)

    # a friendly "gap" field
    if out["obj"] is not None and out["best_bound"] is not None:
        out["gap_bestbound_minus_obj"] = float(out["best_bound"] - out["obj"])
    else:
        out["gap_bestbound_minus_obj"] = None

    return out


# ----------------------------
# Main experiment loop
# ----------------------------
def run_one(example_name: str, T: int, seed: int = 0, ban_vertices: bool = False, ban_gurobi: bool = False) -> Dict[str, Any]:
    system, action_space, remodeling, ubm, z0, bw, cpe, inst = _build_instance_with_T(example_name, T, seed)

    # 3-stage cascade (strict): prior -> strong -> subspace
    # Stop at the first stage whose *certificate passes*.
    stage_out = None

    if ban_vertices:
        # (C2) subspace ablation (no B': no buckets, no truncation)
        outC = try_subspace_stage(system, action_space, remodeling, ubm, z0, T, bw, cpe, time_limit_sec=600.0, use_bprime=False)
    else:
        # (C) subspace (default: with B')
        outC = try_subspace_stage(system, action_space, remodeling, ubm, z0, T, bw, cpe, time_limit_sec=600.0, use_bprime=True)
    
    stage_out = outC
    
    # # (A) prior
    # outA = try_prior_stage(system, action_space, remodeling, ubm, z0, T, bw, cpe)
    # if bool(outA.get("true_optimal_certified", False)):
    #     stage_out = outA
    # else:
        # # (B) strong (hard timeout)
        # strong_timeout_sec = 600.0
        # rr = _run_with_hard_timeout(
        #     try_strong_stage,
        #     args=(system, action_space, remodeling, ubm, z0, T, bw, cpe),
        #     timeout_sec=strong_timeout_sec,
        # )
        # if rr.get("ok", False):
        #     outB = rr["out"]
        # else:
        #     outB = {
        #         "stage": "strong_all_buckets",
        #         "true_optimal_certified": False,
        #         "anytime_timeout": bool(rr.get("timeout", False)),
        #         "anytime_timeout_sec": float(rr.get("timeout_sec", strong_timeout_sec)) if rr.get("timeout", False) else None,
        #         "error": rr.get("error", None),
        #     }

        # if bool(outB.get("true_optimal_certified", False)):
        #     stage_out = outB
        # else:
        #     # (C) subspace
        #     outC = try_subspace_stage(system, action_space, remodeling, ubm, z0, T, bw, cpe, time_limit_sec=600.0)
        #     stage_out = outC



    beam_res = stage_out.get("beam_res", {})
    beam_JT = beam_res.get("best_JT", None)
    beam_time = float(stage_out.get("beam_time", beam_res.get("runtime_sec_total", 0.0) or 0.0))
    total_time = float(stage_out.get("total_time", beam_time))

    # Determine "method name" for CSV
    method_name = stage_out.get("true_optimal_cert_route", None)
    if not method_name:
        # if not certified, use last stage name
        method_name = stage_out.get("stage", "unknown")

    # gap to record
    gap = stage_out.get("gap", None)
    if gap is None and beam_res:
        gap = _gap_from_result(beam_res)

    if ban_gurobi:
        row = {
            "example": example_name,
            "T": int(T),

            # our method
            "our_cert_method": method_name,
            "our_true_optimal_certified": bool(stage_out.get("true_optimal_certified", False)),
            "our_beam_JT": _safe_float(beam_JT),
            "our_gap_UB_minus_LB": _safe_float(gap),
            "our_time_total_sec": float(total_time),
            "our_time_beam_sec": float(beam_time),
            "our_time_cert_sec": _safe_float(stage_out.get("cert_time", None)),
        }

        return row

    # ASGD baselines (using ubm)
    z0_modes = ubm.init_z_modes(z0)
    K, D, B = int(action_space.K), int(action_space.D), int(action_space.B)
    m_nodelay = np.zeros(D + 1, dtype=int); m_nodelay[0] = K

    asgd_no_delay_obj = _rollout_obj(ubm, z0_modes, T, lambda t: m_nodelay)

    rng = np.random.default_rng(0)
    asgd_random_obj = _rollout_obj(ubm, z0_modes, T, lambda t: _sample_hist_random(K, D, B, rng))

    # Gurobi baselines
    # Gurobi system
    H = inst["H"]
    eta = float(inst["eta"])
    obj = QuadraticObjective(H=H)
    system = QuadraticASGDSystem(
        objective=obj,
        K=K,
        D=D,
        eta=eta,
        seed=seed,
        init_subspace_mode="none",
        init_subspace_dim=0,
    )
    remodeling = system.make_remodeling()

    warmstart_m_hist = beam_res.get("best_m_hist", None)
    gurobi_calc = _run_gurobi(system, action_space, z0, T, warmstart_m_hist=None)
    gurobi_verify = _run_gurobi(system, action_space, z0, T, warmstart_m_hist=warmstart_m_hist)

    row = {
        "example": example_name,
        "T": int(T),

        # our method
        "our_cert_method": method_name,
        "our_true_optimal_certified": bool(stage_out.get("true_optimal_certified", False)),
        "our_beam_JT": _safe_float(beam_JT),
        "our_gap_UB_minus_LB": _safe_float(gap),
        "our_time_total_sec": float(total_time),
        "our_time_beam_sec": float(beam_time),
        "our_time_cert_sec": _safe_float(stage_out.get("cert_time", None)),

        # gurobi-calculate
        "gurobi_calculate_available": bool(gurobi_calc.get("available", False)),
        "gurobi_calculate_status": gurobi_calc.get("status", None),
        "gurobi_calculate_obj": gurobi_calc.get("obj", None),
        "gurobi_calculate_best_bound": gurobi_calc.get("best_bound", None),
        "gurobi_calculate_mip_gap": gurobi_calc.get("mip_gap", None),
        "gurobi_calculate_gap_bb_minus_obj": gurobi_calc.get("gap_bestbound_minus_obj", None),
        "gurobi_calculate_time_sec": gurobi_calc.get("runtime_wall", None),

        # gurobi-verify (warm-start)
        "gurobi_verify_available": bool(gurobi_verify.get("available", False)),
        "gurobi_verify_status": gurobi_verify.get("status", None),
        "gurobi_verify_obj": gurobi_verify.get("obj", None),
        "gurobi_verify_best_bound": gurobi_verify.get("best_bound", None),
        "gurobi_verify_mip_gap": gurobi_verify.get("mip_gap", None),
        "gurobi_verify_gap_bb_minus_obj": gurobi_verify.get("gap_bestbound_minus_obj", None),
        "gurobi_verify_time_sec": gurobi_verify.get("runtime_wall", None),

        # ASGD baselines
        "asgd_no_delay_obj": float(asgd_no_delay_obj),
        "asgd_random_delay_obj": float(asgd_random_obj),
    }

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default="experiment_results.csv")
    ap.add_argument("--T_list", type=str, default="5,10,20,40",
                    help="comma-separated T values, e.g. 5,10,20,40")
    ap.add_argument("--examples", type=str, default="all",
                    help="comma-separated example names, or 'all'")
    ap.add_argument("--ban_vertices", action="store_true",
                    help="Not use integer vertices in candidate set constructiona and truncation")
    ap.add_argument("--ban_gurobi", action="store_true",
                    help="Not run gurobi")
    args = ap.parse_args()

    T_list = [int(x.strip()) for x in args.T_list.split(",") if x.strip()]
    if args.examples.strip().lower() == "all":
        examples = sorted(main_verify_beam.EXAMPLES.keys())
    else:
        examples = [x.strip() for x in args.examples.split(",") if x.strip()]

    out_csv = args.out_csv
    # If file exists and is non-empty, we will append without header.
    need_header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)

    num_written = 0
    for ex in examples:
        for T in T_list:
            print(f"[run] example={ex}  T={T}")
            row = run_one(ex, T, ban_vertices=args.ban_vertices, ban_gurobi=args.ban_gurobi)

            df_row = pd.DataFrame([row])
            df_row.to_csv(
                out_csv,
                mode="a",
                header=need_header,
                index=False,
            )
            need_header = False
            num_written += 1

            # Best-effort flush (helps on network FS)
            try:
                with open(out_csv, "a", encoding="utf-8") as f:
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass

            print(f"[saved] appended 1 row -> {out_csv}  (total={num_written})")

    print(f"[done] wrote {num_written} rows -> {out_csv}")



if __name__ == "__main__":
    main()

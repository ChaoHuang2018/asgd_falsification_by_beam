#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from types import SimpleNamespace
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass

# ===== Imports you required (keep exactly) =====
import alg.bound_search as bs
from core.action_space import ActionSpace
from core.remodeling import QuadraticRemodeling
import alg.beam_certificate as cert
import alg.beam_certificate_subspace as cert_sub
from examples.quadratic_asgd import QuadraticASGDSystem, QuadraticObjective
# ==============================================

# -------------------------
# Minimal quadratic system/objective fallback
# -------------------------
@dataclass
class _FallbackQuadraticSystem:
    """Enough fields for bound_search_v11._get_H(...) and remodeling."""
    H: np.ndarray

# ----------------------------
# Example library (includes budget_one/two/three)
# ----------------------------

def _example_budget_one() -> Dict[str, Any]:
    """
    From v1: D can be >1, but B=1 collapses histograms.
    """
    K, D, B = 4, 5, 1
    eta = 0.25
    T = 12

    H = np.diag([1.0, 4.0, 9.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([1.0, -0.5, 0.25])
    z0[1] = np.array([0.7, -0.35, 0.18])

    beam_width = 5
    candidates_per_expand = 10

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_budget_two() -> Dict[str, Any]:
    """
    From v1: D=1, B=2 (nontrivial action count 3)
    """
    K, D, B = 4, 1, 2
    eta = 0.22
    T = 12

    H = np.diag([1.0, 4.0, 9.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([1.0, -0.5, 0.25])
    z0[1] = np.array([0.7, -0.35, 0.18])

    beam_width = 5
    candidates_per_expand = 10

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_budget_three() -> Dict[str, Any]:
    """
    From v1: D=1, B=3 (nontrivial action count 4)
    """
    K, D, B = 6, 1, 3
    eta = 0.18
    T = 14

    H = np.diag([0.5, 2.0, 8.0, 18.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([0.8, -0.6, 0.35, -0.2])
    z0[1] = np.array([0.55, -0.4, 0.22, -0.12])

    beam_width = 5
    candidates_per_expand = 10

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_d3_b3_k3() -> Dict[str, Any]:
    """
    From v3 earlier: D>1,B>1,K>1, moderate eta.
    """
    K, D, B = 3, 3, 3
    eta = 0.12
    T = 20

    H = np.diag([2.0, 5.0, 11.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([0.9, -0.4, 0.2])
    z0[1] = np.array([0.6, -0.25, 0.15])
    z0[2] = np.array([0.35, -0.18, 0.10])
    z0[3] = np.array([0.20, -0.10, 0.06])

    beam_width = 5
    candidates_per_expand = 10

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_d4_b5_k4() -> Dict[str, Any]:
    """
    From v3 earlier: larger D,B,K.
    """
    K, D, B = 4, 4, 5
    eta = 0.08
    T = 12

    H = np.diag([1.0, 3.0, 7.0, 13.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([1.0, -0.6, 0.35, -0.2])
    z0[1] = np.array([0.7, -0.42, 0.24, -0.14])
    z0[2] = np.array([0.45, -0.28, 0.16, -0.09])
    z0[3] = np.array([0.30, -0.18, 0.10, -0.06])
    z0[4] = np.array([0.20, -0.12, 0.07, -0.04])

    beam_width = 5
    candidates_per_expand = 10

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_eta01_d3_b2_k5() -> Dict[str, Any]:
    """
    Extra: eta=1e-2 class, D>1,B>1,K>1, lambdas in [1e-2,1e0]
    """
    K, D, B = 5, 3, 2
    eta = 1e-2
    T = 40

    H = np.diag([0.02, 0.2, 0.9])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([0.30, -0.20, 0.10])
    z0[1] = np.array([0.18, -0.10, 0.05])
    z0[2] = np.array([0.10, -0.06, 0.03])
    z0[3] = np.array([0.06, -0.03, 0.02])

    beam_width = 5
    candidates_per_expand = 10

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_eta01_d4_b3_k6() -> Dict[str, Any]:
    """
    Extra: eta=1e-2 class, D>1,B>1,K>1, lambdas in [1e-2,1e0]
    """
    K, D, B = 6, 4, 3
    eta = 1e-2
    T = 9

    H = np.diag([0.03, 0.15, 0.6, 1.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([0.25, -0.18, 0.12, -0.08])
    z0[1] = np.array([0.16, -0.11, 0.07, -0.05])
    z0[2] = np.array([0.10, -0.07, 0.05, -0.03])
    z0[3] = np.array([0.06, -0.04, 0.03, -0.02])
    z0[4] = np.array([0.04, -0.03, 0.02, -0.01])

    beam_width = 15
    candidates_per_expand = 20

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)

def _example_T40() -> Dict[str, Any]:
    K, D, B = 5, 3, 2
    eta = 1e-2
    T = 40

    H = np.diag([0.02, 0.2, 0.9])

    n = H.shape[0]
    z0 = np.array([
        [ 0.60, -0.40,  0.25],
        [ 0.42, -0.28,  0.18],
        [ 0.28, -0.20,  0.12],
        [ 0.18, -0.13,  0.08],
        ], float)

    beam_width = 10
    candidates_per_expand = 20

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)

def _example_T60() -> Dict[str, Any]:
    """
    Extra: eta=1e-2 class, D>1,B>1,K>1, lambdas in [1e-2,1e0]
    """
    K, D, B = 6, 4, 3
    eta = 1e-2
    T = 60

    H = np.diag([0.03, 0.15, 0.6, 1.0])

    z0 = np.array([
        [ 0.55, -0.35,  0.22, -0.15],
        [ 0.40, -0.26,  0.16, -0.11],
        [ 0.28, -0.19,  0.12, -0.08],
        [ 0.20, -0.14,  0.09, -0.06],
        [ 0.14, -0.10,  0.06, -0.04],
        ], float)

    beam_width = 15
    candidates_per_expand = 20

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0, beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_gap_1() -> Dict[str, Any]:
    K, D, B = 10, 5, 12
    eta = 0.06
    T = 80

    # More dimensions => reachable subspace/bucket explosion
    H = np.diag([0.6, 1.8, 4.0, 7.5, 13.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([0.9, -0.5, 0.28, -0.18, 0.12])
    z0[1] = np.array([0.65, -0.36, 0.20, -0.13, 0.49])
    z0[2] = np.array([0.48, -0.28, 0.15, -0.10, 0.87])
    z0[3] = np.array([0.34, -0.20, 0.11, -0.27, 0.35])
    z0[4] = np.array([0.24, -0.14, 0.38, -0.15, 0.73])
    z0[5] = np.array([0.17, -0.70, 0.36, -0.53, 0.42])

    beam_width = 12
    candidates_per_expand = 25

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0,
                beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_gap_2() -> Dict[str, Any]:
    """
    Gap-heavy regime: very large action space + long horizon + higher dimension.
    Empirically tends to make:
      - prior fail (nontrivial D,B,K,eta)
      - strong not applicable if you run cprime_only_on_certified (scope != all)
      - subspace restriction exists but leak-UB gap stays > 0 (cannot certify)
    """
    K, D, B = 12, 6, 15
    eta = 0.05
    T = 80

    # higher dimension => subspace/bucket mapping harder to cover tightly
    H = np.diag([0.4, 1.0, 2.2, 4.5, 9.0, 16.0])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)

    # a decaying but not-too-small history; avoid being too close to 0
    z0[0] = np.array([ 0.95, -0.55,  0.33, -0.22,  0.15, -0.10])
    z0[1] = np.array([ 0.70, -0.40,  0.24, -0.16,  0.11, -0.07])
    z0[2] = np.array([ 0.52, -0.30,  0.18, -0.12,  0.08, -0.05])
    z0[3] = np.array([ 0.38, -0.22,  0.13, -0.09,  0.06, -0.04])
    z0[4] = np.array([ 0.28, -0.16,  0.10, -0.06,  0.04, -0.03])
    z0[5] = np.array([ 0.20, -0.12,  0.07, -0.05,  0.03, -0.02])
    z0[6] = np.array([ 0.15, -0.09,  0.05, -0.03,  0.02, -0.015])

    # Keep beam modest so leakage via truncation/coverage is more likely
    beam_width = 12
    candidates_per_expand = 25

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0,
                beam_width=beam_width, candidates_per_expand=candidates_per_expand)


def _example_subspace_fail() -> Dict[str, Any]:
    K, D, B = 8, 5, 10
    eta = 0.08
    T = 40

    # non-diagonal, competing modes
    H = np.array([
        [4.0,  1.8, 0.0],
        [1.8,  3.6, 1.2],
        [0.0,  1.2, 2.0],
    ])

    n = H.shape[0]
    z0 = np.zeros((D + 1, n), dtype=float)
    z0[0] = np.array([ 0.9,  0.4, -0.2])
    z0[1] = np.array([ 0.6,  0.3, -0.15])
    z0[2] = np.array([ 0.4,  0.2, -0.10])
    z0[3] = np.array([ 0.3,  0.15, -0.08])
    z0[4] = np.array([ 0.2,  0.10, -0.05])
    z0[5] = np.array([ 0.15, 0.07, -0.03])

    beam_width = 6
    candidates_per_expand = 3

    return dict(H=H, K=K, D=D, B=B, eta=eta, T=T, z0=z0,
                beam_width=beam_width, candidates_per_expand=candidates_per_expand)


EXAMPLES = {
    # "budget_one": _example_budget_one,
    # "budget_two": _example_budget_two,
    # "budget_three": _example_budget_three,
    "EX1": _example_d3_b3_k3,
    "EX4": _example_d4_b5_k4,
    "EX2": _example_eta01_d3_b2_k5,
    "EX5": _example_eta01_d4_b3_k6,
    # "T40": _example_T40,
    # "T60": _example_T60,
    "EX6": _example_gap_1,
    "EX7": _example_gap_2,
    "EX3": _example_subspace_fail,
    # "_example_cprime_gap0_1": _example_cprime_gap0_1,
    # "_example_cprime_gap0_2": _example_cprime_gap0_2,
}


# ----------------------------
# Builders
# ----------------------------
def build_instance(name: str) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any]]:
    inst = EXAMPLES[name]()
    H = inst["H"]
    K, D, B = int(inst["K"]), int(inst["D"]), int(inst["B"])
    eta, T = float(inst["eta"]), int(inst["T"])
    z0 = np.asarray(inst["z0"], dtype=float)
    beam_width = int(inst["beam_width"])
    candidates_per_expand = int(inst["candidates_per_expand"])

    system = _FallbackQuadraticSystem(H=H)
    action_space = ActionSpace(K=K, D=D, B=B)
    remodeling = QuadraticRemodeling(H=H, eta=eta, K=K, D=D)
    bp = bs.BoundParams(
        relax_vertices=True,
        cache_rho=False,
        margin=1e-12,
        lookahead_steps=0,              # IMPORTANT: fix lookahead=0
        lookahead_use_exact_actions=True,
    )
    ubm = bs.UpperBoundModel(system=system, remodeling=remodeling, action_space=action_space, T_max=T, params=bp)
    return system, action_space, remodeling, ubm, z0, beam_width, candidates_per_expand, inst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="eta01_d3_b2_k5", choices=sorted(EXAMPLES.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cert_mode", type=str, default="subspace", choices=["none", "prior", "subspace", "both"])
    parser.add_argument("--subspace_depth", type=int, default=-1, help="reachable-subspace depth; -1 means T-1")
    parser.add_argument("--tol_new_vec", type=float, default=1e-9)
    parser.add_argument("--psd_tol", type=float, default=1e-10)
    parser.add_argument("--dump_dir", type=str, default="")

    parser.add_argument("--bucket_mode", type=str, default="vertices+cprime",
                        choices=["vertices", "vertices+cprime"],
                        help="vertices: only proxy-coverage buckets; vertices+cprime: also build C' bucket mapping")
    parser.add_argument("--cprime_delta", type=float, default=1e-6)
    parser.add_argument("--cprime_max_buckets", type=int, default=200)
    parser.add_argument("--cprime_only_on_certified", action="store_true",
                        help="If set, only bucketize actions in union_k S_k (faster)")

    args = parser.parse_args()

    system, action_space, remodeling, ubm, z0, beam_width, candidates_per_expand, inst = build_instance(args.example)
    T = int(inst["T"])

    # run certificates
    if args.cert_mode in ("prior", "both"):
        rep = cert.verify_prior_bellman_sufficient_condition(
            action_space=action_space,
            lambdas=ubm.lambdas,
            G=ubm.G,
            D=ubm.D,
            K=ubm.K,
            B=action_space.B,
            eta=ubm.eta,
            T=T,
        )
        print("\n=== Prior certificate report ===")
        for k, v in rep.items():
            print(f"{k}: {v}")

    bucket_actions = None
    enable_bucket_coverage = True if args.cert_mode in ("subspace", "both") else False

    rep2_prior = None
    rep2_gap = None

    # ----------------------------
    # prior subspace action certificate (no candidate_hist)
    # ----------------------------
    if args.cert_mode in ("subspace", "both"):
        depth = (T - 1) if args.subspace_depth < 0 else int(args.subspace_depth)

        rep2_prior = cert_sub.verify_prior_subspace_candidate_certificate(
            ubm=ubm,
            action_space=action_space,
            z0=z0,
            T=T,
            depth=depth,
            tol_new_vec=float(args.tol_new_vec),
            psd_tol=float(args.psd_tol),
        )
        bucket_actions = rep2_prior.bucket_actions

        cprime_bucket_id_by_action = None
        if args.bucket_mode == "vertices+cprime":
            U = rep2_prior.diagnostics.get("U", None)
            if U is None:
                raise ValueError("rep2_prior.diagnostics['U'] missing; please apply the v2 certificate change to store U.")

            restrict = None
            if bool(args.cprime_only_on_certified):
                idx_union = set()
                for _, idxs in rep2_prior.S_by_k.items():
                    idx_union.update(list(idxs))
                restrict = sorted(idx_union)

            cprime = cert_sub.build_cprime_buckets_by_projected_operator(
                ubm=ubm,
                action_space=action_space,
                U=np.asarray(U, dtype=float),
                delta=float(args.cprime_delta),
                max_buckets=int(args.cprime_max_buckets),
                restrict_to_action_indices=restrict,
            )

            print("\n=== C' bucket build report ===")
            print(cprime["report"])
            cprime_bucket_id_by_action = cprime["bucket_id_by_action"]
        else:
            cprime_bucket_id_by_action = None


        print("\n=== subspace-candidate (Phase A: action restriction only) ===]")
        # These are PROXY-level certificate flags only (do NOT interpret as true-optimal soundness).
        print("proxy_cert_overall_pass =", rep2_prior.overall_pass)
        print("proxy_cert_action_pass  =", rep2_prior.action_cert_pass)

        print("subspace_dim =", rep2_prior.subspace_dim, " depth =", rep2_prior.depth)
        print("required_candidates_per_expand =", rep2_prior.required_candidates_per_expand)
        print("required_beam_width_outer =", rep2_prior.required_beam_width_outer)
        sizes = {k: len(v) for k, v in rep2_prior.S_by_k.items()}
        print("  |S_k| by k =", sizes)

        # if int(beam_width) < int(rep2_prior.required_beam_width_outer):
        #     raise ValueError(
        #         f"beam_width={beam_width} < required_beam_width_outer={rep2_prior.required_beam_width_outer}"
        #     )

        # if int(candidates_per_expand) < int(rep2_prior.required_candidates_per_expand):
        #     print(
        #         f"[WARN] candidates_per_expand={candidates_per_expand} < "
        #         f"required_candidates_per_expand={rep2_prior.required_candidates_per_expand}"
        #     )

    # ----------------------------
    # Run beam (lookahead=0)
    # ----------------------------
    bp = bs.BoundParams(
        relax_vertices=True,
        cache_rho=False,
        margin=1e-12,
        lookahead_steps=0,              # IMPORTANT: fix lookahead=0
        lookahead_use_exact_actions=True,
    )

    sp = bs.SearchParams(
        mode="heuristic",
        T=T,
        seed=int(args.seed),
        beam_width=int(beam_width),
        candidates_per_expand=int(candidates_per_expand),
        enable_pruning=True,
        prune_eps=0.0,
        bound=bp,
        dump_word_dir=str(args.dump_dir),
    )
    sp.time_limit_sec = float(600.0)
    

    if args.cert_mode in ("subspace", "both"):
        sp.enable_certified_action_set = True
        sp.certified_action_indices_by_k = rep2_prior.S_by_k
        # sp.enable_leakage_ub = bool(args.cprime_only_on_certified) and bool(sp.enable_certified_action_set)
        # We want a unified "gap-only" optimality decision for all levels.
        # This requires computing truncation-leakage U2 whenever the run is expand-all (sound),
        # and restriction-leakage U1 when S_k is enabled.
        sp.enable_leakage_ub = True


    # STRONG mode: ALL-action C' buckets -> do NOT restrict beam actions by proxy S_k
    if args.bucket_mode == "vertices+cprime" and cprime.get("scope", None) == "all":
        sp.enable_certified_action_set = False
        sp.certified_action_indices_by_k = None
        # --- Structural completeness via C'-driven truncation ---
        sp.enable_bucket_coverage = True
        sp.bucket_coverage_mode = "cprime"
        sp.bucket_coverage_expand_all = True
        sp.bucket_actions = None  # B' anchors not used

        all_bucket_ids = cprime.get("all_bucket_ids", None)
        if all_bucket_ids is not None:
            sp.beam_width = max(int(sp.beam_width), len(list(all_bucket_ids)))



    # Build S1b required C' bucket ids per k_tail.
    # - If C' was built in ALL scope, required buckets are ALL C' buckets (strong sound certificate).
    # - Otherwise, fall back to the old conditional required set induced by proxy S_k (not strong).
    cprime_required_bucket_ids_by_k = None
    if (args.bucket_mode == "vertices+cprime") and (cprime_bucket_id_by_action is not None) and (rep2_prior is not None):
        # If you built C' with restrict=None, build_cprime... returns scope="all" and all_bucket_ids.
        # We can detect ALL scope via the printed report, but here we also reconstruct from the returned dict.
        # The variable `cprime` exists earlier in this function only when bucket_mode=="vertices+cprime".
        cprime_scope = None
        all_bucket_ids = None
        try:
            cprime_scope = cprime.get("scope", None)
            all_bucket_ids = cprime.get("all_bucket_ids", None)
        except Exception:
            cprime_scope = None
            all_bucket_ids = None

        if cprime_scope == "all" and all_bucket_ids is not None:
            # STRONG: require coverage of ALL C' buckets at every depth.
            all_ids = [int(x) for x in list(all_bucket_ids)]
            cprime_required_bucket_ids_by_k = {int(k_tail): all_ids for k_tail in range(T)}
        else:
            # Conditional (old): require only buckets induced by proxy certified action sets S_by_k.
            actions_full = [np.asarray(m, dtype=int).reshape(-1) for m in action_space.enumerate_histograms()]
            cprime_required_bucket_ids_by_k = {}
            for k_tail, idxs in rep2_prior.S_by_k.items():
                req = set()
                for i in idxs:
                    m = actions_full[int(i)]
                    cbid = cprime_bucket_id_by_action.get(tuple(m.tolist()), -1)
                    if int(cbid) >= 0:
                        req.add(int(cbid))
                cprime_required_bucket_ids_by_k[int(k_tail)] = sorted(req)

    sp.cprime_required_bucket_ids_by_k = cprime_required_bucket_ids_by_k


    # Truncation rule:
    #  - Strong (C' scope == "all"): use C'-truncation (structural coverage over ALL actions).
    #  - Otherwise (e.g., Level III with S_k restriction): use B'-anchors truncation if available.
    bucket_actions = getattr(rep2_prior, "bucket_actions", None) if rep2_prior is not None else None

    if (args.bucket_mode == "vertices+cprime") and (cprime_bucket_id_by_action is not None) and (cprime.get("scope", None) == "all"):
        # STRONG
        sp.enable_bucket_coverage = True
        sp.bucket_coverage_mode = "cprime"
        sp.bucket_actions = None
    else:
        # Level III (subspace): revert to B' if we have anchors; otherwise disable
        if bucket_actions is not None:
            sp.enable_bucket_coverage = True
            sp.bucket_coverage_mode = "bprime"
            sp.bucket_actions = bucket_actions
        else:
            sp.enable_bucket_coverage = False
            sp.bucket_actions = None



    sp.cprime_bucket_id_by_action = cprime_bucket_id_by_action
    print("enable_bucket_coverage:    ", sp.enable_bucket_coverage)

    res = bs.run_bound_search(
        system=system,
        remodeling=remodeling,
        action_space=action_space,
        z0=z0,
        params=sp,
    )

    # ----------------------------
    # Final certification flags (TRUE-opt)
    # ----------------------------
    proxy_optimality_certified = False
    if args.cert_mode in ("subspace", "both") and rep2_prior is not None:
        proxy_optimality_certified = (
            bool(rep2_prior.action_cert_pass)
            and (int(beam_width) >= int(rep2_prior.required_beam_width_outer))
            and (int(candidates_per_expand) >= int(rep2_prior.required_candidates_per_expand))
        )

    consistency_s1b_pass = res.get("consistency_s1b_pass", None)
    consistency_s1b_applicable = res.get("consistency_s1b_applicable", False)

    # --- Determine "STRONG" applicability: ALL-action C' buckets (full scope) ---
    cprime_scope = None
    try:
        cprime_scope = cprime.get("scope", None)
    except Exception:
        cprime_scope = None

    strong_applicable = (args.bucket_mode == "vertices+cprime") and (cprime_scope == "all")
    strong_pass = bool(strong_applicable) and bool(consistency_s1b_pass)

    # ----------------------------
    # Final certification (TRUE-opt): GAP-ONLY (sound UB)
    # ----------------------------
    gap_eps = 1e-12
    lb = res.get("best_JT", None)

    ub_all = res.get("UB_all", None)  # always sound (root spectral UB)
    ub_leak = res.get("leakage_ub_global_max", None)
    gap_applicable = bool(res.get("leakage_ub_applicable", False))

    anytime_timeout = bool(res.get("anytime_timeout", False))

    ub_final = None
    gap_value = None
    gap0_pass = False

    if lb is not None:
        if anytime_timeout:
            # On timeout, rely on UB_all only (safe)
            if ub_all is not None:
                ub_final = float(ub_all)
        else:
            # Normal case: combine leakage UB with LB, then tighten with UB_all
            if gap_applicable and (ub_leak is not None):
                ub_candidate = max(float(lb), float(ub_leak))  # sound
                if ub_all is not None:
                    ub_final = min(float(ub_all), float(ub_candidate))  # still sound
                else:
                    ub_final = float(ub_candidate)
            else:
                # Fallback to UB_all if leakage UB not applicable
                if ub_all is not None:
                    ub_final = float(ub_all)

        if ub_final is not None:
            gap_value = float(ub_final) - float(lb)
            gap0_pass = bool(float(ub_final) <= float(lb) + gap_eps)

    true_optimal_certified = bool(gap0_pass)
    true_optimal_cert_route = ("gap0" if true_optimal_certified else None)

    print("\n=== Certification summary (gap-only, sound UB) ===")
    print("anytime_timeout          =", anytime_timeout)
    print("gap_applicable           =", gap_applicable)
    if ub_final is not None:
        print("UB_final                 =", ub_final)
    if gap_value is not None:
        print("gap_UB_minus_LB          =", gap_value)
    print("true_optimal_certified   =", true_optimal_certified)
    print("true_optimal_cert_route  =", true_optimal_cert_route)



    print("\n=== Beam result ===")
    print(f"mode: {res.get('mode')}")
    print(f"best_JT: {res.get('best_JT')}")
    print(f"beam_width: {res.get('beam_width')}, candidates_per_expand: {res.get('candidates_per_expand')}")
    print(f"runtime_sec_total: {res.get('runtime_sec_total')}")

    if res.get("leakage_ub_applicable", False):
        print("\n=== Leakage UB accounting (U1/U2) ===")
        u1 = res.get("leakage_U1_restriction_max", None)
        u2 = res.get("leakage_U2_truncation_max", None)
        ub = res.get("leakage_ub_global_max", None)
        lb = res.get("best_JT", None)

        print("U1_restriction_one_step_outside_Sk =", u1)
        print("U2_truncation_best_dropped_node    =", u2)
        print("Global_UB_run = max(U1,U2)         =", ub)

        if (ub is not None) and (lb is not None):
            print("gap_UB_minus_LB =", float(ub) - float(lb))

        b1 = res.get("leakage_U1_restriction_best", None)
        b2 = res.get("leakage_U2_truncation_best", None)
        if b1 is not None:
            print("U1_best_meta =", b1)
        if b2 is not None:
            print("U2_best_meta =", b2)


    _use_gurobi = True
    if _use_gurobi == True:
        beam_m_hist = res.get("best_m_hist", None)

        # Lazy import so the rest of the code runs without gurobipy installed.
        from alg.solver_baseline_gurobi import (
            SolverBaselineParams,
            solve_worst_sequence_miqcp_gurobi,
        )

        inst = EXAMPLES[args.example]()
        H = inst["H"]
        K, D, B = int(inst["K"]), int(inst["D"]), int(inst["B"])
        eta, T = float(inst["eta"]), int(inst["T"])
        z0 = np.asarray(inst["z0"], dtype=float)

        obj = QuadraticObjective(H=H)
        system = QuadraticASGDSystem(
            objective=obj,
            K=K,
            D=D,
            eta=eta,
            seed=args.seed,
            init_subspace_mode="none",
            init_subspace_dim=0,
        )
        remodeling = system.make_remodeling()

        gp = SolverBaselineParams(
            time_limit=float(600),
            mip_gap=float(1e-8),
            output_flag=int(1),
            bound_scale=float(2.0),
            basis="original",
            bound_mode="global",
            m_encoding="hist_miqcp",
        )
        res = solve_worst_sequence_miqcp_gurobi(
            system=system,
            action_space=action_space,
            z0=z0,
            T=T,
            params=gp,
        )

        print("=== gurobi baseline result ===")
        print(f"{"[warmstart]":12s}: {False}")
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
            # return
        else: 
            # Print sequence (convert histogram->delay vector)
            print(f"=== Best sequence (length T={m_hist.shape[0]}) ===")
            print("Format: t | delay-vector d_t | sum(d_t) | histogram m_t")
            for t in range(m_hist.shape[0]):
                m = np.asarray(m_hist[t], dtype=int).reshape(-1)
                d = np.asarray(action_space.histogram_to_delay(m), dtype=int).reshape(-1)
                s = int(d.sum())
                ok = ("OK" if (np.all((0 <= d) & (d <= D)) and s <= B) else "INVALID")
                print(f"{t:3d} | {d.tolist()} | sum={s:2d}/{B:2d} | m={m.tolist()} | {ok}")

        # Warm start with beam results
        res = solve_worst_sequence_miqcp_gurobi(
            system=system,
            action_space=action_space,
            z0=z0,
            T=T,
            params=gp,
            warm_start_m_hist=beam_m_hist,
        )

        print("=== gurobi baseline result ===")
        print(f"{"[warmstart]":12s}: {True}")
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
            # return
        else:
            # Print sequence (convert histogram->delay vector)
            print(f"=== Best sequence (length T={m_hist.shape[0]}) ===")
            print("Format: t | delay-vector d_t | sum(d_t) | histogram m_t")
            for t in range(m_hist.shape[0]):
                m = np.asarray(m_hist[t], dtype=int).reshape(-1)
                d = np.asarray(action_space.histogram_to_delay(m), dtype=int).reshape(-1)
                s = int(d.sum())
                ok = ("OK" if (np.all((0 <= d) & (d <= D)) and s <= B) else "INVALID")
                print(f"{t:3d} | {d.tolist()} | sum={s:2d}/{B:2d} | m={m.tolist()} | {ok}")

if __name__ == "__main__":
    main()

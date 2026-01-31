#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from core.action_space import ActionSpace
from core.schedules import StaticSchedule
from examples.quadratic_asgd import QuadraticASGDSystem, QuadraticObjective
from alg.synthesize import SynthesizeParams, synthesize_schedule
from core.subspace import make_basis, lift_basis


# -------------------------
# H generators
# -------------------------
def make_spd_H(n: int, condition: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.linspace(1.0, float(condition), n)
    H = Q @ np.diag(eigs) @ Q.T
    H = 0.5 * (H + H.T)
    return H


def make_iso_H(n: int, scale: float = 1.0) -> np.ndarray:
    return float(scale) * np.eye(n, dtype=float)


def make_diag_H(vals: List[float]) -> np.ndarray:
    return np.diag(np.array(vals, dtype=float))


def build_H(cfg: Dict[str, Any]) -> np.ndarray:
    n = int(cfg["n"])
    kind = cfg["H_kind"]
    hp = cfg["H_params"]

    if kind == "iso":
        return make_iso_H(n, scale=float(hp.get("scale", 1.0)))
    if kind == "diag":
        vals = hp["vals"]
        assert len(vals) == n, f"diag vals length {len(vals)} != n {n}"
        return make_diag_H([float(x) for x in vals])
    if kind == "rot_spd":
        return make_spd_H(n, condition=float(hp["condition"]), seed=int(hp["seed"]))
    raise ValueError(f"Unknown H_kind: {kind}")


# -------------------------
# Helpers
# -------------------------
def _np_to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def _extract_best_static_from_meta(meta: Dict[str, Any]) -> Optional[np.ndarray]:
    ss = meta.get("static_search", None)
    if not isinstance(ss, dict):
        return None
    if "best_d" not in ss:
        return None
    return np.asarray(ss["best_d"], dtype=int)


def _eval_static_JT(system: Any, d: np.ndarray, T: int) -> float:
    return float(system.eval_JT(StaticSchedule(np.asarray(d, dtype=int)), int(T)))


def _classify(
    cfg: Dict[str, Any],
    result: Any,
    system: Any,
    T: int,
    gap_min: float,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    branch = getattr(result, "branch", None)
    meta = getattr(result, "meta", {}) or {}

    if branch == "uniform_certified":
        return ("uniform_global", {})

    if branch == "static_certified":
        return ("static_global", {})

    # non_static: allow ANY H_kind when we are using subspace sweeps;
    # keep the old "rot_spd-only" restriction only when no subspace is used.
    sub_mode = cfg.get("init_subspace_mode", "none")
    if sub_mode == "none" and cfg.get("H_kind") != "rot_spd":
        return None

    Sdiag = meta.get("S", None)
    NotS = meta.get("NotS", None) or meta.get("not_static", None) or meta.get("NotStatic", None)

    s_failed = bool(Sdiag.get("failed", False)) if isinstance(Sdiag, dict) else True
    nots_cert = bool(NotS.get("not_static_certified", False)) if isinstance(NotS, dict) else False

    if not (s_failed and nots_cert and branch == "bb"):
        return None

    best_d = _extract_best_static_from_meta(meta)
    if best_d is None:
        return None

    JT_best_static = _eval_static_JT(system, best_d, T)
    JT_bb = float(result.JT)

    gap = (JT_bb / JT_best_static) - 1.0 if JT_best_static > 0 else float("inf")
    if gap < gap_min:
        return None

    return (
        "non_static",
        {
            "gap": float(gap),
            "JT_best_static": float(JT_best_static),
            "best_static_d": best_d.tolist(),
        },
    )


def _record_example(
    cfg_with_bucket: Dict[str, Any],
    result: Any,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    meta = getattr(result, "meta", {}) or {}

    rec: Dict[str, Any] = {
        "bucket": cfg_with_bucket["bucket"],
        "cfg": {k: _np_to_jsonable(v) for k, v in cfg_with_bucket.items() if k != "bucket"},
        "result": {
            "branch": result.branch,
            "JT": float(result.JT),
            "meta_keys": list(meta.keys()),
        },
        "cert": {
            "S": meta.get("S", None),
            "U": meta.get("U", None),
            "NotS": meta.get("NotS", None),
        },
        "extra": {k: _np_to_jsonable(v) for k, v in extra.items()},
    }

    sch = result.schedule
    if hasattr(sch, "d"):
        rec["schedule"] = {"type": "static", "d": sch.d.tolist(), "sum": int(np.sum(sch.d))}
    else:
        word = sch.word
        rec["schedule"] = {
            "type": "word",
            "L": len(word),
            "word": [a.tolist() for a in word],
            "sums": [int(np.sum(a)) for a in word],
        }

    if "static_search" in meta and isinstance(meta["static_search"], dict):
        rec["static_search"] = {k: _np_to_jsonable(v) for k, v in meta["static_search"].items()}

    return rec


# -------------------------
# Candidate config generator (base configs, before subspace sweep)
# -------------------------
def candidate_configs() -> Iterator[Dict[str, Any]]:
    dims_priority = [4, 3, 2, 1]

    # 1) Uniform-global candidates: isotropic H, small action space
    for n in dims_priority:
        for K, D, B in [
            (2, 1, 1),
            (2, 1, 2),
            (3, 1, 1),
            (3, 1, 2),
        ]:
            for eta in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
                for T in [5, 10, 20, 50]:
                    yield dict(
                        family="uniform_iso_small",
                        n=n,
                        H_kind="iso",
                        H_params=dict(scale=1.0),
                        seed=0,
                        cond=1.0,
                        K=K,
                        D=D,
                        B=B,
                        eta=eta,
                        T=T,
                    )

    # 2) Static-global candidates: diagonal anisotropy, small action space
    for n in dims_priority:
        for c in [3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            vals = [1.0] * max(1, n - 1) + [float(c)]
            vals = (vals + [1.0] * n)[:n]
            vals[-1] = float(c)

            for K, D, B in [
                (2, 1, 1),
                (2, 1, 2),
                (3, 1, 1),
                (3, 1, 2),
            ]:
                for eta in [0.002, 0.005, 0.01, 0.02, 0.05]:
                    for T in [5, 10, 20, 50]:
                        yield dict(
                            family="static_diag_small",
                            n=n,
                            H_kind="diag",
                            H_params=dict(vals=vals),
                            seed=0,
                            cond=c,
                            K=K,
                            D=D,
                            B=B,
                            eta=eta,
                            T=T,
                        )

    # 3) Non-static candidates: rotated SPD, your known "easy non-static" space
    for n in [4, 3, 5, 2, 1]:
        for cond in [10.0, 20.0, 50.0, 100.0]:
            for seed in [0, 1, 2, 3, 4, 5]:
                for eta in [0.03, 0.05, 0.07, 0.1, 0.12]:
                    yield dict(
                        family="nonstatic_rot",
                        n=n,
                        H_kind="rot_spd",
                        H_params=dict(condition=cond, seed=seed),
                        seed=seed,
                        cond=cond,
                        K=4,
                        D=3,
                        B=6,
                        eta=eta,
                        T=50,
                    )


def subspace_sweep_specs(n: int) -> List[Tuple[str, int]]:
    """
    Return list of (mode, r) in the requested order:
      r = n-1, n-2, ..., 1
      mode order per r: mask then ones_perp
    """
    specs: List[Tuple[str, int]] = []
    if n < 2:
        return specs
    for r in range(n - 1, 0, -1):
        specs.append(("mask", r))
        specs.append(("ones_perp", r))
    return specs


def compatible_with_strict_invariance(H_kind: str, mode: str) -> bool:
    """
    For strict 'reachable subspace' semantics:
      - mask requires H diagonal (we accept iso or diag families; rot_spd is NOT diagonal)
      - ones_perp requires H scalar*I (we accept iso only)
    """
    H_kind = str(H_kind)
    mode = str(mode).lower()
    if mode == "mask":
        return H_kind in ("iso", "diag")
    if mode == "ones_perp":
        return H_kind == "iso"
    return False


# -------------------------
# Main collection loop
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="buckets_quadratic_6.jsonl")
    ap.add_argument("--need_per_bucket", type=int, default=2)
    ap.add_argument("--max_trials", type=int, default=6000)
    ap.add_argument("--gap_min", type=float, default=0.05)

    ap.add_argument("--Lmax", type=int, default=4)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--cand", type=int, default=50)

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    targets = {
        "uniform_global": args.need_per_bucket,
        "static_global": args.need_per_bucket,
        "non_static": args.need_per_bucket,
    }
    found: Dict[str, List[Dict[str, Any]]] = {k: [] for k in targets}

    def done() -> bool:
        return all(len(found[k]) >= targets[k] for k in targets)

    out_f = open(args.out, "w", encoding="utf-8")

    trials = 0

    for base_cfg in candidate_configs():
        if done() or trials >= args.max_trials:
            break

        n = int(base_cfg["n"])

        # Decide sweep list:
        # - rot_spd: only run in full-space (no subspace), because invariance doesn't hold
        # - iso/diag: run subspace sweep from n-1 down to 1, both constructions
        if base_cfg["H_kind"] == "rot_spd":
            cfg_list = [dict(base_cfg, init_subspace_mode="none", init_subspace_dim=0)]
        else:
            cfg_list = []
            for mode, r in subspace_sweep_specs(n):
                if not compatible_with_strict_invariance(base_cfg["H_kind"], mode):
                    continue
                cfg_list.append(dict(base_cfg, init_subspace_mode=mode, init_subspace_dim=r))

        for cfg in cfg_list:
            if done() or trials >= args.max_trials:
                break
            trials += 1

            # Build H
            try:
                H = build_H(cfg)
            except Exception as e:
                if args.verbose:
                    print(f"[trial {trials}] build_H exception: {repr(e)} cfg={cfg}")
                continue

            obj = QuadraticObjective(H=H)

            # System with strict init-state projection
            system = QuadraticASGDSystem(
                objective=obj,
                K=int(cfg["K"]),
                D=int(cfg["D"]),
                eta=float(cfg["eta"]),
                seed=int(cfg["seed"]),
                init_subspace_mode=str(cfg.get("init_subspace_mode", "none")),
                init_subspace_dim=int(cfg.get("init_subspace_dim", 0)),
            )

            action_space = ActionSpace(K=int(cfg["K"]), D=int(cfg["D"]), B=int(cfg["B"]))
            remodeling = system.make_remodeling()

            # Params
            params = SynthesizeParams(T=int(cfg["T"]), B=int(cfg["B"]))
            params.bb_params.L_max = int(args.Lmax)
            params.bb_params.beam_width = int(args.beam)
            params.bb_params.candidates_per_expand = int(args.cand)
            params.bb_params.seed = int(cfg["seed"])
            params.run_not_s_check = True  # per your synth: only runs if S FAIL

            # Attach certificate subspace basis (lifted)
            mode = str(cfg.get("init_subspace_mode", "none")).lower()
            r = int(cfg.get("init_subspace_dim", 0))
            if mode != "none" and r > 0:
                U = make_basis(n=int(cfg["n"]), r=r, mode=mode)
                U_lift = lift_basis(U, D=int(cfg["D"]))
                params.static_params.subspace_basis = U_lift
                params.static_params.subspace_name = f"{mode}(n={cfg['n']}, r={r})"

            # Run
            try:
                result = synthesize_schedule(system, remodeling, action_space, params)
            except Exception as e:
                if args.verbose:
                    print(f"[trial {trials}] synthesize_schedule exception: {repr(e)} cfg={cfg}")
                continue

            cls = _classify(cfg, result, system, int(cfg["T"]), gap_min=float(args.gap_min))
            if cls is None:
                if args.verbose:
                    print(f"[trial {trials}] miss: branch={result.branch} family={cfg.get('family')} n={cfg['n']} sub={cfg.get('init_subspace_mode')}/{cfg.get('init_subspace_dim')}")
                continue

            bucket, extra = cls
            if len(found[bucket]) >= targets[bucket]:
                continue

            cfg_with_bucket = dict(cfg)
            cfg_with_bucket["bucket"] = bucket
            rec = _record_example(cfg_with_bucket, result, extra)

            found[bucket].append(rec)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"[HIT {bucket}] {len(found[bucket])}/{targets[bucket]}  "
                f"branch={result.branch} JT={float(result.JT):.6f}  "
                f"family={cfg.get('family')} n={cfg['n']} "
                f"K={cfg['K']} D={cfg['D']} B={cfg['B']} eta={cfg['eta']} T={cfg['T']} "
                f"H_kind={cfg['H_kind']} sub={cfg.get('init_subspace_mode')}/{cfg.get('init_subspace_dim')}"
            )

    out_f.close()

    print("\n=== Summary ===")
    print(f"trials: {trials}")
    for b in targets:
        print(f"{b}: {len(found[b])}/{targets[b]}")
    print(f"saved to: {args.out}")

    if not done():
        print("\nNot all buckets filled.")
        print("Notes:")
        print("- Subspace sweep is strict: mask requires iso/diag; ones_perp requires iso; rot_spd is only tried with subspace=none.")
        print("- If uniform_global/static_global remain 0 even under subspaces, it likely means dominators do not exist on those subspaces for your action spaces.")


if __name__ == "__main__":
    main()

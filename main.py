# main.py
from __future__ import annotations

import argparse
import numpy as np

from core.action_space import ActionSpace
from alg.synthesize import SynthesizeParams, synthesize_schedule
from examples.quadratic_asgd import QuadraticASGDSystem, QuadraticObjective
from core.schedules import StaticSchedule
from core.subspace import make_basis, lift_basis

from alg.static_search import static_worst_search
from alg.bb_search import BBSearchParams, bb_search
from core.schedules import WordSchedule

from alg.exhaustive_search import ExhaustiveSearchParams, exhaustive_sequence_search


def make_spd_H(n: int, condition: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.linspace(1.0, condition, n)
    H = Q @ np.diag(eigs) @ Q.T
    H = 0.5 * (H + H.T)
    return H


def make_diag_mask_last3_H(cond_last: float) -> np.ndarray:
    # H = diag(1,1,1,cond_last) (4D)
    return np.diag([1.0, 1.0, 1.0, float(cond_last)])


def make_iso_H(scale: float, n: int = 4) -> np.ndarray:
    return float(scale) * np.eye(n, dtype=float)


def subspace_basis_4d(mode: str) -> np.ndarray:
    """
    Return U ∈ R^{4×3} with orthonormal columns spanning the desired 3D subspace.

    mode:
      - "mask_last3": span(e1,e2,e3)  <=> x4 = 0
      - "ones_perp3": {x : sum_i x_i = 0} = 1^⊥
    """
    if mode == "mask_last3":
        U = np.zeros((4, 3), dtype=float)
        U[0, 0] = 1.0
        U[1, 1] = 1.0
        U[2, 2] = 1.0
        return U  # already orthonormal

    if mode == "ones_perp3":
        V = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, -1.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )
        # Orthonormalize (QR)
        Q, _ = np.linalg.qr(V)
        return Q[:, :3]

    raise ValueError(f"Unknown subspace mode: {mode}")


def lift_subspace_basis(U: np.ndarray, D: int) -> np.ndarray:
    """
    Lift U (n×r) to U_lift (m×((D+1)r)) in history-stack state s = vec([x_t, x_{t-1},...,x_{t-D}]).
    We use kron(I_{D+1}, U).
    """
    I = np.eye(D + 1, dtype=float)
    return np.kron(I, U)


def try_set_system_initial_x(system, x0: np.ndarray) -> bool:
    """
    Best-effort: try to force the system to use an initial x0 in the chosen subspace.
    This is OPTIONAL but recommended for full semantic alignment with your "initial set ⊂ subspace" statement.
    """
    # common patterns
    if hasattr(system, "set_x0"):
        try:
            system.set_x0(x0)
            return True
        except Exception:
            pass

    for attr in ("x0", "x_init", "init_x", "init_x0"):
        if hasattr(system, attr):
            try:
                setattr(system, attr, x0)
                return True
            except Exception:
                pass

    # If QuadraticASGDSystem supports passing x0 at construction, this won't help here,
    # but we keep this hook in case your class exposes a setter/attribute.
    return False


def _pretty_print_schedule(schedule) -> None:
    if hasattr(schedule, "d"):
        d = schedule.d
        print("schedule: static d =", d.tolist(), "sum=", int(np.sum(d)))
    else:
        word = schedule.word
        print("schedule: word length =", len(word))
        for i, a in enumerate(word):
            print(f"  a[{i}] = {a.tolist()} sum={int(np.sum(a))}")


def _pretty_print_S_diag(meta: dict) -> None:
    Sdiag = meta.get("S", None)
    if Sdiag is None:
        print("S run: False")
        return
    print("S run: True")
    print("S pass:", not Sdiag.get("failed", False))
    print("S diagnostics:", Sdiag)


def _pretty_print_U_diag(meta: dict) -> None:
    Udiag = meta.get("U", None)
    if Udiag is None:
        print("U run: False (skipped because S failed)")
        return
    print("U run: True")
    print("U pass:", not Udiag.get("failed", False))
    print("U diagnostics:", Udiag)


def _pretty_print_pointwise_cert(meta: dict) -> None:
    C = meta.get("PointwiseStaticCert", None)
    if C is None:
        print("Pointwise static certificate: not run")
        return

    # 兼容两种存法：直接存整包 or 只存 diagnostics
    if isinstance(C, dict) and "pass" in C:
        passed = bool(C.get("pass", False))
        diag = C.get("diagnostics", {}) or {}
    else:
        diag = C if isinstance(C, dict) else {}
        passed = not bool(diag.get("failed", True))

    print("Pointwise static certificate: run")
    print("Pointwise pass:", passed)
    print("Pointwise diagnostics:", diag)


def _get_not_s_diag(meta: dict):
    return (
        meta.get("NotS", None)
        or meta.get("notS", None)
        or meta.get("not_static", None)
        or meta.get("NotStatic", None)
        or meta.get("notStatic", None)
    )


def _pretty_print_NotS_diag(meta: dict, schedule) -> None:
    NotS = _get_not_s_diag(meta)
    if NotS is None:
        print("Not-S run: False")
        return

    print("Not-S run: True")

    not_static_certified = bool(NotS.get("not_static_certified", False))
    print("Not-S certified (static impossible):", not_static_certified)

    diag = NotS.get("diagnostics", {}) or {}
    failed = bool(diag.get("failed", False))
    reason = diag.get("reason", None)
    print("Not-S diagnostics.failed:", failed)
    print("Not-S diagnostics.reason :", reason)

    details = (diag.get("details", {}) or {})
    if "certificate" in details:
        print("Not-S certificate:", details["certificate"])
    if "note" in details:
        print("Not-S note:", details["note"])
    if "num_actions_checked" in details:
        print("Not-S num_actions_checked:", int(details["num_actions_checked"]))
    if "exception" in details:
        print("Not-S exception:", details["exception"])

    wp = details.get("witness_pair", None)
    if wp is not None:
        print("Not-S witness_pair:")
        print("  a_candidate :", wp.get("a_candidate", None))
        print("  a_competitor:", wp.get("a_competitor", None))
        if "min_eig(M_candidate - M_competitor)" in wp:
            print("  min_eig(M_candidate - M_competitor):", float(wp["min_eig(M_candidate - M_competitor)"]))

    doms = details.get("dominators", None) or NotS.get("dominators", None)
    if doms:
        doms_list = []
        for x in doms:
            if hasattr(x, "tolist"):
                doms_list.append(x.tolist())
            else:
                doms_list.append(x)
        print("Not-S 2-step dominators (exist):")
        for i, d in enumerate(doms_list):
            print(f"  dom[{i}] =", d, "sum=", int(np.sum(d)))

    if not_static_certified and hasattr(schedule, "d"):
        print("WARNING: Not-S certified static-impossible, but returned schedule is static."
              " Check pipeline ordering/branch logic.")

import numpy as np

def _hist_from_delay(d: np.ndarray, D: int):
    d = np.asarray(d, dtype=int).reshape(-1)
    m = np.zeros(D + 1, dtype=int)
    for x in d:
        m[int(x)] += 1
    return m

def print_best_sequence(res, K: int, D: int, B: int, max_lines: int = 200):
    """
    res: output dict from exhaustive_sequence_search (or bb_search if it returns best_word similarly)
    """
    word = res.get("best_word", None)
    if not word:
        print("[best_sequence] <empty>")
        return

    T = len(word)
    print("=== Best sequence (length T=%d) ===" % T)
    print("best_JT:", res.get("best_JT", None))
    print("num_actions:", res.get("num_actions", None), "visited_nodes:", res.get("visited_nodes", None), "complete:", res.get("complete", None))
    print("Format: t | delay-vector d_t | sum(d_t) | histogram m_t")

    for t, d in enumerate(word):
        if t >= max_lines:
            print(f"... (truncated after {max_lines} steps)")
            break
        d = np.asarray(d).astype(int).reshape(-1)
        if d.shape[0] != K:
            print(f"{t:3d} | <bad shape {d.shape}>")
            continue
        s = int(d.sum())
        m = _hist_from_delay(d, D)
        ok = ("OK" if (np.all((0 <= d) & (d <= D)) and s <= B) else "INVALID")
        print(f"{t:3d} | {d.tolist()} | sum={s:2d}/{B:2d} | m={m.tolist()} | {ok}")

def print_best_sequence_compact(res, D: int):
    """
    Compact view: print only histogram per step (useful when K is large).
    """
    word = res.get("best_word", None)
    if not word:
        print("[best_sequence] <empty>")
        return
    print("=== Best sequence (compact histogram view) ===")
    for t, d in enumerate(word):
        d = np.asarray(d).astype(int).reshape(-1)
        m = _hist_from_delay(d, D)
        print(f"{t:3d}: m={m.tolist()}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--cond", type=float, default=10.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--B", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--init_subspace_mode",
        type=str,
        default="none",
        choices=["none", "mask", "ones_perp"],
        help="Initial-set subspace mode for certificates & init_state projection.",
    )
    parser.add_argument(
        "--init_subspace_dim",
        type=int,
        default=0,
        help="Subspace dimension r. If 0, use n-1 for ones_perp/mask (when applicable).",
    )

    # pointwise sampling controls
    parser.add_argument("--num_samples", type=int, default=1, help="pointwise: number of x0 samples from initial set")
    parser.add_argument("--x0_seed", type=int, default=0, help="pointwise: RNG seed for sampling x0 from initial set")
    parser.add_argument("--x0_radius", type=float, default=1.0, help="pointwise: L2-ball radius for initial set sampling")


    # BB-search params
    parser.add_argument("--Lmax", type=int, default=4)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--cand", type=int, default=50)

    parser.add_argument("--no_not_s", action="store_true", help="Disable Not-S certificate (if supported).")

    parser.add_argument(
        "--mode",
        type=str,
        default="global",
        choices=["global", "pointwise"],
        help="global: run S/U/NotS global-cert pipeline; pointwise: run pointwise-cert pipeline on a fixed init x0",
    )

    parser.add_argument("--point_relax_vertices", action="store_true", help="pointwise: use continuous polytope vertices (safe upper bound)")
    parser.add_argument("--point_no_cache_rho", action="store_true", help="pointwise: do not cache rho_lambda")
    parser.add_argument("--point_margin", type=float, default=1e-12, help="pointwise: compare margin for certificate")
    parser.add_argument("--point_no_skip_bb", action="store_true", help="pointwise: do not skip bb even if pointwise cert passes")


    args = parser.parse_args()

    # Decide subspace dimension r
    mode = args.init_subspace_mode.lower()
    r = int(args.init_subspace_dim)
    if mode != "none":
        if r <= 0:
            # default sweep-like choice: start from n-1
            r = max(1, args.n - 1)
        if mode == "ones_perp" and r > args.n - 1:
            raise ValueError(f"ones_perp requires r<=n-1, got n={args.n}, r={r}")
    else:
        r = 0

    # Build H
    # IMPORTANT: for strict "reachable subspace" semantics, we only support:
    #   - ones_perp: H must be scalar*I (so ANY linear subspace is invariant)
    #   - mask: H must be diagonal (so coordinate-masked subspace is invariant)
    if mode == "ones_perp":
        H = float(args.cond) * np.eye(args.n, dtype=float)
    elif mode == "mask":
        # diagonal SPD, mild anisotropy controlled by cond
        eigs = np.linspace(1.0, float(args.cond), args.n)
        H = np.diag(eigs)
    else:
        H = make_spd_H(args.n, args.cond, args.seed)

    # Build subspace basis for certificates (lifted state)
    U_lift = None
    subspace_name = None
    if mode != "none":
        U = make_basis(n=args.n, r=r, mode=mode)          # (n x r)
        U_lift = lift_basis(U, D=args.D)                  # ((D+1)n x (D+1)r)
        subspace_name = f"{mode}(n={args.n}, r={r})"
        print(f"[init_subspace] enabled: {subspace_name} | U_lift shape={U_lift.shape}")


    # ----- build example system -----
    obj = QuadraticObjective(H=H)
    system = QuadraticASGDSystem(
        objective=obj,
        K=args.K,
        D=args.D,
        eta=args.eta,
        seed=args.seed,
        init_subspace_mode=mode,
        init_subspace_dim=r,
    )



    # OPTIONAL: nothing to do here anymore.
    # Initial-set subspace semantics are enforced inside QuadraticASGDSystem.init_state()
    # via (init_subspace_mode, init_subspace_dim). This guarantees rollout + certificates
    # are aligned, so we do NOT inject x0 from main.
    if args.init_subspace_mode != "none" and args.init_subspace_dim > 0:
        print(
            f"[init_state] system will project initial history to subspace: "
            f"mode={args.init_subspace_mode}, r={args.init_subspace_dim}"
        )
    
    # ----- initial set description -----
    if args.mode == "pointwise":
        print("[init_set]", system.describe_initial_set(kind="ball", radius=args.x0_radius))
        print(f"[pointwise] num_samples={args.num_samples} x0_seed={args.x0_seed} (repeatable sampling)")
    else:
        # global mode rollout still uses a single init_state() (Gaussian by default)
        print("[init_set]", system.describe_initial_set(kind="gaussian", radius=None))


    action_space = ActionSpace(K=args.K, D=args.D, B=args.B)
    remodeling = system.make_remodeling()

    if args.mode == "global":
        from alg.synthesize import SynthesizeParams, synthesize_schedule
        # ----- synth params -----
        params = SynthesizeParams(T=args.T, B=args.B)
        params.bb_params.L_max = args.Lmax
        params.bb_params.beam_width = args.beam
        params.bb_params.candidates_per_expand = args.cand
        params.bb_params.seed = args.seed

        if U_lift is not None:
            # your static_search / S-check code must read this field and do U^T Δ U PSD checks
            params.static_params.subspace_basis = U_lift
            params.static_params.subspace_name = subspace_name
            print(f"[init_subspace] enabled: {subspace_name} | U_lift shape={U_lift.shape}")

        # Enable Not-S check if supported
        if args.no_not_s:
            if hasattr(params, "run_not_s_check"):
                params.run_not_s_check = False
        else:
            if hasattr(params, "run_not_s_check"):
                params.run_not_s_check = True

        # ----- run -----
        result = synthesize_schedule(system, remodeling, action_space, params)

        # ----- print -----
        print("=== Result ===")
        print("branch:", result.branch)
        print("JT:", result.JT)
        print("meta keys:", list(result.meta.keys()))
        print("first 10 losses:", [float(x) for x in result.losses[:10]])
        print("last 10 losses:", [float(x) for x in result.losses[-10:]])

        _pretty_print_schedule(result.schedule)

        _pretty_print_S_diag(result.meta)
        _pretty_print_U_diag(result.meta)
        _pretty_print_NotS_diag(result.meta, result.schedule)
        # _pretty_print_pointwise_cert(result.meta)

        # ----- Compare static baselines -----
        d_bal = action_space.balanced(min(args.B, args.K * args.D))
        d_zero = np.zeros(args.K, dtype=int)

        if hasattr(result.schedule, "d"):
            d_worst_static = result.schedule.d
        else:
            d_worst_static = None
            if "static_search" in result.meta and isinstance(result.meta["static_search"], dict):
                if "best_d" in result.meta["static_search"]:
                    d_worst_static = np.asarray(result.meta["static_search"]["best_d"], dtype=int)

        baselines = [("balanced", d_bal), ("all_zero", d_zero)]
        if d_worst_static is not None:
            baselines.insert(0, ("worst_static", d_worst_static))

        print("=== Static baselines ===")
        for name, d in baselines:
            JT = system.eval_JT(StaticSchedule(d), args.T)
            print(f"{name:12s} d={d.tolist()} sum={int(np.sum(d))}  JT={JT:.6f}")


    else:
        from alg.synthesize_pointwise import PointwiseSynthesizeParams, synthesize_schedule_pointwise
        from alg.pointwise_static_cert import PointwiseStaticCertParams

        rng = np.random.default_rng(args.x0_seed)

        pcert = PointwiseStaticCertParams(
            margin=float(args.point_margin),
            relax_vertices=bool(args.point_relax_vertices),
            cache_rho=not bool(args.point_no_cache_rho),
        )
        params = PointwiseSynthesizeParams(T=args.T, B=args.B, point_cert_params=pcert)
        params.skip_bb_if_point_cert_passes = not bool(args.point_no_skip_bb)

        # bb 参数沿用你已有的 args.Lmax/beam/cand/seed
        params.bb_params.L_max = args.Lmax
        params.bb_params.beam_width = args.beam
        params.bb_params.candidates_per_expand = args.cand
        params.bb_params.seed = args.seed

        # ---- counters ----
        branch_counts = {}          # e.g. point_static_certified / point_static / point_bb
        cert_pass = 0               # how many times pointwise cert passes
        total = int(args.num_samples)

        # optional: track mean JT
        JT_by_branch = {}

        # ---- sample loop ----
        for i in range(total):
            # Make BB-search randomness repeatable per-sample (and different across samples)
            params.bb_params.seed = int(args.seed) + int(i)

            # Sample x0 from the initial ball
            if hasattr(system, "sample_init_state_ball"):
                z0 = system.sample_init_state_ball(rng, radius=float(args.x0_radius))
            else:
                # fallback: uniform in L2-ball in R^{(D+1)*n}, then reshape
                n = int(system.objective.n)
                dim = int((system.D + 1) * n)
                v = rng.normal(size=(dim,))
                nv = float(np.linalg.norm(v))
                if nv < 1e-15:
                    v[0] = 1.0
                    nv = 1.0
                v = v / nv
                r = float(args.x0_radius) * float(rng.random() ** (1.0 / dim))
                z0 = (v * r).reshape(system.D + 1, n)

            # Force this x0 for the whole pipeline call
            if hasattr(system, "set_fixed_init_state"):
                system.set_fixed_init_state(z0)
            else:
                raise RuntimeError("Pointwise sampling requires system.set_fixed_init_state(z0). Please add it to QuadraticASGDSystem.")

            # Run pointwise synth for this x0
            res_i = synthesize_schedule_pointwise(system, remodeling, action_space, params)

            # Clear fixed init (important for safety)
            if hasattr(system, "clear_fixed_init_state"):
                system.clear_fixed_init_state()

            # Count branches
            b = getattr(res_i, "branch", "unknown")
            branch_counts[b] = branch_counts.get(b, 0) + 1
            JT_by_branch.setdefault(b, []).append(float(getattr(res_i, "JT", float("nan"))))

            # Count cert pass (robustly read from meta)
            meta = getattr(res_i, "meta", {}) or {}
            pc = meta.get("PointwiseStaticCert", None)
            if isinstance(pc, dict):
                # depending on how you stored it: diagnostics-only or whole dict
                if "pass" in pc:
                    if bool(pc.get("pass", False)):
                        cert_pass += 1
                else:
                    # diagnostics-only: failed==False means pass
                    if not bool(pc.get("failed", True)):
                        cert_pass += 1

            if i <= 10:
                # print("sample0 cert diag:", meta.get("PointwiseStaticCert"))
                print("sample", i, "z0[0,0]=", float(z0[0,0]), "||z0||=", float(np.linalg.norm(z0)))
                res = exhaustive_sequence_search(system, action_space, T=args.T,
                                params=ExhaustiveSearchParams(max_nodes=None, use_histograms=True), z0=z0)
                # print("best_JT:", res["best_JT"], "visited:", res["visited_nodes"], "complete:", res["complete"])
                print_best_sequence(res, K=args.K, D=args.D, B=args.B)

        # ---- summary printing ----
        print("=== Pointwise sampling summary ===")
        print(f"samples={total}  x0_seed={args.x0_seed}  radius={args.x0_radius}")
        print(f"pointwise_cert_pass: {cert_pass}/{total} ({cert_pass/total:.3f})")

        # branch proportions
        for k in sorted(branch_counts.keys()):
            c = branch_counts[k]
            print(f"{k:22s}: {c:5d}  ({c/total:.3f})")

        # mean JT per branch (optional but useful)
        print("=== Mean JT per branch ===")
        for k in sorted(JT_by_branch.keys()):
            arr = np.asarray(JT_by_branch[k], dtype=float)
            print(f"{k:22s}: mean={float(arr.mean()):.6f}")

        # In pointwise mode we don't return a single 'result' like global mode
        result = None

if __name__ == "__main__":
    main()

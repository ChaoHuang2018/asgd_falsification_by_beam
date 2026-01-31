# alg/synthesize.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.results import ScheduleResult
from core.schedules import StaticSchedule, WordSchedule

from alg.uniform_check import UniformCheckParams, check_uniform_worst
from alg.static_search import StaticSearchParams, check_S_area_static_worst_dp, static_worst_search, check_not_static_worst_dp, check_S_area_static_worst_dp_structured, check_not_static_worst_dp_structured
from alg.bb_search import BBSearchParams, bb_search  # keep your existing bb_search


def _eval_schedule(system: Any, schedule: Any, T: int) -> Tuple[float, List[float]]:
    """
    Return (JT, losses). Supports:
      - system.eval_JT(schedule, T, return_losses=True) if implemented
      - system.rollout(schedule, T) returning:
          * dict with 'losses'
          * tuple whose 2nd element is losses
          * RolloutResult dataclass with .losses
    """
    # Preferred: an enhanced eval_JT that returns losses
    if hasattr(system, "eval_JT"):
        try:
            JT, losses = system.eval_JT(schedule, T, return_losses=True)
            return float(JT), [float(x) for x in losses]
        except TypeError:
            # eval_JT exists but doesn't support return_losses
            JT = float(system.eval_JT(schedule, T))

    # Fallback: rollout
    if hasattr(system, "rollout"):
        out = system.rollout(schedule, T)

        # core.system.RolloutResult
        if hasattr(out, "losses"):
            losses = [float(x) for x in out.losses]
            return float(np.sum(losses)), losses

        # dict
        if isinstance(out, dict) and "losses" in out:
            losses = [float(x) for x in out["losses"]]
            return float(np.sum(losses)), losses

        # tuple
        if isinstance(out, tuple) and len(out) >= 2:
            losses = [float(x) for x in out[1]]
            return float(np.sum(losses)), losses

        # If rollout returns something unexpected, at least return JT computed above if available
        if "JT" in locals():
            return float(JT), []

    # Last resort
    if hasattr(system, "eval_JT"):
        JT = float(system.eval_JT(schedule, T))
        return float(JT), []

    raise AttributeError("system must provide eval_JT or rollout to evaluate schedules.")



@dataclass
class SynthesizeParams:
    T: int = 100
    B: int = 0  # must match action_space budget; set from CLI
    uniform_params: UniformCheckParams = field(default_factory=UniformCheckParams)
    static_params: StaticSearchParams = field(default_factory=StaticSearchParams)
    bb_params: BBSearchParams = field(default_factory=BBSearchParams)

    run_not_s_check: bool = True

    # If True, skip BB-search when S-area-T passes (since then static a* is globally worst)
    skip_bb_if_S_area_passes: bool = True


def synthesize_schedule(
    system: Any,
    remodeling: Any,
    action_space: Any,
    params: Optional[SynthesizeParams] = None,
) -> ScheduleResult:
    if params is None:
        params = SynthesizeParams()

    meta: Dict[str, Any] = {}

    # 1) S-area-T (global static worst certificate for area)
    try:
        S = check_S_area_static_worst_dp_structured(system, remodeling, action_space, T=params.T, params=params.static_params)
    except Exception:
        S = check_S_area_static_worst_dp(system, remodeling, action_space, T=params.T, params=params.static_params)

    meta["S"] = S.get("diagnostics", S)

    # --- NEW: Not-S runs ONLY if S FAILS (and enabled) ---
    S_failed = not bool(S.get("pass", False))
    if S_failed and getattr(params, "run_not_s_check", True):
        try:
            # Use keyword args to avoid signature mismatch / positional mistakes
            try:
                NotS = check_not_static_worst_dp_structured(system, remodeling, action_space, params=params.static_params)
            except Exception:
                NotS = check_not_static_worst_dp(system, remodeling, action_space, params=params.static_params)
            meta["NotS"] = NotS

        except Exception as e:
            NotS = {
                "not_static_certified": False,
                "diagnostics": {
                    "failed": True,
                    "reason": "exception",
                    "details": {"exception": repr(e)},
                },
                "dominators": [],
            }
        meta["NotS"] = NotS


    if S.get("pass", False):
        # static is globally worst; now U-area can be used to decide if balanced is that worst static
        U = check_uniform_worst(
            system, action_space, T=params.T, B=params.B, params=params.uniform_params
        )
        meta["U"] = U.get("diagnostics", U)

        if U.get("pass", False):
            # certified chain: S + U => globally worst is uniform balanced(B)
            d = U["witness_d"]
            schedule = StaticSchedule(d)
            JT, losses = _eval_schedule(system, schedule, params.T)
            return ScheduleResult(
                branch="uniform_certified",
                schedule=schedule,
                JT=float(JT),
                losses=losses,
                meta=meta,
            )
        else:
            # S says global worst is static a* (not necessarily balanced)
            d = S["witness_d"]
            schedule = StaticSchedule(d)
            JT, losses = _eval_schedule(system, schedule, params.T)
            return ScheduleResult(
                branch="static_certified",
                schedule=schedule,
                JT=float(JT),
                losses=losses,
                meta=meta,
            )

    # 2) If S-area-T fails, we cannot elevate U-area to a global claim.
    #    Fall back to: static best vs BB-search, pick larger JT.
    Sbest = static_worst_search(system, action_space, params.T)
    meta["static_search"] = Sbest
    static_schedule = StaticSchedule(Sbest["best_d"])
    JT_static, losses_static = _eval_schedule(system, static_schedule, params.T)

    BB = bb_search(system, action_space, params.T, params.bb_params)
    meta["bb_search"] = BB

    # Interpret bb_search output flexibly
    if "best_word" in BB and BB["best_word"] is not None:
        bb_schedule = WordSchedule(BB["best_word"])
    elif "schedule" in BB:
        bb_schedule = BB["schedule"]
    else:
        # no bb result; return best static
        return ScheduleResult(
            branch="static",
            schedule=static_schedule,
            JT=float(JT_static),
            losses=losses_static,
            meta=meta,
        )

    JT_bb, losses_bb = _eval_schedule(system, bb_schedule, params.T)

    if JT_static >= JT_bb:
        return ScheduleResult(
            branch="static",
            schedule=static_schedule,
            JT=float(JT_static),
            losses=losses_static,
            meta=meta,
        )
    else:
        return ScheduleResult(
            branch="bb",
            schedule=bb_schedule,
            JT=float(JT_bb),
            losses=losses_bb,
            meta=meta,
        )


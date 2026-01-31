# alg/synthesize_pointwise.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

from core.results import ScheduleResult
from core.schedules import StaticSchedule, WordSchedule

from alg.bb_search import BBSearchParams, bb_search
from alg.static_search import static_worst_search
from alg.pointwise_static_cert import PointwiseStaticCertParams, certify_pointwise_static_worst_upperbound


def _eval_schedule(system: Any, schedule: Any, T: int) -> Tuple[float, List[float]]:
    # same helper as alg/synthesize.py (copy here to avoid mixing pipelines)
    import numpy as np
    if hasattr(system, "eval_JT"):
        try:
            JT, losses = system.eval_JT(schedule, T, return_losses=True)
            return float(JT), [float(x) for x in losses]
        except TypeError:
            JT = float(system.eval_JT(schedule, T))

    if hasattr(system, "rollout"):
        out = system.rollout(schedule, T)
        if hasattr(out, "losses"):
            losses = [float(x) for x in out.losses]
            return float(sum(losses)), losses
        if isinstance(out, dict) and "losses" in out:
            losses = [float(x) for x in out["losses"]]
            return float(sum(losses)), losses
        if isinstance(out, tuple) and len(out) >= 2:
            losses = [float(x) for x in out[1]]
            return float(sum(losses)), losses

    return float(JT), []


@dataclass
class PointwiseSynthesizeParams:
    T: int = 100
    B: int = 0
    bb_params: BBSearchParams = field(default_factory=BBSearchParams)
    point_cert_params: PointwiseStaticCertParams = field(default_factory=PointwiseStaticCertParams)

    # When pointwise certificate passes, skip BB-search
    skip_bb_if_point_cert_passes: bool = True


def synthesize_schedule_pointwise(
    system: Any,
    remodeling: Any,
    action_space: Any,
    params: Optional[PointwiseSynthesizeParams] = None,
) -> ScheduleResult:
    if params is None:
        params = PointwiseSynthesizeParams()

    meta: Dict[str, Any] = {}

    # 1) best static for this specific x0 (baseline)
    Sbest = static_worst_search(system, action_space, params.T)
    meta["static_search"] = Sbest
    static_schedule = StaticSchedule(Sbest["best_d"])
    JT_static, losses_static = _eval_schedule(system, static_schedule, params.T)

    # 2) pointwise sufficient certificate (independent module)
    Cert = certify_pointwise_static_worst_upperbound(
        system, remodeling, action_space, T=params.T, params=params.point_cert_params
    )
    meta["PointwiseStaticCert"] = Cert.get("diagnostics", Cert)

    if Cert.get("pass", False) and params.skip_bb_if_point_cert_passes:
        return ScheduleResult(
            branch="point_static_certified",
            schedule=StaticSchedule(Cert["witness_d"]),
            JT=float(JT_static),
            losses=losses_static,
            meta=meta,
        )

    # 3) fallback: BB-search and compare
    BB = bb_search(system, action_space, params.T, params.bb_params)
    meta["bb_search"] = BB

    if "best_word" in BB and BB["best_word"] is not None:
        bb_schedule = WordSchedule(BB["best_word"])
    elif "schedule" in BB:
        bb_schedule = BB["schedule"]
    else:
        return ScheduleResult(
            branch="point_static",
            schedule=static_schedule,
            JT=float(JT_static),
            losses=losses_static,
            meta=meta,
        )

    JT_bb, losses_bb = _eval_schedule(system, bb_schedule, params.T)

    if JT_static >= JT_bb:
        return ScheduleResult(
            branch="point_static",
            schedule=static_schedule,
            JT=float(JT_static),
            losses=losses_static,
            meta=meta,
        )
    else:
        return ScheduleResult(
            branch="point_bb",
            schedule=bb_schedule,
            JT=float(JT_bb),
            losses=losses_bb,
            meta=meta,
        )

"""COLREGs Maritime Collision Risk & Search-and-Rescue (SAR) Priority Evaluator.

Evaluates detected marine targets using International Regulations for Preventing
Collisions at Sea (COLREGs, 1972) and IAMSAR / SOLAS Search-and-Rescue criteria
based on target classification, optical azimuth bearing angle (deg), and
monocular estimated range (m).
"""

import math
from typing import Any, Dict, List, Optional

RISK_LEVELS = ("CRITICAL", "WARNING", "CAUTION", "NOMINAL")
SAR_PRIORITIES = ("P1_IMMEDIATE_RESCUE", "P2_ACTIVE_ASSIST", "NONE")

# Assumed relative closure speeds (m/s) by canonical target category for TCPA estimation
CLOSING_SPEED_MPS: Dict[str, float] = {
    "swimmer": 0.8,          # Drift / swimmer relative motion
    "rescue_board": 2.5,     # Windsurf / SUP / light craft
    "commercial_ship": 7.5,  # ~14.5 knots commercial vessel
    "motor_craft": 6.0,      # ~11.5 knots power/fishing craft
    "sailing_vessel": 4.0,   # ~7.8 knots under sail
    "navigation_buoy": 1.2,  # Relative drift / mooring watch
    "surface_obstacle": 2.0, # Unclassified floating hazard
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert value to a finite float or return default."""
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def normalize_bearing(bearing_deg: Any) -> float:
    """Normalize azimuth bearing angle into [-180.0, +180.0] degrees (0 = dead ahead, + = starboard, - = port)."""
    b = _safe_float(bearing_deg, 0.0)
    normalized = ((b + 180.0) % 360.0) - 180.0
    if normalized == -180.0 and b > 0:
        return 180.0
    return round(normalized, 2)


def classify_bearing_sector(bearing_deg: Any) -> str:
    """Classify relative azimuth bearing angle into standard maritime watch sectors.

    Conventions:
    - [-6°, +6°]: Dead Ahead (Bow) — COLREGs Rule 14 Head-On sector
    - (+6°, +22.5°]: Starboard Bow — COLREGs Rule 15 Crossing (Give-Way) sector
    - [-22.5°, -6°): Port Bow — COLREGs Rule 15/17 Crossing (Stand-On) sector
    - (+22.5°, +67.5°]: Starboard Beam
    - [-67.5°, -22.5°): Port Beam
    - > +67.5°: Starboard Quarter
    - < -67.5°: Port Quarter
    """
    b = normalize_bearing(bearing_deg)
    if abs(b) <= 6.0:
        return "Dead Ahead (Bow)"
    if 6.0 < b <= 22.5:
        return "Starboard Bow"
    if -22.5 <= b < -6.0:
        return "Port Bow"
    if 22.5 < b <= 67.5:
        return "Starboard Beam"
    if -67.5 <= b < -22.5:
        return "Port Beam"
    if b > 67.5:
        return "Starboard Quarter"
    return "Port Quarter"


def resolve_target_category(class_name: Optional[str]) -> str:
    """Map raw detector class label to a canonical maritime target category."""
    if not class_name or not isinstance(class_name, str):
        return "surface_obstacle"
    low = class_name.strip().lower()
    if any(k in low for k in ("swimmer", "human", "person", "mob", "overboard")):
        return "swimmer"
    if any(k in low for k in ("windsurf", "sup", "board", "kayak", "raft", "float")):
        return "rescue_board"
    if any(k in low for k in ("commercial", "ferry", "cargo", "tanker", "ship", "liner")):
        return "commercial_ship"
    if any(k in low for k in ("sail", "yacht", "catamaran")):
        return "sailing_vessel"
    if any(k in low for k in ("motor", "fishing", "craft", "boat", "vessel", "speedboat")):
        return "motor_craft"
    if any(k in low for k in ("buoy", "marker", "beacon", "aton")):
        return "navigation_buoy"
    return "surface_obstacle"


def evaluate_target_risk(
    class_name: Optional[str],
    bearing_deg: Any,
    est_range_m: Any,
    confidence: Any = 1.0,
) -> Dict[str, Any]:
    """Evaluate COLREGs collision risk and SAR priority for a single detected marine target.

    Args:
        class_name: Target class label from YOLOv11m detector.
        bearing_deg: Optical azimuth angle in degrees (- = Port, 0 = Bow, + = Starboard).
        est_range_m: Monocular estimated distance to target in meters.
        confidence: Detector confidence in [0, 1] (or [0, 100]).

    Returns:
        Dictionary with COLREGs rule, risk score [0..100], risk level, SAR priority,
        sector, CPA/TCPA estimates, and recommended tactical action.
    """
    category = resolve_target_category(class_name)
    b_deg = normalize_bearing(bearing_deg)
    sector = classify_bearing_sector(b_deg)

    raw_range = _safe_float(est_range_m, 150.0)
    range_m = max(1.0, min(2000.0, raw_range if raw_range > 0 else 1.0))

    raw_conf = _safe_float(confidence, 1.0)
    if raw_conf > 1.0:
        raw_conf = raw_conf / 100.0
    conf = max(0.0, min(1.0, raw_conf))

    # Proximity factor in [0.0, 1.0]: exponential decay with range + close-quarters boost
    proximity = math.exp(-range_m / 110.0)
    if range_m <= 35.0:
        proximity = min(1.0, proximity + 0.18)

    # Bearing / CPA geometry factor: targets near dead ahead or starboard bow carry higher collision weight
    abs_b = abs(b_deg)
    bearing_rad = math.radians(min(85.0, abs_b))
    cpa_estimate_m = round(range_m * math.sin(bearing_rad) * 0.65, 1)
    closing_speed = CLOSING_SPEED_MPS.get(category, 2.5)
    axial_closure = max(0.25, closing_speed * math.cos(math.radians(min(75.0, abs_b))))
    tcpa_estimate_s = round(range_m / axial_closure, 1)

    if abs_b <= 6.0:
        bearing_weight = 1.00  # Dead ahead
    elif 6.0 < b_deg <= 25.0:
        bearing_weight = 0.94  # Starboard bow give-way zone
    elif -25.0 <= b_deg < -6.0:
        bearing_weight = 0.84  # Port bow stand-on zone
    elif b_deg > 25.0:
        bearing_weight = 0.72  # Starboard beam/quarter
    else:
        bearing_weight = 0.65  # Port beam/quarter

    # Determine SAR priority, COLREGs rule, base risk score, and tactical action
    is_sar_target = category in ("swimmer", "rescue_board")
    if category == "swimmer":
        sar_priority = "P1_IMMEDIATE_RESCUE"
        colregs_rule = "SOLAS Ch. V Reg. 33 / IAMSAR — Person in Water (MOB)"
        # Person in water is always a high/critical SAR emergency
        base_score = 78.0 + 22.0 * proximity
        recommended_action = (
            f"MAYDAY RELAY / MOB ALERT: Person in water at {range_m:.1f}m ({b_deg:+.1f}° {sector}). "
            "Lock optical tracking, broadcast GPS drift vector & alert Coast Guard SAR."
        )
    elif category == "rescue_board":
        sar_priority = "P2_ACTIVE_ASSIST"
        colregs_rule = "IAMSAR / COLREGs Rule 18 — Light Surface Craft / Rescue Board"
        base_score = 52.0 + 38.0 * (0.75 * proximity + 0.25 * bearing_weight)
        recommended_action = (
            f"SAR WATCH: Light board/craft at {range_m:.1f}m ({b_deg:+.1f}° {sector}). "
            "Maintain continuous visual track and monitor occupant distress signals."
        )
    elif category == "commercial_ship":
        sar_priority = "NONE"
        if abs_b <= 6.0:
            colregs_rule = "COLREGs Rule 14 & 18 — Head-On Commercial Vessel"
            recommended_action = (
                f"COLLISION WARNING: Heavy vessel dead ahead at {range_m:.1f}m (TCPA ~{tcpa_estimate_s:.0f}s). "
                "Alter course to starboard (port-to-port pass) and sound AIS/horn warning."
            )
        elif b_deg > 6.0:
            colregs_rule = "COLREGs Rule 15 & 16 — Starboard Crossing (Give-Way)"
            recommended_action = (
                f"GIVE-WAY ALERT: Commercial ship on starboard ({b_deg:+.1f}°, {range_m:.1f}m). "
                "Take early, substantial action to keep well clear; avoid crossing ahead."
            )
        else:
            colregs_rule = "COLREGs Rule 15 & 17 — Port Crossing (Stand-On)"
            recommended_action = (
                f"STAND-ON WATCH: Commercial ship on port ({b_deg:+.1f}°, {range_m:.1f}m). "
                "Maintain course/speed; maneuver under Rule 17(b) if vessel fails to give way."
            )
        base_score = 28.0 + 70.0 * (0.72 * proximity + 0.28 * bearing_weight)
    elif category == "sailing_vessel":
        sar_priority = "NONE"
        colregs_rule = "COLREGs Rule 12 & 18(a)(iv) — Sailing Vessel Right-of-Way"
        base_score = 22.0 + 66.0 * (0.70 * proximity + 0.30 * bearing_weight)
        recommended_action = (
            f"SAIL PRIORITY: Vessel under sail at {range_m:.1f}m ({b_deg:+.1f}° {sector}). "
            "Power-driven vessels must keep out of the way per COLREGs Rule 18."
        )
    elif category == "motor_craft":
        sar_priority = "NONE"
        if abs_b <= 6.0:
            colregs_rule = "COLREGs Rule 14 — Head-On Power-Driven Craft"
            recommended_action = (
                f"HEAD-ON APPROACH: Motor/fishing craft at {range_m:.1f}m ({b_deg:+.1f}°). "
                "Pass port-to-port and monitor sudden course changes."
            )
        elif b_deg > 6.0:
            colregs_rule = "COLREGs Rule 15 — Starboard Crossing (Give-Way)"
            recommended_action = (
                f"STARBOARD CROSSING: Motor craft at {range_m:.1f}m ({b_deg:+.1f}°). "
                "Give way and pass astern of target."
            )
        else:
            colregs_rule = "COLREGs Rule 15/17 — Port Crossing (Stand-On)"
            recommended_action = (
                f"PORT CONTACT: Motor/fishing craft at {range_m:.1f}m ({b_deg:+.1f}°). "
                "Monitor bearing drift (CPA) and fishing gear clearance."
            )
        base_score = 20.0 + 68.0 * (0.72 * proximity + 0.28 * bearing_weight)
    elif category == "navigation_buoy":
        sar_priority = "NONE"
        colregs_rule = "IALA Maritime Buoyage / COLREGs Rule 5 & 6 — Aid to Navigation"
        base_score = 12.0 + 58.0 * (0.80 * proximity + 0.20 * bearing_weight)
        recommended_action = (
            f"ATON REFERENCE: Navigation buoy at {range_m:.1f}m ({b_deg:+.1f}° {sector}). "
            "Verify channel lateral mark and maintain safe mooring clearance."
        )
    else:
        sar_priority = "NONE"
        colregs_rule = "COLREGs Rule 5 & 6 — Unclassified Surface Obstacle (Look-out & Safe Speed)"
        base_score = 24.0 + 64.0 * (0.75 * proximity + 0.25 * bearing_weight)
        recommended_action = (
            f"SURFACE HAZARD: Obstacle detected at {range_m:.1f}m ({b_deg:+.1f}° {sector}). "
            "Reduce speed and verify target clearance."
        )

    # Confidence modulation (slight damping for very low confidence while preserving safety floor)
    conf_factor = 0.88 + 0.12 * conf
    risk_score = round(max(0.0, min(100.0, base_score * conf_factor)), 1)

    if category == "swimmer" and risk_score < 75.0:
        risk_score = 75.0

    if risk_score >= 75.0:
        risk_level = "CRITICAL"
    elif risk_score >= 52.0:
        risk_level = "WARNING"
    elif risk_score >= 30.0:
        risk_level = "CAUTION"
    else:
        risk_level = "NOMINAL"

    return {
        "category": category,
        "sector": sector,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "is_sar_target": is_sar_target,
        "sar_priority": sar_priority,
        "colregs_rule": colregs_rule,
        "cpa_estimate_m": cpa_estimate_m,
        "tcpa_estimate_s": tcpa_estimate_s,
        "recommended_action": recommended_action,
    }


def evaluate_frame_risk(
    detections: List[Dict[str, Any]],
    roll_deg: float = 0.0,
    pitch_offset_pct: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate and enrich all detections in a frame and produce a tactical COLREGs/SAR summary.

    Mutates each detection dict in `detections` in-place to attach risk fields and returns
    an aggregate frame-level risk assessment dictionary.
    """
    counts_by_level: Dict[str, int] = {lvl: 0 for lvl in RISK_LEVELS}
    counts_by_sector: Dict[str, int] = {}
    sar_targets_count = 0
    collision_watch_count = 0
    highest_target: Optional[Dict[str, Any]] = None
    max_risk_score = 0.0

    for det in detections:
        eval_res = evaluate_target_risk(
            class_name=det.get("class_name"),
            bearing_deg=det.get("bearing_deg", 0.0),
            est_range_m=det.get("est_range_m", 150.0),
            confidence=det.get("confidence", 1.0),
        )
        det.update(eval_res)

        lvl = eval_res["risk_level"]
        sec = eval_res["sector"]
        score = eval_res["risk_score"]

        counts_by_level[lvl] = counts_by_level.get(lvl, 0) + 1
        counts_by_sector[sec] = counts_by_sector.get(sec, 0) + 1

        if eval_res["is_sar_target"]:
            sar_targets_count += 1
        if lvl in ("CRITICAL", "WARNING"):
            collision_watch_count += 1

        if highest_target is None or score > max_risk_score:
            max_risk_score = score
            highest_target = {
                "id": det.get("id"),
                "class_name": det.get("class_name"),
                "category": eval_res["category"],
                "bearing_deg": det.get("bearing_deg"),
                "est_range_m": det.get("est_range_m"),
                "sector": sec,
                "risk_score": score,
                "risk_level": lvl,
                "sar_priority": eval_res["sar_priority"],
                "colregs_rule": eval_res["colregs_rule"],
                "cpa_estimate_m": eval_res["cpa_estimate_m"],
                "tcpa_estimate_s": eval_res["tcpa_estimate_s"],
                "recommended_action": eval_res["recommended_action"],
            }

    if not detections:
        overall_status = "CLEAR"
        advisory = "All optical azimuth sectors clear. Maintain standard COLREGs Rule 5 look-out."
    elif counts_by_level["CRITICAL"] > 0:
        overall_status = "CRITICAL"
        assert highest_target is not None
        advisory = highest_target["recommended_action"]
    elif counts_by_level["WARNING"] > 0:
        overall_status = "WARNING"
        assert highest_target is not None
        advisory = highest_target["recommended_action"]
    elif counts_by_level["CAUTION"] > 0:
        overall_status = "CAUTION"
        assert highest_target is not None
        advisory = highest_target["recommended_action"]
    else:
        overall_status = "NOMINAL"
        advisory = f"{len(detections)} target(s) tracked at safe range. Continue routine radar/optical watch."

    r_deg = abs(_safe_float(roll_deg, 0.0))
    p_pct = abs(_safe_float(pitch_offset_pct, 0.0))
    if r_deg > 8.0 or p_pct > 15.0:
        sea_state_note = (
            f"Elevated platform motion (Roll {roll_deg:+.1f}°, Pitch {pitch_offset_pct:+.1f}%): "
            "apply horizon-stabilized bearing compensation."
        )
    else:
        sea_state_note = (
            f"Optical horizon stable (Roll {roll_deg:+.1f}°, Pitch {pitch_offset_pct:+.1f}%)."
        )

    return {
        "overall_status": overall_status,
        "max_risk_score": round(max_risk_score, 1),
        "total_targets": len(detections),
        "sar_alert_active": sar_targets_count > 0,
        "sar_targets_count": sar_targets_count,
        "collision_watch_count": collision_watch_count,
        "counts_by_level": counts_by_level,
        "counts_by_sector": counts_by_sector,
        "highest_priority_target": highest_target,
        "advisory": advisory,
        "sea_state_stability_note": sea_state_note,
    }


class COLREGsRiskEvaluator:
    """Stateful/stateless evaluator for COLREGs collision risk and SAR priority scoring."""

    def evaluate_target(
        self,
        class_name: Optional[str],
        bearing_deg: Any,
        est_range_m: Any,
        confidence: Any = 1.0,
    ) -> Dict[str, Any]:
        """Evaluate risk for a single target."""
        return evaluate_target_risk(
            class_name=class_name,
            bearing_deg=bearing_deg,
            est_range_m=est_range_m,
            confidence=confidence,
        )

    def evaluate_detections(
        self,
        detections: List[Dict[str, Any]],
        roll_deg: float = 0.0,
        pitch_offset_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluate and enrich all detections in a frame and return summary."""
        return evaluate_frame_risk(
            detections=detections,
            roll_deg=roll_deg,
            pitch_offset_pct=pitch_offset_pct,
        )

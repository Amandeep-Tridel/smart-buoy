"""Oceanographic Spectral Wave Physics, Sea-State Classifications & Rogue Wave Risk Engine.

Provides physics-based oceanographic computations for NOAA NDBC & NERRS buoy telemetry:
1. WMO Code 3700 / Douglas Sea State scale classification (Codes 0–9)
2. WMO Beaufort Wind Force scale classification (Forces 0–12)
3. Significant & Mean Wave Steepness (S_p = 2*pi*H_m0 / (g * T_p^2)) and dispersion kinematics
4. Rogue Wave Crest Risk Index (Rayleigh/Tayfun nonlinear crest statistics & Benjamin-Feir Index)
5. Pierson-Moskowitz (1964) vs. JONSWAP (1973) theoretical spectral comparison & gamma optimization
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

GRAVITY_MPS2: float = 9.80665
EULER_GAMMA: float = 0.5772156649015329
MPS_TO_KNOTS: float = 1.9438444924406

# WMO Code 3700 / Douglas Sea State Table: (code, upper_bound_m, label, range_str, severity)
WMO_SEA_STATE_TABLE: List[Tuple[int, float, str, str, str]] = [
    (0, 0.0, "Calm (glassy)", "0.0 m", "calm"),
    (1, 0.1, "Calm (rippled)", "0.0-0.1 m", "calm"),
    (2, 0.5, "Smooth (wavelets)", "0.1-0.5 m", "calm"),
    (3, 1.25, "Slight", "0.5-1.25 m", "moderate"),
    (4, 2.5, "Moderate", "1.25-2.5 m", "moderate"),
    (5, 4.0, "Rough", "2.5-4.0 m", "rough"),
    (6, 6.0, "Very rough", "4.0-6.0 m", "rough"),
    (7, 9.0, "High", "6.0-9.0 m", "severe"),
    (8, 14.0, "Very high", "9.0-14.0 m", "severe"),
    (9, float("inf"), "Phenomenal", ">14.0 m", "severe"),
]

# WMO Beaufort Wind Scale Table: (force, upper_bound_mps, label, range_mps, advisory)
BEAUFORT_WIND_TABLE: List[Tuple[int, float, str, str, str]] = [
    (0, 0.5, "Calm", "<0.5 m/s", "None"),
    (1, 1.6, "Light air", "0.5-1.5 m/s", "None"),
    (2, 3.4, "Light breeze", "1.6-3.3 m/s", "None"),
    (3, 5.5, "Gentle breeze", "3.4-5.4 m/s", "None"),
    (4, 8.0, "Moderate breeze", "5.5-7.9 m/s", "None"),
    (5, 10.8, "Fresh breeze", "8.0-10.7 m/s", "None"),
    (6, 13.9, "Strong breeze", "10.8-13.8 m/s", "Small Craft Advisory"),
    (7, 17.2, "Near gale", "13.9-17.1 m/s", "Small Craft Advisory"),
    (8, 20.8, "Gale", "17.2-20.7 m/s", "Gale Warning"),
    (9, 24.5, "Strong gale", "20.8-24.4 m/s", "Gale Warning"),
    (10, 28.5, "Storm", "24.5-28.4 m/s", "Storm Warning"),
    (11, 32.7, "Violent storm", "28.5-32.6 m/s", "Storm Warning"),
    (12, float("inf"), "Hurricane force", ">=32.7 m/s", "Hurricane Force Wind Warning"),
]


def _sanitize_non_negative(value: Optional[float], default: float = 0.0) -> float:
    """Convert value to a finite non-negative float, handling None, NaN, and negative values."""
    if value is None:
        return default
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(val) or val < 0.0:
        return default
    return val


def classify_wmo_sea_state(hs_m: Optional[float]) -> Dict[str, Any]:
    """Classify Significant Wave Height H_m0 (m) on the WMO / Douglas Sea State Scale (0–9)."""
    if hs_m is not None:
        try:
            raw_val = float(hs_m)
        except (TypeError, ValueError):
            raw_val = 0.0
    else:
        raw_val = 0.0

    if math.isnan(raw_val) or raw_val <= 0.0:
        code, _, label, range_str, severity = WMO_SEA_STATE_TABLE[0]
        clean_hs = 0.0
    elif math.isinf(raw_val):
        code, _, label, range_str, severity = WMO_SEA_STATE_TABLE[-1]
        clean_hs = raw_val
    else:
        clean_hs = raw_val
        code, _, label, range_str, severity = WMO_SEA_STATE_TABLE[-1]
        for c, upper, lbl, rng, sev in WMO_SEA_STATE_TABLE:
            if clean_hs <= upper:
                code, label, range_str, severity = c, lbl, rng, sev
                break

    return {
        "code": int(code),
        "label": label,
        "range_m": range_str,
        "severity": severity,
        "hs_m": round(clean_hs, 3) if math.isfinite(clean_hs) else clean_hs,
        "description": f"WMO Sea State {code}: {label} ({range_str})",
    }


def classify_beaufort_wind(wspd_mps: Optional[float]) -> Dict[str, Any]:
    """Classify 10-meter wind speed (m/s) on the WMO Beaufort Wind Scale (0–12)."""
    if wspd_mps is not None:
        try:
            raw_val = float(wspd_mps)
        except (TypeError, ValueError):
            raw_val = 0.0
    else:
        raw_val = 0.0

    if math.isnan(raw_val) or raw_val < 0.0:
        clean_u = 0.0
    else:
        clean_u = raw_val

    force, _, label, range_mps, advisory = BEAUFORT_WIND_TABLE[-1]
    if not math.isinf(clean_u):
        for f_code, upper, lbl, rng, adv in BEAUFORT_WIND_TABLE:
            if clean_u < upper:
                force, label, range_mps, advisory = f_code, lbl, rng, adv
                break

    knots = clean_u * MPS_TO_KNOTS
    return {
        "force": int(force),
        "label": label,
        "range_mps": range_mps,
        "advisory": advisory,
        "wspd_mps": round(clean_u, 2) if math.isfinite(clean_u) else clean_u,
        "wspd_knots": round(knots, 2) if math.isfinite(knots) else knots,
        "description": f"Beaufort Force {force}: {label} ({range_mps})",
    }


def compute_wave_steepness(
    hm0_m: Optional[float],
    tp_s: Optional[float],
    tm02_s: Optional[float] = None,
    g: float = GRAVITY_MPS2,
) -> Dict[str, Any]:
    """Compute significant wave steepness S_p = 2*pi*H_m0 / (g * T_p^2) and deep-water dispersion metrics."""
    hs = _sanitize_non_negative(hm0_m, 0.0)
    tp = _sanitize_non_negative(tp_s, 0.0)
    tm02 = _sanitize_non_negative(tm02_s, 0.0)
    g_val = g if (g is not None and math.isfinite(g) and g > 0.0) else GRAVITY_MPS2

    if hs <= 0.0 or tp <= 0.0 or not math.isfinite(hs) or not math.isfinite(tp):
        sp = 0.0
        wavelength_p = 0.0
        wavenumber_kp = 0.0
        phase_speed_cp = 0.0
        group_speed_cg = 0.0
    else:
        wavelength_p = (g_val * (tp ** 2)) / (2.0 * math.pi)
        sp = (2.0 * math.pi * hs) / (g_val * (tp ** 2))
        wavenumber_kp = (2.0 * math.pi) / wavelength_p if wavelength_p > 0.0 else 0.0
        phase_speed_cp = (g_val * tp) / (2.0 * math.pi)
        group_speed_cg = 0.5 * phase_speed_cp

    if hs <= 0.0 or tm02 <= 0.0 or not math.isfinite(hs) or not math.isfinite(tm02):
        sm02 = round(sp, 5)
    else:
        sm02 = (2.0 * math.pi * hs) / (g_val * (tm02 ** 2))

    stokes_limit = 1.0 / 7.0  # Miche/Stokes deep-water wave breaking limit (~0.14286)
    breaking_fraction = min(1.0, sp / stokes_limit) if stokes_limit > 0 else 0.0

    if sp <= 0.0:
        regime = "Calm / Flat"
    elif sp < 0.01:
        regime = "Swell-dominated (long-period swell)"
    elif sp < 0.025:
        regime = "Mixed wind-sea and swell"
    elif sp < 0.04:
        regime = "Developing wind-sea"
    elif sp < 0.06:
        regime = "Steep wind-sea (energetic)"
    else:
        regime = "Very steep / breaking-prone sea"

    return {
        "steepness_sp": round(float(sp), 5),
        "steepness_sm02": round(float(sm02), 5),
        "inverse_steepness": round(float(1.0 / sp), 2) if sp > 1e-6 else None,
        "wavelength_p_m": round(float(wavelength_p), 2),
        "wavenumber_kp_rad_m": round(float(wavenumber_kp), 5),
        "phase_speed_cp_mps": round(float(phase_speed_cp), 2),
        "group_speed_cg_mps": round(float(group_speed_cg), 2),
        "stokes_breaking_limit": round(stokes_limit, 5),
        "breaking_proximity_pct": round(float(breaking_fraction * 100.0), 2),
        "regime": regime,
    }


def compute_goda_peakedness(freqs: np.ndarray, spectrum: np.ndarray) -> float:
    """Compute Goda's spectral peakedness factor Q_p = (2 / m_0^2) * integral(f * S(f)^2 df)."""
    f = np.asarray(freqs, dtype=np.float64)
    s = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
    if f.ndim != 1 or s.ndim != 1 or len(f) < 2 or len(f) != len(s):
        return 2.0
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    m0 = float(trapz_fn(s, f))
    if m0 <= 1e-12 or not math.isfinite(m0):
        return 2.0
    num = float(trapz_fn(f * (s ** 2), f))
    qp = (2.0 * num) / (m0 ** 2)
    return float(np.clip(qp, 1.0, 12.0))


def compute_rogue_wave_risk(
    hm0_m: Optional[float],
    tp_s: Optional[float],
    tm02_s: Optional[float] = None,
    bandwidth_eps: Optional[float] = None,
    freqs: Optional[np.ndarray] = None,
    spectrum: Optional[np.ndarray] = None,
    duration_s: float = 10800.0,
) -> Dict[str, Any]:
    """Compute Rogue Wave Crest Risk Index using Tayfun second-order crest statistics and Benjamin-Feir Index."""
    hs = _sanitize_non_negative(hm0_m, 0.0)
    tp = _sanitize_non_negative(tp_s, 0.0)
    tm02 = _sanitize_non_negative(tm02_s, tp * 0.78 if tp > 0 else 0.0)
    eps = _sanitize_non_negative(bandwidth_eps, 0.55)
    if eps <= 0.0:
        eps = 0.55

    if hs <= 0.0 or tp <= 0.0 or not math.isfinite(hs) or not math.isfinite(tp):
        return {
            "risk_index": 0.0,
            "risk_score": 0.0,
            "risk_level": "LOW",
            "benjamin_feir_index": 0.0,
            "goda_peakedness_qp": 1.0,
            "kurtosis_excess": 0.0,
            "expected_hmax_3hr_m": 0.0,
            "expected_crest_max_3hr_m": 0.0,
            "crest_to_hm0_ratio": 0.0,
            "hmax_to_hm0_ratio": 0.0,
            "rogue_threshold_wave_m": 0.0,
            "rogue_threshold_crest_m": 0.0,
            "rogue_prob_3hr": 0.0,
            "num_zero_crossings_3hr": 0,
        }

    steep = compute_wave_steepness(hs, tp, tm02)
    sp = float(steep["steepness_sp"])
    kp = float(steep["wavenumber_kp_rad_m"])

    if freqs is not None and spectrum is not None:
        qp = compute_goda_peakedness(freqs, spectrum)
    else:
        # Estimate peakedness from spectral bandwidth parameter eps
        qp = float(np.clip(2.0 / max(eps, 0.25), 1.2, 8.0))

    # Benjamin-Feir Index (BFI): ratio of steepness to relative spectral width (Janssen 2003)
    bfi = float(math.pi * math.sqrt(math.pi) * sp * qp)
    # Excess fourth-order cumulant (kurtosis) from four-wave quasi-resonant interactions
    kurtosis_excess = float((math.pi / math.sqrt(3.0)) * (bfi ** 2))

    eff_tm02 = tm02 if tm02 > 0.5 else max(tp * 0.78, 1.0)
    n_waves = max(1.0, float(duration_s) / eff_tm02)
    ln_n = max(math.log(n_waves), 1.0)

    # Longuet-Higgins expected maximum wave height in duration_s (with kurtosis nonlinear correction)
    rayleigh_factor = math.sqrt(0.5 * ln_n) + EULER_GAMMA / (2.0 * math.sqrt(2.0 * ln_n))
    nonlinear_h_boost = 1.0 + 0.04 * min(kurtosis_excess, 2.5)
    expected_hmax = hs * rayleigh_factor * nonlinear_h_boost

    # Tayfun second-order nonlinear maximum crest elevation: eta_c = 0.5 * H_max * (1 + 0.5 * k_p * H_max)
    linear_crest = 0.5 * expected_hmax
    expected_crest_max = linear_crest * (1.0 + 0.5 * kp * expected_hmax)

    crest_ratio = expected_crest_max / hs
    hmax_ratio = expected_hmax / hs

    # Mori & Janssen (2006) single-wave exceedance probability P(H > 2 * Hm0)
    p_single_rayleigh = math.exp(-8.0)  # ~3.3546e-4
    p_single_rogue = p_single_rayleigh * (1.0 + 8.0 * min(kurtosis_excess, 3.0))
    p_state_rogue = float(1.0 - math.exp(-n_waves * p_single_rogue))
    p_state_rogue = float(np.clip(p_state_rogue, 0.0, 1.0))

    # Composite Rogue Wave Crest Risk Index in [0, 1]
    # Combines steepness (Sp), BFI modulational instability, nonlinear crest amplification, and wave energy scale
    steep_component = min(1.0, sp / 0.055)
    bfi_component = min(1.0, bfi / 1.0)
    crest_component = float(np.clip((crest_ratio - 0.90) / (1.35 - 0.90), 0.0, 1.0))
    energy_component = min(1.0, hs / 7.5)

    risk_index = (
        0.35 * steep_component
        + 0.25 * bfi_component
        + 0.20 * crest_component
        + 0.20 * energy_component
    )
    risk_index = float(np.clip(risk_index, 0.0, 1.0))
    risk_score = round(risk_index * 100.0, 1)

    if risk_index < 0.20:
        risk_level = "LOW"
    elif risk_index < 0.40:
        risk_level = "MODERATE"
    elif risk_index < 0.65:
        risk_level = "ELEVATED"
    elif risk_index < 0.82:
        risk_level = "HIGH"
    else:
        risk_level = "EXTREME"

    return {
        "risk_index": round(risk_index, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "benjamin_feir_index": round(bfi, 4),
        "goda_peakedness_qp": round(qp, 3),
        "kurtosis_excess": round(kurtosis_excess, 4),
        "expected_hmax_3hr_m": round(expected_hmax, 3),
        "expected_crest_max_3hr_m": round(expected_crest_max, 3),
        "crest_to_hm0_ratio": round(crest_ratio, 4),
        "hmax_to_hm0_ratio": round(hmax_ratio, 4),
        "rogue_threshold_wave_m": round(2.0 * hs, 3),
        "rogue_threshold_crest_m": round(1.25 * hs, 3),
        "rogue_prob_3hr": round(p_state_rogue, 4),
        "num_zero_crossings_3hr": int(round(n_waves)),
    }


def pierson_moskowitz_spectrum(f: np.ndarray, hs: float, tp: float) -> np.ndarray:
    """Compute the two-parameter Pierson-Moskowitz (1964) / Bretschneider wave spectrum S_PM(f) (gamma = 1.0)."""
    f_arr = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f_arr, dtype=np.float64)
    hs_val = _sanitize_non_negative(hs, 0.0)
    tp_val = _sanitize_non_negative(tp, 0.0)
    if hs_val <= 0.0 or tp_val <= 0.0 or not math.isfinite(hs_val) or not math.isfinite(tp_val):
        return out

    fp = 1.0 / max(tp_val, 1e-4)
    valid = np.isfinite(f_arr) & (f_arr > 1e-6)
    if not np.any(valid):
        return out

    fv = f_arr[valid]
    ratio = fv / fp
    # Prevent underflow/overflow in exp(-1.25 * (f / fp)^-4) for tiny frequencies
    exponent = np.clip(-1.25 * (ratio ** -4), -700.0, 0.0)
    s_pm = (5.0 / 16.0) * (hs_val ** 2) * (fp ** 4) * (fv ** -5) * np.exp(exponent)
    out[valid] = np.where(np.isfinite(s_pm), np.maximum(s_pm, 0.0), 0.0)
    return out


def jonswap_spectrum(f: np.ndarray, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
    """Compute the fetch-limited JONSWAP (Hasselmann et al., 1973) wave energy density spectrum S_J(f)."""
    f_arr = np.asarray(f, dtype=np.float64)
    s_pm = pierson_moskowitz_spectrum(f_arr, hs, tp)
    if not np.any(s_pm > 0.0):
        return s_pm

    gamma_val = float(gamma) if (gamma is not None and math.isfinite(gamma) and gamma >= 1.0) else 3.3
    if abs(gamma_val - 1.0) < 1e-9:
        return s_pm

    fp = 1.0 / max(_sanitize_non_negative(tp, 1.0), 1e-4)
    sigma = np.where(f_arr <= fp, 0.07, 0.09)
    r = np.exp(-((f_arr - fp) ** 2) / (2.0 * (sigma ** 2) * (fp ** 2)))
    c_gamma = max(0.1, 1.0 - 0.287 * math.log(gamma_val))
    s_j = c_gamma * s_pm * (gamma_val ** r)
    return np.where(np.isfinite(s_j), np.maximum(s_j, 0.0), 0.0)


def compare_pm_vs_jonswap(
    f: np.ndarray,
    s_measured: np.ndarray,
    hs: float,
    tp: float,
    gamma: float = 3.3,
) -> Dict[str, Any]:
    """Compare measured buoy wave spectrum S(f) against Pierson-Moskowitz (fully developed) and JONSWAP models."""
    f_arr = np.asarray(f, dtype=np.float64)
    s_meas = np.maximum(np.asarray(s_measured, dtype=np.float64), 0.0)
    s_pm = pierson_moskowitz_spectrum(f_arr, hs, tp)
    s_js = jonswap_spectrum(f_arr, hs, tp, gamma=gamma)

    if len(f_arr) == 0 or len(s_meas) != len(f_arr):
        return {
            "pm_rmse_m2_hz": 0.0,
            "jonswap_rmse_m2_hz": 0.0,
            "pm_mae_m2_hz": 0.0,
            "jonswap_mae_m2_hz": 0.0,
            "pm_r2": 0.0,
            "jonswap_r2": 0.0,
            "pm_corr_r": 0.0,
            "jonswap_corr_r": 0.0,
            "measured_peak_m2_hz": 0.0,
            "pm_peak_m2_hz": 0.0,
            "jonswap_peak_m2_hz": 0.0,
            "peak_enhancement_ratio": 1.0,
            "fitted_gamma": 1.0,
            "fitted_jonswap_rmse_m2_hz": 0.0,
            "best_fit_model": "JONSWAP",
            "sea_development_stage": "Calm / Flat",
            "pierson_moskowitz_m2_hz": [],
            "jonswap_m2_hz": [],
        }

    pm_rmse = float(np.sqrt(np.mean((s_meas - s_pm) ** 2)))
    js_rmse = float(np.sqrt(np.mean((s_meas - s_js) ** 2)))
    pm_mae = float(np.mean(np.abs(s_meas - s_pm)))
    js_mae = float(np.mean(np.abs(s_meas - s_js)))

    ss_tot = float(np.sum((s_meas - np.mean(s_meas)) ** 2))
    if ss_tot > 1e-12:
        pm_r2 = float(1.0 - np.sum((s_meas - s_pm) ** 2) / ss_tot)
        js_r2 = float(1.0 - np.sum((s_meas - s_js) ** 2) / ss_tot)
    else:
        pm_r2 = 0.0
        js_r2 = 0.0

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        c = float(np.corrcoef(a, b)[0, 1])
        return c if math.isfinite(c) else 0.0

    pm_corr = _safe_corr(s_meas, s_pm)
    js_corr = _safe_corr(s_meas, s_js)

    meas_peak = float(np.max(s_meas))
    pm_peak = float(np.max(s_pm))
    js_peak = float(np.max(s_js))
    peak_ratio = meas_peak / pm_peak if pm_peak > 1e-9 else 1.0

    # Grid search optimal peak enhancement factor gamma in [1.0, 7.0]
    gamma_grid = np.linspace(1.0, 7.0, 61)
    best_gamma = 1.0
    best_rmse = pm_rmse
    if meas_peak > 1e-6 and hs > 0.0 and tp > 0.0:
        for g_cand in gamma_grid:
            s_cand = jonswap_spectrum(f_arr, hs, tp, gamma=float(g_cand))
            cand_rmse = float(np.sqrt(np.mean((s_meas - s_cand) ** 2)))
            if cand_rmse < best_rmse:
                best_rmse = cand_rmse
                best_gamma = float(g_cand)

    best_model = "JONSWAP" if js_rmse <= pm_rmse else "Pierson-Moskowitz"
    if meas_peak <= 1e-6 or hs <= 0.0:
        stage = "Calm / Flat"
    elif best_gamma >= 1.8 or best_model == "JONSWAP":
        stage = "Fetch-limited / Developing Sea (JONSWAP)"
    else:
        stage = "Fully Developed Sea (Pierson-Moskowitz)"

    return {
        "pm_rmse_m2_hz": round(pm_rmse, 4),
        "jonswap_rmse_m2_hz": round(js_rmse, 4),
        "pm_mae_m2_hz": round(pm_mae, 4),
        "jonswap_mae_m2_hz": round(js_mae, 4),
        "pm_r2": round(pm_r2, 4),
        "jonswap_r2": round(js_r2, 4),
        "pm_corr_r": round(pm_corr, 4),
        "jonswap_corr_r": round(js_corr, 4),
        "measured_peak_m2_hz": round(meas_peak, 4),
        "pm_peak_m2_hz": round(pm_peak, 4),
        "jonswap_peak_m2_hz": round(js_peak, 4),
        "peak_enhancement_ratio": round(peak_ratio, 3),
        "fitted_gamma": round(best_gamma, 2),
        "fitted_jonswap_rmse_m2_hz": round(best_rmse, 4),
        "best_fit_model": best_model,
        "sea_development_stage": stage,
        "pierson_moskowitz_m2_hz": [round(float(x), 4) for x in s_pm],
        "jonswap_m2_hz": [round(float(x), 4) for x in s_js],
    }

"""Maritime Hazard & Environmental Alert Engine for the Smart Buoy AI Platform.

Evaluates five core operational oceanographic and environmental hazard domains:
1. Significant Wave Height thresholds (WMO Douglas Sea State & AI multi-step forecast)
2. Rapid Barometric Pressure drops (cyclonic bombogenesis / squall tendency dP/dt)
3. Gale-force Wind Gusts (Beaufort Scale 6-10 marine wind hazards)
4. Hypoxic Dissolved Oxygen (< 2.0 mg/L severe estuarine hypoxia & < 3.0 mg/L stress)
5. Deep LSTM Autoencoder Water Quality / Turbidity Plume anomalies
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .telemetry import BuoyTelemetryStore, WQ_COLS


SEVERITY_ORDER: Dict[str, int] = {
    "NOMINAL": 0,
    "ADVISORY": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}


@dataclass(frozen=True)
class AlertThresholds:
    """Configurable oceanographic and environmental hazard thresholds."""

    # 1. Wave height thresholds (meters, WMO Douglas Sea Scale)
    wave_advisory_m: float = 2.0      # Moderate-to-rough sea state (Small Craft Advisory)
    wave_warning_m: float = 2.5       # Rough sea state warning
    wave_critical_m: float = 3.5      # Very rough / high seas storm hazard
    wave_rapid_build_m_1h: float = 0.35  # Rapid 1-hour wave height increase

    # 2. Barometric pressure drop thresholds (hPa over 1h and 3h windows)
    pressure_drop_1h_warning_hpa: float = -1.0   # Rapid 1h pressure fall
    pressure_drop_3h_advisory_hpa: float = -1.5  # Noticeable 3h barometric fall
    pressure_drop_3h_warning_hpa: float = -2.0   # Rapid 3h frontal pressure drop
    pressure_drop_3h_critical_hpa: float = -3.5  # Severe cyclonic / squall pressure drop
    pressure_low_trough_hpa: float = 1006.0      # Deep low-pressure trough threshold

    # 3. Wind & gale gust thresholds (m/s, WMO / Beaufort scale)
    gust_advisory_mps: float = 10.8   # Beaufort 6 Strong Breeze (~21 kt)
    gust_near_gale_mps: float = 13.9  # Beaufort 7 Near Gale (~27 kt)
    gust_gale_mps: float = 17.2       # Beaufort 8 Gale Force (~34 kt)
    gust_storm_mps: float = 24.5      # Beaufort 10 Storm Force (~48 kt)

    # 4. Dissolved Oxygen (DO) hypoxia thresholds (mg/L == ppm)
    hypoxia_severe_mg_l: float = 1.5  # Acute lethal hypoxia threshold (< 1.5 mg/L)
    hypoxia_do_mg_l: float = 2.0      # Standard EPA/NOAA estuarine hypoxia (< 2.0 mg/L)
    hypoxia_stress_mg_l: float = 3.0  # Biological stress advisory (< 3.0 mg/L)

    # 5. Turbidity & LSTM Autoencoder anomaly thresholds
    turbidity_advisory_ntu: float = 35.0   # Elevated estuarine sediment resuspension
    turbidity_warning_ntu: float = 50.0    # Estuarine sediment runoff plume
    turbidity_critical_ntu: float = 100.0  # Severe turbidity plume event
    ae_default_mse_threshold: float = 0.5357  # H100 LSTM Autoencoder 95th percentile MSE

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class MaritimeAlert:
    """Structured representation of a single maritime or environmental alert."""

    id: str
    category: str
    severity: str
    scope: str  # "live" | "forecast" | "historical"
    title: str
    message: str
    station: str
    timestamp: str
    metric_name: str
    observed_value: float
    threshold_value: float
    unit: str
    recommended_action: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def wmo_sea_state(hs_m: float) -> Tuple[int, str]:
    """Map Significant Wave Height Hs (m) to WMO Douglas Sea State code and label."""
    if not np.isfinite(hs_m) or hs_m < 0.1:
        return 1, "Calm (Rippled)"
    if hs_m < 0.5:
        return 2, "Smooth (Wavelets)"
    if hs_m < 1.25:
        return 3, "Slight"
    if hs_m < 2.5:
        return 4, "Moderate"
    if hs_m < 4.0:
        return 5, "Rough"
    if hs_m < 6.0:
        return 6, "Very Rough"
    if hs_m < 9.0:
        return 7, "High"
    return 8, "Very High / Phenomenal"


def beaufort_scale(wind_mps: float) -> Tuple[int, str]:
    """Map wind or gust speed (m/s) to WMO Beaufort number and description."""
    if not np.isfinite(wind_mps) or wind_mps < 0.5:
        return 0, "Calm"
    if wind_mps < 1.6:
        return 1, "Light Air"
    if wind_mps < 3.4:
        return 2, "Light Breeze"
    if wind_mps < 5.5:
        return 3, "Gentle Breeze"
    if wind_mps < 8.0:
        return 4, "Moderate Breeze"
    if wind_mps < 10.8:
        return 5, "Fresh Breeze"
    if wind_mps < 13.9:
        return 6, "Strong Breeze"
    if wind_mps < 17.2:
        return 7, "Near Gale"
    if wind_mps < 20.8:
        return 8, "Gale"
    if wind_mps < 24.5:
        return 9, "Strong Gale"
    if wind_mps < 28.5:
        return 10, "Storm"
    return 11, "Violent Storm / Hurricane"


def _fmt_ts(ts: Any) -> str:
    """Format a pandas Timestamp, datetime, or string consistently."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return "N/A"
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    return str(ts)


def _max_severity(*severities: str) -> str:
    best = "NOMINAL"
    for s in severities:
        if SEVERITY_ORDER.get(s, 0) > SEVERITY_ORDER.get(best, 0):
            best = s
    return best


def _compute_pressure_tendency(df_met: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute 1-hour and 3-hour barometric pressure changes (hPa) from df_met."""
    if df_met is None or len(df_met) == 0 or "PRES_hpa" not in df_met.columns:
        empty = pd.Series(dtype=np.float64)
        return empty, empty

    pres = pd.to_numeric(df_met["PRES_hpa"], errors="coerce").ffill().bfill()
    if len(pres) <= 1:
        zeros = pd.Series(np.zeros(len(pres), dtype=np.float64), index=df_met.index)
        return zeros, zeros

    # Determine median sampling interval in minutes if timestamps are present
    step_1h, step_3h = 6, 18
    if "timestamp" in df_met.columns:
        ts = pd.to_datetime(df_met["timestamp"], errors="coerce")
        diffs_min = ts.diff().dt.total_seconds().dropna() / 60.0
        diffs_min = diffs_min[diffs_min > 0]
        if len(diffs_min) > 0:
            dt_min = float(np.median(diffs_min))
            if dt_min > 0:
                step_1h = max(1, int(round(60.0 / dt_min)))
                step_3h = max(1, int(round(180.0 / dt_min)))

    s1 = min(step_1h, max(1, len(pres) - 1))
    s3 = min(step_3h, max(1, len(pres) - 1))
    dp_1h = pres.diff(s1).fillna(0.0)
    dp_3h = pres.diff(s3).fillna(0.0)
    return dp_1h, dp_3h


class MaritimeAlertEngine:
    """Evaluates live, forecasted, and historical maritime & environmental hazards."""

    def __init__(self, thresholds: Optional[AlertThresholds] = None) -> None:
        self.thresholds = thresholds or AlertThresholds()

    def evaluate_wave_hazards(
        self,
        df_met: Optional[pd.DataFrame],
        forecast_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[MaritimeAlert], List[MaritimeAlert]]:
        """Evaluate wave height thresholds (live, AI forecast, and historical storm peaks)."""
        t = self.thresholds
        live_alerts: List[MaritimeAlert] = []
        hist_alerts: List[MaritimeAlert] = []

        if df_met is None or len(df_met) == 0 or "WVHT_m" not in df_met.columns:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_value": 0.0,
                    "forecast_1h_value": 0.0,
                    "peak_value": 0.0,
                    "threshold": t.wave_warning_m,
                    "unit": "m",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "wmo_sea_state_code": 1,
                    "wmo_sea_state_label": "Calm (Rippled)",
                    "summary": "No wave telemetry available.",
                },
                live_alerts,
                hist_alerts,
            )

        wvht = pd.to_numeric(df_met["WVHT_m"], errors="coerce").dropna()
        if len(wvht) == 0:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_value": 0.0,
                    "forecast_1h_value": 0.0,
                    "peak_value": 0.0,
                    "threshold": t.wave_warning_m,
                    "unit": "m",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "wmo_sea_state_code": 1,
                    "wmo_sea_state_label": "Calm (Rippled)",
                    "summary": "No valid wave height measurements.",
                },
                live_alerts,
                hist_alerts,
            )

        latest_idx = wvht.index[-1]
        current_hs = float(wvht.loc[latest_idx])
        latest_ts = _fmt_ts(df_met.loc[latest_idx, "timestamp"]) if "timestamp" in df_met.columns else "Latest"
        current_dpd = float(df_met.loc[latest_idx, "DPD_s"]) if "DPD_s" in df_met.columns else 0.0

        peak_idx = wvht.idxmax()
        peak_hs = float(wvht.loc[peak_idx])
        peak_ts = _fmt_ts(df_met.loc[peak_idx, "timestamp"]) if "timestamp" in df_met.columns else "Peak"

        # Extract AI forecast metadata if available
        wf = (forecast_payload or {}).get("wave_forecast", {})
        live_pred = wf.get("live_prediction", {})
        forecast_1h_hs = float(live_pred.get("forecast_hs_1h_m", current_hs))
        delta_hs_1h = float(live_pred.get("delta_hs_m", forecast_1h_hs - current_hs))
        ci95_upper_hs = float(live_pred.get("ci95_upper_hs_m", forecast_1h_hs))

        multi_step = wf.get("multi_step_forecast", {}).get("horizons", [])
        max_multi_hs = max(
            [forecast_1h_hs] + [float(h.get("forecast_hs_m", 0.0)) for h in multi_step if isinstance(h, dict)]
        )

        # Evaluate live wave height
        live_sev = "NOMINAL"
        effective_live_hs = max(current_hs, forecast_1h_hs, max_multi_hs)
        if effective_live_hs >= t.wave_critical_m:
            live_sev = "CRITICAL"
            thresh_used = t.wave_critical_m
        elif effective_live_hs >= t.wave_warning_m:
            live_sev = "WARNING"
            thresh_used = t.wave_warning_m
        elif effective_live_hs >= t.wave_advisory_m:
            live_sev = "ADVISORY"
            thresh_used = t.wave_advisory_m
        else:
            thresh_used = t.wave_warning_m

        sea_code, sea_label = wmo_sea_state(current_hs)

        if live_sev != "NOMINAL":
            is_forecast_driven = forecast_1h_hs > current_hs and current_hs < thresh_used
            live_alerts.append(
                MaritimeAlert(
                    id="WAVE-LIVE-01",
                    category="wave_height",
                    severity=live_sev,
                    scope="forecast" if is_forecast_driven else "live",
                    title=f"WMO Sea State {sea_code} ({sea_label}) — Wave Height {live_sev.title()}",
                    message=(
                        f"Significant wave height Hs = {current_hs:.2f} m (1h AI forecast: {forecast_1h_hs:.2f} m, "
                        f"95% CI upper: {ci95_upper_hs:.2f} m, dominant period Tp = {current_dpd:.1f} s) "
                        f"exceeds {thresh_used:.1f} m threshold."
                    ),
                    station="NOAA NDBC 46050 (Stonewall Bank, OR)",
                    timestamp=latest_ts,
                    metric_name="WVHT_m",
                    observed_value=round(effective_live_hs, 2),
                    threshold_value=round(thresh_used, 2),
                    unit="m",
                    recommended_action="Restrict small-craft coastal transit; monitor buoy heave & mooring line tension.",
                    metadata={
                        "current_hs_m": round(current_hs, 2),
                        "forecast_hs_1h_m": round(forecast_1h_hs, 2),
                        "ci95_upper_hs_m": round(ci95_upper_hs, 2),
                        "wmo_sea_state_code": sea_code,
                        "wmo_sea_state_label": sea_label,
                    },
                )
            )

        if delta_hs_1h >= t.wave_rapid_build_m_1h:
            live_sev = _max_severity(live_sev, "WARNING")
            live_alerts.append(
                MaritimeAlert(
                    id="WAVE-FORECAST-BUILD",
                    category="wave_height",
                    severity="WARNING",
                    scope="forecast",
                    title="Rapid Sea-State Escalation Forecast (+1h)",
                    message=(
                        f"ResidualWaveForecaster projects rapid wave growth of +{delta_hs_1h:.2f} m/hr "
                        f"(from {current_hs:.2f} m to {forecast_1h_hs:.2f} m)."
                    ),
                    station="NOAA NDBC 46050 (Stonewall Bank, OR)",
                    timestamp=latest_ts,
                    metric_name="delta_hs_1h_m",
                    observed_value=round(delta_hs_1h, 3),
                    threshold_value=round(t.wave_rapid_build_m_1h, 3),
                    unit="m/h",
                    recommended_action="Pre-alert SAR and vessel traffic services for rapidly building swell.",
                    metadata={"current_hs_m": round(current_hs, 2), "forecast_hs_1h_m": round(forecast_1h_hs, 2)},
                )
            )

        # Evaluate historical wave storm peaks across dataset
        warn_count = int((wvht >= t.wave_warning_m).sum())
        crit_count = int((wvht >= t.wave_critical_m).sum())
        hist_sev = "NOMINAL"
        if peak_hs >= t.wave_critical_m:
            hist_sev = "CRITICAL"
            peak_thresh = t.wave_critical_m
        elif peak_hs >= t.wave_warning_m:
            hist_sev = "WARNING"
            peak_thresh = t.wave_warning_m
        elif peak_hs >= t.wave_advisory_m:
            hist_sev = "ADVISORY"
            peak_thresh = t.wave_advisory_m
        else:
            peak_thresh = t.wave_warning_m

        if hist_sev != "NOMINAL":
            peak_code, peak_label = wmo_sea_state(peak_hs)
            hist_alerts.append(
                MaritimeAlert(
                    id="WAVE-HIST-PEAK",
                    category="wave_height",
                    severity=hist_sev,
                    scope="historical",
                    title=f"Offshore Storm Wave Peak — WMO Sea State {peak_code} ({peak_label})",
                    message=(
                        f"Peak significant wave height reached {peak_hs:.2f} m on {peak_ts} "
                        f"({warn_count} observations >= {t.wave_warning_m:.1f} m; "
                        f"{crit_count} observations >= {t.wave_critical_m:.1f} m)."
                    ),
                    station="NOAA NDBC 46050 (Stonewall Bank, OR)",
                    timestamp=peak_ts,
                    metric_name="WVHT_m",
                    observed_value=round(peak_hs, 2),
                    threshold_value=round(peak_thresh, 2),
                    unit="m",
                    recommended_action="Issue High Seas / Rough Bar Advisory for Newport & Stonewall Bank approaches.",
                    metadata={
                        "warning_observations": warn_count,
                        "critical_observations": crit_count,
                        "wmo_sea_state_code": peak_code,
                        "wmo_sea_state_label": peak_label,
                    },
                )
            )

        overall_sev = _max_severity(live_sev, hist_sev)
        cat_info = {
            "triggered": overall_sev != "NOMINAL",
            "live_triggered": live_sev != "NOMINAL",
            "severity": overall_sev,
            "live_severity": live_sev,
            "current_value": round(current_hs, 2),
            "forecast_1h_value": round(forecast_1h_hs, 2),
            "ci95_upper_hs_m": round(ci95_upper_hs, 2),
            "peak_value": round(peak_hs, 2),
            "threshold": round(t.wave_warning_m, 2),
            "critical_threshold": round(t.wave_critical_m, 2),
            "unit": "m",
            "event_count": warn_count,
            "critical_event_count": crit_count,
            "peak_timestamp": peak_ts,
            "wmo_sea_state_code": sea_code,
            "wmo_sea_state_label": sea_label,
            "summary": (
                f"Live Hs = {current_hs:.2f} m ({sea_label}, 1h forecast {forecast_1h_hs:.2f} m); "
                f"dataset storm peak = {peak_hs:.2f} m ({warn_count} rough-sea intervals >= {t.wave_warning_m:.1f} m)."
            ),
        }
        return cat_info, live_alerts, hist_alerts

    def evaluate_pressure_drops(
        self,
        df_met: Optional[pd.DataFrame],
    ) -> Tuple[Dict[str, Any], List[MaritimeAlert], List[MaritimeAlert]]:
        """Evaluate rapid barometric pressure drops (1h and 3h dP/dt tendency) and cyclonic troughs."""
        t = self.thresholds
        live_alerts: List[MaritimeAlert] = []
        hist_alerts: List[MaritimeAlert] = []

        if df_met is None or len(df_met) == 0 or "PRES_hpa" not in df_met.columns:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_pressure_hpa": 1013.25,
                    "current_drop_1h_hpa": 0.0,
                    "current_drop_3h_hpa": 0.0,
                    "max_drop_3h_hpa": 0.0,
                    "min_pressure_hpa": 1013.25,
                    "threshold": t.pressure_drop_3h_warning_hpa,
                    "unit": "hPa/3h",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "summary": "No barometric telemetry available.",
                },
                live_alerts,
                hist_alerts,
            )

        pres = pd.to_numeric(df_met["PRES_hpa"], errors="coerce").ffill().bfill()
        if pres.isna().all():
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_pressure_hpa": 1013.25,
                    "current_drop_1h_hpa": 0.0,
                    "current_drop_3h_hpa": 0.0,
                    "max_drop_3h_hpa": 0.0,
                    "min_pressure_hpa": 1013.25,
                    "threshold": t.pressure_drop_3h_warning_hpa,
                    "unit": "hPa/3h",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "summary": "No valid barometric pressure observations.",
                },
                live_alerts,
                hist_alerts,
            )

        dp_1h, dp_3h = _compute_pressure_tendency(df_met)
        latest_idx = pres.index[-1]
        curr_pres = float(pres.loc[latest_idx])
        curr_dp1 = float(dp_1h.loc[latest_idx])
        curr_dp3 = float(dp_3h.loc[latest_idx])
        latest_ts = _fmt_ts(df_met.loc[latest_idx, "timestamp"]) if "timestamp" in df_met.columns else "Latest"

        min_dp3_idx = dp_3h.idxmin()
        max_drop_3h = float(dp_3h.loc[min_dp3_idx])
        max_drop_ts = _fmt_ts(df_met.loc[min_dp3_idx, "timestamp"]) if "timestamp" in df_met.columns else "Peak"

        min_pres_idx = pres.idxmin()
        min_pres = float(pres.loc[min_pres_idx])
        min_pres_ts = _fmt_ts(df_met.loc[min_pres_idx, "timestamp"]) if "timestamp" in df_met.columns else "Min"

        # Live barometric tendency check
        live_sev = "NOMINAL"
        if curr_dp3 <= t.pressure_drop_3h_critical_hpa:
            live_sev = "CRITICAL"
            live_thresh = t.pressure_drop_3h_critical_hpa
        elif curr_dp3 <= t.pressure_drop_3h_warning_hpa or curr_dp1 <= t.pressure_drop_1h_warning_hpa:
            live_sev = "WARNING"
            live_thresh = t.pressure_drop_3h_warning_hpa
        elif curr_dp3 <= t.pressure_drop_3h_advisory_hpa or curr_pres <= t.pressure_low_trough_hpa:
            live_sev = "ADVISORY"
            live_thresh = t.pressure_drop_3h_advisory_hpa
        else:
            live_thresh = t.pressure_drop_3h_warning_hpa

        if live_sev != "NOMINAL":
            live_alerts.append(
                MaritimeAlert(
                    id="BARO-LIVE-01",
                    category="pressure_drop",
                    severity=live_sev,
                    scope="live",
                    title=f"Rapid Barometric Pressure Fall ({curr_dp3:+.1f} hPa / 3h)",
                    message=(
                        f"Barometric pressure at {curr_pres:.1f} hPa with 3h change {curr_dp3:+.1f} hPa "
                        f"(1h change: {curr_dp1:+.1f} hPa) signals approaching frontal squall or cyclonic trough."
                    ),
                    station="NOAA NDBC 46050 (Stonewall Bank, OR)",
                    timestamp=latest_ts,
                    metric_name="dPRES_3h_hpa",
                    observed_value=round(curr_dp3, 2),
                    threshold_value=round(live_thresh, 2),
                    unit="hPa/3h",
                    recommended_action="Monitor wind shift and squall line propagation; secure offshore deck gear.",
                    metadata={
                        "current_pres_hpa": round(curr_pres, 1),
                        "dp_1h_hpa": round(curr_dp1, 2),
                        "dp_3h_hpa": round(curr_dp3, 2),
                    },
                )
            )

        # Historical rapid pressure drop check
        drop_warn_count = int((dp_3h <= t.pressure_drop_3h_warning_hpa).sum())
        drop_adv_count = int((dp_3h <= t.pressure_drop_3h_advisory_hpa).sum())
        hist_sev = "NOMINAL"
        if max_drop_3h <= t.pressure_drop_3h_critical_hpa:
            hist_sev = "CRITICAL"
            hist_thresh = t.pressure_drop_3h_critical_hpa
        elif max_drop_3h <= t.pressure_drop_3h_warning_hpa:
            hist_sev = "WARNING"
            hist_thresh = t.pressure_drop_3h_warning_hpa
        elif max_drop_3h <= t.pressure_drop_3h_advisory_hpa:
            hist_sev = "ADVISORY"
            hist_thresh = t.pressure_drop_3h_advisory_hpa
        else:
            hist_thresh = t.pressure_drop_3h_warning_hpa

        if hist_sev != "NOMINAL":
            hist_alerts.append(
                MaritimeAlert(
                    id="BARO-HIST-DROP",
                    category="pressure_drop",
                    severity=hist_sev,
                    scope="historical",
                    title=f"Frontal Cyclogenesis Pressure Drop ({max_drop_3h:+.1f} hPa / 3h)",
                    message=(
                        f"Rapid barometric pressure fall of {max_drop_3h:+.1f} hPa over 3 hours recorded on {max_drop_ts} "
                        f"({drop_warn_count} intervals <= {t.pressure_drop_3h_warning_hpa:+.1f} hPa/3h; "
                        f"storm trough minimum {min_pres:.1f} hPa on {min_pres_ts})."
                    ),
                    station="NOAA NDBC 46050 (Stonewall Bank, OR)",
                    timestamp=max_drop_ts,
                    metric_name="dPRES_3h_hpa",
                    observed_value=round(max_drop_3h, 2),
                    threshold_value=round(hist_thresh, 2),
                    unit="hPa/3h",
                    recommended_action="Correlate barometric trough with peak gust and wave height escalation.",
                    metadata={
                        "min_pressure_hpa": round(min_pres, 1),
                        "min_pressure_timestamp": min_pres_ts,
                        "warning_intervals": drop_warn_count,
                        "advisory_intervals": drop_adv_count,
                    },
                )
            )

        overall_sev = _max_severity(live_sev, hist_sev)
        cat_info = {
            "triggered": overall_sev != "NOMINAL",
            "live_triggered": live_sev != "NOMINAL",
            "severity": overall_sev,
            "live_severity": live_sev,
            "current_value": round(curr_dp3, 2),
            "current_pressure_hpa": round(curr_pres, 1),
            "current_drop_1h_hpa": round(curr_dp1, 2),
            "current_drop_3h_hpa": round(curr_dp3, 2),
            "max_drop_3h_hpa": round(max_drop_3h, 2),
            "min_pressure_hpa": round(min_pres, 1),
            "threshold": round(t.pressure_drop_3h_warning_hpa, 2),
            "unit": "hPa/3h",
            "event_count": drop_warn_count,
            "peak_timestamp": max_drop_ts,
            "summary": (
                f"Current barometer {curr_pres:.1f} hPa (3h tendency {curr_dp3:+.1f} hPa); "
                f"steepest 3h drop = {max_drop_3h:+.1f} hPa ({drop_warn_count} rapid-fall intervals, min {min_pres:.1f} hPa)."
            ),
        }
        return cat_info, live_alerts, hist_alerts

    def evaluate_gale_gusts(
        self,
        df_met: Optional[pd.DataFrame],
    ) -> Tuple[Dict[str, Any], List[MaritimeAlert], List[MaritimeAlert]]:
        """Evaluate wind speed and gale-force gust thresholds (Beaufort scale)."""
        t = self.thresholds
        live_alerts: List[MaritimeAlert] = []
        hist_alerts: List[MaritimeAlert] = []

        if df_met is None or len(df_met) == 0 or "GST_mps" not in df_met.columns:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_value": 0.0,
                    "current_wspd_mps": 0.0,
                    "peak_value": 0.0,
                    "threshold": t.gust_gale_mps,
                    "unit": "m/s",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "beaufort_number": 0,
                    "beaufort_label": "Calm",
                    "summary": "No wind/gust telemetry available.",
                },
                live_alerts,
                hist_alerts,
            )

        gst = pd.to_numeric(df_met["GST_mps"], errors="coerce").dropna()
        if len(gst) == 0:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_value": 0.0,
                    "current_wspd_mps": 0.0,
                    "peak_value": 0.0,
                    "threshold": t.gust_gale_mps,
                    "unit": "m/s",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "beaufort_number": 0,
                    "beaufort_label": "Calm",
                    "summary": "No valid wind gust observations.",
                },
                live_alerts,
                hist_alerts,
            )

        latest_idx = gst.index[-1]
        curr_gst = float(gst.loc[latest_idx])
        curr_wspd = float(df_met.loc[latest_idx, "WSPD_mps"]) if "WSPD_mps" in df_met.columns else curr_gst
        curr_wdir = float(df_met.loc[latest_idx, "WDIR_deg"]) if "WDIR_deg" in df_met.columns else 0.0
        latest_ts = _fmt_ts(df_met.loc[latest_idx, "timestamp"]) if "timestamp" in df_met.columns else "Latest"

        peak_idx = gst.idxmax()
        peak_gst = float(gst.loc[peak_idx])
        peak_wspd = float(df_met.loc[peak_idx, "WSPD_mps"]) if "WSPD_mps" in df_met.columns else peak_gst
        peak_ts = _fmt_ts(df_met.loc[peak_idx, "timestamp"]) if "timestamp" in df_met.columns else "Peak"

        bft_num, bft_label = beaufort_scale(curr_gst)
        peak_bft_num, peak_bft_label = beaufort_scale(peak_gst)

        # Live wind gust evaluation
        live_sev = "NOMINAL"
        if curr_gst >= t.gust_gale_mps:
            live_sev = "CRITICAL"
            live_thresh = t.gust_gale_mps
        elif curr_gst >= t.gust_near_gale_mps:
            live_sev = "WARNING"
            live_thresh = t.gust_near_gale_mps
        elif curr_gst >= t.gust_advisory_mps:
            live_sev = "ADVISORY"
            live_thresh = t.gust_advisory_mps
        else:
            live_thresh = t.gust_gale_mps

        if live_sev != "NOMINAL":
            live_alerts.append(
                MaritimeAlert(
                    id="WIND-LIVE-01",
                    category="gale_gust",
                    severity=live_sev,
                    scope="live",
                    title=f"Beaufort {bft_num} ({bft_label}) — Offshore Wind Gust Alert",
                    message=(
                        f"Peak wind gust of {curr_gst:.1f} m/s ({curr_gst * 1.94384:.1f} kt) with sustained wind "
                        f"{curr_wspd:.1f} m/s from {curr_wdir:.0f}° exceeds {live_thresh:.1f} m/s threshold."
                    ),
                    station="NOAA NDBC 46050 (Stonewall Bank, OR)",
                    timestamp=latest_ts,
                    metric_name="GST_mps",
                    observed_value=round(curr_gst, 2),
                    threshold_value=round(live_thresh, 2),
                    unit="m/s",
                    recommended_action="Issue Marine Gale Warning; verify navigation buoy watch circle & solar array loads.",
                    metadata={
                        "wspd_mps": round(curr_wspd, 2),
                        "wdir_deg": round(curr_wdir, 1),
                        "gust_knots": round(curr_gst * 1.94384, 1),
                        "beaufort_number": bft_num,
                        "beaufort_label": bft_label,
                    },
                )
            )

        # Historical gale gust evaluation
        gale_count = int((gst >= t.gust_gale_mps).sum())
        near_gale_count = int((gst >= t.gust_near_gale_mps).sum())
        hist_sev = "NOMINAL"
        if peak_gst >= t.gust_gale_mps:
            hist_sev = "CRITICAL"
            hist_thresh = t.gust_gale_mps
        elif peak_gst >= t.gust_near_gale_mps:
            hist_sev = "WARNING"
            hist_thresh = t.gust_near_gale_mps
        elif peak_gst >= t.gust_advisory_mps:
            hist_sev = "ADVISORY"
            hist_thresh = t.gust_advisory_mps
        else:
            hist_thresh = t.gust_gale_mps

        if hist_sev != "NOMINAL":
            hist_alerts.append(
                MaritimeAlert(
                    id="WIND-HIST-GALE",
                    category="gale_gust",
                    severity=hist_sev,
                    scope="historical",
                    title=f"Beaufort {peak_bft_num} ({peak_bft_label}) — Gale Force Gust Event",
                    message=(
                        f"Peak gale gust reached {peak_gst:.1f} m/s ({peak_gst * 1.94384:.1f} kt, sustained {peak_wspd:.1f} m/s) "
                        f"on {peak_ts} ({gale_count} gale records >= {t.gust_gale_mps:.1f} m/s; "
                        f"{near_gale_count} near-gale records >= {t.gust_near_gale_mps:.1f} m/s)."
                    ),
                    station="NOAA NDBC 46050 (Stonewall Bank, OR)",
                    timestamp=peak_ts,
                    metric_name="GST_mps",
                    observed_value=round(peak_gst, 2),
                    threshold_value=round(hist_thresh, 2),
                    unit="m/s",
                    recommended_action="Maintain Gale Force Warning logs for offshore vessel traffic & SAR readiness.",
                    metadata={
                        "peak_wspd_mps": round(peak_wspd, 2),
                        "peak_gust_knots": round(peak_gst * 1.94384, 1),
                        "gale_observations": gale_count,
                        "near_gale_observations": near_gale_count,
                        "beaufort_number": peak_bft_num,
                        "beaufort_label": peak_bft_label,
                    },
                )
            )

        overall_sev = _max_severity(live_sev, hist_sev)
        cat_info = {
            "triggered": overall_sev != "NOMINAL",
            "live_triggered": live_sev != "NOMINAL",
            "severity": overall_sev,
            "live_severity": live_sev,
            "current_value": round(curr_gst, 2),
            "current_wspd_mps": round(curr_wspd, 2),
            "peak_value": round(peak_gst, 2),
            "peak_knots": round(peak_gst * 1.94384, 1),
            "threshold": round(t.gust_gale_mps, 2),
            "near_gale_threshold": round(t.gust_near_gale_mps, 2),
            "unit": "m/s",
            "event_count": gale_count,
            "near_gale_event_count": near_gale_count,
            "peak_timestamp": peak_ts,
            "beaufort_number": bft_num,
            "beaufort_label": bft_label,
            "peak_beaufort_number": peak_bft_num,
            "peak_beaufort_label": peak_bft_label,
            "summary": (
                f"Current gust {curr_gst:.1f} m/s (Bft {bft_num} {bft_label}); "
                f"peak gust = {peak_gst:.1f} m/s / {peak_gst * 1.94384:.1f} kt ({gale_count} Gale Force intervals >= {t.gust_gale_mps:.1f} m/s)."
            ),
        }
        return cat_info, live_alerts, hist_alerts

    def evaluate_hypoxia(
        self,
        df_wq: Optional[pd.DataFrame],
    ) -> Tuple[Dict[str, Any], List[MaritimeAlert], List[MaritimeAlert]]:
        """Evaluate hypoxic dissolved oxygen (< 2.0 mg/L critical hypoxia & < 3.0 mg/L biological stress)."""
        t = self.thresholds
        live_alerts: List[MaritimeAlert] = []
        hist_alerts: List[MaritimeAlert] = []

        if df_wq is None or len(df_wq) == 0 or "O2_ppm" not in df_wq.columns:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_value": 0.0,
                    "min_value": 0.0,
                    "threshold": t.hypoxia_do_mg_l,
                    "unit": "mg/L",
                    "event_count": 0,
                    "stress_event_count": 0,
                    "peak_timestamp": "N/A",
                    "summary": "No dissolved oxygen sonde telemetry available.",
                },
                live_alerts,
                hist_alerts,
            )

        o2 = pd.to_numeric(df_wq["O2_ppm"], errors="coerce").dropna()
        if len(o2) == 0:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_value": 0.0,
                    "min_value": 0.0,
                    "threshold": t.hypoxia_do_mg_l,
                    "unit": "mg/L",
                    "event_count": 0,
                    "stress_event_count": 0,
                    "peak_timestamp": "N/A",
                    "summary": "No valid dissolved oxygen readings.",
                },
                live_alerts,
                hist_alerts,
            )

        latest_idx = o2.index[-1]
        curr_o2 = float(o2.loc[latest_idx])
        curr_o2_pct = float(df_wq.loc[latest_idx, "O2_pct"]) if "O2_pct" in df_wq.columns else 0.0
        curr_otmp = float(df_wq.loc[latest_idx, "OTMP_c"]) if "OTMP_c" in df_wq.columns else 0.0
        latest_ts = _fmt_ts(df_wq.loc[latest_idx, "timestamp"]) if "timestamp" in df_wq.columns else "Latest"

        min_idx = o2.idxmin()
        min_o2 = float(o2.loc[min_idx])
        min_o2_pct = float(df_wq.loc[min_idx, "O2_pct"]) if "O2_pct" in df_wq.columns else 0.0
        min_ts = _fmt_ts(df_wq.loc[min_idx, "timestamp"]) if "timestamp" in df_wq.columns else "Min"

        # Live hypoxia check
        live_sev = "NOMINAL"
        if curr_o2 < t.hypoxia_do_mg_l:
            live_sev = "CRITICAL"
            live_thresh = t.hypoxia_do_mg_l
        elif curr_o2 < t.hypoxia_stress_mg_l:
            live_sev = "WARNING"
            live_thresh = t.hypoxia_stress_mg_l
        else:
            live_thresh = t.hypoxia_do_mg_l

        if live_sev != "NOMINAL":
            live_alerts.append(
                MaritimeAlert(
                    id="HYPOXIA-LIVE-01",
                    category="hypoxia",
                    severity=live_sev,
                    scope="live",
                    title=(
                        "Severe Estuarine Hypoxia Alert (DO < 2.0 mg/L)"
                        if curr_o2 < t.hypoxia_do_mg_l
                        else "Estuarine Dissolved Oxygen Stress Advisory (DO < 3.0 mg/L)"
                    ),
                    message=(
                        f"Sonde dissolved oxygen dropped to {curr_o2:.2f} mg/L ({curr_o2_pct:.1f}% sat at {curr_otmp:.1f} °C), "
                        f"below the {live_thresh:.1f} mg/L ecological threshold."
                    ),
                    station="NOAA NERRS ACQS1 (Estuarine Sonde)",
                    timestamp=latest_ts,
                    metric_name="O2_ppm",
                    observed_value=round(curr_o2, 2),
                    threshold_value=round(live_thresh, 2),
                    unit="mg/L",
                    recommended_action="Notify coastal fisheries & estuarine water-quality authorities of hypoxic water mass.",
                    metadata={
                        "o2_pct": round(curr_o2_pct, 1),
                        "water_temp_c": round(curr_otmp, 2),
                        "acute_lethal": bool(curr_o2 < t.hypoxia_severe_mg_l),
                    },
                )
            )

        # Historical hypoxia episodes across ACQS1 record
        hypoxic_count = int((o2 < t.hypoxia_do_mg_l).sum())
        severe_hypoxic_count = int((o2 < t.hypoxia_severe_mg_l).sum())
        stress_count = int((o2 < t.hypoxia_stress_mg_l).sum())

        hist_sev = "NOMINAL"
        if min_o2 < t.hypoxia_do_mg_l:
            hist_sev = "CRITICAL"
            hist_thresh = t.hypoxia_do_mg_l
        elif min_o2 < t.hypoxia_stress_mg_l:
            hist_sev = "WARNING"
            hist_thresh = t.hypoxia_stress_mg_l
        else:
            hist_thresh = t.hypoxia_do_mg_l

        if hist_sev != "NOMINAL":
            hist_alerts.append(
                MaritimeAlert(
                    id="HYPOXIA-HIST-MIN",
                    category="hypoxia",
                    severity=hist_sev,
                    scope="historical",
                    title="Estuarine Hypoxia Episode (DO < 2.0 mg/L)",
                    message=(
                        f"Dissolved oxygen reached a hypoxic minimum of {min_o2:.2f} mg/L ({min_o2_pct:.1f}% saturation) "
                        f"on {min_ts} ({hypoxic_count} sonde observations < {t.hypoxia_do_mg_l:.1f} mg/L; "
                        f"{severe_hypoxic_count} acute readings < {t.hypoxia_severe_mg_l:.1f} mg/L)."
                    ),
                    station="NOAA NERRS ACQS1 (Estuarine Sonde)",
                    timestamp=min_ts,
                    metric_name="O2_ppm",
                    observed_value=round(min_o2, 2),
                    threshold_value=round(hist_thresh, 2),
                    unit="mg/L",
                    recommended_action="Correlate hypoxic dip with tidal salinity intrusion and benthic oxygen demand.",
                    metadata={
                        "hypoxic_observations_lt_2mg": hypoxic_count,
                        "acute_observations_lt_1_5mg": severe_hypoxic_count,
                        "stress_observations_lt_3mg": stress_count,
                        "min_o2_pct": round(min_o2_pct, 1),
                    },
                )
            )

        overall_sev = _max_severity(live_sev, hist_sev)
        cat_info = {
            "triggered": overall_sev != "NOMINAL",
            "live_triggered": live_sev != "NOMINAL",
            "severity": overall_sev,
            "live_severity": live_sev,
            "current_value": round(curr_o2, 2),
            "current_o2_pct": round(curr_o2_pct, 1),
            "min_value": round(min_o2, 2),
            "threshold": round(t.hypoxia_do_mg_l, 2),
            "stress_threshold": round(t.hypoxia_stress_mg_l, 2),
            "unit": "mg/L",
            "event_count": hypoxic_count,
            "severe_event_count": severe_hypoxic_count,
            "stress_event_count": stress_count,
            "peak_timestamp": min_ts,
            "summary": (
                f"Current DO = {curr_o2:.2f} mg/L ({curr_o2_pct:.1f}% sat); "
                f"dataset minimum = {min_o2:.2f} mg/L ({hypoxic_count} hypoxic readings < {t.hypoxia_do_mg_l:.1f} mg/L)."
            ),
        }
        return cat_info, live_alerts, hist_alerts

    def evaluate_turbidity_plumes(
        self,
        df_wq: Optional[pd.DataFrame],
        forecast_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[MaritimeAlert], List[MaritimeAlert]]:
        """Evaluate LSTM Autoencoder water quality reconstruction anomalies and turbidity plumes."""
        t = self.thresholds
        live_alerts: List[MaritimeAlert] = []
        hist_alerts: List[MaritimeAlert] = []

        if df_wq is None or len(df_wq) == 0 or "TURB_ntu" not in df_wq.columns:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_turbidity_ntu": 0.0,
                    "peak_turbidity_ntu": 0.0,
                    "latest_ae_mse": 0.0,
                    "ae_threshold_mse": t.ae_default_mse_threshold,
                    "latest_is_ae_anomaly": False,
                    "flagged_ae_windows": 0,
                    "threshold": t.turbidity_warning_ntu,
                    "unit": "NTU",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "summary": "No turbidity or water quality autoencoder data available.",
                },
                live_alerts,
                hist_alerts,
            )

        turb = pd.to_numeric(df_wq["TURB_ntu"], errors="coerce").dropna()
        if len(turb) == 0:
            return (
                {
                    "triggered": False,
                    "live_triggered": False,
                    "severity": "NOMINAL",
                    "live_severity": "NOMINAL",
                    "current_turbidity_ntu": 0.0,
                    "peak_turbidity_ntu": 0.0,
                    "latest_ae_mse": 0.0,
                    "ae_threshold_mse": t.ae_default_mse_threshold,
                    "latest_is_ae_anomaly": False,
                    "flagged_ae_windows": 0,
                    "threshold": t.turbidity_warning_ntu,
                    "unit": "NTU",
                    "event_count": 0,
                    "peak_timestamp": "N/A",
                    "summary": "No valid turbidity measurements.",
                },
                live_alerts,
                hist_alerts,
            )

        latest_idx = turb.index[-1]
        curr_turb = float(turb.loc[latest_idx])
        curr_sal = float(df_wq.loc[latest_idx, "SAL_psu"]) if "SAL_psu" in df_wq.columns else 0.0
        curr_ph = float(df_wq.loc[latest_idx, "PH"]) if "PH" in df_wq.columns else 0.0
        latest_ts = _fmt_ts(df_wq.loc[latest_idx, "timestamp"]) if "timestamp" in df_wq.columns else "Latest"

        peak_idx = turb.idxmax()
        peak_turb = float(turb.loc[peak_idx])
        peak_ts = _fmt_ts(df_wq.loc[peak_idx, "timestamp"]) if "timestamp" in df_wq.columns else "Peak"

        # Extract LSTM Autoencoder metrics from forecast_payload (or compute lightweight fallback)
        wqa = (forecast_payload or {}).get("water_quality_anomaly", {})
        ae_thresh = float(wqa.get("threshold_mse", t.ae_default_mse_threshold))
        if not np.isfinite(ae_thresh) or ae_thresh <= 0:
            ae_thresh = t.ae_default_mse_threshold

        latest_mse = float(wqa.get("latest_mse", 0.0))
        if not np.isfinite(latest_mse):
            latest_mse = 0.0
        latest_is_ae_anom = bool(wqa.get("latest_is_anomaly", latest_mse > ae_thresh))
        flagged_ae_windows = int(wqa.get("flagged_anomalies", 0))
        max_ae_mse = float(wqa.get("max_mse", max(latest_mse, ae_thresh)))

        # Evaluate live turbidity & autoencoder anomaly status
        live_sev = "NOMINAL"
        if curr_turb >= t.turbidity_critical_ntu or (latest_is_ae_anom and latest_mse >= ae_thresh * 1.5):
            live_sev = "CRITICAL"
            live_thresh = t.turbidity_critical_ntu
        elif latest_is_ae_anom or curr_turb >= t.turbidity_warning_ntu:
            live_sev = "WARNING"
            live_thresh = t.turbidity_warning_ntu
        elif curr_turb >= t.turbidity_advisory_ntu:
            live_sev = "ADVISORY"
            live_thresh = t.turbidity_advisory_ntu
        else:
            live_thresh = t.turbidity_warning_ntu

        if live_sev != "NOMINAL":
            live_alerts.append(
                MaritimeAlert(
                    id="PLUME-LIVE-01",
                    category="turbidity_plume",
                    severity=live_sev,
                    scope="live",
                    title="LSTM Autoencoder Turbidity Plume Anomaly Detected",
                    message=(
                        f"Live estuarine turbidity reached {curr_turb:.1f} NTU (Salinity: {curr_sal:.1f} PSU, pH: {curr_ph:.2f}; "
                        f"WaterQualityAE MSE = {latest_mse:.4f} vs. threshold {ae_thresh:.4f})."
                    ),
                    station="NOAA NERRS ACQS1 (Estuarine Sonde)",
                    timestamp=latest_ts,
                    metric_name="TURB_ntu",
                    observed_value=round(curr_turb, 1),
                    threshold_value=round(live_thresh, 1),
                    unit="NTU",
                    recommended_action="Trigger automated water sampler; inspect watershed storm-runoff / dredge plume dispersion.",
                    metadata={
                        "latest_ae_mse": round(latest_mse, 4),
                        "ae_threshold_mse": round(ae_thresh, 4),
                        "latest_is_ae_anomaly": latest_is_ae_anom,
                        "salinity_psu": round(curr_sal, 2),
                        "ph": round(curr_ph, 2),
                    },
                )
            )

        # Evaluate historical turbidity plumes & LSTM Autoencoder flagged windows
        plume_warn_count = int((turb >= t.turbidity_warning_ntu).sum())
        plume_crit_count = int((turb >= t.turbidity_critical_ntu).sum())
        effective_anom_count = flagged_ae_windows if flagged_ae_windows > 0 else plume_warn_count

        hist_sev = "NOMINAL"
        if peak_turb >= t.turbidity_critical_ntu or flagged_ae_windows > 0:
            hist_sev = "CRITICAL" if peak_turb >= t.turbidity_critical_ntu else "WARNING"
            hist_thresh = t.turbidity_critical_ntu if peak_turb >= t.turbidity_critical_ntu else t.turbidity_warning_ntu
        elif peak_turb >= t.turbidity_warning_ntu:
            hist_sev = "WARNING"
            hist_thresh = t.turbidity_warning_ntu
        else:
            hist_thresh = t.turbidity_warning_ntu

        if hist_sev != "NOMINAL":
            hist_alerts.append(
                MaritimeAlert(
                    id="PLUME-HIST-AE",
                    category="turbidity_plume",
                    severity=hist_sev,
                    scope="historical",
                    title=f"Deep LSTM Autoencoder Plume Detections ({effective_anom_count} Windows Flagged)",
                    message=(
                        f"WaterQualityAE flagged {effective_anom_count} multivariate sonde windows (MSE > {ae_thresh:.4f}), "
                        f"with peak estuarine sediment plume reaching {peak_turb:.1f} NTU on {peak_ts} "
                        f"({plume_warn_count} readings >= {t.turbidity_warning_ntu:.0f} NTU)."
                    ),
                    station="NOAA NERRS ACQS1 (Estuarine Sonde)",
                    timestamp=peak_ts,
                    metric_name="TURB_ntu",
                    observed_value=round(peak_turb, 1),
                    threshold_value=round(hist_thresh, 1),
                    unit="NTU",
                    recommended_action="Review 7-channel autoencoder reconstruction error profile for storm-runoff events.",
                    metadata={
                        "flagged_ae_windows": effective_anom_count,
                        "ae_threshold_mse": round(ae_thresh, 4),
                        "max_ae_mse": round(max_ae_mse, 4),
                        "plume_warning_readings": plume_warn_count,
                        "plume_critical_readings": plume_crit_count,
                    },
                )
            )

        overall_sev = _max_severity(live_sev, hist_sev)
        cat_info = {
            "triggered": overall_sev != "NOMINAL",
            "live_triggered": live_sev != "NOMINAL",
            "severity": overall_sev,
            "live_severity": live_sev,
            "current_value": round(curr_turb, 1),
            "current_turbidity_ntu": round(curr_turb, 1),
            "peak_value": round(peak_turb, 1),
            "peak_turbidity_ntu": round(peak_turb, 1),
            "latest_ae_mse": round(latest_mse, 4),
            "ae_threshold_mse": round(ae_thresh, 4),
            "latest_is_ae_anomaly": latest_is_ae_anom,
            "flagged_ae_windows": effective_anom_count,
            "threshold": round(t.turbidity_warning_ntu, 1),
            "critical_threshold": round(t.turbidity_critical_ntu, 1),
            "unit": "NTU",
            "event_count": effective_anom_count,
            "turbidity_exceedance_count": plume_warn_count,
            "peak_timestamp": peak_ts,
            "summary": (
                f"Live turbidity = {curr_turb:.1f} NTU (AE MSE = {latest_mse:.4f}, threshold = {ae_thresh:.4f}); "
                f"{effective_anom_count} autoencoder plume windows flagged (peak = {peak_turb:.1f} NTU)."
            ),
        }
        return cat_info, live_alerts, hist_alerts

    def evaluate(
        self,
        store: Optional[BuoyTelemetryStore] = None,
        forecast_payload: Optional[Dict[str, Any]] = None,
        df_met: Optional[pd.DataFrame] = None,
        df_wq: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Run full 5-domain maritime hazard and environmental alert evaluation."""
        if df_met is None and store is not None:
            df_met = store.df_met
        if df_wq is None and store is not None:
            df_wq = store.df_wq

        wave_cat, wave_live, wave_hist = self.evaluate_wave_hazards(df_met, forecast_payload=forecast_payload)
        pres_cat, pres_live, pres_hist = self.evaluate_pressure_drops(df_met)
        gust_cat, gust_live, gust_hist = self.evaluate_gale_gusts(df_met)
        hypo_cat, hypo_live, hypo_hist = self.evaluate_hypoxia(df_wq)
        plume_cat, plume_live, plume_hist = self.evaluate_turbidity_plumes(df_wq, forecast_payload=forecast_payload)

        active_alerts = wave_live + pres_live + gust_live + hypo_live + plume_live
        historical_alerts = wave_hist + pres_hist + gust_hist + hypo_hist + plume_hist

        # Sort alerts by severity descending
        active_alerts.sort(key=lambda a: SEVERITY_ORDER.get(a.severity, 0), reverse=True)
        historical_alerts.sort(key=lambda a: SEVERITY_ORDER.get(a.severity, 0), reverse=True)
        all_alerts = active_alerts + historical_alerts

        categories = {
            "wave_height": wave_cat,
            "pressure_drop": pres_cat,
            "gale_gust": gust_cat,
            "hypoxia": hypo_cat,
            "turbidity_plume": plume_cat,
        }

        live_status = _max_severity(*(c["live_severity"] for c in categories.values()))
        overall_status = _max_severity(*(c["severity"] for c in categories.values()))

        # Compute composite 0-100 Maritime & Environmental Risk Index
        sev_weights = {"NOMINAL": 8.0, "ADVISORY": 35.0, "WARNING": 68.0, "CRITICAL": 95.0}
        cat_scores = [
            0.55 * sev_weights.get(c["live_severity"], 8.0) + 0.45 * sev_weights.get(c["severity"], 8.0)
            for c in categories.values()
        ]
        risk_score = round(float(np.clip(np.mean(cat_scores) + 0.25 * np.max(cat_scores), 0.0, 100.0)), 1)

        crit_count = sum(1 for a in all_alerts if a.severity == "CRITICAL")
        warn_count = sum(1 for a in all_alerts if a.severity == "WARNING")
        adv_count = sum(1 for a in all_alerts if a.severity == "ADVISORY")

        return {
            "status": overall_status,
            "live_status": live_status,
            "risk_score": risk_score,
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "thresholds": self.thresholds.to_dict(),
            "summary": {
                "total_alerts": len(all_alerts),
                "active_count": len(active_alerts),
                "historical_episode_count": len(historical_alerts),
                "critical_count": crit_count,
                "warning_count": warn_count,
                "advisory_count": adv_count,
                "triggered_categories": [k for k, v in categories.items() if v["triggered"]],
                "live_triggered_categories": [k for k, v in categories.items() if v["live_triggered"]],
                "counts_by_category": {
                    k: int(v.get("event_count", 0)) for k, v in categories.items()
                },
            },
            "categories": categories,
            "active_alerts": [a.to_dict() for a in active_alerts],
            "historical_alerts": [a.to_dict() for a in historical_alerts],
            "alerts": [a.to_dict() for a in all_alerts],
        }

    # Convenience aliases for flexible caller integration
    evaluate_alerts = evaluate
    get_alerts = evaluate


# Aliases for class discovery across modules/tests
BuoyAlertEngine = MaritimeAlertEngine
AlertEngine = MaritimeAlertEngine


def evaluate_alerts(
    store: Optional[BuoyTelemetryStore] = None,
    forecast_payload: Optional[Dict[str, Any]] = None,
    df_met: Optional[pd.DataFrame] = None,
    df_wq: Optional[pd.DataFrame] = None,
    thresholds: Optional[AlertThresholds] = None,
) -> Dict[str, Any]:
    """Top-level helper to evaluate all maritime & environmental alerts."""
    engine = MaritimeAlertEngine(thresholds=thresholds)
    return engine.evaluate(store=store, forecast_payload=forecast_payload, df_met=df_met, df_wq=df_wq)


evaluate_maritime_alerts = evaluate_alerts
generate_alerts = evaluate_alerts
get_alerts_payload = evaluate_alerts

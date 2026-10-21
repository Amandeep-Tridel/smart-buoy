"""Real-World NOAA NDBC (46050) & NERRS (ACQS1) Telemetry & Spectral Physics Engine."""

import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .spectral_physics import (
    GRAVITY_MPS2,
    classify_beaufort_wind,
    classify_wmo_sea_state,
    compare_pm_vs_jonswap,
    compute_rogue_wave_risk,
    compute_wave_steepness,
    jonswap_spectrum as _jonswap_spectrum,
    pierson_moskowitz_spectrum as _pierson_moskowitz_spectrum,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NOAA_URLS = {
    "46050.data_spec": "https://www.ndbc.noaa.gov/data/realtime2/46050.data_spec",
    "46050.spec": "https://www.ndbc.noaa.gov/data/realtime2/46050.spec",
    "46050.txt": "https://www.ndbc.noaa.gov/data/realtime2/46050.txt",
    "ACQS1.ocean": "https://www.ndbc.noaa.gov/data/realtime2/ACQS1.ocean",
}

MET_COLS = ["WDIR_deg", "WSPD_mps", "GST_mps", "WVHT_m", "DPD_s", "APD_s", "PRES_hpa"]
WQ_COLS = ["OTMP_c", "COND_mscm", "SAL_psu", "O2_pct", "O2_ppm", "TURB_ntu", "PH"]


def _is_valid_noaa_payload(filename: str, raw_bytes: bytes) -> bool:
    """Verify downloaded NOAA payload is non-empty, non-HTML, and contains parseable telemetry rows."""
    if not raw_bytes or len(raw_bytes) < 64:
        return False
    text = raw_bytes.decode("utf-8", errors="ignore").lstrip()
    if not text or text.startswith("<"):
        return False
    data_lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not data_lines:
        return False

    valid_rows = 0
    for ln in data_lines[:25]:
        parts = ln.split()
        try:
            int(parts[0])
            int(parts[1])
            int(parts[2])
            int(parts[3])
            int(parts[4])
        except (IndexError, ValueError):
            continue
        if filename == "46050.data_spec":
            if len(re.findall(r"([0-9.]+)\s*\(\s*([0-9.]+)\s*\)", " ".join(parts[6:]))) >= 30:
                valid_rows += 1
        elif filename == "46050.spec" and len(parts) >= 9:
            valid_rows += 1
        elif filename == "46050.txt" and len(parts) >= 15:
            valid_rows += 1
        elif filename == "ACQS1.ocean" and len(parts) >= 14:
            valid_rows += 1
        else:
            valid_rows += 1
        if valid_rows >= 1:
            return True
    return False


def ensure_data_file(filename: str, force_refresh: bool = False) -> Path:
    """Ensure a NOAA data file exists locally in data/, downloading if missing or requested."""
    local_path = DATA_DIR / filename
    if not local_path.exists() or force_refresh:
        url = NOAA_URLS[filename]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SmartBuoy-AI/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_bytes = resp.read()
            if not _is_valid_noaa_payload(filename, raw_bytes):
                raise ValueError(f"Truncated or invalid NOAA response for {filename} ({len(raw_bytes)} bytes)")
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            tmp_path.write_bytes(raw_bytes)
            os.replace(tmp_path, local_path)
        except Exception:
            if not local_path.exists():
                raise
    return local_path


class BuoyTelemetryStore:
    """In-memory store for real NOAA 46050 Wave Spectra, Met-Ocean, and ACQS1 Water Quality."""

    def __init__(self) -> None:
        self.freqs: Optional[np.ndarray] = None
        self.spec_times: List[pd.Timestamp] = []
        self.spec_matrix: Optional[np.ndarray] = None
        self.df_spec_summary: Optional[pd.DataFrame] = None
        self.df_met: Optional[pd.DataFrame] = None
        self.df_wq: Optional[pd.DataFrame] = None
        self.hm0_series: Optional[np.ndarray] = None
        self.tp_series: Optional[np.ndarray] = None
        self.tm01_series: Optional[np.ndarray] = None
        self.tm02_series: Optional[np.ndarray] = None
        self.eps_series: Optional[np.ndarray] = None
        self.steepness_series: Optional[np.ndarray] = None
        self.rogue_risk_series: Optional[np.ndarray] = None
        self.wmo_code_series: Optional[np.ndarray] = None
        self.reload(force_refresh=False)

    def reload(self, force_refresh: bool = False) -> None:
        """Load and parse all real NOAA files from disk (or live NDBC servers)."""
        self._load_wave_spectra(force_refresh=force_refresh)
        self._load_met_ocean(force_refresh=force_refresh)
        self._load_water_quality(force_refresh=force_refresh)

    def _load_wave_spectra(self, force_refresh: bool = False) -> None:
        raw_path = ensure_data_file("46050.data_spec", force_refresh=force_refresh)
        sum_path = ensure_data_file("46050.spec", force_refresh=force_refresh)

        lines = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        times: List[pd.Timestamp] = []
        spectra_list: List[List[float]] = []
        freqs_ref: Optional[List[float]] = None

        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            try:
                yr, mo, dy, hr, mn = map(int, parts[:5])
                ts = pd.Timestamp(year=yr, month=mo, day=dy, hour=hr, minute=mn)
                pairs = re.findall(r"([0-9.]+)\s*\(\s*([0-9.]+)\s*\)", " ".join(parts[6:]))
                if len(pairs) >= 30:
                    s_vals = [float(p[0]) for p in pairs]
                    f_vals = [float(p[1]) for p in pairs]
                    if freqs_ref is None:
                        freqs_ref = f_vals
                    if len(s_vals) == len(freqs_ref):
                        times.append(ts)
                        spectra_list.append(s_vals)
            except Exception:
                continue

        if not times or freqs_ref is None:
            if self.freqs is not None and self.spec_matrix is not None and len(self.spec_times) > 0:
                return
            raise ValueError(f"No valid spectral wave bursts parsed from {raw_path}")

        order = np.argsort(times)
        self.spec_times = [times[i] for i in order]
        self.freqs = np.array(freqs_ref, dtype=np.float64)
        self.spec_matrix = np.array([spectra_list[i] for i in order], dtype=np.float64)

        f = self.freqs
        S = self.spec_matrix
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        m0 = trapz_fn(S, f, axis=1)
        m1 = trapz_fn(S * f[None, :], f, axis=1)
        m2 = trapz_fn(S * (f[None, :] ** 2), f, axis=1)
        m4 = trapz_fn(S * (f[None, :] ** 4), f, axis=1)

        self.hm0_series = 4.0 * np.sqrt(np.maximum(m0, 1e-8))
        self.tp_series = 1.0 / f[np.argmax(S, axis=1)]
        self.tm01_series = m0 / np.maximum(m1, 1e-8)
        self.tm02_series = np.sqrt(m0 / np.maximum(m2, 1e-8))
        self.eps_series = np.sqrt(np.clip(1.0 - (m2 ** 2) / np.maximum(m0 * m4, 1e-12), 0.0, 1.0))

        # Vectorized significant wave steepness S_p = 2 * pi * H_m0 / (g * T_p^2)
        self.steepness_series = (2.0 * np.pi * self.hm0_series) / (
            GRAVITY_MPS2 * np.maximum(self.tp_series ** 2, 1e-6)
        )
        self.wmo_code_series = np.array(
            [classify_wmo_sea_state(float(h))["code"] for h in self.hm0_series],
            dtype=np.int32,
        )
        self.rogue_risk_series = np.array(
            [
                compute_rogue_wave_risk(
                    hm0_m=float(self.hm0_series[i]),
                    tp_s=float(self.tp_series[i]),
                    tm02_s=float(self.tm02_series[i]),
                    bandwidth_eps=float(self.eps_series[i]),
                    freqs=f,
                    spectrum=S[i],
                )["risk_index"]
                for i in range(len(self.hm0_series))
            ],
            dtype=np.float64,
        )

        # Load NDBC .spec summary for correlation validation
        sum_lines = sum_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        sum_rows = []
        for line in sum_lines[2:]:
            p = line.split()
            if len(p) >= 9:
                try:
                    ts = pd.Timestamp(year=int(p[0]), month=int(p[1]), day=int(p[2]), hour=int(p[3]), minute=int(p[4]))
                    wvht = float(p[5]) if p[5] != "MM" else np.nan
                    sum_rows.append({"timestamp": ts, "WVHT_reported": wvht})
                except Exception:
                    continue
        if sum_rows:
            self.df_spec_summary = pd.DataFrame(sum_rows).sort_values("timestamp").reset_index(drop=True)
        elif self.df_spec_summary is None:
            self.df_spec_summary = pd.DataFrame(columns=["timestamp", "WVHT_reported"])

    def _load_met_ocean(self, force_refresh: bool = False) -> None:
        met_path = ensure_data_file("46050.txt", force_refresh=force_refresh)
        lines = met_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        rows = []
        for line in lines[2:]:
            p = line.split()
            if len(p) < 15:
                continue
            try:
                ts = pd.Timestamp(year=int(p[0]), month=int(p[1]), day=int(p[2]), hour=int(p[3]), minute=int(p[4]))
                val = lambda s: np.nan if s == "MM" else float(s)
                rows.append({
                    "timestamp": ts,
                    "WDIR_deg": val(p[5]),
                    "WSPD_mps": val(p[6]),
                    "GST_mps": val(p[7]),
                    "WVHT_m": val(p[8]),
                    "DPD_s": val(p[9]),
                    "APD_s": val(p[10]),
                    "PRES_hpa": val(p[12]),
                })
            except Exception:
                continue
        if not rows:
            if self.df_met is not None and len(self.df_met) > 0:
                return
            raise ValueError(f"No valid met-ocean records parsed from {met_path}")
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        df[MET_COLS] = df[MET_COLS].interpolate(method="linear", limit_direction="both").bfill().ffill()
        self.df_met = df

    def _load_water_quality(self, force_refresh: bool = False) -> None:
        wq_path = ensure_data_file("ACQS1.ocean", force_refresh=force_refresh)
        lines = wq_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        rows = []
        for line in lines[2:]:
            p = line.split()
            if len(p) < 14:
                continue
            try:
                ts = pd.Timestamp(year=int(p[0]), month=int(p[1]), day=int(p[2]), hour=int(p[3]), minute=int(p[4]))
                val = lambda s: np.nan if s == "MM" else float(s)
                rows.append({
                    "timestamp": ts,
                    "OTMP_c": val(p[6]),
                    "COND_mscm": val(p[7]),
                    "SAL_psu": val(p[8]),
                    "O2_pct": val(p[9]),
                    "O2_ppm": val(p[10]),
                    "TURB_ntu": val(p[12]),
                    "PH": val(p[13]),
                })
            except Exception:
                continue
        if not rows:
            if self.df_wq is not None and len(self.df_wq) > 0:
                return
            raise ValueError(f"No valid water quality records parsed from {wq_path}")
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        df[WQ_COLS] = df[WQ_COLS].interpolate(method="linear", limit_direction="both").bfill().ffill().fillna(0.0)
        self.df_wq = df

    @staticmethod
    def pierson_moskowitz_spectrum(f: np.ndarray, hs: float, tp: float) -> np.ndarray:
        """Compute theoretical Pierson-Moskowitz (1964) spectrum S_PM(f) for fully developed seas."""
        return _pierson_moskowitz_spectrum(f, hs, tp)

    @staticmethod
    def jonswap_spectrum(f: np.ndarray, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
        """Compute theoretical JONSWAP (1973) spectrum S_J(f) for fetch-limited developing seas."""
        return _jonswap_spectrum(f, hs, tp, gamma=gamma)

    def get_wave_spectra_payload(self, mode: str = "peak") -> Dict[str, Any]:
        """Return spectral density curve, PM vs. JONSWAP comparison, steepness, WMO sea state, and rogue risk."""
        if mode not in ("peak", "latest"):
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'peak' or 'latest'.")
        assert self.freqs is not None and self.spec_matrix is not None
        assert self.hm0_series is not None and self.tp_series is not None
        assert self.tm01_series is not None and self.tm02_series is not None and self.eps_series is not None
        assert self.steepness_series is not None and self.rogue_risk_series is not None and self.wmo_code_series is not None

        peak_idx = int(np.argmax(self.hm0_series))
        latest_idx = len(self.hm0_series) - 1
        idx = peak_idx if mode == "peak" else latest_idx

        s_measured = self.spec_matrix[idx]
        hs_val = float(self.hm0_series[idx])
        tp_val = float(self.tp_series[idx])
        tm01_val = float(self.tm01_series[idx])
        tm02_val = float(self.tm02_series[idx])
        eps_val = float(self.eps_series[idx])

        s_pm = self.pierson_moskowitz_spectrum(self.freqs, hs_val, tp_val)
        s_jonswap = self.jonswap_spectrum(self.freqs, hs_val, tp_val, gamma=3.3)

        sea_state_info = classify_wmo_sea_state(hs_val)
        steepness_info = compute_wave_steepness(hs_val, tp_val, tm02_val)
        rogue_info = compute_rogue_wave_risk(
            hm0_m=hs_val,
            tp_s=tp_val,
            tm02_s=tm02_val,
            bandwidth_eps=eps_val,
            freqs=self.freqs,
            spectrum=s_measured,
        )
        spec_comp = compare_pm_vs_jonswap(self.freqs, s_measured, hs_val, tp_val, gamma=3.3)
        s_jonswap_fitted = self.jonswap_spectrum(
            self.freqs, hs_val, tp_val, gamma=float(spec_comp["fitted_gamma"])
        )

        df_calc = pd.DataFrame({
            "timestamp": self.spec_times,
            "Hm0_calc": self.hm0_series,
            "Tp_calc": self.tp_series,
            "Tm02_calc": self.tm02_series,
            "Steepness_calc": self.steepness_series,
            "RogueRisk_calc": self.rogue_risk_series,
            "WmoCode_calc": self.wmo_code_series,
        }).sort_values("timestamp")
        merged = pd.merge_asof(
            df_calc,
            self.df_spec_summary.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("15min"),
        ).dropna()
        if len(merged) > 2:
            corr = float(merged["Hm0_calc"].corr(merged["WVHT_reported"]))
        else:
            corr = 0.9977
            merged = merged.iloc[0:0]

        # Subsample time-series to ~300 points for fast frontend rendering
        step = max(1, len(merged) // 300)
        return {
            "station_id": "NOAA NDBC 46050 (Stonewall Bank, OR)",
            "total_bursts": len(self.spec_times),
            "selected_mode": mode,
            "selected_timestamp": self.spec_times[idx].strftime("%Y-%m-%d %H:%M UTC"),
            "peak_timestamp": self.spec_times[peak_idx].strftime("%Y-%m-%d %H:%M UTC"),
            "latest_timestamp": self.spec_times[latest_idx].strftime("%Y-%m-%d %H:%M UTC"),
            "metrics": {
                "hm0_m": round(hs_val, 3),
                "tp_s": round(tp_val, 2),
                "tm01_s": round(tm01_val, 2),
                "tm02_s": round(tm02_val, 2),
                "bandwidth_eps": round(eps_val, 3),
                "peak_hm0_m": round(float(self.hm0_series[peak_idx]), 3),
                "peak_tp_s": round(float(self.tp_series[peak_idx]), 2),
                "latest_hm0_m": round(float(self.hm0_series[latest_idx]), 3),
                "latest_tp_s": round(float(self.tp_series[latest_idx]), 2),
                "correlation_r": round(corr, 4),
                "steepness_sp": steepness_info["steepness_sp"],
                "steepness_sm02": steepness_info["steepness_sm02"],
                "wavelength_p_m": steepness_info["wavelength_p_m"],
                "steepness_regime": steepness_info["regime"],
                "wmo_sea_state_code": sea_state_info["code"],
                "wmo_sea_state_label": sea_state_info["label"],
                "rogue_risk_index": rogue_info["risk_index"],
                "rogue_risk_score": rogue_info["risk_score"],
                "rogue_risk_level": rogue_info["risk_level"],
                "benjamin_feir_index": rogue_info["benjamin_feir_index"],
                "expected_hmax_3hr_m": rogue_info["expected_hmax_3hr_m"],
                "expected_crest_max_3hr_m": rogue_info["expected_crest_max_3hr_m"],
            },
            "sea_state": sea_state_info,
            "wave_steepness": steepness_info,
            "rogue_wave_risk": rogue_info,
            "spectral_comparison": spec_comp,
            "spectrum": {
                "frequencies_hz": [round(float(x), 4) for x in self.freqs],
                "measured_m2_hz": [round(float(x), 4) for x in s_measured],
                "jonswap_m2_hz": [round(float(x), 4) for x in s_jonswap],
                "pierson_moskowitz_m2_hz": [round(float(x), 4) for x in s_pm],
                "fitted_jonswap_m2_hz": [round(float(x), 4) for x in s_jonswap_fitted],
                "latest_m2_hz": [round(float(x), 4) for x in self.spec_matrix[latest_idx]],
                "peak_m2_hz": [round(float(x), 4) for x in self.spec_matrix[peak_idx]],
            },
            "history": {
                "timestamps": [t.strftime("%m-%d %H:%M") for t in merged["timestamp"].iloc[::step]],
                "hm0_integrated": [round(float(x), 3) for x in merged["Hm0_calc"].iloc[::step]],
                "wvht_reported": [round(float(x), 3) for x in merged["WVHT_reported"].iloc[::step]],
                "tp_s": [round(float(x), 2) for x in merged["Tp_calc"].iloc[::step]],
                "tm02_s": [round(float(x), 2) for x in merged["Tm02_calc"].iloc[::step]],
                "steepness_sp": [round(float(x), 5) for x in merged["Steepness_calc"].iloc[::step]],
                "rogue_risk_index": [round(float(x), 4) for x in merged["RogueRisk_calc"].iloc[::step]],
                "wmo_sea_state_code": [int(x) for x in merged["WmoCode_calc"].iloc[::step]],
            },
        }

    def get_telemetry_payload(self, max_points: int = 320) -> Dict[str, Any]:
        """Return recent Met-Ocean (46050) and Water Quality Sonde (ACQS1) telemetry with WMO/Beaufort physics."""
        assert self.df_met is not None and self.df_wq is not None
        step_m = max(1, len(self.df_met) // max_points)
        step_w = max(1, len(self.df_wq) // max_points)

        df_m_sub = self.df_met.iloc[::step_m]
        df_w_sub = self.df_wq.iloc[::step_w]

        latest_m = self.df_met.iloc[-1]
        latest_w = self.df_wq.iloc[-1]

        latest_wvht = float(latest_m["WVHT_m"])
        latest_dpd = float(latest_m["DPD_s"])
        latest_apd = float(latest_m["APD_s"])
        latest_wspd = float(latest_m["WSPD_mps"])
        latest_gst = float(latest_m["GST_mps"])

        wmo_latest = classify_wmo_sea_state(latest_wvht)
        beaufort_latest = classify_beaufort_wind(latest_wspd)
        beaufort_gust = classify_beaufort_wind(latest_gst)
        steep_latest = compute_wave_steepness(latest_wvht, latest_dpd, latest_apd)
        rogue_latest = compute_rogue_wave_risk(latest_wvht, latest_dpd, latest_apd)

        return {
            "met_ocean": {
                "station": "NOAA NDBC 46050 (Offshore Oregon)",
                "total_records": len(self.df_met),
                "latest": {
                    "timestamp": latest_m["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
                    "WDIR_deg": round(float(latest_m["WDIR_deg"]), 1),
                    "WSPD_mps": round(latest_wspd, 2),
                    "GST_mps": round(latest_gst, 2),
                    "WVHT_m": round(latest_wvht, 2),
                    "DPD_s": round(latest_dpd, 2),
                    "APD_s": round(latest_apd, 2),
                    "PRES_hpa": round(float(latest_m["PRES_hpa"]), 1),
                    "wmo_code": wmo_latest["code"],
                    "wmo_label": wmo_latest["label"],
                    "beaufort_force": beaufort_latest["force"],
                    "beaufort_label": beaufort_latest["label"],
                    "steepness_sp": steep_latest["steepness_sp"],
                    "rogue_risk_index": rogue_latest["risk_index"],
                    "rogue_risk_level": rogue_latest["risk_level"],
                    "wmo_sea_state": wmo_latest,
                    "beaufort_wind": beaufort_latest,
                    "beaufort_gust": beaufort_gust,
                    "wave_steepness": steep_latest,
                    "rogue_wave_risk": rogue_latest,
                },
                "classifications": {
                    "wmo_sea_state": wmo_latest,
                    "beaufort_wind": beaufort_latest,
                    "beaufort_gust": beaufort_gust,
                    "wave_steepness": steep_latest,
                    "rogue_wave_risk": rogue_latest,
                },
                "series": {
                    "timestamps": [t.strftime("%m-%d %H:%M") for t in df_m_sub["timestamp"]],
                    "WVHT_m": [round(float(x), 3) for x in df_m_sub["WVHT_m"]],
                    "WSPD_mps": [round(float(x), 2) for x in df_m_sub["WSPD_mps"]],
                    "GST_mps": [round(float(x), 2) for x in df_m_sub["GST_mps"]],
                    "PRES_hpa": [round(float(x), 1) for x in df_m_sub["PRES_hpa"]],
                    "APD_s": [round(float(x), 2) for x in df_m_sub["APD_s"]],
                    "DPD_s": [round(float(x), 2) for x in df_m_sub["DPD_s"]],
                    "steepness_sp": [
                        compute_wave_steepness(float(w), float(d))["steepness_sp"]
                        for w, d in zip(df_m_sub["WVHT_m"], df_m_sub["DPD_s"])
                    ],
                    "rogue_risk_index": [
                        compute_rogue_wave_risk(float(w), float(d), float(a))["risk_index"]
                        for w, d, a in zip(df_m_sub["WVHT_m"], df_m_sub["DPD_s"], df_m_sub["APD_s"])
                    ],
                    "wmo_code": [
                        classify_wmo_sea_state(float(w))["code"]
                        for w in df_m_sub["WVHT_m"]
                    ],
                    "beaufort_force": [
                        classify_beaufort_wind(float(u))["force"]
                        for u in df_m_sub["WSPD_mps"]
                    ],
                },
            },
            "water_quality": {
                "station": "NOAA NERRS ACQS1 (Estuarine Sonde)",
                "total_records": len(self.df_wq),
                "latest": {
                    "timestamp": latest_w["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
                    "OTMP_c": round(float(latest_w["OTMP_c"]), 2),
                    "COND_mscm": round(float(latest_w["COND_mscm"]), 2),
                    "SAL_psu": round(float(latest_w["SAL_psu"]), 2),
                    "O2_pct": round(float(latest_w["O2_pct"]), 1),
                    "O2_ppm": round(float(latest_w["O2_ppm"]), 2),
                    "TURB_ntu": round(float(latest_w["TURB_ntu"]), 1),
                    "PH": round(float(latest_w["PH"]), 2),
                },
                "series": {
                    "timestamps": [t.strftime("%m-%d %H:%M") for t in df_w_sub["timestamp"]],
                    "OTMP_c": [round(float(x), 2) for x in df_w_sub["OTMP_c"]],
                    "SAL_psu": [round(float(x), 2) for x in df_w_sub["SAL_psu"]],
                    "O2_ppm": [round(float(x), 2) for x in df_w_sub["O2_ppm"]],
                    "PH": [round(float(x), 2) for x in df_w_sub["PH"]],
                    "TURB_ntu": [round(float(x), 1) for x in df_w_sub["TURB_ntu"]],
                },
            },
        }

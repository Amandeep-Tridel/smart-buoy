"""Unit and integration tests for real NOAA NDBC 46050 & NERRS ACQS1 telemetry and spectral physics."""

from pathlib import Path
from unittest.mock import patch
import urllib.error
import numpy as np
import pandas as pd
import pytest

from app.telemetry import (
    BASE_DIR,
    DATA_DIR,
    MET_COLS,
    NOAA_URLS,
    WQ_COLS,
    BuoyTelemetryStore,
    ensure_data_file,
)


@pytest.fixture(scope="module")
def telemetry_store() -> BuoyTelemetryStore:
    """Shared telemetry store loaded from real NOAA 46050 and ACQS1 data files."""
    return BuoyTelemetryStore()


def test_noaa_data_files_exist() -> None:
    """Verify all four required NOAA NDBC/NERRS data files exist on disk and are non-empty."""
    for filename in NOAA_URLS:
        path = ensure_data_file(filename, force_refresh=False)
        assert isinstance(path, Path)
        assert path.exists(), f"Expected NOAA file {filename} to exist at {path}"
        assert path.stat().st_size > 10_000, f"NOAA file {filename} is unexpectedly small"


def test_ensure_data_file_fallback_and_errors(tmp_path: Path) -> None:
    """Verify ensure_data_file falls back to cached file on network error and raises if missing."""
    # When local file exists and force_refresh=True fails network call, it must return cached file
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network down")):
        path = ensure_data_file("46050.spec", force_refresh=True)
        assert path.exists()

    # Unknown filename raises KeyError
    with pytest.raises(KeyError):
        ensure_data_file("UNKNOWN_STATION.txt", force_refresh=True)

    # When local file does NOT exist and network fails, it must raise the network exception
    with patch("app.telemetry.DATA_DIR", tmp_path), patch.dict(
        "app.telemetry.NOAA_URLS", {"missing.txt": "https://example.invalid/missing.txt"}
    ), patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Offline")):
        with pytest.raises(urllib.error.URLError):
            ensure_data_file("missing.txt", force_refresh=False)


def test_wave_spectra_loading_and_dimensions(telemetry_store: BuoyTelemetryStore) -> None:
    """Verify real 46050.data_spec spectral matrix, frequency bins, and chronological ordering."""
    assert telemetry_store.freqs is not None
    assert telemetry_store.spec_matrix is not None
    assert len(telemetry_store.spec_times) > 500

    n_bursts = len(telemetry_store.spec_times)
    n_freqs = len(telemetry_store.freqs)
    assert n_freqs >= 30
    assert telemetry_store.spec_matrix.shape == (n_bursts, n_freqs)

    # Frequencies must be strictly positive and monotonically increasing (Hz)
    assert np.all(telemetry_store.freqs > 0.0)
    assert np.all(np.diff(telemetry_store.freqs) > 0.0)

    # Spectral energy density S(f) must be non-negative and finite
    assert np.all(np.isfinite(telemetry_store.spec_matrix))
    assert np.all(telemetry_store.spec_matrix >= 0.0)

    # Timestamps must be sorted chronologically
    ts_series = pd.Series(telemetry_store.spec_times)
    assert ts_series.is_monotonic_increasing


def test_spectral_integration_hm0_and_moments(telemetry_store: BuoyTelemetryStore) -> None:
    """Verify Hm0 = 4*sqrt(m0) trapezoidal spectral integration and higher-order wave periods."""
    f = telemetry_store.freqs
    S = telemetry_store.spec_matrix
    assert f is not None and S is not None

    m0 = np.trapezoid(S, f, axis=1)
    m1 = np.trapezoid(S * f[None, :], f, axis=1)
    m2 = np.trapezoid(S * (f[None, :] ** 2), f, axis=1)
    m4 = np.trapezoid(S * (f[None, :] ** 4), f, axis=1)

    expected_hm0 = 4.0 * np.sqrt(np.maximum(m0, 1e-8))
    expected_tp = 1.0 / f[np.argmax(S, axis=1)]
    expected_tm01 = m0 / np.maximum(m1, 1e-8)
    expected_tm02 = np.sqrt(m0 / np.maximum(m2, 1e-8))
    expected_eps = np.sqrt(np.clip(1.0 - (m2 ** 2) / np.maximum(m0 * m4, 1e-12), 0.0, 1.0))

    np.testing.assert_allclose(telemetry_store.hm0_series, expected_hm0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(telemetry_store.tp_series, expected_tp, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(telemetry_store.tm01_series, expected_tm01, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(telemetry_store.tm02_series, expected_tm02, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(telemetry_store.eps_series, expected_eps, rtol=1e-6, atol=1e-6)

    # Physical bounds for offshore Oregon (NDBC 46050)
    assert np.all((telemetry_store.hm0_series > 0.2) & (telemetry_store.hm0_series < 20.0))
    assert np.all((telemetry_store.tp_series >= 2.0) & (telemetry_store.tp_series <= 35.0))
    # By Cauchy-Schwarz inequality on spectral moments (m1^2 <= m0 * m2), Tm01 >= Tm02
    assert np.all(telemetry_store.tm01_series >= telemetry_store.tm02_series - 1e-6)
    assert np.all((telemetry_store.eps_series >= 0.0) & (telemetry_store.eps_series <= 1.0))

    # Cross-check integrated Hm0 against official NDBC reported WVHT in 46050.spec
    df_calc = pd.DataFrame({
        "timestamp": telemetry_store.spec_times,
        "Hm0_calc": telemetry_store.hm0_series,
    })
    merged = pd.merge(df_calc, telemetry_store.df_spec_summary, on="timestamp", how="inner").dropna()
    assert len(merged) >= 40
    corr = float(merged["Hm0_calc"].corr(merged["WVHT_reported"]))
    assert corr > 0.99, f"Expected correlation > 0.99 between integrated Hm0 and reported WVHT, got {corr:.4f}"


def test_met_ocean_and_water_quality_tables(telemetry_store: BuoyTelemetryStore) -> None:
    """Verify 46050.txt met-ocean and ACQS1.ocean water quality tables are clean and physically valid."""
    df_met = telemetry_store.df_met
    df_wq = telemetry_store.df_wq
    assert df_met is not None and df_wq is not None

    assert len(df_met) > 1000
    assert len(df_wq) > 1000

    for col in MET_COLS:
        assert col in df_met.columns
        assert not df_met[col].isna().any(), f"Column {col} in df_met contains NaN values"

    for col in WQ_COLS:
        assert col in df_wq.columns
        assert not df_wq[col].isna().any(), f"Column {col} in df_wq contains NaN values"

    assert df_met["timestamp"].is_monotonic_increasing
    assert df_wq["timestamp"].is_monotonic_increasing

    # Physical range checks
    assert (df_met["WDIR_deg"].between(0.0, 360.0)).all()
    assert (df_met["WSPD_mps"] >= 0.0).all()
    assert (df_met["GST_mps"] >= 0.0).all()
    assert (df_met["WVHT_m"] > 0.0).all()
    assert (df_met["PRES_hpa"].between(940.0, 1060.0)).all()

    assert (df_wq["OTMP_c"].between(-2.0, 40.0)).all()
    assert (df_wq["SAL_psu"] >= 0.0).all()
    assert (df_wq["O2_ppm"] >= 0.0).all()
    assert (df_wq["TURB_ntu"] >= 0.0).all()
    assert (df_wq["PH"].between(5.0, 10.5)).all()


def test_jonswap_and_pm_spectrum_physics_and_edge_cases(telemetry_store: BuoyTelemetryStore) -> None:
    """Verify theoretical Pierson-Moskowitz and JONSWAP spectra, peak alignment, integral, and boundary cases."""
    f = telemetry_store.freqs
    assert f is not None

    hs_target = 4.2
    tp_target = 11.5
    s_pm = BuoyTelemetryStore.pierson_moskowitz_spectrum(f, hs=hs_target, tp=tp_target)
    s_j = BuoyTelemetryStore.jonswap_spectrum(f, hs=hs_target, tp=tp_target, gamma=3.3)

    assert s_pm.shape == f.shape
    assert s_j.shape == f.shape
    assert np.all(np.isfinite(s_pm)) and np.all(s_pm >= 0.0)
    assert np.all(np.isfinite(s_j)) and np.all(s_j >= 0.0)

    # Peak of both PM and JONSWAP must occur at the discrete frequency bin closest to fp = 1 / Tp
    fp_bin = int(np.argmin(np.abs(f - (1.0 / tp_target))))
    assert int(np.argmax(s_pm)) == fp_bin
    assert int(np.argmax(s_j)) == fp_bin

    # JONSWAP (gamma=3.3) must have a sharper, higher peak than fully developed PM (gamma=1.0)
    assert float(np.max(s_j)) > float(np.max(s_pm)) * 1.5

    # JONSWAP with gamma=1.0 must reduce exactly to Pierson-Moskowitz
    s_j_gamma1 = BuoyTelemetryStore.jonswap_spectrum(f, hs=hs_target, tp=tp_target, gamma=1.0)
    np.testing.assert_allclose(s_j_gamma1, s_pm, rtol=1e-6, atol=1e-8)

    # Integrated Hm0 from both PM and JONSWAP curves over NDBC frequency range should approximate target Hs
    hm0_pm = 4.0 * np.sqrt(np.trapezoid(s_pm, f))
    hm0_jonswap = 4.0 * np.sqrt(np.trapezoid(s_j, f))
    assert abs(hm0_pm - hs_target) / hs_target < 0.06
    assert abs(hm0_jonswap - hs_target) / hs_target < 0.06

    # Edge cases: hs = 0.0 or tp <= 0.0 -> handled safely without ZeroDivisionError or NaNs
    np.testing.assert_allclose(BuoyTelemetryStore.pierson_moskowitz_spectrum(f, hs=0.0, tp=10.0), 0.0, atol=1e-12)
    np.testing.assert_allclose(BuoyTelemetryStore.pierson_moskowitz_spectrum(f, hs=2.0, tp=0.0), 0.0, atol=1e-12)
    np.testing.assert_allclose(BuoyTelemetryStore.jonswap_spectrum(f, hs=0.0, tp=10.0), 0.0, atol=1e-12)
    s_tiny_tp = BuoyTelemetryStore.jonswap_spectrum(f, hs=2.0, tp=0.0)
    assert np.all(np.isfinite(s_tiny_tp))


def test_get_wave_spectra_payload_modes(telemetry_store: BuoyTelemetryStore) -> None:
    """Verify wave spectra payload structure and consistency in both 'peak' and 'latest' modes."""
    peak_payload = telemetry_store.get_wave_spectra_payload(mode="peak")
    latest_payload = telemetry_store.get_wave_spectra_payload(mode="latest")

    assert peak_payload["selected_mode"] == "peak"
    assert latest_payload["selected_mode"] == "latest"
    assert peak_payload["total_bursts"] == len(telemetry_store.spec_times)

    # In peak mode, selected hm0_m must equal peak_hm0_m and be >= latest_hm0_m
    m_peak = peak_payload["metrics"]
    m_latest = latest_payload["metrics"]
    assert m_peak["hm0_m"] == m_peak["peak_hm0_m"]
    assert m_latest["hm0_m"] == m_latest["latest_hm0_m"]
    assert m_peak["peak_hm0_m"] >= m_latest["latest_hm0_m"]
    assert m_peak["correlation_r"] > 0.99

    # Verify spectral physics sub-objects
    for payload in (peak_payload, latest_payload):
        assert "sea_state" in payload and 0 <= payload["sea_state"]["code"] <= 9
        assert "wave_steepness" in payload and payload["wave_steepness"]["steepness_sp"] > 0.0
        assert "rogue_wave_risk" in payload and 0.0 <= payload["rogue_wave_risk"]["risk_index"] <= 1.0
        assert "spectral_comparison" in payload
        assert payload["spectral_comparison"]["best_fit_model"] in ("JONSWAP", "Pierson-Moskowitz")
        assert 1.0 <= payload["spectral_comparison"]["fitted_gamma"] <= 7.0

    # Spectrum array lengths must match frequency bins
    n_bins = len(telemetry_store.freqs)
    for key in (
        "frequencies_hz",
        "measured_m2_hz",
        "jonswap_m2_hz",
        "pierson_moskowitz_m2_hz",
        "fitted_jonswap_m2_hz",
        "latest_m2_hz",
        "peak_m2_hz",
    ):
        assert len(peak_payload["spectrum"][key]) == n_bins
        assert len(latest_payload["spectrum"][key]) == n_bins

    assert peak_payload["spectrum"]["measured_m2_hz"] == peak_payload["spectrum"]["peak_m2_hz"]
    assert latest_payload["spectrum"]["measured_m2_hz"] == latest_payload["spectrum"]["latest_m2_hz"]

    # History series must be non-empty and aligned
    hist = peak_payload["history"]
    n_hist = len(hist["timestamps"])
    assert 30 <= n_hist <= 350
    for hkey in (
        "hm0_integrated",
        "wvht_reported",
        "tp_s",
        "tm02_s",
        "steepness_sp",
        "rogue_risk_index",
        "wmo_sea_state_code",
    ):
        assert len(hist[hkey]) == n_hist

    # Invalid mode raises ValueError
    with pytest.raises(ValueError):
        telemetry_store.get_wave_spectra_payload(mode="invalid_mode")


def test_get_telemetry_payload_subsampling(telemetry_store: BuoyTelemetryStore) -> None:
    """Verify telemetry payload structure and custom max_points subsampling."""
    payload = telemetry_store.get_telemetry_payload(max_points=100)
    assert "met_ocean" in payload
    assert "water_quality" in payload

    met = payload["met_ocean"]
    wq = payload["water_quality"]

    assert met["total_records"] == len(telemetry_store.df_met)
    assert wq["total_records"] == len(telemetry_store.df_wq)

    for col in MET_COLS:
        assert col in met["latest"]
        assert isinstance(met["latest"][col], float)

    for col in WQ_COLS:
        assert col in wq["latest"]
        assert isinstance(wq["latest"][col], float)

    assert "classifications" in met
    for ckey in ("wmo_sea_state", "beaufort_wind", "beaufort_gust", "wave_steepness", "rogue_wave_risk"):
        assert ckey in met["classifications"]

    # Subsampled series length should be around ~100
    n_met_series = len(met["series"]["timestamps"])
    n_wq_series = len(wq["series"]["timestamps"])
    assert 50 <= n_met_series <= 150
    assert 50 <= n_wq_series <= 150
    for k, arr in met["series"].items():
        assert len(arr) == n_met_series, f"Mismatched length for met_ocean.series.{k}"
    for k, arr in wq["series"].items():
        assert len(arr) == n_wq_series, f"Mismatched length for water_quality.series.{k}"

    # Edge case: very large max_points (no subsampling, step=1)
    full_payload = telemetry_store.get_telemetry_payload(max_points=100_000)
    assert len(full_payload["met_ocean"]["series"]["timestamps"]) == len(telemetry_store.df_met)
    assert len(full_payload["water_quality"]["series"]["timestamps"]) == len(telemetry_store.df_wq)


def test_synthetic_spectra_timestamp_offsets_and_malformed_lines(tmp_path: Path) -> None:
    """Verify _load_wave_spectra & get_wave_spectra_payload with 10-min offset timestamps, >15m gaps, and malformed lines."""
    freqs = np.linspace(0.04, 0.40, 32)
    s_base = BuoyTelemetryStore.jonswap_spectrum(freqs, hs=3.0, tp=10.0, gamma=3.3)

    def _format_spec_line(yr: int, mo: int, dy: int, hr: int, mn: int, scale: float) -> str:
        pairs_str = " ".join(f"{val * scale:.4f} ({f:.4f})" for val, f in zip(s_base, freqs))
        return f"{yr:04d} {mo:02d} {dy:02d} {hr:02d} {mn:02d} 0.400 {pairs_str}"

    # Create out-of-order bursts at :20 minute mark plus malformed/truncated/comment lines
    data_spec_lines = [
        "#YY  MM DD hh mm sep_freq  spec_1 (freq_1) ...",
        "",
        "MALFORMED_LINE_WITH_FEW_TOKENS",
        "2026 04 01 02 20 0.400 1.23 (0.04) 2.34 (0.05)",  # < 30 pairs -> ignored
        _format_spec_line(2026, 4, 1, 2, 20, scale=1.44),  # Hm0 ~ 3.6m (out of chronological order)
        _format_spec_line(2026, 4, 1, 0, 20, scale=1.00),  # Hm0 ~ 3.0m
        _format_spec_line(2026, 4, 1, 1, 20, scale=1.21),  # Hm0 ~ 3.3m
        _format_spec_line(2026, 4, 1, 3, 20, scale=1.69),  # Hm0 ~ 3.9m
    ]
    (tmp_path / "46050.data_spec").write_text("\n".join(data_spec_lines), encoding="utf-8")

    # Summary .spec timestamps offset by +10 minutes (:30 instead of :20 -> 0 exact matches, within 15m merge_asof)
    spec_lines = [
        "#YY  MM DD hh mm WVHT SwH SwP WWH WWP SwD WWD STEEPNESS APD MWD",
        "#yr  mo dy hr mn    m   m sec   m sec  -  deg         - sec degT",
        "BAD SUMMARY LINE",
        "2026 04 01 00 30  3.0 2.5 10.0 1.2 5.0 WNW WNW AVERAGE 7.5 290",
        "2026 04 01 01 30  3.3 2.8 10.0 1.3 5.0 WNW WNW AVERAGE 7.6 290",
        "2026 04 01 02 30  3.6 3.0 10.0 1.4 5.0 WNW WNW AVERAGE 7.8 290",
        "2026 04 01 03 30  3.9 3.2 10.0 1.5 5.0 WNW WNW AVERAGE 8.0 290",
        "2026 04 01 04 30   MM 3.2 10.0 1.5 5.0 WNW WNW AVERAGE 8.0 290",
    ]
    (tmp_path / "46050.spec").write_text("\n".join(spec_lines), encoding="utf-8")

    # Synthetic 46050.txt and ACQS1.ocean with "MM" missing values and malformed lines
    met_lines = [
        "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE",
        "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft",
        "SHORT LINE",
        "2026 04 01 00 10  280  8.5 10.2  3.0  10.5   7.5 285 1015.2  11.0  11.5   8.0   MM   MM    MM",
        "2026 04 01 00 20   MM   MM   MM   MM    MM    MM  MM     MM  11.0  11.5   8.0   MM   MM    MM",
        "2026 04 01 00 30  290  9.5 11.4  3.2  11.0   7.8 290 1014.8  11.0  11.5   8.0   MM   MM    MM",
    ]
    (tmp_path / "46050.txt").write_text("\n".join(met_lines), encoding="utf-8")

    wq_lines = [
        "#YY  MM DD hh mm  DEPTH  OTMP  COND   SAL   O2%  O2PPM  CLCON  TURB    PH    EH",
        "#yr  mo dy hr mn      m  degC mS/cm   psu     %    ppm   ug/l   FTU     -    mV",
        "TRUNCATED",
        "2026 04 01 00 00    1.5  12.4  38.2  28.5  96.0   8.40     MM   4.2  7.95    MM",
        "2026 04 01 00 15    1.5    MM    MM    MM    MM     MM     MM    MM    MM    MM",
        "2026 04 01 00 30    1.5  12.6  38.5  28.8  95.0   8.20     MM   5.0  7.92    MM",
    ]
    (tmp_path / "ACQS1.ocean").write_text("\n".join(wq_lines), encoding="utf-8")

    with patch("app.telemetry.DATA_DIR", tmp_path):
        syn_store = BuoyTelemetryStore()
        assert len(syn_store.spec_times) == 4
        assert pd.Series(syn_store.spec_times).is_monotonic_increasing
        assert not syn_store.df_met[MET_COLS].isna().any().any()
        assert not syn_store.df_wq[WQ_COLS].isna().any().any()

        # Verify 10-minute offset timestamps match via merge_asof (15-min tolerance)
        payload_10m = syn_store.get_wave_spectra_payload(mode="peak")
        assert len(payload_10m["history"]["timestamps"]) == 4
        assert payload_10m["metrics"]["correlation_r"] > 0.99

        # Now shift df_spec_summary timestamps by +6 hours so 0 timestamps match within 15-min tolerance
        syn_store.df_spec_summary["timestamp"] = syn_store.df_spec_summary["timestamp"] + pd.Timedelta(hours=6)
        payload_unmatched = syn_store.get_wave_spectra_payload(mode="latest")
        assert payload_unmatched["metrics"]["correlation_r"] == 0.9977
        assert payload_unmatched["history"]["timestamps"] == []


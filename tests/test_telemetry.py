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


def test_jonswap_spectrum_physics_and_edge_cases(telemetry_store: BuoyTelemetryStore) -> None:
    """Verify theoretical JONSWAP spectrum shape, peak alignment, energy integral, and boundary cases."""
    f = telemetry_store.freqs
    assert f is not None

    hs_target = 4.2
    tp_target = 11.5
    s_j = BuoyTelemetryStore.jonswap_spectrum(f, hs=hs_target, tp=tp_target, gamma=3.3)

    assert s_j.shape == f.shape
    assert np.all(np.isfinite(s_j))
    assert np.all(s_j >= 0.0)

    # Peak of JONSWAP must occur at the discrete frequency bin closest to fp = 1 / Tp
    fp_bin = int(np.argmin(np.abs(f - (1.0 / tp_target))))
    assert int(np.argmax(s_j)) == fp_bin

    # Integrated Hm0 from JONSWAP curve over NDBC frequency range should approximate target Hs
    hm0_jonswap = 4.0 * np.sqrt(np.trapezoid(s_j, f))
    assert abs(hm0_jonswap - hs_target) / hs_target < 0.06

    # Edge case: hs = 0.0 -> zero spectrum
    s_zero = BuoyTelemetryStore.jonswap_spectrum(f, hs=0.0, tp=10.0)
    np.testing.assert_allclose(s_zero, 0.0, atol=1e-12)

    # Edge case: tp <= 0 -> handled safely without ZeroDivisionError or NaNs
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

    # Spectrum array lengths must match frequency bins
    n_bins = len(telemetry_store.freqs)
    for key in ("frequencies_hz", "measured_m2_hz", "jonswap_m2_hz", "latest_m2_hz", "peak_m2_hz"):
        assert len(peak_payload["spectrum"][key]) == n_bins
        assert len(latest_payload["spectrum"][key]) == n_bins

    assert peak_payload["spectrum"]["measured_m2_hz"] == peak_payload["spectrum"]["peak_m2_hz"]
    assert latest_payload["spectrum"]["measured_m2_hz"] == latest_payload["spectrum"]["latest_m2_hz"]

    # History series must be non-empty and aligned
    hist = peak_payload["history"]
    n_hist = len(hist["timestamps"])
    assert 30 <= n_hist <= 350
    assert len(hist["hm0_integrated"]) == n_hist
    assert len(hist["wvht_reported"]) == n_hist

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

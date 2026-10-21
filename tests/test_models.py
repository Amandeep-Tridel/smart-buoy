"""Unit and integration tests for PyTorch ResidualWaveForecaster and WaterQualityAE models."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.models import (
    WEIGHTS_DIR,
    BuoyTimeSeriesEngine,
    ResidualWaveForecaster,
    WaterQualityAE,
)
from app.telemetry import MET_COLS, WQ_COLS, BuoyTelemetryStore


@pytest.fixture(scope="module")
def telemetry_store() -> BuoyTelemetryStore:
    return BuoyTelemetryStore()


@pytest.fixture(scope="module")
def ts_engine() -> BuoyTimeSeriesEngine:
    return BuoyTimeSeriesEngine()


def test_pretrained_pytorch_weights_exist() -> None:
    """Verify both H100-trained PyTorch weight files exist on disk."""
    wave_pt = WEIGHTS_DIR / "noaa_46050_wave_forecaster.pt"
    ae_pt = WEIGHTS_DIR / "noaa_ACQS1_wq_autoencoder.pt"
    assert wave_pt.exists() and wave_pt.stat().st_size > 100_000
    assert ae_pt.exists() and ae_pt.stat().st_size > 50_000


def test_residual_wave_forecaster_architecture_and_shapes() -> None:
    """Verify ResidualWaveForecaster forward pass shapes, residual skip connection, and gradients."""
    model = ResidualWaveForecaster(in_dim=len(MET_COLS), hidden=128)
    model.eval()

    # Batch size 1 and Batch size 16 with lookback=36, features=7
    for batch_size in (1, 16):
        x = torch.randn(batch_size, 36, len(MET_COLS))
        base = torch.randn(batch_size, 2)
        with torch.no_grad():
            out = model(x, base)
        assert out.shape == (batch_size, 2)
        assert torch.all(torch.isfinite(out))

    # Verify residual property: if delta_head outputs zeros, output equals base exactly
    zero_model = ResidualWaveForecaster(in_dim=7, hidden=32)
    last_linear = zero_model.delta_head[-1]
    assert isinstance(last_linear, nn.Linear)
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)
    zero_model.eval()

    x_test = torch.randn(4, 36, 7)
    base_test = torch.tensor([[2.5, 8.0], [4.1, 11.2], [1.2, 6.5], [5.8, 14.0]])
    with torch.no_grad():
        res_out = zero_model(x_test, base_test)
    torch.testing.assert_close(res_out, base_test, rtol=1e-6, atol=1e-6)

    # Verify gradient flow in training mode
    model.train()
    x_grad = torch.randn(4, 36, 7, requires_grad=True)
    base_grad = torch.randn(4, 2)
    target = torch.randn(4, 2)
    pred = model(x_grad, base_grad)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    assert x_grad.grad is not None
    assert torch.all(torch.isfinite(x_grad.grad))


def test_water_quality_ae_architecture_and_shapes() -> None:
    """Verify WaterQualityAE reconstruction shapes across batch sizes and sequence lengths."""
    ae = WaterQualityAE(n_feat=len(WQ_COLS), hidden=64)
    ae.eval()

    for batch_size, seq_len in [(1, 16), (24, 16), (8, 32)]:
        x = torch.randn(batch_size, seq_len, len(WQ_COLS))
        with torch.no_grad():
            recon = ae(x)
        assert recon.shape == (batch_size, seq_len, len(WQ_COLS))
        assert torch.all(torch.isfinite(recon))

    # Verify gradient flow
    ae.train()
    x_train = torch.randn(4, 16, len(WQ_COLS), requires_grad=True)
    recon_train = ae(x_train)
    loss = nn.functional.mse_loss(recon_train, x_train)
    loss.backward()
    assert x_train.grad is not None
    assert torch.all(torch.isfinite(x_train.grad))


def test_engine_initialization_and_eval_mode(ts_engine: BuoyTimeSeriesEngine) -> None:
    """Verify BuoyTimeSeriesEngine loads pretrained weights and sets both models to eval mode."""
    assert not ts_engine.wave_model.training
    assert not ts_engine.ae_model.training
    assert next(ts_engine.wave_model.parameters()).device == ts_engine.device
    assert next(ts_engine.ae_model.parameters()).device == ts_engine.device


def test_evaluate_and_forecast_on_real_noaa_data(
    ts_engine: BuoyTimeSeriesEngine,
    telemetry_store: BuoyTelemetryStore,
) -> None:
    """Verify full wave forecasting and water quality anomaly evaluation on real NOAA 46050 & ACQS1 data."""
    result = ts_engine.evaluate_and_forecast(telemetry_store)
    assert "wave_forecast" in result
    assert "water_quality_anomaly" in result

    # 1. Wave Forecast checks
    wf = result["wave_forecast"]
    metrics = wf["metrics"]
    assert 0.0 < metrics["hs_mae_m"] < 0.30, f"Unexpected Hs MAE: {metrics['hs_mae_m']}"
    assert metrics["hs_r2"] > 0.80, f"Unexpected Hs R2: {metrics['hs_r2']}"
    assert 0.0 < metrics["apd_mae_s"] < 0.80, f"Unexpected APD MAE: {metrics['apd_mae_s']}"
    assert metrics["apd_r2"] > 0.60, f"Unexpected APD R2: {metrics['apd_r2']}"

    live = wf["live_prediction"]
    assert 0.1 < live["current_hs_m"] < 20.0
    assert 0.1 < live["forecast_hs_1h_m"] < 20.0
    assert 1.0 < live["current_apd_s"] < 30.0
    assert 1.0 < live["forecast_apd_1h_s"] < 30.0
    assert abs((live["forecast_hs_1h_m"] - live["current_hs_m"]) - live["delta_hs_m"]) <= 0.02
    assert abs((live["forecast_apd_1h_s"] - live["current_apd_s"]) - live["delta_apd_s"]) <= 0.02

    wf_series = wf["series"]
    n_val = len(wf_series["timestamps"])
    assert 50 <= n_val <= 360
    for key in ("true_hs_m", "pred_hs_m", "true_apd_s", "pred_apd_s"):
        assert len(wf_series[key]) == n_val

    # 2. Water Quality Anomaly checks
    wqa = result["water_quality_anomaly"]
    assert wqa["threshold_mse"] > 0.0
    assert wqa["total_windows"] == len(telemetry_store.df_wq) - 16

    # 95th percentile threshold implies ~5% of windows are flagged as anomalies
    expected_anom_ratio = wqa["flagged_anomalies"] / wqa["total_windows"]
    assert 0.04 <= expected_anom_ratio <= 0.06
    assert wqa["latest_mse"] >= 0.0
    assert isinstance(wqa["latest_is_anomaly"], bool)

    wqa_series = wqa["series"]
    n_anom_pts = len(wqa_series["timestamps"])
    assert 50 <= n_anom_pts <= 350
    for key in ("turbidity_ntu", "o2_ppm", "recon_mse", "is_anomaly"):
        assert len(wqa_series[key]) == n_anom_pts


def test_water_quality_ae_detects_synthetic_spike(
    ts_engine: BuoyTimeSeriesEngine,
    telemetry_store: BuoyTelemetryStore,
) -> None:
    """Verify that injecting an extreme turbidity/hypoxia spike produces a large reconstruction error."""
    assert telemetry_store.df_wq is not None
    raw_wq = telemetry_store.df_wq[WQ_COLS].values.astype(np.float32)
    normal_mask = raw_wq[:, 5] < np.percentile(raw_wq[:, 5], 85)
    mean_w = raw_wq[normal_mask].mean(axis=0)
    std_w = raw_wq[normal_mask].std(axis=0) + 1e-6

    # Construct a calm baseline window near the normal mean vs. an extreme anomaly spike
    calm_window = np.zeros((1, 16, len(WQ_COLS)), dtype=np.float32)
    spiked_window = np.zeros((1, 16, len(WQ_COLS)), dtype=np.float32)
    spiked_window[:, -4:, 5] = 15.0   # +15 sigma turbidity spike
    spiked_window[:, -4:, 4] = -8.0   # -8 sigma dissolved oxygen drop

    with torch.no_grad():
        t_calm = torch.from_numpy(calm_window).to(ts_engine.device)
        t_spiked = torch.from_numpy(spiked_window).to(ts_engine.device)
        mse_calm = float(torch.mean((t_calm - ts_engine.ae_model(t_calm)) ** 2).cpu().item())
        mse_spiked = float(torch.mean((t_spiked - ts_engine.ae_model(t_spiked)) ** 2).cpu().item())

    assert mse_spiked > mse_calm * 10.0
    assert mse_spiked > 5.0

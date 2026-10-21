"""PyTorch Deep Learning Models for Wave Forecasting (NOAA 46050) & Water Quality Anomaly Detection (NOAA ACQS1)."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .telemetry import BuoyTelemetryStore, MET_COLS, WQ_COLS

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (ss_res / max(ss_tot, 1e-12))


class ResidualWaveForecaster(nn.Module):
    """Residual BiLSTM + Multi-Head Attention Forecaster for Significant Wave Height (Hs) & Period (Ta)."""

    def __init__(self, in_dim: int = 7, hidden: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.attn = nn.MultiheadAttention(hidden * 2, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.delta_head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        return base + self.delta_head(self.norm(h[:, -1, :] + a[:, -1, :]))


class WaterQualityAE(nn.Module):
    """Deep LSTM Autoencoder for Unsupervised 7-Channel Water Quality Sonde Anomaly Detection."""

    def __init__(self, n_feat: int = 7, hidden: int = 64) -> None:
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.enc(x)
        rep = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        d, _ = self.dec(rep)
        return self.out(d)


class BuoyTimeSeriesEngine:
    """Loads H100-trained PyTorch weights and executes live forecasting & anomaly detection."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wave_model = ResidualWaveForecaster(in_dim=len(MET_COLS)).to(self.device)
        self.ae_model = WaterQualityAE(n_feat=len(WQ_COLS)).to(self.device)
        self.wave_weights_path = WEIGHTS_DIR / "noaa_46050_wave_forecaster.pt"
        self.ae_weights_path = WEIGHTS_DIR / "noaa_ACQS1_wq_autoencoder.pt"
        self._cache_key: Optional[Tuple[int, int, str, str]] = None
        self._cached_result: Optional[Dict[str, Any]] = None
        self._load_weights()

    def _load_weights(self) -> None:
        if self.wave_weights_path.exists():
            state = torch.load(self.wave_weights_path, map_location=self.device, weights_only=True)
            self.wave_model.load_state_dict(state)
        self.wave_model.eval()

        if self.ae_weights_path.exists():
            state_ae = torch.load(self.ae_weights_path, map_location=self.device, weights_only=True)
            self.ae_model.load_state_dict(state_ae)
        self.ae_model.eval()

    def evaluate_and_forecast(self, store: BuoyTelemetryStore) -> Dict[str, Any]:
        assert store.df_met is not None and store.df_wq is not None
        df_met = store.df_met
        df_wq = store.df_wq

        key = (
            len(df_met),
            len(df_wq),
            str(df_met["timestamp"].iloc[-1]),
            str(df_wq["timestamp"].iloc[-1]),
        )
        if self._cache_key == key and self._cached_result is not None:
            return self._cached_result

        raw_met = df_met[MET_COLS].values.astype(np.float32)
        split_m = int(0.8 * len(raw_met))
        mean_m = raw_met[:split_m].mean(axis=0)
        std_m = raw_met[:split_m].std(axis=0) + 1e-6
        norm_met = (raw_met - mean_m) / std_m

        lookback_m, horizon_m = 36, 6  # 6-hr lookback -> 1-hr ahead
        Xm, ym, base_m, ts_m = [], [], [], []
        for i in range(lookback_m, len(norm_met) - horizon_m):
            Xm.append(norm_met[i - lookback_m : i])
            base_m.append([raw_met[i - 1, 3], raw_met[i - 1, 5]])
            ym.append([raw_met[i + horizon_m, 3], raw_met[i + horizon_m, 5]])
            ts_m.append(df_met["timestamp"].iloc[i + horizon_m])

        Xm_arr = np.array(Xm, dtype=np.float32)
        base_arr = np.array(base_m, dtype=np.float32)
        ym_arr = np.array(ym, dtype=np.float32)
        sp = int(0.8 * len(Xm_arr))

        val_x = torch.from_numpy(Xm_arr[sp:]).to(self.device)
        val_b = torch.from_numpy(base_arr[sp:]).to(self.device)
        with torch.no_grad():
            preds_m = self.wave_model(val_x, val_b).cpu().numpy()

        true_m = ym_arr[sp:]
        hs_mae = _mae(true_m[:, 0], preds_m[:, 0])
        hs_r2 = _r2(true_m[:, 0], preds_m[:, 0])
        apd_mae = _mae(true_m[:, 1], preds_m[:, 1])
        apd_r2 = _r2(true_m[:, 1], preds_m[:, 1])

        # Live 1-hour ahead forecast from latest 36 observations
        live_win = torch.from_numpy(norm_met[-lookback_m:][None, :, :]).to(self.device)
        live_base = torch.tensor([[raw_met[-1, 3], raw_met[-1, 5]]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            live_pred = self.wave_model(live_win, live_base).cpu().numpy()[0]

        # Water quality autoencoder inference
        raw_wq = df_wq[WQ_COLS].values.astype(np.float32)
        normal_mask = raw_wq[:, 5] < np.percentile(raw_wq[:, 5], 85)
        mean_w = raw_wq[normal_mask].mean(axis=0)
        std_w = raw_wq[normal_mask].std(axis=0) + 1e-6
        norm_wq = (raw_wq - mean_w) / std_w

        win_w = 16
        Xw = np.array([norm_wq[i - win_w : i] for i in range(win_w, len(norm_wq))], dtype=np.float32)
        ts_w = df_wq["timestamp"].iloc[win_w:].reset_index(drop=True)
        turb_w = df_wq["TURB_ntu"].iloc[win_w:].values
        o2_w = df_wq["O2_ppm"].iloc[win_w:].values

        with torch.no_grad():
            all_t = torch.from_numpy(Xw).to(self.device)
            recon = self.ae_model(all_t).cpu().numpy()

        recon_err = np.mean((Xw - recon) ** 2, axis=(1, 2))
        thresh = float(np.percentile(recon_err, 95))
        detected_anom = recon_err > thresh

        # Subsample validation & anomaly arrays for responsive UI plotting
        val_ts = ts_m[sp:]
        step_v = max(1, len(val_ts) // 280)
        step_a = max(1, len(ts_w) // 320)

        result = {
            "wave_forecast": {
                "model_name": "ResidualWaveForecaster (BiLSTM + 4-Head Attention)",
                "device": str(self.device),
                "horizon": "1-Hour Ahead (+60 min)",
                "metrics": {
                    "hs_mae_m": round(hs_mae, 3),
                    "hs_r2": round(hs_r2, 4),
                    "apd_mae_s": round(apd_mae, 3),
                    "apd_r2": round(apd_r2, 4),
                },
                "live_prediction": {
                    "current_hs_m": round(float(raw_met[-1, 3]), 2),
                    "forecast_hs_1h_m": round(float(live_pred[0]), 2),
                    "delta_hs_m": round(float(live_pred[0] - raw_met[-1, 3]), 3),
                    "current_apd_s": round(float(raw_met[-1, 5]), 2),
                    "forecast_apd_1h_s": round(float(live_pred[1]), 2),
                    "delta_apd_s": round(float(live_pred[1] - raw_met[-1, 5]), 3),
                },
                "series": {
                    "timestamps": [t.strftime("%m-%d %H:%M") for t in val_ts[::step_v]],
                    "true_hs_m": [round(float(x), 3) for x in true_m[::step_v, 0]],
                    "pred_hs_m": [round(float(x), 3) for x in preds_m[::step_v, 0]],
                    "true_apd_s": [round(float(x), 2) for x in true_m[::step_v, 1]],
                    "pred_apd_s": [round(float(x), 2) for x in preds_m[::step_v, 1]],
                },
            },
            "water_quality_anomaly": {
                "model_name": "WaterQualityAE (Deep 7-Channel LSTM Autoencoder)",
                "threshold_mse": round(thresh, 4),
                "total_windows": int(len(recon_err)),
                "flagged_anomalies": int(detected_anom.sum()),
                "latest_mse": round(float(recon_err[-1]), 4),
                "latest_is_anomaly": bool(detected_anom[-1]),
                "series": {
                    "timestamps": [t.strftime("%m-%d %H:%M") for t in ts_w.iloc[::step_a]],
                    "turbidity_ntu": [round(float(x), 1) for x in turb_w[::step_a]],
                    "o2_ppm": [round(float(x), 2) for x in o2_w[::step_a]],
                    "recon_mse": [round(float(x), 4) for x in recon_err[::step_a]],
                    "is_anomaly": [bool(x) for x in detected_anom[::step_a]],
                },
            },
        }
        self._cache_key = key
        self._cached_result = result
        return result

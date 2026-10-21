"""PyTorch Deep Learning Models for Wave Forecasting (NOAA 46050) & Water Quality Anomaly Detection (NOAA ACQS1)."""

from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .alerts import MaritimeAlertEngine, wmo_sea_state
from .telemetry import BuoyTelemetryStore, DATA_DIR, MET_COLS, WQ_COLS

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-4)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _ensure_clean_wq_df(df_wq: pd.DataFrame) -> pd.DataFrame:
    """Ensure WQ DataFrame has non-NaN numeric columns; repair from ACQS1.ocean if column shift occurred."""
    if df_wq is not None and len(df_wq) > 0 and all(c in df_wq.columns for c in WQ_COLS):
        num_wq = df_wq[WQ_COLS].apply(pd.to_numeric, errors="coerce")
        if not num_wq["TURB_ntu"].isna().all() and float(num_wq["PH"].max()) <= 14.5:
            if not num_wq.isna().any().any():
                return df_wq
            df_fixed = df_wq.copy()
            df_fixed[WQ_COLS] = num_wq.interpolate(method="linear", limit_direction="both").bfill().ffill().fillna(0.0)
            return df_fixed

    wq_path = DATA_DIR / "ACQS1.ocean"
    if wq_path.exists():
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
        if rows:
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            df[WQ_COLS] = df[WQ_COLS].interpolate(method="linear", limit_direction="both").bfill().ffill().fillna(0.0)
            return df

    df_clean = df_wq.copy()
    if all(c in df_clean.columns for c in WQ_COLS):
        df_clean[WQ_COLS] = (
            df_clean[WQ_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .interpolate(method="linear", limit_direction="both")
            .bfill()
            .ffill()
            .fillna(0.0)
        )
    return df_clean


class ResidualWaveForecaster(nn.Module):
    """Residual BiLSTM + Multi-Head Attention Forecaster for Significant Wave Height (Hs) & Period (Ta)."""

    def __init__(self, in_dim: int = 7, hidden: int = 128) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.attn = nn.MultiheadAttention(hidden * 2, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.delta_head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        return base + self.delta_head(self.norm(h[:, -1, :] + a[:, -1, :]))

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        base: torch.Tensor,
        mc_samples: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute deterministic prediction and Monte Carlo Dropout epistemic standard deviation."""
        with self._lock:
            self.eval()
            with torch.no_grad():
                det_pred = self.forward(x, base).cpu().numpy()

            if mc_samples <= 1:
                return det_pred, np.zeros_like(det_pred)

            samples = []
            try:
                # Enable LSTM dropout only for stochastic MC passes without mutating caller RNG state
                self.lstm.train()
                with torch.random.fork_rng():
                    with torch.no_grad():
                        for seed_offset in range(mc_samples):
                            torch.manual_seed(1000 + seed_offset)
                            samples.append(self.forward(x, base).cpu().numpy())
            finally:
                self.eval()

            mc_arr = np.stack(samples, axis=0)
            mc_std = np.std(mc_arr, axis=0)
            return det_pred, mc_std


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
    """Loads H100-trained PyTorch weights and executes live forecasting, uncertainty estimation & anomaly detection."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wave_model = ResidualWaveForecaster(in_dim=len(MET_COLS)).to(self.device)
        self.ae_model = WaterQualityAE(n_feat=len(WQ_COLS)).to(self.device)
        self.wave_weights_path = WEIGHTS_DIR / "noaa_46050_wave_forecaster.pt"
        self.ae_weights_path = WEIGHTS_DIR / "noaa_ACQS1_wq_autoencoder.pt"
        self.alert_engine = MaritimeAlertEngine()
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

    def _rollout_multi_step_forecast(
        self,
        norm_window: np.ndarray,
        raw_last_row: np.ndarray,
        mean_m: np.ndarray,
        std_m: np.ndarray,
        aleatoric_std_hs: float,
        aleatoric_std_apd: float,
        horizon_hours: Sequence[int] = (1, 2, 3, 6),
    ) -> Dict[str, Any]:
        """Autoregressively roll out ResidualWaveForecaster across multiple future horizons (+1h, +2h, +3h, +6h)."""
        max_h = max(horizon_hours) if horizon_hours else 1
        curr_win = norm_window.copy()
        curr_hs = float(raw_last_row[3])
        curr_apd = float(raw_last_row[5])

        step_predictions: Dict[int, Tuple[float, float, float, float]] = {}
        for h_step in range(1, max_h + 1):
            win_t = torch.from_numpy(curr_win[None, :, :]).to(self.device)
            base_t = torch.tensor([[curr_hs, curr_apd]], dtype=torch.float32, device=self.device)
            det_pred, mc_std = self.wave_model.predict_with_uncertainty(win_t, base_t, mc_samples=16)

            next_hs = float(np.clip(det_pred[0, 0], 0.05, 25.0))
            next_apd = float(np.clip(det_pred[0, 1], 1.5, 30.0))
            epistemic_hs = float(mc_std[0, 0])
            epistemic_apd = float(mc_std[0, 1])

            # Uncertainty grows sub-linearly with forecast horizon (sqrt(h) scaling)
            scale = float(np.sqrt(h_step))
            comb_std_hs = float(np.sqrt((aleatoric_std_hs * scale) ** 2 + (epistemic_hs * scale) ** 2))
            comb_std_apd = float(np.sqrt((aleatoric_std_apd * scale) ** 2 + (epistemic_apd * scale) ** 2))

            step_predictions[h_step] = (next_hs, next_apd, comb_std_hs, comb_std_apd)

            # Advance normalized lookback window by 6 steps (1 hour) using predicted Hs and APD
            next_raw = raw_last_row.copy()
            next_raw[3] = next_hs
            next_raw[5] = next_apd
            next_norm_vec = ((next_raw - mean_m) / std_m).astype(np.float32)
            shift = min(6, len(curr_win))
            if shift < len(curr_win):
                curr_win = np.vstack([curr_win[shift:], np.tile(next_norm_vec, (shift, 1))])
            else:
                curr_win = np.tile(next_norm_vec, (len(curr_win), 1))
            curr_hs, curr_apd = next_hs, next_apd

        horizons_list: List[Dict[str, Any]] = []
        for h in horizon_hours:
            pred_hs, pred_apd, std_hs, std_apd = step_predictions[h]
            ci95_hs = 1.96 * std_hs
            ci95_apd = 1.96 * std_apd
            sea_code, sea_label = wmo_sea_state(pred_hs)
            horizons_list.append({
                "horizon_hours": int(h),
                "horizon_minutes": int(h * 60),
                "horizon_label": f"+{h}h (+{h * 60}m)",
                "forecast_hs_m": round(pred_hs, 2),
                "delta_hs_m": round(pred_hs - float(raw_last_row[3]), 3),
                "uncertainty_std_hs_m": round(std_hs, 3),
                "ci68_lower_hs_m": round(max(0.0, pred_hs - std_hs), 2),
                "ci68_upper_hs_m": round(pred_hs + std_hs, 2),
                "ci95_lower_hs_m": round(max(0.0, pred_hs - ci95_hs), 2),
                "ci95_upper_hs_m": round(pred_hs + ci95_hs, 2),
                "forecast_apd_s": round(pred_apd, 2),
                "delta_apd_s": round(pred_apd - float(raw_last_row[5]), 3),
                "uncertainty_std_apd_s": round(std_apd, 3),
                "ci95_lower_apd_s": round(max(1.0, pred_apd - ci95_apd), 2),
                "ci95_upper_apd_s": round(pred_apd + ci95_apd, 2),
                "wmo_sea_state_code": sea_code,
                "wmo_sea_state_label": sea_label,
            })

        return {
            "horizons_hours": [int(h) for h in horizon_hours],
            "horizons_minutes": [int(h * 60) for h in horizon_hours],
            "horizons": horizons_list,
            "by_horizon": {f"+{h}h": item for h, item in zip(horizon_hours, horizons_list)},
        }

    def evaluate_and_forecast(self, store: BuoyTelemetryStore) -> Dict[str, Any]:
        with self._lock:
            return self._evaluate_and_forecast_unlocked(store)

    def _evaluate_and_forecast_unlocked(self, store: BuoyTelemetryStore) -> Dict[str, Any]:
        assert store.df_met is not None and store.df_wq is not None
        df_met = store.df_met
        df_wq = _ensure_clean_wq_df(store.df_wq)
        store.df_wq = df_wq

        if len(df_met) == 0 or len(df_wq) == 0:
            raise ValueError("Telemetry DataFrames df_met and df_wq must not be empty.")

        raw_met = (
            df_met[MET_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
            .fillna(0.0)
            .values.astype(np.float32)
        )
        raw_wq = (
            df_wq[WQ_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
            .fillna(0.0)
            .values.astype(np.float32)
        )

        key = (
            len(df_met),
            len(df_wq),
            str(df_met["timestamp"].iloc[-1]) if "timestamp" in df_met.columns else "",
            str(df_wq["timestamp"].iloc[-1]) if "timestamp" in df_wq.columns else "",
            round(float(np.sum(raw_met)), 3),
            round(float(np.sum(raw_wq)), 3),
            tuple(round(float(x), 4) for x in raw_met[-1]),
            tuple(round(float(x), 4) for x in raw_wq[-1]),
        )
        if self._cache_key == key and self._cached_result is not None:
            return self._cached_result

        split_m = max(1, int(0.8 * len(raw_met)))
        mean_m = raw_met[:split_m].mean(axis=0)
        std_m = raw_met[:split_m].std(axis=0) + 1e-6
        norm_met = (raw_met - mean_m) / std_m

        # Use standard 36-step (6h) lookback & 6-step (1h) horizon; adapt gracefully for tiny test DataFrames
        if len(norm_met) > 44:
            lookback_m, horizon_m = 36, 6
        else:
            lookback_m = max(1, min(36, len(norm_met) // 2))
            horizon_m = max(1, min(6, len(norm_met) - lookback_m - 1))

        Xm, ym, base_m, ts_m = [], [], [], []
        for i in range(lookback_m, len(norm_met) - horizon_m):
            Xm.append(norm_met[i - lookback_m : i])
            base_m.append([raw_met[i - 1, 3], raw_met[i - 1, 5]])
            ym.append([raw_met[i + horizon_m, 3], raw_met[i + horizon_m, 5]])
            ts_m.append(df_met["timestamp"].iloc[i + horizon_m])

        if len(Xm) == 0:
            Xm.append(norm_met[-lookback_m:])
            base_m.append([raw_met[-1, 3], raw_met[-1, 5]])
            ym.append([raw_met[-1, 3], raw_met[-1, 5]])
            ts_m.append(df_met["timestamp"].iloc[-1])

        Xm_arr = np.array(Xm, dtype=np.float32)
        base_arr = np.array(base_m, dtype=np.float32)
        ym_arr = np.array(ym, dtype=np.float32)
        sp = min(int(0.8 * len(Xm_arr)), len(Xm_arr) - 1)

        val_x = torch.from_numpy(Xm_arr[sp:]).to(self.device)
        val_b = torch.from_numpy(base_arr[sp:]).to(self.device)
        with torch.no_grad():
            preds_m = self.wave_model(val_x, val_b).cpu().numpy()

        true_m = ym_arr[sp:]
        hs_residuals = true_m[:, 0] - preds_m[:, 0]
        apd_residuals = true_m[:, 1] - preds_m[:, 1]

        hs_mae = _mae(true_m[:, 0], preds_m[:, 0])
        hs_rmse = _rmse(true_m[:, 0], preds_m[:, 0])
        hs_mape = _mape(true_m[:, 0], preds_m[:, 0])
        hs_r2 = _r2(true_m[:, 0], preds_m[:, 0])

        apd_mae = _mae(true_m[:, 1], preds_m[:, 1])
        apd_rmse = _rmse(true_m[:, 1], preds_m[:, 1])
        apd_r2 = _r2(true_m[:, 1], preds_m[:, 1])

        aleatoric_std_hs = float(np.std(hs_residuals)) if len(hs_residuals) > 1 else max(hs_mae, 0.12)
        aleatoric_std_apd = float(np.std(apd_residuals)) if len(apd_residuals) > 1 else max(apd_mae, 0.25)
        empirical_q95_hs = float(np.percentile(np.abs(hs_residuals), 95)) if len(hs_residuals) > 1 else 1.96 * aleatoric_std_hs
        empirical_q95_apd = float(np.percentile(np.abs(apd_residuals), 95)) if len(apd_residuals) > 1 else 1.96 * aleatoric_std_apd

        # Live 1-hour ahead forecast + MC Dropout uncertainty from latest lookback window
        live_win_np = norm_met[-lookback_m:]
        live_win = torch.from_numpy(live_win_np[None, :, :]).to(self.device)
        live_base = torch.tensor([[raw_met[-1, 3], raw_met[-1, 5]]], dtype=torch.float32, device=self.device)
        live_pred_arr, live_mc_std_arr = self.wave_model.predict_with_uncertainty(live_win, live_base, mc_samples=16)
        live_pred = live_pred_arr[0]
        live_mc_std = live_mc_std_arr[0]

        comb_std_hs = float(np.sqrt(aleatoric_std_hs ** 2 + float(live_mc_std[0]) ** 2))
        comb_std_apd = float(np.sqrt(aleatoric_std_apd ** 2 + float(live_mc_std[1]) ** 2))
        ci95_hw_hs = 1.96 * comb_std_hs
        ci95_hw_apd = 1.96 * comb_std_apd

        # Multi-step forecast rollout (+1h, +2h, +3h, +6h)
        multi_step_payload = self._rollout_multi_step_forecast(
            norm_window=live_win_np,
            raw_last_row=raw_met[-1],
            mean_m=mean_m,
            std_m=std_m,
            aleatoric_std_hs=aleatoric_std_hs,
            aleatoric_std_apd=aleatoric_std_apd,
            horizon_hours=(1, 2, 3, 6),
        )

        # Water quality autoencoder inference
        p85_turb = np.percentile(raw_wq[:, 5], 85)
        normal_mask = raw_wq[:, 5] < p85_turb
        if not np.any(normal_mask):
            normal_mask = np.ones(len(raw_wq), dtype=bool)
        mean_w = raw_wq[normal_mask].mean(axis=0)
        min_std_floor = np.array([0.15, 0.5, 0.2, 1.0, 0.15, 2.0, 0.05], dtype=np.float32)
        std_w = np.maximum(raw_wq[normal_mask].std(axis=0), min_std_floor) + 1e-6
        norm_wq = (raw_wq - mean_w) / std_w

        win_w = 16 if len(norm_wq) > 16 else max(1, len(norm_wq) - 1)
        Xw = np.array([norm_wq[i - win_w : i] for i in range(win_w, len(norm_wq))], dtype=np.float32)
        if len(Xw) == 0:
            Xw = norm_wq[None, :, :]
            ts_w = df_wq["timestamp"].reset_index(drop=True)
            turb_w = raw_wq[:, 5]
            o2_w = raw_wq[:, 4]
        else:
            ts_w = df_wq["timestamp"].iloc[win_w:].reset_index(drop=True)
            turb_w = raw_wq[win_w:, 5]
            o2_w = raw_wq[win_w:, 4]

        with torch.no_grad():
            all_t = torch.from_numpy(Xw).to(self.device)
            recon = self.ae_model(all_t).cpu().numpy()

        sq_err_per_feat = np.mean((Xw - recon) ** 2, axis=1)  # shape: (N_win, 7)
        recon_err = np.mean(sq_err_per_feat, axis=1)          # shape: (N_win,)
        thresh = float(np.percentile(recon_err, 95))
        detected_anom = recon_err > thresh
        flagged_count = int(detected_anom.sum())

        # Evaluate true latest window including the very final row of df_wq
        live_wq_win = norm_wq[-win_w:][None, :, :]
        with torch.no_grad():
            live_wq_recon = self.ae_model(torch.from_numpy(live_wq_win).to(self.device)).cpu().numpy()
        live_feat_mse = np.mean((live_wq_win - live_wq_recon) ** 2, axis=1)[0]
        latest_mse = float(np.mean(live_feat_mse))
        latest_is_anom = bool(latest_mse > thresh)
        max_mse_val = max(float(np.max(recon_err)), latest_mse)

        # Update the final entry of series arrays so the live endpoint aligns with the final timestamp
        recon_err_series = recon_err.copy()
        detected_anom_series = detected_anom.copy()
        if len(recon_err_series) > 0:
            recon_err_series[-1] = latest_mse
            detected_anom_series[-1] = latest_is_anom

        mean_feat_mse = np.mean(sq_err_per_feat, axis=0)
        channel_breakdown = {
            col: {
                "mean_mse": round(float(mean_feat_mse[idx]), 4),
                "latest_mse": round(float(live_feat_mse[idx]), 4),
                "share_pct": round(float(live_feat_mse[idx] / max(np.sum(live_feat_mse), 1e-8) * 100.0), 1),
            }
            for idx, col in enumerate(WQ_COLS)
        }

        recon_mean = float(np.mean(recon_err))
        recon_std = float(np.std(recon_err)) + 1e-6
        latest_z = (latest_mse - recon_mean) / recon_std

        # Subsample validation & anomaly arrays for responsive UI plotting
        val_ts = ts_m[sp:] if sp < len(ts_m) else ts_m
        step_v = max(1, len(val_ts) // 280)
        step_a = max(1, len(ts_w) // 320)

        sub_pred_hs = preds_m[::step_v, 0]
        sub_pred_apd = preds_m[::step_v, 1]
        sea_code, sea_label = wmo_sea_state(float(live_pred[0]))

        result: Dict[str, Any] = {
            "wave_forecast": {
                "model_name": "ResidualWaveForecaster (BiLSTM + 4-Head Attention)",
                "device": str(self.device),
                "horizon": "1-Hour Ahead (+60 min) & Multi-Step (+1h to +6h)",
                "lookback_steps": int(lookback_m),
                "horizon_steps": int(horizon_m),
                "metrics": {
                    "hs_mae_m": round(hs_mae, 3),
                    "hs_rmse_m": round(hs_rmse, 3),
                    "hs_mape_pct": round(hs_mape, 2),
                    "hs_r2": round(hs_r2, 4),
                    "apd_mae_s": round(apd_mae, 3),
                    "apd_rmse_s": round(apd_rmse, 3),
                    "apd_r2": round(apd_r2, 4),
                    "hs_residual_std_m": round(aleatoric_std_hs, 3),
                    "apd_residual_std_s": round(aleatoric_std_apd, 3),
                    "ci95_halfwidth_hs_m": round(ci95_hw_hs, 3),
                    "ci95_halfwidth_apd_s": round(ci95_hw_apd, 3),
                },
                "live_prediction": {
                    "current_hs_m": round(float(raw_met[-1, 3]), 2),
                    "forecast_hs_1h_m": round(float(live_pred[0]), 2),
                    "delta_hs_m": round(float(live_pred[0] - raw_met[-1, 3]), 3),
                    "ci95_lower_hs_m": round(max(0.0, float(live_pred[0]) - ci95_hw_hs), 2),
                    "ci95_upper_hs_m": round(float(live_pred[0]) + ci95_hw_hs, 2),
                    "uncertainty_std_hs_m": round(comb_std_hs, 3),
                    "mc_epistemic_std_hs_m": round(float(live_mc_std[0]), 4),
                    "current_apd_s": round(float(raw_met[-1, 5]), 2),
                    "forecast_apd_1h_s": round(float(live_pred[1]), 2),
                    "delta_apd_s": round(float(live_pred[1] - raw_met[-1, 5]), 3),
                    "ci95_lower_apd_s": round(max(1.0, float(live_pred[1]) - ci95_hw_apd), 2),
                    "ci95_upper_apd_s": round(float(live_pred[1]) + ci95_hw_apd, 2),
                    "uncertainty_std_apd_s": round(comb_std_apd, 3),
                    "wmo_sea_state_code": sea_code,
                    "wmo_sea_state_label": sea_label,
                },
                "multi_step_forecast": multi_step_payload,
                "uncertainty": {
                    "method": "Monte Carlo LSTM Dropout (16 passes) + Empirical Validation Residual Variance",
                    "confidence_level": 0.95,
                    "mc_dropout_samples": 16,
                    "aleatoric_std_hs_m": round(aleatoric_std_hs, 4),
                    "epistemic_std_hs_m": round(float(live_mc_std[0]), 4),
                    "combined_std_hs_m": round(comb_std_hs, 4),
                    "empirical_q95_error_hs_m": round(empirical_q95_hs, 3),
                    "aleatoric_std_apd_s": round(aleatoric_std_apd, 4),
                    "epistemic_std_apd_s": round(float(live_mc_std[1]), 4),
                    "combined_std_apd_s": round(comb_std_apd, 4),
                    "empirical_q95_error_apd_s": round(empirical_q95_apd, 3),
                },
                "series": {
                    "timestamps": [t.strftime("%m-%d %H:%M") for t in val_ts[::step_v]],
                    "true_hs_m": [round(float(x), 3) for x in true_m[::step_v, 0]],
                    "pred_hs_m": [round(float(x), 3) for x in sub_pred_hs],
                    "pred_hs_upper_m": [round(float(x + ci95_hw_hs), 3) for x in sub_pred_hs],
                    "pred_hs_lower_m": [round(max(0.0, float(x - ci95_hw_hs)), 3) for x in sub_pred_hs],
                    "true_apd_s": [round(float(x), 2) for x in true_m[::step_v, 1]],
                    "pred_apd_s": [round(float(x), 2) for x in sub_pred_apd],
                    "pred_apd_upper_s": [round(float(x + ci95_hw_apd), 2) for x in sub_pred_apd],
                    "pred_apd_lower_s": [round(max(1.0, float(x - ci95_hw_apd)), 2) for x in sub_pred_apd],
                },
            },
            "water_quality_anomaly": {
                "model_name": "WaterQualityAE (Deep 7-Channel LSTM Autoencoder)",
                "threshold_mse": round(thresh, 4),
                "threshold_percentile": 95.0,
                "total_windows": int(len(recon_err)),
                "flagged_anomalies": flagged_count,
                "latest_mse": round(latest_mse, 4),
                "max_mse": round(max_mse_val, 4),
                "latest_is_anomaly": latest_is_anom,
                "hypoxic_threshold_mg_l": 2.0,
                "hypoxic_event_count": int((raw_wq[:, 4] < 2.0).sum()),
                "channel_mse_breakdown": channel_breakdown,
                "uncertainty": {
                    "recon_mse_mean": round(recon_mean, 4),
                    "recon_mse_std": round(recon_std, 4),
                    "recon_mse_p90": round(float(np.percentile(recon_err, 90)), 4),
                    "recon_mse_p95": round(thresh, 4),
                    "recon_mse_p99": round(float(np.percentile(recon_err, 99)), 4),
                    "latest_z_score": round(float(latest_z), 2),
                },
                "series": {
                    "timestamps": [t.strftime("%m-%d %H:%M") for t in ts_w.iloc[::step_a]],
                    "turbidity_ntu": [round(float(x), 1) for x in turb_w[::step_a]],
                    "o2_ppm": [round(float(x), 2) for x in o2_w[::step_a]],
                    "recon_mse": [round(float(x), 4) for x in recon_err_series[::step_a]],
                    "is_anomaly": [bool(x) for x in detected_anom_series[::step_a]],
                },
            },
        }

        result["alerts"] = self.alert_engine.evaluate(store=store, forecast_payload=result)
        self._cache_key = key
        self._cached_result = result
        return result

    def evaluate_alerts(self, store: BuoyTelemetryStore) -> Dict[str, Any]:
        """Evaluate maritime & environmental hazards using full telemetry + AI forecast context."""
        forecast_payload = self.evaluate_and_forecast(store)
        return forecast_payload["alerts"]

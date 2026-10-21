"""End-to-End Verification & Edge-Case Test Suite for the Smart Buoy Platform."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app, telemetry_store, timeseries_engine, vision_engine

client = TestClient(app)


def test_index_and_static_assets() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "SMART BUOY AI COMMAND CENTER" in res.text

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "loadWaveSpectra" in js_res.text

    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200
    assert "--bg-main" in css_res.text


def test_api_status_all_models_ready() -> None:
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["models"]["wave_forecaster"]["ready"] is True
    assert data["models"]["wq_autoencoder"]["ready"] is True
    for mode_key in ("buoy", "colregs", "sar", "horizon"):
        assert data["models"]["vision_models"][mode_key]["ready"] is True


def test_api_wave_spectra_peak_and_latest() -> None:
    for mode in ("peak", "latest"):
        res = client.get(f"/api/wave-spectra?mode={mode}")
        assert res.status_code == 200
        d = res.json()
        assert d["selected_mode"] == mode
        assert d["total_bursts"] > 2000
        assert len(d["spectrum"]["frequencies_hz"]) == 46
        assert len(d["spectrum"]["measured_m2_hz"]) == 46
        assert len(d["spectrum"]["jonswap_m2_hz"]) == 46
        assert d["metrics"]["correlation_r"] > 0.99

        # Verify all history arrays are identically aligned in length
        hist = d["history"]
        n_pts = len(hist["timestamps"])
        assert n_pts >= 40
        assert len(hist["hm0_integrated"]) == n_pts
        assert len(hist["wvht_reported"]) == n_pts
        assert len(hist["tp_s"]) == n_pts
        assert len(hist["tm02_s"]) == n_pts


def test_api_telemetry_and_forecast() -> None:
    tel_res = client.get("/api/telemetry")
    assert tel_res.status_code == 200
    tel = tel_res.json()
    assert tel["met_ocean"]["total_records"] > 6000
    assert tel["water_quality"]["total_records"] > 4000
    assert tel["met_ocean"]["latest"]["WVHT_m"] > 0
    assert tel["water_quality"]["latest"]["O2_ppm"] > 0

    fc_res = client.get("/api/forecast")
    assert fc_res.status_code == 200
    fc = fc_res.json()
    wf = fc["wave_forecast"]
    wqa = fc["water_quality_anomaly"]

    assert wf["metrics"]["hs_r2"] > 0.85
    assert wf["metrics"]["hs_mae_m"] < 0.20
    assert wf["live_prediction"]["forecast_hs_1h_m"] > 0
    assert wqa["total_windows"] > 4000
    assert wqa["flagged_anomalies"] > 100
    assert len(wqa["series"]["timestamps"]) == len(wqa["series"]["is_anomaly"])


def test_api_vision_samples_and_all_four_modes() -> None:
    s_res = client.get("/api/vision/samples")
    assert s_res.status_code == 200
    samples = s_res.json()["samples"]
    assert len(samples) >= 12

    mode_to_sample = {
        "buoy": "buoy/buoy_sample_1_buoy_00022.png",
        "colregs": "colregs/vessel_sample_1_buoy_00013.png",
        "sar": "sar/sar_sample_1_a_1121.jpg",
        "horizon": "colregs/vessel_sample_3_vessel_002958.jpg",
    }
    for mode, sample_id in mode_to_sample.items():
        det_res = client.post(
            "/api/vision/detect-sample",
            json={"sample_id": sample_id, "mode": mode, "conf": 0.25},
        )
        assert det_res.status_code == 200
        payload = det_res.json()
        assert payload["mode"] == mode
        assert payload["sample_id"] == sample_id
        assert payload["image_base64"].startswith("data:image/jpeg;base64,")
        assert payload["total_detections"] >= 1
        assert "roll_deg" in payload["attitude"]
        assert "water_ratio_pct" in payload["attitude"]


def test_api_vision_upload_valid_image() -> None:
    sample_path = vision_engine.list_samples()[0]
    raw_bytes = (vision_engine.list_samples() and (telemetry_store.df_met is not None))
    img = Image.new("RGB", (320, 240), color=(20, 90, 140))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/vision/detect-upload",
        files={"file": ("synthetic_ocean.png", buf.getvalue(), "image/png")},
        data={"mode": "horizon", "conf": "0.25"},
    )
    assert res.status_code == 200
    d = res.json()
    assert d["filename"] == "synthetic_ocean.png"
    assert d["resolution"] == {"width": 320, "height": 240}
    assert d["image_base64"].startswith("data:image/jpeg;base64,")


def test_edge_cases_and_error_handling() -> None:
    # 1. Invalid wave spectra query param
    assert client.get("/api/wave-spectra?mode=invalid").status_code == 422

    # 2. Non-existent sample file
    assert (
        client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "buoy/does_not_exist.png", "mode": "buoy", "conf": 0.25},
        ).status_code
        == 404
    )

    # 3. Path traversal attempt
    assert (
        client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "../README.md", "mode": "buoy", "conf": 0.25},
        ).status_code
        == 404
    )

    # 4. Directory path instead of file
    assert (
        client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "buoy", "mode": "buoy", "conf": 0.25},
        ).status_code
        == 404
    )

    # 5. Whitespace-only sample_id
    assert (
        client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "   ", "mode": "buoy", "conf": 0.25},
        ).status_code
        == 404
    )

    # 6. Out-of-bounds confidence threshold
    assert (
        client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "buoy/buoy_sample_1_buoy_00022.png", "mode": "buoy", "conf": 1.5},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "buoy/buoy_sample_1_buoy_00022.png", "mode": "buoy", "conf": -0.1},
        ).status_code
        == 422
    )

    # 7. Invalid mode on sample detection
    assert (
        client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "buoy/buoy_sample_1_buoy_00022.png", "mode": "invalid_mode", "conf": 0.25},
        ).status_code
        == 422
    )

    # 8. Empty file upload
    assert (
        client.post(
            "/api/vision/detect-upload",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"mode": "buoy", "conf": "0.25"},
        ).status_code
        == 400
    )

    # 9. Corrupted non-image upload
    assert (
        client.post(
            "/api/vision/detect-upload",
            files={"file": ("corrupt.jpg", b"not-a-valid-image-stream", "image/jpeg")},
            data={"mode": "buoy", "conf": "0.25"},
        ).status_code
        == 400
    )

    # 10. Tiny image upload (< 8x8 px)
    tiny_img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    tbuf = io.BytesIO()
    tiny_img.save(tbuf, format="PNG")
    assert (
        client.post(
            "/api/vision/detect-upload",
            files={"file": ("tiny.png", tbuf.getvalue(), "image/png")},
            data={"mode": "buoy", "conf": "0.25"},
        ).status_code
        == 400
    )


def test_websocket_live_imu_stream() -> None:
    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
        assert "timestamp" in msg
        assert "imu" in msg and "heave_m" in msg["imu"]
        assert "met" in msg and "wvht_m" in msg["met"]
        assert "wq" in msg and "o2_ppm" in msg["wq"]


def test_telemetry_refresh_rejects_truncated_or_html_payload() -> None:
    """Verify POST /api/telemetry/refresh does not overwrite cached NOAA data when upstream returns HTML or truncated HTTP 200."""
    from unittest.mock import MagicMock, patch
    from app.telemetry import DATA_DIR

    spec_file = DATA_DIR / "46050.data_spec"
    orig_size = spec_file.stat().st_size

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"<html>503 Service Temporarily Unavailable</html>"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.post("/api/telemetry/refresh")

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "refreshed"
    assert body["telemetry"]["met_ocean"]["total_records"] > 6000
    assert spec_file.stat().st_size == orig_size


def test_vision_detect_sample_supports_all_and_ensemble_modes() -> None:
    """Verify POST /api/vision/detect-sample accepts mode='all' and mode='ensemble' symmetrically with detect-upload."""
    for mode_alias in ("all", "ensemble"):
        res = client.post(
            "/api/vision/detect-sample",
            json={"sample_id": "buoy/buoy_sample_1_buoy_00022.png", "mode": mode_alias, "conf": 0.25},
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["mode"] == "all"
        assert payload["sample_id"] == "buoy/buoy_sample_1_buoy_00022.png"
        assert payload["active_models"] == [
            "yolo11m_ks_buoy.pt",
            "yolo11m_spscd_vessels.pt",
            "yolo11m_afo_sar.pt",
        ]
        assert payload["total_detections"] >= 1


def test_vision_16bit_grayscale_tiff_and_extreme_aspect_ratios() -> None:
    """Verify 16-bit grayscale TIFF (I;16) dynamic range normalization and narrow/wide image HUD clamping."""
    import numpy as np
    from app.vision import MarineVisionEngine

    arr16 = np.zeros((160, 240), dtype=np.uint16)
    arr16[:80, :] = 48000  # Bright sky
    arr16[80:, :] = 12000  # Darker sea
    img16 = Image.fromarray(arr16)
    buf16 = io.BytesIO()
    img16.save(buf16, format="TIFF")

    _, rgb16 = MarineVisionEngine._decode_rgb_image(buf16.getvalue())
    assert rgb16.dtype == np.uint8
    assert float(rgb16[:80].mean()) > 180.0
    assert float(rgb16[80:].mean()) < 60.0

    res_tiff = client.post(
        "/api/vision/detect-upload",
        files={"file": ("sar_16bit.tiff", buf16.getvalue(), "image/tiff")},
        data={"mode": "horizon", "conf": "0.25"},
    )
    assert res_tiff.status_code == 200
    assert 35.0 <= res_tiff.json()["attitude"]["water_ratio_pct"] <= 65.0

    for width, height in ((10, 4000), (4000, 10)):
        extreme_img = Image.new("RGB", (width, height), color=(25, 85, 135))
        ebuf = io.BytesIO()
        extreme_img.save(ebuf, format="PNG")
        res_ext = client.post(
            "/api/vision/detect-upload",
            files={"file": (f"extreme_{width}x{height}.png", ebuf.getvalue(), "image/png")},
            data={"mode": "buoy", "conf": "0.05"},
        )
        assert res_ext.status_code == 200
        assert res_ext.json()["resolution"] == {"width": width, "height": height}


def test_concurrent_forecast_and_vision_thread_safety() -> None:
    """Verify concurrent threads invoking BuoyTimeSeriesEngine and MarineVisionEngine do not race."""
    from concurrent.futures import ThreadPoolExecutor

    img = Image.new("RGB", (240, 160), color=(30, 95, 145))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    def _worker(idx: int) -> float:
        if idx % 2 == 0:
            fc = timeseries_engine.evaluate_and_forecast(telemetry_store)
            return float(fc["wave_forecast"]["live_prediction"]["forecast_hs_1h_m"])
        det = vision_engine.run_detection(raw_bytes, mode="horizon", conf_threshold=0.25)
        return float(det["attitude"]["water_ratio_pct"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_worker, range(8)))
    assert len(results) == 8
    assert all(r >= 0.0 for r in results)


"""End-to-end integration tests for FastAPI REST and WebSocket endpoints in app/main.py."""

from unittest.mock import patch
import urllib.error
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.vision import SAMPLES_DIR


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_index_dashboard_route(client: TestClient) -> None:
    """Verify GET / serves the main HTML dashboard."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Smart Buoy" in resp.text


def test_api_status(client: TestClient) -> None:
    """Verify GET /api/status reports online status and all model weights ready."""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["compute_device"] in ("cuda", "cpu")
    assert "46050" in data["stations"]["offshore_wave_met"]
    assert "ACQS1" in data["stations"]["coastal_water_quality"]

    models = data["models"]
    assert models["wave_forecaster"]["ready"] is True
    assert models["wq_autoencoder"]["ready"] is True
    for mode in ("buoy", "colregs", "sar", "horizon"):
        assert mode in models["vision_models"]
        assert models["vision_models"][mode]["ready"] is True


def test_api_wave_spectra_modes_and_validation(client: TestClient) -> None:
    """Verify GET /api/wave-spectra supports peak/latest and rejects invalid modes with 422."""
    resp_peak = client.get("/api/wave-spectra?mode=peak")
    assert resp_peak.status_code == 200
    d_peak = resp_peak.json()
    assert d_peak["selected_mode"] == "peak"
    assert d_peak["metrics"]["hm0_m"] == d_peak["metrics"]["peak_hm0_m"]

    resp_latest = client.get("/api/wave-spectra?mode=latest")
    assert resp_latest.status_code == 200
    d_latest = resp_latest.json()
    assert d_latest["selected_mode"] == "latest"
    assert d_latest["metrics"]["hm0_m"] == d_latest["metrics"]["latest_hm0_m"]

    # Invalid query parameter must fail pattern validation
    resp_invalid = client.get("/api/wave-spectra?mode=invalid")
    assert resp_invalid.status_code == 422


def test_api_telemetry_and_refresh(client: TestClient) -> None:
    """Verify GET /api/telemetry and POST /api/telemetry/refresh return full NOAA telemetry."""
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert "met_ocean" in data
    assert "water_quality" in data
    assert data["met_ocean"]["total_records"] > 1000
    assert data["water_quality"]["total_records"] > 1000

    # Test refresh endpoint (with network mocked to fall back cleanly to local NOAA files)
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Mock offline")):
        resp_ref = client.post("/api/telemetry/refresh")
    assert resp_ref.status_code == 200
    ref_data = resp_ref.json()
    assert ref_data["status"] == "refreshed"
    assert "met_ocean" in ref_data["telemetry"]
    assert "water_quality" in ref_data["telemetry"]


def test_api_forecast(client: TestClient) -> None:
    """Verify GET /api/forecast returns PyTorch wave forecast and water quality anomaly payloads."""
    resp = client.get("/api/forecast")
    assert resp.status_code == 200
    data = resp.json()
    assert "wave_forecast" in data
    assert "water_quality_anomaly" in data
    assert data["wave_forecast"]["metrics"]["hs_r2"] > 0.80
    assert data["water_quality_anomaly"]["total_windows"] > 1000


def test_api_vision_samples(client: TestClient) -> None:
    """Verify GET /api/vision/samples returns sample list and model specifications."""
    resp = client.get("/api/vision/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["samples"]) >= 12
    assert set(data["modes"].keys()) == {"buoy", "colregs", "sar", "horizon"}


def test_api_vision_detect_sample_and_security(client: TestClient) -> None:
    """Verify POST /api/vision/detect-sample runs YOLOv11m and blocks path traversal / missing files."""
    resp = client.post(
        "/api/vision/detect-sample",
        json={"sample_id": "buoy/buoy_sample_1_buoy_00022.png", "mode": "buoy", "conf": 0.25},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sample_id"] == "buoy/buoy_sample_1_buoy_00022.png"
    assert data["mode"] == "buoy"
    assert data["total_detections"] >= 1
    assert data["image_base64"].startswith("data:image/jpeg;base64,")

    # Nonexistent sample -> 404
    resp_missing = client.post(
        "/api/vision/detect-sample",
        json={"sample_id": "buoy/nonexistent_frame.png", "mode": "buoy", "conf": 0.25},
    )
    assert resp_missing.status_code == 404

    # Directory traversal attempt -> 404
    resp_traversal = client.post(
        "/api/vision/detect-sample",
        json={"sample_id": "../requirements.txt", "mode": "buoy", "conf": 0.25},
    )
    assert resp_traversal.status_code == 404


def test_api_vision_detect_upload_and_empty_error(client: TestClient) -> None:
    """Verify POST /api/vision/detect-upload handles multipart image upload and rejects empty files."""
    sample_path = SAMPLES_DIR / "sar" / "sar_sample_1_a_1121.jpg"
    img_bytes = sample_path.read_bytes()

    resp = client.post(
        "/api/vision/detect-upload",
        data={"mode": "sar", "conf": "0.25"},
        files={"file": ("sar_upload.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "sar_upload.jpg"
    assert data["mode"] == "sar"
    assert data["total_detections"] >= 1
    assert data["image_base64"].startswith("data:image/jpeg;base64,")

    # Empty upload -> 400 Bad Request
    resp_empty = client.post(
        "/api/vision/detect-upload",
        data={"mode": "buoy", "conf": "0.25"},
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp_empty.status_code == 400


def test_websocket_live_buoy_stream(client: TestClient) -> None:
    """Verify WebSocket /ws/live streams real-time 6-DOF IMU heave/roll/pitch and NOAA telemetry."""
    with client.websocket_connect("/ws/live") as ws:
        frame1 = ws.receive_json()
        frame2 = ws.receive_json()

    for frame in (frame1, frame2):
        assert "timestamp" in frame
        assert "imu" in frame and "met" in frame and "wq" in frame

        imu = frame["imu"]
        assert all(k in imu for k in ("heave_m", "roll_deg", "pitch_deg"))
        assert -15.0 <= imu["heave_m"] <= 15.0
        assert -30.0 <= imu["roll_deg"] <= 30.0
        assert -30.0 <= imu["pitch_deg"] <= 30.0

        met = frame["met"]
        assert all(k in met for k in ("wvht_m", "dpd_s", "wspd_mps", "gst_mps", "pres_hpa"))
        assert met["wvht_m"] > 0.0

        wq = frame["wq"]
        assert all(k in wq for k in ("otmp_c", "sal_psu", "o2_ppm", "turb_ntu", "ph"))
        assert 5.0 <= wq["ph"] <= 10.5

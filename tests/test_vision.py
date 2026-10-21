"""Unit and integration tests for YOLOv11m multi-task marine computer vision and optical horizon attitude."""

import base64
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from app.vision import MODEL_SPECS, SAMPLES_DIR, WEIGHTS_DIR, MarineVisionEngine


@pytest.fixture(scope="module")
def vision_engine() -> MarineVisionEngine:
    return MarineVisionEngine()


def test_model_specs_and_yolo_weights_exist() -> None:
    """Verify all vision modes (including 'all'/'ensemble') are specified and YOLOv11m weight files exist."""
    expected_modes = {"buoy", "colregs", "sar", "horizon"}
    assert expected_modes.issubset(set(MODEL_SPECS.keys()))
    assert "all" in MODEL_SPECS and "ensemble" in MODEL_SPECS
    assert MODEL_SPECS["all"] == MODEL_SPECS["ensemble"]
    assert MODEL_SPECS.get("all") == MODEL_SPECS["all"]
    assert "ensemble_weights" in MODEL_SPECS["all"]

    for mode, spec in MODEL_SPECS.items():
        assert "title" in spec
        assert "weights" in spec
        assert "dataset" in spec
        assert "imgsz" in spec
        assert "metrics" in spec
        assert "classes" in spec
        wpath = WEIGHTS_DIR / spec["weights"]
        assert wpath.exists(), f"Missing weight file {wpath} for mode {mode}"
        assert wpath.stat().st_size > 10_000_000, f"Weight file {wpath} is too small to be YOLOv11m"


def test_list_samples(vision_engine: MarineVisionEngine) -> None:
    """Verify bundled sample enumeration covers buoy, colregs, and sar validation frames."""
    samples = vision_engine.list_samples()
    assert len(samples) >= 12

    categories = {s["category"] for s in samples}
    assert categories == {"buoy", "colregs", "sar"}

    for s in samples:
        assert "id" in s and "name" in s and "filename" in s and "url" in s
        sample_file = SAMPLES_DIR / s["id"]
        assert sample_file.exists(), f"Listed sample file does not exist: {sample_file}"


def test_estimate_horizon_flat_tilted_uniform_and_edge_shapes() -> None:
    """Verify optical horizon line estimation on flat, tilted, uniform, 2D grayscale, and tiny images."""
    H, W = 240, 320

    # 1. Flat horizon at y = 120 (mid-height)
    flat_img = np.zeros((H, W, 3), dtype=np.uint8)
    flat_img[:120, :, :] = [210, 230, 250]  # Bright sky
    flat_img[120:, :, :] = [15, 45, 75]     # Dark ocean water

    slope, intercept, roll_deg, pitch_pct, water_mask = MarineVisionEngine.estimate_horizon(flat_img)
    assert abs(slope) < 0.02
    assert 116.0 <= intercept <= 124.0
    assert abs(roll_deg) < 1.5
    assert abs(pitch_pct) < 3.0
    assert water_mask.shape == (H, W)
    assert water_mask.dtype == np.bool_
    assert 0.45 <= water_mask.mean() <= 0.55

    # 2. Tilted horizon: y = 0.20 * x + 90
    tilted_img = np.zeros((H, W, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    true_line = 0.20 * xx + 90.0
    tilted_img[yy < true_line] = [220, 235, 255]
    tilted_img[yy >= true_line] = [10, 40, 70]

    slope_t, intercept_t, roll_t, _, _ = MarineVisionEngine.estimate_horizon(tilted_img)
    assert 0.16 <= slope_t <= 0.24
    assert 84.0 <= intercept_t <= 96.0
    assert 9.0 <= roll_t <= 14.0

    # 3. Uniform black image (edge case: zero gradients everywhere)
    uniform_img = np.zeros((H, W, 3), dtype=np.uint8)
    s_u, i_u, r_u, p_u, mask_u = MarineVisionEngine.estimate_horizon(uniform_img)
    assert np.isfinite(s_u) and np.isfinite(i_u) and np.isfinite(r_u) and np.isfinite(p_u)
    assert mask_u.shape == (H, W)

    # 4. 2D grayscale image input (ndim == 2)
    gray_2d = np.zeros((H, W), dtype=np.uint8)
    gray_2d[:120, :] = 220
    gray_2d[120:, :] = 30
    s_g, i_g, r_g, _, mask_g = MarineVisionEngine.estimate_horizon(gray_2d)
    assert abs(s_g) < 0.02 and 116.0 <= i_g <= 124.0 and abs(r_g) < 1.5
    assert mask_g.shape == (H, W)

    # 5. Tiny image (< 8x8 pixels fallback)
    tiny_img = np.zeros((4, 6, 3), dtype=np.uint8)
    s_sm, i_sm, r_sm, p_sm, mask_sm = MarineVisionEngine.estimate_horizon(tiny_img)
    assert s_sm == 0.0 and i_sm == 2.0 and r_sm == 0.0 and p_sm == 0.0
    assert mask_sm.shape == (4, 6)


def test_iou_and_ensemble_box_deduplication() -> None:
    """Verify 2D bounding-box IoU and cross-model Non-Maximum Suppression (NMS)."""
    assert MarineVisionEngine._compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)
    assert MarineVisionEngine._compute_iou([0, 0, 50, 50], [60, 60, 100, 100]) == pytest.approx(0.0)
    # 50x100 intersection (5000) / (10000 + 10000 - 5000 = 15000) = 1/3
    assert MarineVisionEngine._compute_iou([0, 0, 100, 100], [50, 0, 150, 100]) == pytest.approx(1.0 / 3.0)

    candidates = [
        {"class_name": "Navigation Buoy", "confidence": 0.92, "bbox": [10, 10, 110, 110]},
        # Same class with high overlap (IoU > 0.45) -> suppressed
        {"class_name": "Navigation Buoy", "confidence": 0.78, "bbox": [15, 15, 112, 112]},
        # Different class with very high overlap (IoU > 0.60) -> suppressed
        {"class_name": "Motor / Fishing Craft", "confidence": 0.65, "bbox": [12, 12, 108, 108]},
        # Distinct target far away -> kept
        {"class_name": "Commercial Ship / Ferry", "confidence": 0.88, "bbox": [300, 200, 450, 320]},
    ]
    kept = MarineVisionEngine._deduplicate_ensemble_boxes(candidates)
    assert len(kept) == 2
    assert kept[0]["class_name"] == "Navigation Buoy" and kept[0]["confidence"] == 0.92
    assert kept[1]["class_name"] == "Commercial Ship / Ferry" and kept[1]["confidence"] == 0.88


def _assert_valid_detection_response(res: dict, expected_mode: str) -> None:
    """Helper to validate detection output structure, COLREGs risk, and decoded annotated base64 JPEG."""
    assert res["mode"] == expected_mode
    assert res["inference_ms"] > 0.0
    W = res["resolution"]["width"]
    H = res["resolution"]["height"]
    assert W > 0 and H > 0

    att = res["attitude"]
    assert -90.0 <= att["roll_deg"] <= 90.0
    assert -100.0 <= att["pitch_offset_pct"] <= 100.0
    assert abs((att["sky_ratio_pct"] + att["water_ratio_pct"]) - 100.0) <= 0.2

    assert "risk_summary" in res and "colregs_risk" in res
    assert res["risk_summary"] == res["colregs_risk"]
    assert res["risk_summary"]["total_targets"] == res["total_detections"]

    assert res["total_detections"] == len(res["detections"])
    for det in res["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        assert x2 >= x1 and y2 >= y1
        assert 0.0 <= det["confidence"] <= 1.0
        assert -90.0 <= det["bearing_deg"] <= 90.0
        assert 8.0 <= det["est_range_m"] <= 450.0
        assert det["risk_level"] in ("CRITICAL", "WARNING", "CAUTION", "NOMINAL")
        assert 0.0 <= det["risk_score"] <= 100.0
        assert "colregs_rule" in det and "recommended_action" in det

    assert res["image_base64"].startswith("data:image/jpeg;base64,")
    b64_payload = res["image_base64"].split(",", 1)[1]
    raw_jpeg = base64.b64decode(b64_payload)
    decoded_img = Image.open(io.BytesIO(raw_jpeg))
    assert decoded_img.size == (W, H)


@pytest.mark.parametrize(
    "mode,sample_relpath",
    [
        ("buoy", "buoy/buoy_sample_1_buoy_00022.png"),
        ("colregs", "colregs/vessel_sample_3_vessel_002958.jpg"),
        ("sar", "sar/sar_sample_1_a_1121.jpg"),
        ("horizon", "buoy/buoy_sample_1_buoy_00022.png"),
        ("horizon", "colregs/vessel_sample_4_vessel_003106.jpg"),
        ("all", "colregs/vessel_sample_3_vessel_002958.jpg"),
    ],
)
def test_run_detection_all_modes_on_real_samples(
    vision_engine: MarineVisionEngine,
    mode: str,
    sample_relpath: str,
) -> None:
    """Run real YOLOv11m inference across buoy, colregs, sar, horizon, and all modes on sample frames."""
    sample_path = SAMPLES_DIR / sample_relpath
    assert sample_path.exists()
    img_bytes = sample_path.read_bytes()

    res = vision_engine.run_detection(
        image_bytes=img_bytes,
        mode=mode,
        conf_threshold=0.25,
        source_hint=sample_relpath,
    )
    _assert_valid_detection_response(res, expected_mode=mode)
    assert res["total_detections"] >= 1, f"Expected >= 1 detection for {mode} on {sample_relpath}"

    if mode == "buoy" or (mode == "horizon" and sample_relpath.startswith("buoy/")):
        assert all(d["class_name"] == "Navigation Buoy" for d in res["detections"])


def test_vision_edge_cases(vision_engine: MarineVisionEngine) -> None:
    """Verify mode aliases/fallback, empty/corrupt/tiny image ValueErrors, zero-detection threshold, and missing weight error."""
    sample_bytes = (SAMPLES_DIR / "buoy/buoy_sample_1_buoy_00022.png").read_bytes()

    # Unknown mode falls back cleanly to 'buoy'
    res_fallback = vision_engine.run_detection(image_bytes=sample_bytes, mode="unknown_mode_xyz")
    assert res_fallback["mode"] == "buoy"

    # 'ensemble' alias normalizes to 'all'
    res_ens = vision_engine.run_detection(image_bytes=sample_bytes, mode="ensemble")
    assert res_ens["mode"] == "all"
    assert len(res_ens["active_models"]) == 3

    # Empty image_bytes raises ValueError
    with pytest.raises(ValueError, match="Empty image_bytes"):
        vision_engine.run_detection(image_bytes=b"", mode="buoy")

    # Corrupted non-image bytes raises ValueError
    with pytest.raises(ValueError, match="Invalid or corrupted image file"):
        vision_engine.run_detection(image_bytes=b"not-a-valid-image-payload", mode="buoy")

    # Tiny (< 8x8) image raises ValueError
    tiny_img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    tbuf = io.BytesIO()
    tiny_img.save(tbuf, format="PNG")
    with pytest.raises(ValueError, match="too small"):
        vision_engine.run_detection(image_bytes=tbuf.getvalue(), mode="buoy")

    # Impossible confidence threshold yields 0 detections cleanly
    blank_img = Image.new("RGB", (320, 240), color=(20, 50, 80))
    buf = io.BytesIO()
    blank_img.save(buf, format="PNG")
    res_empty = vision_engine.run_detection(
        image_bytes=buf.getvalue(),
        mode="buoy",
        conf_threshold=0.99,
    )
    _assert_valid_detection_response(res_empty, expected_mode="buoy")
    assert res_empty["total_detections"] == 0
    assert res_empty["detections"] == []
    assert res_empty["risk_summary"]["overall_status"] == "CLEAR"

    # Missing YOLO weight file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        vision_engine._get_yolo("nonexistent_yolo11m_weights.pt")


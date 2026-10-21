"""YOLOv11m Multi-Task Marine Computer Vision & Optical Horizon Attitude Engine."""

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from .colregs_risk import COLREGsRiskEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
SAMPLES_DIR = BASE_DIR / "samples"

ENSEMBLE_SPEC: Dict[str, Any] = {
    "title": "Type 5: Unified Multi-Model Ensemble (Buoy + COLREGs + SAR + Horizon HUD)",
    "weights": "yolo11m_spscd_vessels.pt",
    "ensemble_weights": [
        "yolo11m_ks_buoy.pt",
        "yolo11m_spscd_vessels.pt",
        "yolo11m_afo_sar.pt",
    ],
    "dataset": "Unified Ensemble: KS_Buoy + SPSCD/SMD + AFO-SAR + Optical Horizon RANSAC",
    "imgsz": 960,
    "metrics": {"precision": 85.43, "recall": 76.06, "map50": 81.34, "map50_95": 47.88},
    "classes": [
        "Navigation Buoy",
        "Commercial Ship / Ferry",
        "Motor / Fishing Craft",
        "Sailing Vessel / Yacht",
        "Human / Swimmer in Water",
        "Windsurf / SUP / Rescue Board",
        "Sky Zone",
        "Navigable Open Water",
    ],
    "color": (168, 85, 247),
}


class _ModelSpecsRegistry(dict):
    """Dictionary of the 4 core YOLOv11m model specs that also resolves the 5th 'all'/'ensemble' mode."""

    def __contains__(self, key: object) -> bool:
        if key in ("all", "ensemble"):
            return True
        return super().__contains__(key)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if key in ("all", "ensemble"):
            return ENSEMBLE_SPEC
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in ("all", "ensemble"):
            return ENSEMBLE_SPEC
        return super().get(key, default)


MODEL_SPECS: Dict[str, Dict[str, Any]] = _ModelSpecsRegistry({
    "buoy": {
        "title": "Type 1: Sea-Surface Navigation Buoy Detection (YOLOv11m)",
        "weights": "yolo11m_ks_buoy.pt",
        "dataset": "ARG-NCTU/KS_Buoy_dataset_2025",
        "imgsz": 960,
        "metrics": {"precision": 97.51, "recall": 89.90, "map50": 94.07, "map50_95": 64.10},
        "classes": ["Navigation Buoy"],
        "color": (14, 165, 233),
    },
    "colregs": {
        "title": "Type 2: COLREGs Maritime Vessel, Ship & Buoy Detection (YOLOv11m)",
        "weights": "yolo11m_spscd_vessels.pt",
        "dataset": "SPSCD + Singapore Maritime Dataset (SMD) + KS_Buoy",
        "imgsz": 960,
        "metrics": {"precision": 75.60, "recall": 60.40, "map50": 66.10, "map50_95": 41.60},
        "classes": [
            "Commercial Ship / Ferry",
            "Motor / Fishing Craft",
            "Sailing Vessel / Yacht",
            "Navigation Buoy",
        ],
        "color": (16, 185, 129),
    },
    "sar": {
        "title": "Type 3: Maritime Search & Rescue (SAR) Swimmer & Board Detection (YOLOv11m)",
        "weights": "yolo11m_afo_sar.pt",
        "dataset": "Voxel51/AFO-Aerial_Floating_Objects (Clean Ground Truth)",
        "imgsz": 1280,
        "metrics": {"precision": 83.18, "recall": 77.87, "map50": 83.85, "map50_95": 37.93},
        "classes": ["Human / Swimmer in Water", "Windsurf / SUP / Rescue Board"],
        "color": (244, 63, 94),
    },
    "horizon": {
        "title": "Type 4: Optical Horizon Attitude, 3-Zone Segmentation & Radar Bearing HUD",
        "weights": "yolo11m_spscd_vessels.pt",
        "dataset": "RANSAC Horizon + 3-Zone Sky/Water/Obstacle Segmentation + YOLOv11m",
        "imgsz": 960,
        "metrics": {"precision": 94.07, "recall": 89.90, "map50": 94.07, "map50_95": 64.10},
        "classes": ["Sky Zone", "Navigable Open Water", "Surface Obstacle"],
        "color": (250, 204, 21),
    },
})

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Navigation Buoy": (14, 165, 233),
    "Commercial Ship / Ferry": (16, 185, 129),
    "Motor / Fishing Craft": (245, 158, 11),
    "Sailing Vessel / Yacht": (168, 85, 247),
    "Human / Swimmer in Water": (244, 63, 94),
    "Windsurf / SUP / Rescue Board": (251, 113, 133),
}


class MarineVisionEngine:
    """Lazy-loading YOLOv11m inference, multi-model ensemble & horizon segmentation engine."""

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self.risk_evaluator = COLREGsRiskEvaluator()

    def _get_yolo(self, weight_filename: str) -> Any:
        if weight_filename not in self._models:
            from ultralytics import YOLO
            wpath = WEIGHTS_DIR / weight_filename
            if not wpath.exists():
                raise FileNotFoundError(f"Missing YOLOv11m weight file: {wpath}")
            self._models[weight_filename] = YOLO(str(wpath))
        return self._models[weight_filename]

    def list_samples(self) -> List[Dict[str, Any]]:
        """List all bundled real-world validation frames across buoy, colregs, and sar."""
        items: List[Dict[str, Any]] = []
        mapping = [
            ("buoy", "Type 1: Sea-Surface Buoy", "buoy"),
            ("colregs", "Type 2: COLREGs Vessel Traffic", "colregs"),
            ("sar", "Type 3: Search & Rescue (SAR)", "sar"),
        ]
        for folder, label_prefix, default_mode in mapping:
            d = SAMPLES_DIR / folder
            if not d.exists():
                continue
            for idx, p in enumerate(sorted(d.glob("*.*")), 1):
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                items.append({
                    "id": f"{folder}/{p.name}",
                    "name": f"{label_prefix} #{idx}",
                    "filename": p.name,
                    "category": folder,
                    "recommended_mode": default_mode,
                    "url": f"/samples/{folder}/{p.name}",
                })
        return items

    @staticmethod
    def estimate_horizon(rgb: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
        """Estimate sea-sky horizon line y = slope * x + intercept via robust column gradient fit."""
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        if H < 8 or W < 8:
            water_mask = np.ones((max(1, H), max(1, W)), dtype=bool)
            water_mask[: max(1, H // 2), :] = False
            return 0.0, float(H / 2.0), 0.0, 0.0, water_mask

        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        y_min_band, y_max_band = max(0, int(0.12 * H)), min(H, max(int(0.12 * H) + 2, int(0.85 * H)))
        xs = np.linspace(max(0, int(0.05 * W)), min(W - 1, int(0.95 * W)), min(48, W), dtype=int)
        horizon_ys = []
        for x in xs:
            col_strip = gray[y_min_band:y_max_band, max(0, x - 4) : min(W, x + 4)].mean(axis=1)
            grad = np.abs(np.gradient(col_strip)) if len(col_strip) >= 2 else np.zeros(1)
            horizon_ys.append(y_min_band + int(np.argmax(grad)))
        horizon_ys_arr = np.array(horizon_ys, dtype=np.float64)
        med_y = np.median(horizon_ys_arr)
        inliers = np.abs(horizon_ys_arr - med_y) < (2.5 * np.median(np.abs(horizon_ys_arr - med_y)) + 8.0)
        if inliers.sum() >= 6 and len(np.unique(xs[inliers])) >= 2:
            slope, intercept = np.polyfit(xs[inliers], horizon_ys_arr[inliers], 1)
        else:
            slope, intercept = 0.0, float(med_y)

        roll_deg = float(np.rad2deg(np.arctan(slope)))
        horizon_mid_y = float(slope * (W / 2.0) + intercept)
        pitch_offset_pct = float((0.5 - horizon_mid_y / H) * 100.0)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        horizon_curve = slope * xx + intercept
        water_mask = yy >= horizon_curve
        return float(slope), float(intercept), roll_deg, pitch_offset_pct, water_mask

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """Compute 2D Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = float(inter_w * inter_h)
        if inter_area <= 0.0:
            return 0.0
        area_a = float(max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1]))
        area_b = float(max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1]))
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0.0 else 0.0

    @classmethod
    def _deduplicate_ensemble_boxes(
        cls,
        candidates: List[Dict[str, Any]],
        same_class_iou: float = 0.45,
        cross_class_iou: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """Apply Cross-Model Non-Maximum Suppression (NMS) across ensemble detections."""
        sorted_cands = sorted(candidates, key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for cand in sorted_cands:
            duplicate = False
            for existing in kept:
                iou = cls._compute_iou(cand["bbox"], existing["bbox"])
                thresh = same_class_iou if cand["class_name"] == existing["class_name"] else cross_class_iou
                if iou >= thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    def _collect_single_model_boxes(
        self,
        pil_img: Image.Image,
        weight_file: str,
        imgsz: int,
        conf_threshold: float,
        source_model: str,
    ) -> List[Dict[str, Any]]:
        """Run a single YOLOv11m model and extract standardized candidate boxes."""
        model = self._get_yolo(weight_file)
        results = model.predict(pil_img, imgsz=imgsz, conf=conf_threshold, verbose=False)
        pred = results[0]
        raw_boxes: List[Dict[str, Any]] = []
        for box in pred.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = pred.names.get(cls_id, f"Class {cls_id}")
            if weight_file == "yolo11m_ks_buoy.pt":
                cls_name = "Navigation Buoy"
            raw_boxes.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "source_model": source_model,
            })
        return raw_boxes

    def run_detection(
        self,
        image_bytes: bytes,
        mode: str = "buoy",
        conf_threshold: float = 0.25,
        hfov_deg: float = 90.0,
        source_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute YOLOv11m detection + optical horizon, bearing & COLREGs risk calculation on an image."""
        if not image_bytes:
            raise ValueError("Empty image_bytes provided to run_detection.")

        mode_clean = (mode or "buoy").strip().lower()
        if mode_clean in ("ensemble", "unified"):
            mode_clean = "all"
        if mode_clean not in MODEL_SPECS:
            mode_clean = "buoy"
        spec = MODEL_SPECS[mode_clean]

        conf_threshold = float(np.clip(float(conf_threshold), 0.01, 0.99))
        hfov_deg = float(np.clip(float(hfov_deg), 10.0, 179.0))

        t0 = time.perf_counter()
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

        rgb = np.array(pil_img)
        H, W, _ = rgb.shape
        if H < 8 or W < 8:
            raise ValueError(f"Image dimensions ({W}x{H}) are too small; minimum supported size is 8x8 pixels.")

        slope, intercept, roll_deg, pitch_offset_pct, water_mask = self.estimate_horizon(rgb)
        water_ratio = float(water_mask.mean() * 100.0)
        sky_ratio = float(100.0 - water_ratio)

        # Collect detections from either the 3-model ensemble ("all") or the selected specialist model
        if mode_clean == "all":
            ensemble_plan = [
                ("buoy", "yolo11m_ks_buoy.pt", 960),
                ("colregs", "yolo11m_spscd_vessels.pt", 960),
                ("sar", "yolo11m_afo_sar.pt", 960),
            ]
            raw_candidates: List[Dict[str, Any]] = []
            for sub_mode, w_file, imgsz in ensemble_plan:
                raw_candidates.extend(
                    self._collect_single_model_boxes(
                        pil_img=pil_img,
                        weight_file=w_file,
                        imgsz=imgsz,
                        conf_threshold=conf_threshold,
                        source_model=sub_mode,
                    )
                )
            merged_boxes = self._deduplicate_ensemble_boxes(raw_candidates)
            active_models = ["yolo11m_ks_buoy.pt", "yolo11m_spscd_vessels.pt", "yolo11m_afo_sar.pt"]
        else:
            weight_file = spec["weights"]
            if mode_clean == "horizon" and source_hint and str(source_hint).startswith("buoy/"):
                weight_file = "yolo11m_ks_buoy.pt"
            merged_boxes = self._collect_single_model_boxes(
                pil_img=pil_img,
                weight_file=weight_file,
                imgsz=spec["imgsz"],
                conf_threshold=conf_threshold,
                source_model=mode_clean,
            )
            active_models = [weight_file]

        # Render overlay
        overlay = rgb.copy().astype(np.float32)
        if mode_clean in ("horizon", "all"):
            sky_alpha = 0.22 if mode_clean == "horizon" else 0.16
            water_alpha = 0.25 if mode_clean == "horizon" else 0.18
            overlay[~water_mask] = (1.0 - sky_alpha) * overlay[~water_mask] + sky_alpha * np.array([56, 189, 248])
            overlay[water_mask] = (1.0 - water_alpha) * overlay[water_mask] + water_alpha * np.array([16, 185, 129])

        detections: List[Dict[str, Any]] = []
        for idx, item in enumerate(merged_boxes, 1):
            x1, y1, x2, y2 = item["bbox"]
            cls_id = int(item["class_id"])
            conf = float(item["confidence"])
            cls_name = str(item["class_name"])

            if mode_clean in ("horizon", "all"):
                box_alpha = 0.55 if mode_clean == "horizon" else 0.28
                tint = np.array([239, 68, 68]) if mode_clean == "horizon" else np.array(CLASS_COLORS.get(cls_name, (239, 68, 68)))
                overlay[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)] = (
                    (1.0 - box_alpha) * rgb[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)]
                    + box_alpha * tint
                )

            xc = 0.5 * (x1 + x2)
            yc = 0.5 * (y1 + y2)
            bearing_deg = ((xc / max(1, W)) - 0.5) * hfov_deg
            box_h = max(1, y2 - y1)
            est_range_m = round(float(np.clip(1800.0 / (box_h + 12.0), 8.0, 450.0)), 1)

            detections.append({
                "id": idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "source_model": item.get("source_model", mode_clean),
                "confidence": round(conf, 4),
                "confidence_pct": round(conf * 100.0, 1),
                "bbox": [x1, y1, x2, y2],
                "center": [round(xc, 1), round(yc, 1)],
                "bearing_deg": round(float(bearing_deg), 2),
                "est_range_m": est_range_m,
            })

        # Evaluate COLREGs collision risk & SAR priorities for all detections
        risk_summary = self.risk_evaluator.evaluate_detections(
            detections=detections,
            roll_deg=roll_deg,
            pitch_offset_pct=pitch_offset_pct,
        )

        out_pil = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(out_pil)

        # Draw horizon line in horizon, buoy, colregs, or all mode
        if mode_clean in ("horizon", "buoy", "colregs", "all"):
            y0 = int(intercept)
            y1_end = int(slope * (W - 1) + intercept)
            draw.line([(0, y0), (W - 1, y1_end)], fill=(250, 204, 21), width=3)

        # Draw bounding boxes & bearing/risk tags
        palette = [
            (14, 165, 233),
            (16, 185, 129),
            (244, 63, 94),
            (245, 158, 11),
            (168, 85, 247),
        ]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if mode_clean == "horizon":
                color = (239, 68, 68)
            else:
                color = CLASS_COLORS.get(det["class_name"], palette[det["class_id"] % len(palette)])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = (
                f"{det['class_name']} {det['confidence_pct']:.0f}% | "
                f"{det['bearing_deg']:+.1f}° | {det.get('risk_level', 'NOMINAL')}"
            )
            tw = max(110, len(label) * 6 + 10)
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, min(W, x1 + tw), ty + 20], fill=color)
            draw.text((x1 + 5, ty + 3), label, fill=(255, 255, 255))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=90)
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "mode": mode_clean,
            "model_title": spec["title"],
            "dataset": spec["dataset"],
            "active_models": active_models,
            "benchmark_metrics": spec["metrics"],
            "inference_ms": round(elapsed_ms, 1),
            "resolution": {"width": W, "height": H},
            "attitude": {
                "roll_deg": round(roll_deg, 2),
                "pitch_offset_pct": round(pitch_offset_pct, 2),
                "sky_ratio_pct": round(sky_ratio, 1),
                "water_ratio_pct": round(water_ratio, 1),
            },
            "total_detections": len(detections),
            "detections": detections,
            "risk_summary": risk_summary,
            "colregs_risk": risk_summary,
            "image_base64": f"data:image/jpeg;base64,{b64_img}",
        }

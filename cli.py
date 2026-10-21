#!/usr/bin/env python3
"""Command-Line Interface for the Smart Buoy End-to-End AI & Telemetry Platform.

Supported subcommands:
  python cli.py status
  python cli.py spectra [--mode peak|latest] [--json]
  python cli.py forecast [--json]
  python cli.py detect <image> [--mode buoy|colregs|sar|horizon] [--conf 0.25] [--output out.jpg] [--json]
  python cli.py nmea-stream [--count 5] [--interval 0.1] [--mode latest|peak] [--json]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
SAMPLES_DIR = BASE_DIR / "samples"


def build_status_payload() -> Dict[str, Any]:
    """Assemble system, station, dataset, and model readiness status."""
    import torch
    from app.telemetry import BuoyTelemetryStore
    from app.vision import MODEL_SPECS

    store = BuoyTelemetryStore()
    assert store.df_met is not None and store.df_wq is not None and store.spec_times is not None

    return {
        "status": "online",
        "compute_device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU Inference Engine (H100 Pre-Trained Weights)"
        ),
        "stations": {
            "offshore_wave_met": "NOAA NDBC 46050 (Stonewall Bank, OR)",
            "coastal_water_quality": "NOAA NERRS ACQS1 (Estuarine Sonde)",
        },
        "datasets": {
            "wave_spectral_bursts": len(store.spec_times),
            "met_ocean_records": len(store.df_met),
            "water_quality_records": len(store.df_wq),
        },
        "models": {
            "wave_forecaster": {
                "name": "ResidualWaveForecaster (BiLSTM + 4-Head Attention)",
                "weights": "noaa_46050_wave_forecaster.pt",
                "ready": (WEIGHTS_DIR / "noaa_46050_wave_forecaster.pt").exists(),
                "r2_hs": 0.8862,
                "mae_hs_m": 0.127,
            },
            "wq_autoencoder": {
                "name": "WaterQualityAE (7-Channel LSTM Autoencoder)",
                "weights": "noaa_ACQS1_wq_autoencoder.pt",
                "ready": (WEIGHTS_DIR / "noaa_ACQS1_wq_autoencoder.pt").exists(),
                "threshold_mse": 0.536,
            },
            "vision_models": {
                k: {
                    "title": v["title"],
                    "weights": v["weights"],
                    "ready": (WEIGHTS_DIR / v["weights"]).exists(),
                    "metrics": v["metrics"],
                }
                for k, v in MODEL_SPECS.items()
            },
        },
    }


def cmd_status(args: argparse.Namespace) -> int:
    """Execute `python cli.py status`."""
    payload = build_status_payload()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("=" * 72)
    print("SMART BUOY AI & TELEMETRY PLATFORM -- SYSTEM STATUS")
    print("=" * 72)
    print(f"Status         : {payload['status'].upper()}")
    print(f"Compute Device : {payload['compute_device']} ({payload['gpu_name']})")
    print(f"Offshore Buoy  : {payload['stations']['offshore_wave_met']}")
    print(f"Estuarine Sonde: {payload['stations']['coastal_water_quality']}")
    print("-" * 72)
    ds = payload["datasets"]
    print(
        f"Loaded Records : {ds['wave_spectral_bursts']} spectral bursts | "
        f"{ds['met_ocean_records']} met-ocean rows | "
        f"{ds['water_quality_records']} water quality rows"
    )
    print("-" * 72)
    wf = payload["models"]["wave_forecaster"]
    ae = payload["models"]["wq_autoencoder"]
    print(f"[{'READY' if wf['ready'] else 'MISSING'}] {wf['name']} ({wf['weights']})")
    print(f"[{'READY' if ae['ready'] else 'MISSING'}] {ae['name']} ({ae['weights']})")
    for mode_key, vm in payload["models"]["vision_models"].items():
        tag = "READY" if vm["ready"] else "MISSING"
        m = vm["metrics"]
        print(
            f"[{tag}] [{mode_key.upper():7s}] {vm['title']} "
            f"(mAP@50={m['map50']:.2f}%, P={m['precision']:.2f}%)"
        )
    print("=" * 72)
    return 0


def cmd_spectra(args: argparse.Namespace) -> int:
    """Execute `python cli.py spectra`."""
    from app.telemetry import BuoyTelemetryStore

    store = BuoyTelemetryStore()
    payload = store.get_wave_spectra_payload(mode=args.mode)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    m = payload["metrics"]
    print("=" * 72)
    print(f"WAVE ENERGY DENSITY SPECTRA -- {payload['station_id']}")
    print("=" * 72)
    print(f"Selected Mode      : {payload['selected_mode'].upper()} ({payload['selected_timestamp']})")
    print(f"Total Spectral Runs: {payload['total_bursts']} bursts (46 frequency bands)")
    print(f"Significant Hm0    : {m['hm0_m']:.3f} m (Peak: {m['peak_hm0_m']:.3f} m | Latest: {m['latest_hm0_m']:.3f} m)")
    print(f"Peak Period Tp     : {m['tp_s']:.2f} s  (Tm01: {m['tm01_s']:.2f} s | Tm02: {m['tm02_s']:.2f} s)")
    print(f"Spectral Bandwidth : eps = {m['bandwidth_eps']:.3f}")
    print(f"NDBC WVHT Corr (r) : {m['correlation_r']:.4f}")
    print("-" * 72)
    freqs = payload["spectrum"]["frequencies_hz"]
    meas = payload["spectrum"]["measured_m2_hz"]
    jons = payload["spectrum"]["jonswap_m2_hz"]
    top_indices = sorted(range(len(meas)), key=lambda i: meas[i], reverse=True)[:5]
    print("Top 5 Energetic Frequency Bands:")
    for idx in top_indices:
        f_hz = freqs[idx]
        period_s = 1.0 / f_hz if f_hz > 0 else 0.0
        print(
            f"  f = {f_hz:6.4f} Hz (T = {period_s:5.2f} s) | "
            f"S_measured = {meas[idx]:8.4f} m^2/Hz | S_JONSWAP = {jons[idx]:8.4f} m^2/Hz"
        )
    print("=" * 72)
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    """Execute `python cli.py forecast`."""
    from app.models import BuoyTimeSeriesEngine
    from app.telemetry import BuoyTelemetryStore

    store = BuoyTelemetryStore()
    engine = BuoyTimeSeriesEngine()
    payload = engine.evaluate_and_forecast(store)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    wf = payload["wave_forecast"]
    wm = wf["metrics"]
    lp = wf["live_prediction"]
    wq = payload["water_quality_anomaly"]

    print("=" * 72)
    print("DEEP LEARNING WAVE FORECAST & WATER QUALITY ANOMALY REPORT")
    print("=" * 72)
    print(f"Wave Model         : {wf['model_name']} [{wf['device']}]")
    print(f"Forecast Horizon   : {wf['horizon']}")
    print(f"Validation Hs      : MAE = {wm['hs_mae_m']:.3f} m | R^2 = {wm['hs_r2']:.4f}")
    print(f"Validation Period  : MAE = {wm['apd_mae_s']:.3f} s | R^2 = {wm['apd_r2']:.4f}")
    print(
        f"Live 1-Hr Hs Pred  : {lp['current_hs_m']:.2f} m -> {lp['forecast_hs_1h_m']:.2f} m "
        f"({lp['delta_hs_m']:+.3f} m)"
    )
    print(
        f"Live 1-Hr APD Pred : {lp['current_apd_s']:.2f} s -> {lp['forecast_apd_1h_s']:.2f} s "
        f"({lp['delta_apd_s']:+.3f} s)"
    )
    print("-" * 72)
    print(f"Anomaly Model      : {wq['model_name']}")
    print(f"MSE Threshold (95%): {wq['threshold_mse']:.4f}")
    print(f"Flagged Events     : {wq['flagged_anomalies']} / {wq['total_windows']} windows")
    status_str = "ANOMALY DETECTED" if wq["latest_is_anomaly"] else "NORMAL"
    print(f"Latest Window MSE  : {wq['latest_mse']:.4f} [{status_str}]")
    print("=" * 72)
    return 0


def _resolve_image_path(image_arg: str) -> Optional[Path]:
    """Resolve an image path directly, relative to repo root, or inside samples/."""
    if not isinstance(image_arg, str) or not image_arg.strip():
        return None
    cleaned = image_arg.strip()
    cand = Path(cleaned)
    if cand.is_file():
        return cand.resolve()
    repo_cand = BASE_DIR / cleaned
    if repo_cand.is_file():
        return repo_cand.resolve()
    sample_cand = SAMPLES_DIR / cleaned
    if sample_cand.is_file():
        return sample_cand.resolve()
    # Search by bare filename inside samples/* only when no directory prefix was given
    if cand.parent == Path(".") and cand.name and SAMPLES_DIR.exists():
        for match in sorted(SAMPLES_DIR.rglob("*")):
            if match.is_file() and match.name == cand.name:
                return match.resolve()
    return None


def _infer_vision_mode(image_path: Path, explicit_mode: Optional[str]) -> str:
    """Infer vision mode from image path if not explicitly specified."""
    if explicit_mode:
        return explicit_mode
    lower_parts = "/".join(image_path.parts).lower()
    if "sar" in lower_parts:
        return "sar"
    if "colregs" in lower_parts or "vessel" in lower_parts:
        return "colregs"
    return "buoy"


def cmd_detect(args: argparse.Namespace) -> int:
    """Execute `python cli.py detect <image>`."""
    from app.vision import MarineVisionEngine

    resolved = _resolve_image_path(args.image)
    if resolved is None:
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        return 1

    if not (math.isfinite(args.conf) and 0.0 < args.conf <= 1.0):
        print(f"Error: Confidence threshold must be in (0.0, 1.0], got {args.conf}", file=sys.stderr)
        return 1

    if not (math.isfinite(args.hfov) and 0.0 < args.hfov < 180.0):
        print(f"Error: Horizontal FOV (--hfov) must be in (0.0, 180.0), got {args.hfov}", file=sys.stderr)
        return 1

    mode = _infer_vision_mode(resolved, args.mode)
    source_hint = None
    try:
        rel_to_samples = resolved.relative_to(SAMPLES_DIR.resolve())
        source_hint = rel_to_samples.as_posix()
    except ValueError:
        source_hint = resolved.name

    try:
        img_bytes = resolved.read_bytes()
    except OSError as exc:
        print(f"Error: Unable to read image file {resolved}: {exc}", file=sys.stderr)
        return 1

    if not img_bytes:
        print(f"Error: Image file is empty: {resolved}", file=sys.stderr)
        return 1

    engine = MarineVisionEngine()
    try:
        res = engine.run_detection(
            image_bytes=img_bytes,
            mode=mode,
            conf_threshold=args.conf,
            hfov_deg=args.hfov,
            source_hint=source_hint,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        b64_data = res["image_base64"].split(",", 1)[-1]
        out_path.write_bytes(base64.b64decode(b64_data))
        res["saved_output"] = str(out_path.resolve())

    if args.json:
        json_out = dict(res)
        if not args.include_base64:
            json_out.pop("image_base64", None)
        print(json.dumps(json_out, indent=2))
        return 0

    att = res["attitude"]
    print("=" * 72)
    print(f"MARINE COMPUTER VISION -- {res['model_title']}")
    print("=" * 72)
    print(f"Input Image        : {resolved}")
    print(f"Mode / Resolution  : {res['mode']} | {res['resolution']['width']}x{res['resolution']['height']} px")
    print(f"Inference Latency  : {res['inference_ms']:.1f} ms")
    print(
        f"Optical Attitude   : Roll = {att['roll_deg']:+.2f} deg | Pitch Offset = {att['pitch_offset_pct']:+.2f}% "
        f"(Sky {att['sky_ratio_pct']:.1f}% / Water {att['water_ratio_pct']:.1f}%)"
    )
    print(f"Total Detections   : {res['total_detections']}")
    print("-" * 72)
    if res["detections"]:
        for det in res["detections"]:
            bbox_str = f"[{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]"
            print(
                f"  #{det['id']:02d} {det['class_name']:28s} | "
                f"Conf: {det['confidence_pct']:5.1f}% | "
                f"Bearing: {det['bearing_deg']:+6.2f} deg | "
                f"Range: ~{det['est_range_m']:5.1f} m | BBox: {bbox_str}"
            )
    else:
        print("  No maritime targets detected above confidence threshold.")
    if args.output:
        print("-" * 72)
        print(f"Saved Overlay      : {res['saved_output']}")
    print("=" * 72)
    return 0


def cmd_nmea_stream(args: argparse.Namespace) -> int:
    """Execute `python cli.py nmea-stream`."""
    from app.simulator import SmartBuoySimulator

    if args.count < 0:
        print(f"Error: --count must be >= 0, got {args.count}", file=sys.stderr)
        return 1
    if not math.isfinite(args.interval) or args.interval < 0:
        print(f"Error: --interval must be a finite number >= 0, got {args.interval}", file=sys.stderr)
        return 1

    sim = SmartBuoySimulator(seed=args.seed, mode=args.mode)
    target_count: Optional[int] = None if args.count == 0 else args.count

    try:
        for frame in sim.stream_frames(
            count=target_count,
            interval_sec=args.interval,
            jitter=not args.no_jitter,
        ):
            if args.json:
                print(json.dumps(frame), flush=True)
            else:
                imu = frame["imu_6dof"]
                gps = frame["gps"]
                met = frame["met_ocean"]
                wq = frame["water_quality"]
                print(
                    f"[Frame #{frame['sequence_id']:03d} | {frame['timestamp_utc']}] "
                    f"GPS: ({gps['lat']:.6f}, {gps['lon']:.6f}) drift={gps['watch_circle_offset_m']:.1f}m | "
                    f"IMU: heave={imu['heave_m']:+.3f}m roll={imu['roll_deg']:+.2f}deg pitch={imu['pitch_deg']:+.2f}deg | "
                    f"Met: WVHT={met['WVHT_m']:.2f}m WSPD={met['WSPD_mps']:.1f}m/s | "
                    f"Sonde: T={wq['OTMP_c']:.2f}C SAL={wq['SAL_psu']:.2f}psu O2={wq['O2_ppm']:.2f}ppm pH={wq['PH']:.2f}"
                )
                for nmea in frame["nmea_sentences"]:
                    print(f"  {nmea}", flush=True)
    except KeyboardInterrupt:
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Smart Buoy End-to-End AI, Spectral Wave Physics, Vision & NMEA-0183 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. status
    p_status = subparsers.add_parser("status", help="Show system, station, dataset, and AI model status")
    p_status.add_argument("--json", action="store_true", help="Output status payload as JSON")
    p_status.set_defaults(func=cmd_status)

    # 2. spectra
    p_spectra = subparsers.add_parser("spectra", help="Compute 46-band wave energy spectra & JONSWAP fit")
    p_spectra.add_argument(
        "--mode",
        choices=["peak", "latest"],
        default="peak",
        help="Select storm peak or latest spectral burst (default: peak)",
    )
    p_spectra.add_argument("--json", action="store_true", help="Output spectral payload as JSON")
    p_spectra.set_defaults(func=cmd_spectra)

    # 3. forecast
    p_forecast = subparsers.add_parser(
        "forecast",
        help="Run BiLSTM+Attention wave forecaster and LSTM water quality anomaly detector",
    )
    p_forecast.add_argument("--json", action="store_true", help="Output forecast payload as JSON")
    p_forecast.set_defaults(func=cmd_forecast)

    # 4. detect <image>
    p_detect = subparsers.add_parser(
        "detect",
        help="Run YOLOv11m marine object detection & optical horizon attitude estimation on an image",
    )
    p_detect.add_argument("image", help="Path to input image or sample filename/ID under samples/")
    p_detect.add_argument(
        "--mode",
        choices=["buoy", "colregs", "sar", "horizon", "all", "ensemble"],
        default=None,
        help="Vision mode (auto-inferred from path if omitted)",
    )
    p_detect.add_argument("--conf", type=float, default=0.25, help="Confidence threshold in (0, 1] (default: 0.25)")
    p_detect.add_argument("--hfov", type=float, default=90.0, help="Horizontal camera FOV in degrees (default: 90.0)")
    p_detect.add_argument("-o", "--output", default=None, help="Optional path to save annotated output JPEG")
    p_detect.add_argument("--json", action="store_true", help="Output detection payload as JSON")
    p_detect.add_argument(
        "--include-base64",
        action="store_true",
        help="Keep base64-encoded annotated JPEG in --json output",
    )
    p_detect.set_defaults(func=cmd_detect)

    # 5. nmea-stream
    p_nmea = subparsers.add_parser(
        "nmea-stream",
        help="Stream synthesized 6-DOF IMU, GPS drift, sonde water quality, and NMEA-0183 sentences",
    )
    p_nmea.add_argument(
        "-n",
        "--count",
        type=int,
        default=5,
        help="Number of frames to stream (default: 5; use 0 for continuous stream)",
    )
    p_nmea.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.1,
        help="Interval in seconds between frames (default: 0.1)",
    )
    p_nmea.add_argument(
        "--mode",
        choices=["latest", "peak"],
        default="latest",
        help="Wave spectrum sea-state mode for 6-DOF IMU synthesis (default: latest)",
    )
    p_nmea.add_argument("--seed", type=int, default=42, help="Random seed for phase synthesis (default: 42)")
    p_nmea.add_argument("--no-jitter", action="store_true", help="Disable Gaussian sensor micro-jitter")
    p_nmea.add_argument("--json", action="store_true", help="Output JSON Lines (one JSON object per frame)")
    p_nmea.set_defaults(func=cmd_nmea_stream)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

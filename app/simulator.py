"""Edge Smart Buoy Hardware & NMEA-0183 / JSON Telemetry Simulator.

Synthesizes realistic 6-DOF IMU wave motion (from real 46-band NOAA NDBC 46050
spectral energy density measurements), watch-circle GPS drift, 7-channel water
quality sonde frames (from real NOAA NERRS ACQS1 data), and standard marine
NMEA-0183 sentences ($GPZDA, $WIMWV, $YXMTW).
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .telemetry import BuoyTelemetryStore

# Nominal mooring coordinates for NOAA NDBC Station 46050 (Stonewall Bank, 20 NM West of Newport, OR)
STATION_46050_LAT = 44.6690
STATION_46050_LON = -124.5460
WATCH_CIRCLE_RADIUS_M = 42.0
GRAVITY_MPS2 = 9.80665


def nmea_checksum(sentence_body: str) -> str:
    """Compute the 2-digit uppercase hexadecimal XOR checksum for an NMEA-0183 sentence body.

    Strips leading '$' or '!' and trailing '*..' if present before computing XOR.
    """
    body = sentence_body.strip()
    if body.startswith(("$", "!")):
        body = body[1:]
    if "*" in body:
        body = body.split("*", 1)[0]
    csum = 0
    for ch in body:
        csum ^= ord(ch)
    return f"{csum:02X}"


def format_nmea(sentence_body: str) -> str:
    """Wrap an NMEA-0183 payload in '$<payload>*<XOR_HEX>' format."""
    body = sentence_body.strip()
    if body.startswith(("$", "!")):
        body = body[1:]
    if "*" in body:
        body = body.split("*", 1)[0]
    return f"${body}*{nmea_checksum(body)}"


def verify_nmea_sentence(sentence: str) -> bool:
    """Validate that an NMEA-0183 sentence starts with '$' and has a matching '*XX' XOR checksum."""
    s = sentence.strip()
    if not s.startswith("$") or "*" not in s:
        return False
    body, _, given_hex = s[1:].partition("*")
    if not body or len(given_hex) != 2:
        return False
    return nmea_checksum(body).upper() == given_hex.upper()


def build_gpzda(dt: Optional[datetime] = None) -> str:
    """Build an NMEA-0183 $GPZDA UTC Date & Time sentence.

    Format: $GPZDA,hhmmss.ss,dd,mm,yyyy,xx,yy*CC
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    centisec = int(round(dt.microsecond / 10000.0)) % 100
    hms = f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.{centisec:02d}"
    body = f"GPZDA,{hms},{dt.day:02d},{dt.month:02d},{dt.year:04d},00,00"
    return format_nmea(body)


def build_wimwv(
    wind_dir_deg: float,
    wind_speed_mps: float,
    reference: str = "R",
    units: str = "M",
    status: str = "A",
) -> str:
    """Build an NMEA-0183 $WIMWV Weather Instrument Wind Speed and Angle sentence.

    Format: $WIMWV,x.x,R|T,x.x,M|N|K,A*CC
    """
    ref = reference.upper() if reference.upper() in ("R", "T") else "R"
    unit = units.upper() if units.upper() in ("M", "N", "K") else "M"
    stat = status.upper() if status.upper() in ("A", "V") else "A"
    angle = float(wind_dir_deg) % 360.0
    speed = max(0.0, float(wind_speed_mps))
    body = f"WIMWV,{angle:.1f},{ref},{speed:.2f},{unit},{stat}"
    return format_nmea(body)


def build_yxmtw(water_temp_c: float) -> str:
    """Build an NMEA-0183 $YXMTW Transducer Water Temperature sentence.

    Format: $YXMTW,x.x,C*CC
    """
    temp = float(water_temp_c)
    body = f"YXMTW,{temp:.2f},C"
    return format_nmea(body)


class SmartBuoySimulator:
    """Edge Smart Buoy Hardware & NMEA-0183 / JSON Telemetry Simulator.

    Uses real 46-band spectral density bursts from NOAA NDBC 46050 and 7-channel
    water quality sonde telemetry from NOAA NERRS ACQS1 to synthesize:
      - 6-DOF IMU wave motion (surge, sway, heave, roll, pitch, yaw, 3-axis accel & gyro)
      - Watch-circle GPS mooring position & drift telemetry
      - Sonde water quality & meteorological frames
      - Validated NMEA-0183 sentences ($GPZDA, $WIMWV, $YXMTW)
    """

    def __init__(
        self,
        store: Optional[BuoyTelemetryStore] = None,
        seed: int = 42,
        mode: str = "latest",
    ) -> None:
        self.store = store if store is not None else BuoyTelemetryStore()
        self.rng = np.random.default_rng(seed)
        self.mode = mode if mode in ("latest", "peak") else "latest"
        self._start_monotonic = time.monotonic()
        self._step_counter = 0

        assert self.store.freqs is not None and self.store.spec_matrix is not None
        assert self.store.hm0_series is not None and self.store.tp_series is not None
        assert self.store.df_met is not None and self.store.df_wq is not None

        self.freqs: np.ndarray = self.store.freqs.astype(np.float64)
        self.df_bins: np.ndarray = np.gradient(self.freqs)
        self.omega: np.ndarray = 2.0 * math.pi * self.freqs
        # Deep-water linear dispersion relation: k = omega^2 / g
        self.wavenumbers: np.ndarray = (self.omega ** 2) / GRAVITY_MPS2
        # Deterministic random initial phases across the 46 frequency bands
        self.phases_heave: np.ndarray = self.rng.uniform(0.0, 2.0 * math.pi, size=len(self.freqs))
        self.phases_roll: np.ndarray = self.phases_heave + (math.pi / 2.0)
        self.phases_pitch: np.ndarray = self.rng.uniform(0.0, 2.0 * math.pi, size=len(self.freqs))

    def _select_spectrum_index(self, mode: Optional[str] = None) -> int:
        assert self.store.hm0_series is not None
        m = mode or self.mode
        if m == "peak":
            return int(np.argmax(self.store.hm0_series))
        return len(self.store.hm0_series) - 1

    def sample_imu_6dof(self, t_sec: Optional[float] = None, mode: Optional[str] = None) -> Dict[str, float]:
        """Synthesize 6-DOF IMU buoy motion by superposing all 46 NOAA wave spectral bands."""
        assert self.store.spec_matrix is not None and self.store.df_met is not None
        if t_sec is None:
            t_sec = time.monotonic() - self._start_monotonic

        spec_idx = self._select_spectrum_index(mode=mode)
        s_f = np.maximum(self.store.spec_matrix[spec_idx], 0.0)
        # Wave component amplitudes a_i = sqrt(2 * S(f_i) * df_i)
        amps = np.sqrt(2.0 * s_f * self.df_bins)

        wdir_deg = float(self.store.df_met.iloc[-1]["WDIR_deg"])
        wave_dir_rad = math.radians(wdir_deg)

        phase_t = self.omega * float(t_sec) + self.phases_heave
        cos_phase = np.cos(phase_t)
        sin_phase = np.sin(phase_t)

        # Linear Airy orbital displacements (heave, surge, sway)
        heave_m = float(np.sum(amps * cos_phase))
        orbital_horiz_m = float(np.sum(amps * sin_phase))
        surge_m = orbital_horiz_m * math.cos(wave_dir_rad)
        sway_m = orbital_horiz_m * math.sin(wave_dir_rad)

        # Linear Airy vertical & horizontal accelerations: a_z = -sum(a_i * omega_i^2 * cos(phase))
        accel_z_dyn = float(-np.sum(amps * (self.omega ** 2) * cos_phase))
        accel_horiz = float(np.sum(amps * (self.omega ** 2) * sin_phase))
        accel_x_mps2 = accel_horiz * math.cos(wave_dir_rad)
        accel_y_mps2 = accel_horiz * math.sin(wave_dir_rad)
        accel_z_mps2 = GRAVITY_MPS2 + accel_z_dyn

        # Surface wave slope & hull pitch/roll response (degrees) and angular rates (deg/s)
        slope_components = amps * self.wavenumbers
        roll_rad = float(np.sum(slope_components * np.sin(self.omega * float(t_sec) + self.phases_roll)))
        pitch_rad = float(np.sum(slope_components * np.cos(self.omega * float(t_sec) + self.phases_pitch)))
        roll_deg = math.degrees(roll_rad) * 1.35
        pitch_deg = math.degrees(pitch_rad) * 1.15
        yaw_deg = (wdir_deg + 4.5 * math.sin(0.08 * float(t_sec))) % 360.0

        gyro_x_dps = math.degrees(
            float(np.sum(slope_components * self.omega * np.cos(self.omega * float(t_sec) + self.phases_roll)))
        ) * 1.35
        gyro_y_dps = math.degrees(
            float(-np.sum(slope_components * self.omega * np.sin(self.omega * float(t_sec) + self.phases_pitch)))
        ) * 1.15
        gyro_z_dps = 4.5 * 0.08 * math.cos(0.08 * float(t_sec))

        return {
            "surge_m": round(surge_m, 4),
            "sway_m": round(sway_m, 4),
            "heave_m": round(heave_m, 4),
            "roll_deg": round(roll_deg, 2),
            "pitch_deg": round(pitch_deg, 2),
            "yaw_deg": round(yaw_deg, 2),
            "accel_x_mps2": round(accel_x_mps2, 4),
            "accel_y_mps2": round(accel_y_mps2, 4),
            "accel_z_mps2": round(accel_z_mps2, 4),
            "gyro_x_dps": round(gyro_x_dps, 3),
            "gyro_y_dps": round(gyro_y_dps, 3),
            "gyro_z_dps": round(gyro_z_dps, 3),
        }

    def sample_gps(self, t_sec: Optional[float] = None, imu: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Synthesize GPS watch-circle mooring coordinates around NOAA Station 46050."""
        assert self.store.df_met is not None
        if t_sec is None:
            t_sec = time.monotonic() - self._start_monotonic
        if imu is None:
            imu = self.sample_imu_6dof(t_sec=t_sec)

        latest_m = self.store.df_met.iloc[-1]
        wdir_deg = float(latest_m["WDIR_deg"])
        wspd_mps = float(latest_m["WSPD_mps"])

        # Downwind mooring excursion + slow tidal figure-8 + wave orbital surge/sway
        downwind_rad = math.radians((wdir_deg + 180.0) % 360.0)
        excursion_r = min(WATCH_CIRCLE_RADIUS_M, 12.0 + 2.4 * wspd_mps)
        east_m = (
            excursion_r * math.sin(downwind_rad)
            + 4.5 * math.sin(0.015 * float(t_sec))
            + imu["sway_m"]
        )
        north_m = (
            excursion_r * math.cos(downwind_rad)
            + 3.2 * math.cos(0.011 * float(t_sec))
            + imu["surge_m"]
        )

        # Convert meter offsets to degrees latitude / longitude at 44.669 N
        meters_per_deg_lat = 111_132.0
        meters_per_deg_lon = 111_320.0 * math.cos(math.radians(STATION_46050_LAT))
        lat = STATION_46050_LAT + (north_m / meters_per_deg_lat)
        lon = STATION_46050_LON + (east_m / meters_per_deg_lon)
        offset_m = math.hypot(east_m, north_m)
        drift_bearing_deg = (math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0

        return {
            "station_id": "46050",
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "nominal_lat": STATION_46050_LAT,
            "nominal_lon": STATION_46050_LON,
            "watch_circle_offset_m": round(offset_m, 2),
            "watch_circle_limit_m": WATCH_CIRCLE_RADIUS_M,
            "drift_bearing_deg": round(drift_bearing_deg, 1),
            "sog_knots": round(abs(imu["surge_m"]) * 0.35, 2),
            "fix_quality": 2,  # DGPS
            "satellites": 11,
            "hdop": 0.8,
        }

    def sample_met_ocean(self, index: int = -1, jitter: bool = True) -> Dict[str, Any]:
        """Return a meteorological-oceanographic frame anchored on NOAA NDBC 46050."""
        assert self.store.df_met is not None
        idx = index if -len(self.store.df_met) <= index < len(self.store.df_met) else -1
        row = self.store.df_met.iloc[idx]
        noise = (lambda s: float(self.rng.normal(0.0, s))) if jitter else (lambda _: 0.0)

        wdir = (float(row["WDIR_deg"]) + noise(0.6)) % 360.0
        wspd = max(0.0, float(row["WSPD_mps"]) + noise(0.08))
        gst = max(wspd, float(row["GST_mps"]) + noise(0.10))
        wvht = max(0.05, float(row["WVHT_m"]) + noise(0.01))
        dpd = max(1.0, float(row["DPD_s"]))
        apd = max(1.0, float(row["APD_s"]) + noise(0.02))
        pres = float(row["PRES_hpa"]) + noise(0.05)

        return {
            "station": "NOAA NDBC 46050",
            "source_timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
            "WDIR_deg": round(wdir, 1),
            "WSPD_mps": round(wspd, 2),
            "GST_mps": round(gst, 2),
            "WVHT_m": round(wvht, 2),
            "DPD_s": round(dpd, 2),
            "APD_s": round(apd, 2),
            "PRES_hpa": round(pres, 1),
        }

    def sample_sonde(self, index: int = -1, jitter: bool = True) -> Dict[str, Any]:
        """Return a 7-channel water quality sonde frame anchored on NOAA NERRS ACQS1."""
        assert self.store.df_wq is not None
        idx = index if -len(self.store.df_wq) <= index < len(self.store.df_wq) else -1
        row = self.store.df_wq.iloc[idx]
        noise = (lambda s: float(self.rng.normal(0.0, s))) if jitter else (lambda _: 0.0)

        otmp = float(row["OTMP_c"]) + noise(0.02)
        cond = max(0.0, float(row["COND_mscm"]) + noise(0.03))
        sal = max(0.0, float(row["SAL_psu"]) + noise(0.02))
        o2_pct = max(0.0, float(row["O2_pct"]) + noise(0.15))
        o2_ppm = max(0.0, float(row["O2_ppm"]) + noise(0.02))
        turb = max(0.1, float(row["TURB_ntu"]) + noise(0.1))
        ph = float(np.clip(float(row["PH"]) + noise(0.01), 0.0, 14.0))

        return {
            "station": "NOAA NERRS ACQS1",
            "source_timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
            "OTMP_c": round(otmp, 2),
            "COND_mscm": round(cond, 2),
            "SAL_psu": round(sal, 2),
            "O2_pct": round(o2_pct, 1),
            "O2_ppm": round(o2_ppm, 2),
            "TURB_ntu": round(turb, 1),
            "PH": round(ph, 2),
        }

    def generate_nmea_sentences(
        self,
        met: Optional[Dict[str, Any]] = None,
        sonde: Optional[Dict[str, Any]] = None,
        dt: Optional[datetime] = None,
    ) -> List[str]:
        """Generate standard NMEA-0183 sentences ($GPZDA, $WIMWV, $YXMTW) with XOR checksums."""
        if met is None:
            met = self.sample_met_ocean(jitter=False)
        if sonde is None:
            sonde = self.sample_sonde(jitter=False)

        gpzda = build_gpzda(dt=dt)
        wimwv = build_wimwv(
            wind_dir_deg=float(met["WDIR_deg"]),
            wind_speed_mps=float(met["WSPD_mps"]),
            reference="R",
            units="M",
            status="A",
        )
        yxmtw = build_yxmtw(water_temp_c=float(sonde["OTMP_c"]))
        return [gpzda, wimwv, yxmtw]

    def generate_frame(
        self,
        t_sec: Optional[float] = None,
        index: int = -1,
        dt: Optional[datetime] = None,
        jitter: bool = True,
    ) -> Dict[str, Any]:
        """Synthesize a complete Edge Smart Buoy telemetry frame (IMU + GPS + Met + Sonde + NMEA-0183)."""
        assert self.store.hm0_series is not None and self.store.tp_series is not None
        assert self.store.tm01_series is not None and self.store.tm02_series is not None
        assert self.store.eps_series is not None

        if dt is None:
            dt = datetime.now(timezone.utc)
        if t_sec is None:
            t_sec = self._step_counter * 0.5
        self._step_counter += 1

        spec_idx = self._select_spectrum_index()
        imu = self.sample_imu_6dof(t_sec=t_sec)
        gps = self.sample_gps(t_sec=t_sec, imu=imu)
        met = self.sample_met_ocean(index=index, jitter=jitter)
        sonde = self.sample_sonde(index=index, jitter=jitter)
        nmea = self.generate_nmea_sentences(met=met, sonde=sonde, dt=dt)

        return {
            "sequence_id": self._step_counter,
            "timestamp_utc": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "buoy_id": "NOAA-46050-ACQS1-EDGE",
            "imu_6dof": imu,
            "gps": gps,
            "met_ocean": met,
            "water_quality": sonde,
            "wave_spectra_summary": {
                "mode": self.mode,
                "hm0_m": round(float(self.store.hm0_series[spec_idx]), 3),
                "tp_s": round(float(self.store.tp_series[spec_idx]), 2),
                "tm01_s": round(float(self.store.tm01_series[spec_idx]), 2),
                "tm02_s": round(float(self.store.tm02_series[spec_idx]), 2),
                "bandwidth_eps": round(float(self.store.eps_series[spec_idx]), 3),
            },
            "nmea_sentences": nmea,
        }

    def step(self) -> Dict[str, Any]:
        """Alias for generate_frame() advancing the internal simulation clock."""
        return self.generate_frame()

    def stream_frames(
        self,
        count: Optional[int] = 10,
        interval_sec: float = 0.5,
        jitter: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Yield synthesized Edge Smart Buoy telemetry frames at the requested interval."""
        emitted = 0
        t0 = time.monotonic()
        while count is None or emitted < count:
            elapsed = (time.monotonic() - t0) if interval_sec > 0 else (emitted * 0.5)
            yield self.generate_frame(t_sec=elapsed, jitter=jitter)
            emitted += 1
            if (count is None or emitted < count) and interval_sec > 0:
                time.sleep(interval_sec)

    def to_json(self, frame: Optional[Dict[str, Any]] = None, indent: Optional[int] = None) -> str:
        """Serialize a telemetry frame to a JSON string."""
        if frame is None:
            frame = self.generate_frame()
        return json.dumps(frame, indent=indent)


# Convenience alias
BuoySimulator = SmartBuoySimulator


if __name__ == "__main__":
    sim = SmartBuoySimulator()
    sample_frame = sim.generate_frame(t_sec=0.0, jitter=False)
    for sentence in sample_frame["nmea_sentences"]:
        print(sentence)
    print(sim.to_json(sample_frame, indent=2))

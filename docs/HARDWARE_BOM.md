# 🛠️ Smart Buoy Hardware Bill of Materials (BOM) & Sensor Integration Specification

This document specifies all sensors, edge AI compute boards, communication modules, solar/battery power hardware, and marine enclosure components required to build the physical **Smart Ocean Buoy** mapped 1-to-1 to the software modules in this repository.

---

## 1. System-to-Hardware Mapping Overview

| Buoy Subsystem | Physical Sensors & Hardware | Electrical / Bus Interface | Mapped Software Module & Channels |
| :--- | :--- | :--- | :--- |
| **Edge AI & Vision Compute** | **NVIDIA Jetson Orin Nano Super 8GB** *(or **Raspberry Pi 5 8GB + Hailo-8 26 TOPS AI HAT+**)* + **256 GB Industrial NVMe SSD** | PCIe / CSI-2 / USB3 / Ethernet / UART | [`app/main.py`](../app/main.py), [`app/models.py`](../app/models.py), [`app/vision.py`](../app/vision.py) (`YOLOv11m` 5 modes + PyTorch `ResidualWaveForecaster` & `WaterQualityAE`) |
| **Real-Time Sensor Logger & Power Watchdog** | **STM32H743 / Industrial ESP32-S3 MCU Board** + **Dual Isolated RS-485 & SDI-12 Interface + 4-Ch MOSFET Switch** | UART / SPI / I2C / RS-485 / SDI-12 | [`app/simulator.py`](../app/simulator.py), [`cli.py`](../cli.py) (20–50 Hz continuous IMU logging, sonde polling, NMEA-0183 framing, AI board wake/sleep duty cycling) |
| **Wave Motion & Attitude (6-DOF IMU + GNSS)** | **9-Axis Marine AHRS / Wave IMU** (*WitMotion HWT906 / BNO085 + ADXL355* or *Xsens MTi-630*) + **u-blox ZED-F9P / NEO-M9N GPS** | SPI / UART (`20–50 Hz`) + NMEA UART | [`app/spectral_physics.py`](../app/spectral_physics.py), [`app/telemetry.py`](../app/telemetry.py) (`46050.data_spec` 46-band $S(f)$, $H_{m0}$, $T_p$, $T_{m01}$, $T_{m02}$, Steepness $S_p$, Rogue Risk, `/ws/live` 6-DOF Heave/Roll/Pitch, `$GPZDA`) |
| **Marine Computer Vision Array** | **Sony Starvis 2 (`IMX678` 4K / `IMX462` 1080p) IP67 Wide-Angle Camera (`90°–110° HFOV`)** + Optional **FLIR Lepton 3.5 / InfiRay Thermal Core** | MIPI CSI-2 / USB3 / PoE | [`app/vision.py`](../app/vision.py), [`app/colregs_risk.py`](../app/colregs_risk.py) (Type 1 Buoy, Type 2 COLREGs Vessels, Type 3 SAR Swimmer, Type 4 Optical Horizon Roll/Pitch & Azimuth Radar HUD) |
| **Topside Meteorological Station** | **2-Axis Ultrasonic Marine Anemometer** (`RS-485`) + **Bosch BMP390L / BME280 Barometer & Temp** inside UV Radiation Shield | Isolated RS-485 Modbus / NMEA + I2C | [`app/telemetry.py`](../app/telemetry.py), [`app/alerts.py`](../app/alerts.py) (`46050.txt`: `WSPD_mps`, `GST_mps`, `WDIR_deg`, `PRES_hpa`, `$WIMWV`, Beaufort Scale & 3h Barometric Drop Alert) |
| **Submerged 7-Channel Water Quality Sonde** | **Submersible Multiparameter Sonde / RS-485 Sensor Manifold** (`Temp, Conductivity, Salinity, Optical DO, Wiped Turbidity, pH`) | Isolated RS-485 Modbus RTU / SDI-12 | [`app/telemetry.py`](../app/telemetry.py), [`app/models.py`](../app/models.py), [`app/alerts.py`](../app/alerts.py) (`ACQS1.ocean`: `OTMP_c`, `COND_mscm`, `SAL_psu`, `O2_pct`, `O2_ppm`, `TURB_ntu`, `PH`, `$YXMTW`) |
| **Offshore Telemetry & Marine Conspicuity** | **Teltonika RUT200 4G LTE Router** + **Poynting OMNI-402 Marine Antenna** + Optional **RockBLOCK 9603 Iridium SBD** + **IALA Yellow LED Lantern & Radar Reflector** | Ethernet / UART | Cloud/Shore synchronization, live WebSocket streaming, NMEA-0183 / MQTT backhaul, and IALA nighttime visual/radar safety |
| **Solar Power & Waterproof Buoy Hull** | **4× 25W (`100W`) ETFE Solar Panels** + **Victron SmartSolar MPPT 75/15** + **12.8V 100Ah (`1,280 Wh`) LiFePO4 Battery** + **IP68 M12 Bulkhead Connectors** | `VE.Direct` UART + 12V / 19V / 5V Rails | Provides **14+ days of zero-sun autonomy** under duty-cycled AI operation |

---

## 2. Itemized Procurement Bill of Materials & Cost Breakup (INR `₹`)

All prices below explicitly separate **Basic Price (Excl. GST)**, **GST @ 18%**, and **Total Payable (Incl. GST)**.

### 2.1 Core Compute, Sensors, Sonde, Telemetry & Power (With Option A: NVIDIA Jetson Orin Nano Super 8GB)

| # | Category | Item / Model Specification | Qty | Basic Unit Price (Excl. GST) | Basic Total (Excl. GST) | GST (18%) | Total Payable (Incl. GST) |
| :-: | :--- | :--- | :-: | ---: | ---: | ---: | ---: |
| 1 | **Edge AI Compute** | **NVIDIA Jetson Orin Nano Super Developer Kit (8 GB, 67 TOPS)** with heatsink & carrier | 1 | ₹23,500 | ₹23,500 | ₹4,230 | **₹27,730** |
| 2 | **Storage** | **256 GB Industrial M.2 NVMe SSD** (Wide-temp `-40°C to +85°C`) | 1 | ₹3,800 | ₹3,800 | ₹684 | **₹4,484** |
| 3 | **Watchdog & Logger MCU** | **STM32H743 / Industrial ESP32-S3 Control Board** + Dual Isolated RS-485 (ADM2483), SDI-12 & 4-Ch High-Side MOSFET Load Switch | 1 | ₹4,500 | ₹4,500 | ₹810 | **₹5,310** |
| 4 | **Wave Motion IMU** | **9-Axis Marine AHRS / Wave IMU (`WitMotion HWT906-TTL` IP67 + `ADXL355` Low-Noise MEMS)** (`20–50 Hz` heave/roll/pitch/spectra) | 1 | ₹14,500 | ₹14,500 | ₹2,610 | **₹17,110** |
| 5 | **GNSS / GPS** | **u-blox ZED-F9P / NEO-M9N Multi-GNSS Receiver** + Active IP67 Marine Dome Antenna (SMA) | 1 | ₹6,500 | ₹6,500 | ₹1,170 | **₹7,670** |
| 6 | **Marine Vision Camera** | **Sony Starvis 2 (`IMX678` 4K / `IMX462` 1080p) Wide-Angle (`100° HFOV`) Camera** in IP67 316L/Aluminum Marine Housing with Hydrophobic Window | 2 | ₹11,500 | ₹23,000 | ₹4,140 | **₹27,140** |
| 7 | **Met-Ocean Wind** | **2-Axis Ultrasonic Marine Anemometer (`WSPD`, `GST`, `WDIR`)** — Solid-State IP68 ASA Housing, RS-485 Modbus / NMEA-0183 | 1 | ₹18,500 | ₹18,500 | ₹3,330 | **₹21,830** |
| 8 | **Barometer & Air Temp** | **Bosch BMP390L + SHT35 High-Precision Barometric Pressure & Temp Probe** inside 6-Plate UV Gill Radiation Shield + ePTFE Vent | 1 | ₹3,200 | ₹3,200 | ₹576 | **₹3,776** |
| 9 | **WQ Sonde: Temp & Cond/Salinity** | **Submersible Titanium 4-Electrode / Inductive Marine Conductivity, Salinity & PT1000 Temp Sensor** (`0–70 mS/cm`, `0–42 PSU`, RS-485, IP68) | 1 | ₹16,000 | ₹16,000 | ₹2,880 | **₹18,880** |
| 10 | **WQ Sonde: Dissolved O₂** | **Submersible Optical Fluorescence Dissolved Oxygen (ODO) Sensor** (`0–20 mg/L`, `0–200%`, RS-485 Modbus, IP68 titanium/POM body) | 1 | ₹28,000 | ₹28,000 | ₹5,040 | **₹33,040** |
| 11 | **WQ Sonde: Turbidity** | **Submersible ISO 7027 90° NIR (`860 nm`) Turbidity Sensor (`0–1000 NTU`)** with Self-Cleaning Mechanical Wiper Brush (RS-485, IP68) | 1 | ₹32,000 | ₹32,000 | ₹5,760 | **₹37,760** |
| 12 | **WQ Sonde: Marine pH** | **Submersible Industrial Double-Junction Salt-Bridge Marine pH Sensor** (`0–14 pH`, RS-485 Modbus, IP68) | 1 | ₹9,500 | ₹9,500 | ₹1,710 | **₹11,210** |
| 13 | **4G LTE Comms** | **Teltonika RUT200 Industrial 4G LTE Router** + **High-Gain IP68 Marine Omni MIMO Antenna** | 1 | ₹16,500 | ₹16,500 | ₹2,970 | **₹19,470** |
| 14 | **Marine Safety** | **Self-Contained IALA Yellow (`Fl Y 4s`) Solar/12V LED Marine Lantern** (3 NM) + **Octahedral Aluminum Radar Reflector** | 1 | ₹11,000 | ₹11,000 | ₹1,980 | **₹12,980** |
| 15 | **Solar & MPPT** | **4× 25W (`100W` Total) Marine ETFE Monocrystalline Solar Panels** + **Victron SmartSolar MPPT 75/15** (`VE.Direct` telemetry) | 1 | ₹15,500 | ₹15,500 | ₹2,790 | **₹18,290** |
| 16 | **Battery Bank** | **12.8V 100Ah (`1,280 Wh`) LiFePO4 Marine Deep-Cycle Battery Pack** with 100A Smart BMS | 1 | ₹22,000 | ₹22,000 | ₹3,960 | **₹25,960** |
| 17 | **Marine Connectors & Power Rail** | **IP68 M12 Submersible Bulkhead Connectors (8×)**, Isolated 12V→19V/5V DC-DC Converters, TVS Surge Arrestors, Gore-Tex Vent & 316L Sonde Guard Cage | 1 | ₹14,000 | ₹14,000 | ₹2,520 | **₹16,520** |
| — | **TOTAL (Option A: Jetson Orin Nano Super 8GB)** | **Complete Smart Buoy Hardware Stack** | — | — | **₹2,62,000** | **₹47,160** | **₹3,09,160** |

---

### 2.2 Alternative Compute Option B: Raspberry Pi 5 (8 GB) + Official AI HAT+ (26 TOPS Hailo-8)

Replacing Item #1 (`NVIDIA Jetson Orin Nano Super 8GB`) with the lower-power **Raspberry Pi 5 (8 GB) + Raspberry Pi AI HAT+ (26 TOPS Hailo-8 NPU)** reduces both power draw and hardware cost:

| # | Category | Item / Model Specification | Qty | Basic Total (Excl. GST) | GST (18%) | Total Payable (Incl. GST) |
| :-: | :--- | :--- | :-: | ---: | ---: | ---: |
| 1B | **Edge AI Compute (Low-Power)** | **Raspberry Pi 5 (8 GB)** + **Official Raspberry Pi AI HAT+ (`26 TOPS` Hailo-8 NPU)** + Active Cooler + PCIe M.2 Dual Base | 1 | ₹17,500 | ₹3,150 | **₹20,650** |
| — | **TOTAL (Option B: Pi 5 8GB + Hailo-8 26 TOPS)** | **Complete Smart Buoy Hardware Stack (Items 2–17 + Item 1B)** | — | **₹2,56,000** | **₹46,080** | **₹3,02,080** |

---

## 3. Buoy Electrical Power Budget & Solar Autonomy Calculation

| Subsystem | Continuous Mode Draw | Duty-Cycled Mode (45s every 10 min = 7.5% duty) | Daily Energy (Duty-Cycled) |
| :--- | :---: | :---: | :---: |
| **STM32H7 / ESP32-S3 Watchdog + 9-DOF Wave IMU (`20 Hz`) + GPS + Barometer** | `0.45 W` (Always On) | `0.45 W` | `10.8 Wh / day` |
| **Ultrasonic Anemometer + 7-Channel Water Quality Sonde (Polled every 10 min)** | `1.80 W` | `0.35 W` avg | `8.4 Wh / day` |
| **Edge AI Board (`Jetson Orin Nano` or `Pi 5 + Hailo-8`) + 2× Cameras (`YOLOv11m`)** | `9.50 W` | `0.95 W` avg | `22.8 Wh / day` |
| **Teltonika RUT200 4G LTE Router (Burst Telemetry Sync)** | `2.80 W` | `0.65 W` avg | `15.6 Wh / day` |
| **Total Smart Buoy System Load** | **`14.55 W`** (`349 Wh/day`) | **`2.40 W` avg** | **`57.6 Wh / day`** |

- **Battery Capacity (`12.8V 100Ah LiFePO4`):** `1,280 Wh` (`1,024 Wh` usable at 80% DoD) → **17.7 days of complete darkness/storm autonomy** on a single charge.
- **Solar Harvest (`100W` array @ conservative 3.5 peak sun-hours/day):** `~280 Wh / day` → **4.8× daily energy surplus**, recharging a depleted battery even under overcast maritime conditions.

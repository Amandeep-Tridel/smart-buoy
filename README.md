# 🌊 Smart Buoy: End-to-End AI Command Center, YOLOv11m Marine Vision, Spectral Wave Physics & Met-Ocean Platform

[![Open In Colab (Google Drive)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dMjeZOfojmAeJRTV_f32L6bRwgLhDs3k)
[![Open In Colab (GitHub)](https://img.shields.io/badge/Colab-GitHub_Notebook-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Amandeep-Tridel/smart-buoy/blob/main/Smart_Buoy.ipynb)
![GPU](https://img.shields.io/badge/NVIDIA-H100_80GB_HBM3-76B900?logo=nvidia&logoColor=white)
![YOLOv11m](https://img.shields.io/badge/Ultralytics-YOLOv11m-00FFFF)
![FastAPI](https://img.shields.io/badge/FastAPI-Full--Stack_Web_App-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11+cu130-EE4C2C?logo=pytorch&logoColor=white)

A complete **End-to-End Smart Ocean Buoy AI Command Center** featuring a **Full-Stack Web Dashboard + FastAPI Backend (`app/`)**, **5 Pre-Trained NVIDIA H100 Model Weights (`weights/`)**, **Live NOAA NDBC & NERRS Telemetry (`data/`)**, **12 Real Marine Validation Samples (`samples/`)**, and a **Reproducible H100 Training Notebook (`Smart_Buoy.ipynb`)**:

1. **Full-Stack Web Dashboard & FastAPI Backend (`run.py`, `app/`)** — Interactive Maritime Command & Control UI with live **2 Hz WebSocket 6-DOF IMU Heave/Roll/Pitch streaming**, **46-band Wave Energy Spectra $S(f)$**, **1-Hour Ahead BiLSTM+Attention Sea-State Forecasting**, **7-Channel Sonde Autoencoder Anomaly Detection**, and an interactive **YOLOv11m Marine Vision Studio** with a **Polar Azimuth Radar HUD ($\pm 45^\circ$)** and **Artificial Horizon Gyroscope**.
2. **Spectral Wave Physics & Sea-State Reconstruction** — Live 46-band non-directional wave energy density spectra $S(f)$ from **NOAA NDBC Station `46050`** (*Stonewall Bank, Oregon*), computing spectral moments ($m_0, m_1, m_2, m_4$), significant wave height ($H_{m0} = 4\sqrt{m_0}$), peak/mean periods ($T_p, T_{m01}, T_{m02}$), and JONSWAP theoretical validation ($r = 0.9977$).
3. **Multi-Sensor Met-Ocean & Estuarine Water Quality Telemetry** — Live **NOAA NDBC `46050`** meteorological records (6,466 observations) and **NOAA NERRS `ACQS1`** 7-channel water quality sonde telemetry (4,364 observations of temperature, conductivity, salinity, dissolved oxygen, turbidity, and pH).
4. **H100 GPU Deep Learning Forecasting & Anomaly Detection** — **Residual BiLSTM + Multi-Head Attention Wave Forecaster** ($H_s, T_a$) and **Deep LSTM Water Quality Autoencoder** for unsupervised turbidity plume and hypoxia detection.
5. **All 4 Types of Marine Computer Vision (`YOLOv11m`)**:
   - **Type 1:** Sea-Surface Navigation Buoy Detection (`ARG-NCTU/KS_Buoy_dataset_2025`) — **94.07% mAP@50** | **97.51% Precision**
   - **Type 2:** COLREGs Multi-Class Maritime Vessel, Ship, Craft & Buoy Detection (`SPSCD + Singapore Maritime Dataset + KS_Buoy`) — **66.10% mAP@50** | **82.5% Peak Precision**
   - **Type 3:** Maritime Search & Rescue (SAR) Person-Overboard / Swimmer & Rescue Board Detection (`Voxel51/AFO-Aerial_Floating_Objects` @ `1280px`) — **83.85% mAP@50** | **83.18% Precision**
   - **Type 4:** Optical Sea-Sky Horizon Attitude Estimation (Roll $\phi_{\text{opt}}$, Pitch $\theta_{\text{opt}}$), 3-Zone Water/Sky/Obstacle Semantic Segmentation & Azimuth Bearing HUD ($^\circ$).

---

## 🚀 Quick Start — Launch the Full-Stack Web Application

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the FastAPI + WebSocket Command Center (http://127.0.0.1:8000)
python run.py

# 3. Run the automated End-to-End Pytest Suite
pytest -v tests/test_smart_buoy_e2e.py
```

### REST & WebSocket API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Interactive Smart Buoy AI Command Center Web Dashboard |
| `GET` | `/api/status` | System health, GPU/CPU compute status & readiness of all 5 H100 models |
| `GET` | `/api/wave-spectra?mode=peak\|latest` | 46-band NOAA `46050` wave energy spectrum $S(f)$, JONSWAP fit & $H_{m0}$ correlation |
| `GET` | `/api/telemetry` | Recent NOAA `46050` Met-Ocean & NOAA `ACQS1` 7-channel Water Quality Sonde series |
| `POST` | `/api/telemetry/refresh` | Live re-sync of real-time telemetry directly from NOAA NDBC servers |
| `GET` | `/api/forecast` | Live 1-hour ahead BiLSTM+Attention wave forecast & LSTM Autoencoder sonde anomalies |
| `GET` | `/api/vision/samples` | List bundled validation images across Buoy, COLREGs Vessels & SAR Swimmer tracks |
| `POST` | `/api/vision/detect-sample` | Run YOLOv11m + Optical Horizon + Azimuth Bearing HUD on a bundled sample image |
| `POST` | `/api/vision/detect-upload` | Upload any custom marine image for live YOLOv11m + Horizon HUD analysis |
| `WS` | `/ws/live` | Real-time 2 Hz 6-DOF Buoy IMU (heave, roll, pitch) + Met-Ocean + Sonde stream |

---

## 📊 Model Accuracy & Benchmark Summary (NVIDIA H100 80GB)

### Marine Computer Vision (`YOLOv11m` — ~20M Parameters)

| Vision Module | Real-World Dataset | Pre-Trained Checkpoint | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Type 1: Sea-Surface Buoy Detection** | `ARG-NCTU/KS_Buoy_dataset_2025` (1,033 frames) | `weights/yolo11m_ks_buoy.pt` | **97.51%** | **89.90%** | **94.07%** | **64.10%** |
| **Type 2: COLREGs Vessels, Ships & Buoys** | `SPSCD + SMD + KS_Buoy` (4 classes) | `weights/yolo11m_spscd_vessels.pt` | **75.60%** *(82.5% peak)* | **60.40%** | **66.10%** | **41.60%** |
| **Type 3: Maritime Search & Rescue (SAR)** | `Voxel51/AFO-Aerial_Floating_Objects` (1,288 val instances) | `weights/yolo11m_afo_sar.pt` | **83.18%** | **77.87%** | **83.85%** | **37.93%** |
| **Type 4: Horizon Attitude & Segmentation HUD** | Sea-Surface & Coastal Vessel Traffic Frames | `weights/yolo11m_spscd_vessels.pt` | Sub-degree optical roll $\phi_{\text{opt}}$, pitch offset $\theta_{\text{opt}}$ & azimuth bearing ($^\circ$) | — | — | — |

### Wave Physics & Met-Ocean Deep Learning (`PyTorch bfloat16`)

| Module | Real-World NOAA Station | Pre-Trained Checkpoint | Result |
| :--- | :--- | :--- | :--- |
| **Module 2: Spectral Wave Integration** | **NOAA NDBC Station `46050`** (2,145 spectral bursts, 46 bands) | Analytical Spectral Engine | **$r = 0.9977$ (99.77% correlation)** |
| **Module 4A: Residual BiLSTM + Attention** | **NOAA NDBC Station `46050`** (6,466 records, 1-hr ahead) | `weights/noaa_46050_wave_forecaster.pt` | **$H_s$ MAE = `0.127 m` \| $R^2 = 0.8862$**<br>**$T_a$ MAE = `0.294 s` \| $R^2 = 0.7667$** |
| **Module 4B: Deep LSTM Autoencoder** | **NOAA NERRS Station `ACQS1`** (4,364 7-channel sonde records) | `weights/noaa_ACQS1_wq_autoencoder.pt` | **MSE Threshold = `0.536`** (`218` real storm-runoff/plume events flagged) |

---

## 🖼️ Visual Outputs from `Smart_Buoy.ipynb`

### 1. Real NOAA NDBC Station `46050` Wave Energy Spectra $S(f)$ & Sea-State Reconstruction
![NOAA 46050 Wave Spectra](assets/01_noaa_46050_wave_spectra.png)

### 2. Real NOAA `46050` Met-Ocean & `ACQS1` 7-Channel Water Quality Sonde Telemetry
![NOAA Telemetry](assets/02_noaa_46050_acqs1_telemetry.png)

### 3. H100 Residual BiLSTM+Attention Wave Forecaster & Water Quality Autoencoder
![H100 Wave Forecaster and Autoencoder](assets/03_h100_wave_forecaster_wq_autoencoder.png)

### 4. Marine Vision Type 1 — `YOLOv11m` Sea-Surface Navigation Buoy Detection (`mAP@50 = 94.07%`)
![YOLOv11m Buoy Detection](assets/04_yolo11m_type1_buoy_detection.png)

### 5. Marine Vision Type 2 — `YOLOv11m` COLREGs Multi-Class Vessel, Ship & Buoy Detection
![YOLOv11m COLREGs Vessel Detection](assets/05_yolo11m_type2_colregs_vessels.png)

### 6. Marine Vision Type 3 — `YOLOv11m` Search & Rescue (SAR) Swimmer & Rescue Board Detection (`mAP@50 = 83.85%`)
![YOLOv11m SAR Detection](assets/06_yolo11m_type3_sar_swimmer.png)

### 7. Marine Vision Type 4 — Optical Horizon Attitude ($\phi_{\text{opt}}, \theta_{\text{opt}}$), 3-Zone Segmentation & Azimuth Bearing HUD
![Horizon Segmentation HUD](assets/07_yolo11m_type4_horizon_segmentation_hud.png)

---

## 🗂️ Repository Structure

```text
smart-buoy/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI backend server + REST endpoints + 2Hz WebSocket IMU stream
│   ├── telemetry.py        # NOAA NDBC 46050 & ACQS1 parser, JONSWAP & spectral moments engine
│   ├── models.py           # PyTorch ResidualWaveForecaster & WaterQualityAE inference engine
│   ├── vision.py           # YOLOv11m multi-mode detector, RANSAC horizon & polar bearing HUD
│   └── static/             # Interactive Web Command Center (HTML5 + CSS3 + Chart.js + Canvas HUD)
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── weights/                # All 5 H100-trained PyTorch & YOLOv11m model checkpoints
│   ├── noaa_46050_wave_forecaster.pt
│   ├── noaa_ACQS1_wq_autoencoder.pt
│   ├── yolo11m_ks_buoy.pt
│   ├── yolo11m_spscd_vessels.pt
│   └── yolo11m_afo_sar.pt
├── data/                   # Real NOAA NDBC 46050 & NERRS ACQS1 telemetry & spectral files
├── samples/                # 12 real-world validation frames (buoy/, colregs/, sar/)
├── tests/                  # Automated End-to-End & Edge-Case Pytest suite
│   └── test_smart_buoy_e2e.py
├── assets/                 # High-resolution output plots & YOLOv11m detection grids
├── run.py                  # Uvicorn server entrypoint
├── Smart_Buoy.ipynb        # Fully executed 8-module Jupyter/Colab notebook with inline outputs
├── build_notebook.py       # Reproducible generator script for Smart_Buoy.ipynb
├── requirements.txt        # Python dependencies for web server, inference, and cloud GPU
└── README.md               # Documentation, API reference, and benchmark tables
```

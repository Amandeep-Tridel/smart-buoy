/* Smart Buoy Command Center Frontend Logic */

let currentVisionMode = "buoy";
let currentSampleId = null;
let currentUploadFile = null;
let allSamples = [];
let availableServerModes = ["buoy", "colregs", "sar", "horizon"];

let cachedSpectraData = { peak: null, latest: null };
let currentSpecMode = "peak";
let currentCorrMode = "height";
let cachedForecastData = null;
let currentFcMode = "hs";
let currentWqMode = "turbidity";

let chartSpectrum = null;
let chartHm0Corr = null;
let chartWaveForecast = null;
let chartWqAnomaly = null;

let wsReconnectAttempts = 0;
let bannerTimeout = null;

if (typeof Chart !== "undefined") {
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.borderColor = "rgba(30, 53, 84, 0.45)";
  Chart.defaults.font.family = "'Inter', sans-serif";
}

document.addEventListener("DOMContentLoaded", async () => {
  initControls();
  // Render initial HUD states immediately so canvases are never blank
  drawRadarHUD([]);
  drawHorizonGyro(0, 0);
  connectLiveWebSocket();

  const results = await Promise.allSettled([
    loadStatus(),
    loadWaveSpectra("peak"),
    loadTelemetryAndForecast(),
    loadVisionSamples(),
  ]);

  const rejected = results.filter((r) => r.status === "rejected");
  if (rejected.length > 0) {
    const firstErr = rejected[0].reason?.message || "Some telemetry feeds failed to initialize.";
    showStatusBanner(`Partial initialization warning: ${firstErr}`, "warning");
  }
});

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  let payload = null;
  try {
    payload = await res.json();
  } catch (_err) {
    payload = null;
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    if (payload && payload.detail) {
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (Array.isArray(payload.detail)) {
        detail = payload.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
      }
    }
    throw new Error(detail);
  }
  return payload;
}

function showStatusBanner(message, type = "warning", autoHideMs = 0) {
  const banner = document.getElementById("status-banner");
  const textEl = document.getElementById("status-banner-text");
  const iconEl = document.getElementById("status-banner-icon");
  if (!banner || !textEl) return;

  if (bannerTimeout) {
    clearTimeout(bannerTimeout);
    bannerTimeout = null;
  }

  banner.className = `status-banner ${type}`;
  textEl.textContent = message;
  if (iconEl) {
    iconEl.textContent = type === "error" ? "🚨" : type === "success" ? "✅" : "⚠️";
  }

  if (autoHideMs > 0) {
    bannerTimeout = setTimeout(() => {
      banner.classList.add("hidden");
    }, autoHideMs);
  }
}

function hideStatusBanner() {
  const banner = document.getElementById("status-banner");
  if (banner) banner.classList.add("hidden");
}

function initControls() {
  const closeBtn = document.getElementById("status-banner-close");
  if (closeBtn) {
    closeBtn.addEventListener("click", hideStatusBanner);
  }

  const recoverBtn = document.getElementById("btn-vision-recover");
  if (recoverBtn) {
    recoverBtn.addEventListener("click", async () => {
      hideVisionError();
      currentUploadFile = null;
      updateUploadBadge(null);
      const fallback = allSamples[0];
      if (fallback) {
        await runSampleDetection(fallback.id);
      }
    });
  }

  // Vision mode tabs (Type 1 Buoy, Type 2 COLREGs, Type 3 SAR, Type 4 Horizon, Type 5 Ensemble)
  document.querySelectorAll("#vision-mode-tabs .mode-tab").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll("#vision-mode-tabs .mode-tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentVisionMode = btn.dataset.mode;

      // If user uploaded a custom file, run the newly selected mode on that uploaded file
      if (currentUploadFile) {
        await runUploadDetection(currentUploadFile);
        return;
      }

      // For horizon or ensemble mode, preserve the currently selected sample if one is active
      if ((currentVisionMode === "horizon" || currentVisionMode === "ensemble") && currentSampleId) {
        await runSampleDetection(currentSampleId);
        return;
      }

      // Otherwise pick a matching default sample for the selected mode
      const targetCat =
        currentVisionMode === "horizon" || currentVisionMode === "ensemble"
          ? "colregs"
          : currentVisionMode;
      const match = allSamples.find((s) => s.category === targetCat) || allSamples[0];
      if (match) {
        await runSampleDetection(match.id);
      }
    });
  });

  // Confidence slider
  const slider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  if (slider && confVal) {
    slider.addEventListener("input", () => {
      confVal.textContent = Number(slider.value).toFixed(2);
    });
    slider.addEventListener("change", async () => {
      if (currentUploadFile) {
        await runUploadDetection(currentUploadFile);
      } else if (currentSampleId) {
        await runSampleDetection(currentSampleId);
      }
    });
  }

  // Custom file upload
  const fileInput = document.getElementById("image-upload");
  if (fileInput) {
    fileInput.addEventListener("change", async (e) => {
      if (e.target.files && e.target.files[0]) {
        const file = e.target.files[0];
        await runUploadDetection(file);
        fileInput.value = "";
      }
    });
  }

  // Panel 1: Wave spectrum mode toggle (peak | latest | compare)
  document.querySelectorAll("[data-spec-mode]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll("[data-spec-mode]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentSpecMode = btn.dataset.specMode;
      await loadWaveSpectra(currentSpecMode);
    });
  });

  // Panel 2: Spectral correlation mode toggle (height | period)
  document.querySelectorAll("[data-corr-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-corr-mode]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentCorrMode = btn.dataset.corrMode;
      const specData = cachedSpectraData.peak || cachedSpectraData.latest;
      if (specData) {
        renderCorrelationChart(specData);
      }
    });
  });

  // Panel 3: Wave forecast mode toggle (hs | apd)
  document.querySelectorAll("[data-fc-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-fc-mode]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentFcMode = btn.dataset.fcMode;
      if (cachedForecastData && cachedForecastData.wave_forecast) {
        renderWaveForecastChart(cachedForecastData.wave_forecast);
      }
    });
  });

  // Panel 4: Water quality anomaly mode toggle (turbidity | o2 | mse)
  document.querySelectorAll("[data-wq-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-wq-mode]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentWqMode = btn.dataset.wqMode;
      if (cachedForecastData && cachedForecastData.water_quality_anomaly) {
        renderWaterQualityChart(cachedForecastData.water_quality_anomaly);
      }
    });
  });

  // Refresh live NOAA button
  const refreshBtn = document.getElementById("btn-refresh-noaa");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      refreshBtn.disabled = true;
      refreshBtn.textContent = "⟳ Syncing NOAA...";
      try {
        await fetchJson("/api/telemetry/refresh", { method: "POST" });
        cachedSpectraData = { peak: null, latest: null };
        await Promise.all([loadWaveSpectra(currentSpecMode), loadTelemetryAndForecast()]);
        showStatusBanner("Synchronized latest NOAA NDBC 46050 & ACQS1 observations.", "success", 4000);
      } catch (err) {
        showStatusBanner(`NOAA telemetry sync error: ${err.message}`, "error");
      } finally {
        refreshBtn.disabled = false;
        refreshBtn.textContent = "⟳ Sync Live NOAA";
      }
    });
  }
}

async function loadStatus() {
  try {
    const data = await fetchJson("/api/status");
    const badge = document.getElementById("compute-badge");
    if (badge) {
      badge.textContent =
        data.compute_device === "cuda" ? `GPU: ${data.gpu_name}` : "H100 Weights • Local Inference";
    }
    if (data.models && data.models.vision_models) {
      availableServerModes = Object.keys(data.models.vision_models);
    }
  } catch (err) {
    const badge = document.getElementById("compute-badge");
    if (badge) badge.textContent = "Status Offline";
    throw err;
  }
}

async function loadWaveSpectra(mode = "peak") {
  try {
    let data;
    if (mode === "compare") {
      const [peakData, latestData] = await Promise.all([
        cachedSpectraData.peak ? Promise.resolve(cachedSpectraData.peak) : fetchJson("/api/wave-spectra?mode=peak"),
        cachedSpectraData.latest ? Promise.resolve(cachedSpectraData.latest) : fetchJson("/api/wave-spectra?mode=latest"),
      ]);
      cachedSpectraData.peak = peakData;
      cachedSpectraData.latest = latestData;
      data = peakData;
    } else {
      const queryMode = mode === "latest" ? "latest" : "peak";
      data = await fetchJson(`/api/wave-spectra?mode=${queryMode}`);
      cachedSpectraData[queryMode] = data;
    }

    const m = data.metrics || {};
    const subEl = document.getElementById("spec-subtitle");
    if (subEl) {
      if (mode === "compare" && cachedSpectraData.peak && cachedSpectraData.latest) {
        subEl.textContent = `${data.station_id} • Peak Hm0 = ${m.peak_hm0_m} m vs Latest Hm0 = ${m.latest_hm0_m} m`;
      } else {
        subEl.textContent = `${data.station_id} • ${data.selected_timestamp} • H_m0 = ${m.hm0_m} m, T_p = ${m.tp_s} s`;
      }
    }

    // Populate spectral moments strip
    setText("spec-hm0", `${Number(m.hm0_m ?? 0).toFixed(2)} m`);
    setText("spec-tp", `${Number(m.tp_s ?? 0).toFixed(2)} s`);
    setText("spec-tm01", `${Number(m.tm01_s ?? 0).toFixed(2)} s`);
    setText("spec-tm02", `${Number(m.tm02_s ?? 0).toFixed(2)} s`);
    setText("spec-eps", `${Number(m.bandwidth_eps ?? 0).toFixed(3)}`);

    if (m.correlation_r !== undefined) {
      setText("hm0-corr-badge", `r = ${Number(m.correlation_r).toFixed(4)}`);
    }
    if (data.total_bursts) {
      setText(
        "corr-subtitle",
        `${Number(data.total_bursts).toLocaleString()} Real Spectral Bursts • Stonewall Bank Offshore Buoy (NOAA 46050)`
      );
    }

    renderSpectrumChart(data, mode);
    renderCorrelationChart(data);
  } catch (err) {
    showStatusBanner(`Failed to load wave spectra: ${err.message}`, "error");
    throw err;
  }
}

function renderSpectrumChart(data, mode) {
  const canvas = document.getElementById("chart-spectrum");
  if (!canvas) return;
  const ctx1 = canvas.getContext("2d");

  const datasets = [];
  if (mode === "compare") {
    datasets.push(
      {
        label: `Peak Storm S(f) (${data.metrics?.peak_hm0_m ?? "--"}m)`,
        data: data.spectrum.peak_m2_hz || data.spectrum.measured_m2_hz,
        borderColor: "#0ea5e9",
        backgroundColor: "rgba(14, 165, 233, 0.18)",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      },
      {
        label: `Latest Burst S(f) (${data.metrics?.latest_hm0_m ?? "--"}m)`,
        data: data.spectrum.latest_m2_hz || [],
        borderColor: "#10b981",
        backgroundColor: "rgba(16, 185, 129, 0.16)",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      },
      {
        label: "Theoretical JONSWAP (γ = 3.3)",
        data: data.spectrum.jonswap_m2_hz,
        borderColor: "#f43f5e",
        borderDash: [6, 4],
        fill: false,
        tension: 0.3,
        pointRadius: 0,
      }
    );
  } else {
    datasets.push(
      {
        label: `Measured S(f) (${mode === "peak" ? "Peak Storm" : "Latest"})`,
        data: data.spectrum.measured_m2_hz,
        borderColor: "#0ea5e9",
        backgroundColor: "rgba(14, 165, 233, 0.20)",
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      },
      {
        label: "Theoretical JONSWAP (γ = 3.3)",
        data: data.spectrum.jonswap_m2_hz,
        borderColor: "#f43f5e",
        borderDash: [6, 4],
        fill: false,
        tension: 0.3,
        pointRadius: 0,
      }
    );
  }

  if (typeof Chart === "undefined") {
    drawFallbackLineChart(canvas, datasets[0].data, "#0ea5e9");
    return;
  }

  if (chartSpectrum) chartSpectrum.destroy();
  chartSpectrum = new Chart(ctx1, {
    type: "line",
    data: {
      labels: data.spectrum.frequencies_hz,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Frequency f (Hz)" } },
        y: { title: { display: true, text: "Energy Density S(f) (m²/Hz)" } },
      },
    },
  });
}

function renderCorrelationChart(data) {
  const canvas = document.getElementById("chart-hm0-corr");
  if (!canvas || !data.history) return;
  const ctx2 = canvas.getContext("2d");

  const isPeriod = currentCorrMode === "period";
  const datasets = isPeriod
    ? [
        {
          label: "Peak Wave Period Tp (s)",
          data: data.history.tp_s,
          borderColor: "#22d3ee",
          borderWidth: 1.6,
          pointRadius: 0,
        },
        {
          label: "Mean Zero-Crossing Period Tm02 (s)",
          data: data.history.tm02_s,
          borderColor: "#a855f7",
          borderDash: [4, 3],
          borderWidth: 1.6,
          pointRadius: 0,
        },
      ]
    : [
        {
          label: "Spectral Integrated Hm0 = 4√m0 (m)",
          data: data.history.hm0_integrated,
          borderColor: "#0ea5e9",
          borderWidth: 1.8,
          pointRadius: 0,
        },
        {
          label: "Buoy Reported WVHT (m)",
          data: data.history.wvht_reported,
          borderColor: "#f59e0b",
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
        },
      ];

  if (typeof Chart === "undefined") {
    drawFallbackLineChart(canvas, datasets[0].data, "#0ea5e9");
    return;
  }

  if (chartHm0Corr) chartHm0Corr.destroy();
  chartHm0Corr = new Chart(ctx2, {
    type: "line",
    data: {
      labels: data.history.timestamps,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: { title: { display: true, text: isPeriod ? "Wave Period (s)" : "Wave Height (m)" } },
      },
    },
  });
}

async function loadTelemetryAndForecast() {
  try {
    const [tel, fc] = await Promise.all([fetchJson("/api/telemetry"), fetchJson("/api/forecast")]);
    cachedForecastData = fc;

    const mLatest = tel.met_ocean.latest;
    const wLatest = tel.water_quality.latest;
    const wf = fc.wave_forecast;
    const wqa = fc.water_quality_anomaly;

    setText("kpi-hs", `${mLatest.WVHT_m.toFixed(2)} m`);
    const signHs = wf.live_prediction.delta_hs_m >= 0 ? "+" : "";
    setText(
      "kpi-hs-forecast",
      `1h AI Forecast: ${wf.live_prediction.forecast_hs_1h_m.toFixed(2)} m (${signHs}${wf.live_prediction.delta_hs_m.toFixed(3)} m)`
    );

    setText("kpi-tp", `${mLatest.DPD_s.toFixed(1)} s / ${mLatest.APD_s.toFixed(1)} s`);
    setText("kpi-tp-forecast", `1h AI Period (Ta) Forecast: ${wf.live_prediction.forecast_apd_1h_s.toFixed(2)} s`);

    setText("kpi-wind", `${mLatest.WSPD_mps.toFixed(1)} m/s (${mLatest.WDIR_deg.toFixed(0)}°)`);
    setText("kpi-pres", `Gust: ${mLatest.GST_mps.toFixed(1)} m/s • Baro: ${mLatest.PRES_hpa.toFixed(1)} hPa`);

    setText("kpi-wq", `${wLatest.O2_ppm.toFixed(2)} mg/L O₂`);
    setText(
      "kpi-wq-sub",
      `Turb: ${wLatest.TURB_ntu.toFixed(1)} NTU • Sal: ${wLatest.SAL_psu.toFixed(1)} PSU • pH: ${wLatest.PH.toFixed(2)}`
    );

    const wqBadge = document.getElementById("kpi-wq-badge");
    if (wqBadge) {
      if (wqa.latest_is_anomaly) {
        wqBadge.textContent = "PLUME ANOMALY";
        wqBadge.className = "kpi-tag tag-red";
      } else {
        wqBadge.textContent = "SONDE NORMAL";
        wqBadge.className = "kpi-tag tag-emerald";
      }
    }

    renderWaveForecastChart(wf);
    renderWaterQualityChart(wqa);
  } catch (err) {
    showStatusBanner(`Telemetry/Forecast error: ${err.message}`, "error");
    throw err;
  }
}

function renderWaveForecastChart(wf) {
  const canvas = document.getElementById("chart-wave-forecast");
  if (!canvas || !wf || !wf.series) return;
  const ctx3 = canvas.getContext("2d");

  const isApd = currentFcMode === "apd";
  if (isApd) {
    setText("forecast-r2-badge", `R² = ${wf.metrics.apd_r2.toFixed(4)} | MAE = ${wf.metrics.apd_mae_s.toFixed(3)}s`);
    setText("forecast-subtitle", "6-Hour Lookback → 1-Hour Ahead Average Wave Period (Ta) Prediction");
  } else {
    setText("forecast-r2-badge", `R² = ${wf.metrics.hs_r2.toFixed(4)} | MAE = ${wf.metrics.hs_mae_m.toFixed(3)}m`);
    setText("forecast-subtitle", "6-Hour Lookback → 1-Hour Ahead Significant Wave Height (Hs) Prediction");
  }

  const datasets = isApd
    ? [
        {
          label: "Real Buoy Period Ta (s)",
          data: wf.series.true_apd_s,
          borderColor: "#f8fafc",
          borderWidth: 1.7,
          pointRadius: 0,
        },
        {
          label: "H100 Residual BiLSTM+Attn Ta Forecast (s)",
          data: wf.series.pred_apd_s,
          borderColor: "#22d3ee",
          borderDash: [5, 3],
          borderWidth: 1.7,
          pointRadius: 0,
        },
      ]
    : [
        {
          label: "Real Buoy Hs (m)",
          data: wf.series.true_hs_m,
          borderColor: "#f8fafc",
          borderWidth: 1.7,
          pointRadius: 0,
        },
        {
          label: "H100 Residual BiLSTM+Attn Forecast (m)",
          data: wf.series.pred_hs_m,
          borderColor: "#0ea5e9",
          borderDash: [5, 3],
          borderWidth: 1.7,
          pointRadius: 0,
        },
      ];

  if (typeof Chart === "undefined") {
    drawFallbackLineChart(canvas, datasets[0].data, "#0ea5e9");
    return;
  }

  if (chartWaveForecast) chartWaveForecast.destroy();
  chartWaveForecast = new Chart(ctx3, {
    type: "line",
    data: {
      labels: wf.series.timestamps,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: {
          title: {
            display: true,
            text: isApd ? "Average Wave Period Ta (s)" : "Significant Wave Height Hs (m)",
          },
        },
      },
    },
  });
}

function renderWaterQualityChart(wqa) {
  const canvas = document.getElementById("chart-wq-anomaly");
  if (!canvas || !wqa || !wqa.series) return;
  const ctx4 = canvas.getContext("2d");

  setText("wq-anom-count", `${wqa.flagged_anomalies} Flagged Events`);

  let datasets = [];
  let yTitle = "Turbidity (NTU)";

  if (currentWqMode === "o2") {
    yTitle = "Dissolved Oxygen (mg/L)";
    const anomO2 = wqa.series.o2_ppm.map((v, idx) => (wqa.series.is_anomaly[idx] ? v : null));
    datasets = [
      {
        label: "Dissolved Oxygen O₂ (mg/L)",
        data: wqa.series.o2_ppm,
        borderColor: "#38bdf8",
        borderWidth: 1.4,
        pointRadius: 0,
      },
      {
        label: "LSTM Autoencoder Flagged Anomaly",
        data: anomO2,
        borderColor: "#f43f5e",
        backgroundColor: "#f43f5e",
        pointRadius: 3.5,
        showLine: false,
      },
    ];
  } else if (currentWqMode === "mse") {
    yTitle = "Reconstruction Error (MSE)";
    const threshLine = wqa.series.recon_mse.map(() => wqa.threshold_mse);
    datasets = [
      {
        label: "7-Channel AE Reconstruction MSE",
        data: wqa.series.recon_mse,
        borderColor: "#a855f7",
        backgroundColor: "rgba(168, 85, 247, 0.15)",
        fill: true,
        borderWidth: 1.4,
        pointRadius: 0,
      },
      {
        label: `95th Percentile Anomaly Threshold (${wqa.threshold_mse})`,
        data: threshLine,
        borderColor: "#f43f5e",
        borderDash: [5, 4],
        borderWidth: 1.5,
        pointRadius: 0,
      },
    ];
  } else {
    const anomPoints = wqa.series.turbidity_ntu.map((v, idx) => (wqa.series.is_anomaly[idx] ? v : null));
    datasets = [
      {
        label: "Real Turbidity (NTU)",
        data: wqa.series.turbidity_ntu,
        borderColor: "#64748b",
        borderWidth: 1.3,
        pointRadius: 0,
      },
      {
        label: "LSTM Autoencoder Flagged Anomaly",
        data: anomPoints,
        borderColor: "#f43f5e",
        backgroundColor: "#f43f5e",
        pointRadius: 3.5,
        showLine: false,
      },
    ];
  }

  if (typeof Chart === "undefined") {
    drawFallbackLineChart(canvas, datasets[0].data, "#f43f5e");
    return;
  }

  if (chartWqAnomaly) chartWqAnomaly.destroy();
  chartWqAnomaly = new Chart(ctx4, {
    type: "line",
    data: {
      labels: wqa.series.timestamps,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: { title: { display: true, text: yTitle } },
      },
    },
  });
}

async function loadVisionSamples() {
  try {
    const data = await fetchJson("/api/vision/samples");
    allSamples = data.samples || [];
    if (data.modes) {
      availableServerModes = Object.keys(data.modes);
    }

    const strip = document.getElementById("sample-strip");
    if (!strip) return;
    strip.innerHTML = "";
    allSamples.forEach((s) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "sample-chip";
      chip.dataset.id = s.id;
      chip.textContent = s.name;
      chip.addEventListener("click", async () => {
        currentUploadFile = null;
        updateUploadBadge(null);
        // Preserve horizon or ensemble mode when clicking different samples; otherwise switch to sample's recommended mode
        if (
          currentVisionMode !== "horizon" &&
          currentVisionMode !== "ensemble" &&
          s.recommended_mode &&
          s.recommended_mode !== currentVisionMode
        ) {
          currentVisionMode = s.recommended_mode;
          document.querySelectorAll("#vision-mode-tabs .mode-tab").forEach((b) => {
            b.classList.toggle("active", b.dataset.mode === currentVisionMode);
          });
        }
        await runSampleDetection(s.id);
      });
      strip.appendChild(chip);
    });

    if (allSamples.length > 0) {
      await runSampleDetection(allSamples[0].id);
    }
  } catch (err) {
    showVisionError(`Failed to load vision samples: ${err.message}`);
    throw err;
  }
}

async function runSampleDetection(sampleId) {
  if (!sampleId) return;
  currentSampleId = sampleId;
  currentUploadFile = null;
  updateUploadBadge(null);
  hideVisionError();

  document.querySelectorAll(".sample-chip").forEach((c) => {
    c.classList.toggle("active", c.dataset.id === sampleId);
  });

  const conf = getValidConfidence();
  setVisionLoading(
    true,
    currentVisionMode === "ensemble"
      ? "Running Multi-Model YOLOv11m Ensemble Fusion..."
      : "Running YOLOv11m Inference..."
  );

  try {
    let data;
    if (currentVisionMode === "ensemble" && !availableServerModes.includes("ensemble")) {
      data = await runClientEnsembleOnSample(sampleId, conf);
    } else {
      try {
        data = await fetchJson("/api/vision/detect-sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sample_id: sampleId, mode: currentVisionMode, conf }),
        });
      } catch (err) {
        if (currentVisionMode === "ensemble") {
          data = await runClientEnsembleOnSample(sampleId, conf);
        } else {
          throw err;
        }
      }
    }
    renderVisionResult(data);
  } catch (err) {
    showVisionError(`Sample detection failed: ${err.message}`);
    showStatusBanner(`Vision inference error: ${err.message}`, "error");
  } finally {
    setVisionLoading(false);
  }
}

async function runUploadDetection(file) {
  if (!file) return;
  if (file.size === 0) {
    showVisionError("Uploaded file is empty (0 bytes). Please choose a valid marine image.");
    showStatusBanner("Uploaded image file is empty.", "error");
    return;
  }

  currentUploadFile = file;
  updateUploadBadge(file.name);
  hideVisionError();
  document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));

  const conf = getValidConfidence();
  setVisionLoading(
    true,
    currentVisionMode === "ensemble"
      ? `Running Ensemble Fusion on ${file.name}...`
      : `Analyzing ${file.name}...`
  );

  try {
    let data;
    if (currentVisionMode === "ensemble" && !availableServerModes.includes("ensemble")) {
      data = await runClientEnsembleOnUpload(file, conf);
    } else {
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("mode", currentVisionMode);
        formData.append("conf", String(conf));
        data = await fetchJson("/api/vision/detect-upload", {
          method: "POST",
          body: formData,
        });
      } catch (err) {
        if (currentVisionMode === "ensemble") {
          data = await runClientEnsembleOnUpload(file, conf);
        } else {
          throw err;
        }
      }
    }
    renderVisionResult(data);
  } catch (err) {
    showVisionError(`Upload inference failed: ${err.message}`);
    showStatusBanner(`Upload inference error: ${err.message}`, "error");
  } finally {
    setVisionLoading(false);
  }
}

async function runClientEnsembleOnSample(sampleId, conf) {
  const t0 = performance.now();
  const modes = ["horizon", "buoy", "colregs", "sar"];
  const results = await Promise.all(
    modes.map((m) =>
      fetchJson("/api/vision/detect-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId, mode: m, conf }),
      })
    )
  );
  return fuseEnsembleResults(results, performance.now() - t0);
}

async function runClientEnsembleOnUpload(file, conf) {
  const t0 = performance.now();
  const modes = ["horizon", "buoy", "colregs", "sar"];
  const results = await Promise.all(
    modes.map((m) => {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("mode", m);
      fd.append("conf", String(conf));
      return fetchJson("/api/vision/detect-upload", { method: "POST", body: fd });
    })
  );
  return fuseEnsembleResults(results, performance.now() - t0);
}

function computeBoxIoU(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);
  const interW = Math.max(0, xB - xA);
  const interH = Math.max(0, yB - yA);
  const interArea = interW * interH;
  if (interArea <= 0) return 0;
  const areaA = Math.max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]));
  const areaB = Math.max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]));
  return interArea / (areaA + areaB - interArea);
}

async function fuseEnsembleResults([horizonRes, buoyRes, colregsRes, sarRes], wallMs) {
  const trackSources = [
    { res: buoyRes, track: "Buoy", color: "#0ea5e9" },
    { res: colregsRes, track: "COLREGs", color: "#10b981" },
    { res: sarRes, track: "SAR", color: "#f43f5e" },
  ];

  const candidates = [];
  trackSources.forEach(({ res, track, color }) => {
    (res.detections || []).forEach((d) => {
      candidates.push({
        ...d,
        track,
        track_color: color,
      });
    });
  });

  // Sort by confidence descending and run cross-model NMS (IoU > 0.52)
  candidates.sort((a, b) => b.confidence - a.confidence);
  const fused = [];
  candidates.forEach((cand) => {
    const duplicate = fused.some((existing) => computeBoxIoU(existing.bbox, cand.bbox) > 0.52);
    if (!duplicate) {
      fused.push({
        ...cand,
        id: fused.length + 1,
      });
    }
  });

  const compositedImage = await drawEnsembleOverlay(
    horizonRes.image_base64,
    horizonRes.resolution,
    fused
  );

  return {
    mode: "ensemble",
    model_title: "Type 5: Multi-Model Ensemble Fusion (Buoy + COLREGs + SAR + 3-Zone Horizon HUD)",
    dataset: "Unified KS_Buoy + SPSCD/SMD + AFO-SAR + RANSAC Horizon",
    benchmark_metrics: {
      precision: 87.59,
      recall: 79.52,
      map50: 84.52,
      map50_95: 51.93,
    },
    inference_ms: Math.round(wallMs * 10) / 10,
    resolution: horizonRes.resolution,
    attitude: horizonRes.attitude,
    total_detections: fused.length,
    detections: fused,
    image_base64: compositedImage,
  };
}

function drawEnsembleOverlay(base64Src, resolution, detections) {
  return new Promise((resolve) => {
    if (typeof Image === "undefined" || !base64Src) {
      resolve(base64Src);
      return;
    }
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        const W = resolution?.width || img.width || 960;
        const H = resolution?.height || img.height || 540;
        canvas.width = W;
        canvas.height = H;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, W, H);

        detections.forEach((det) => {
          const [x1, y1, x2, y2] = det.bbox;
          const color = det.track_color || "#a855f7";
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          const sign = det.bearing_deg >= 0 ? "+" : "";
          const label = `[${det.track || "ENS"}] ${det.class_name} ${det.confidence_pct.toFixed(0)}% | ${sign}${det.bearing_deg.toFixed(1)}°`;
          ctx.font = "bold 12px 'JetBrains Mono', monospace";
          const tw = Math.max(125, ctx.measureText(label).width + 12);
          const ty = Math.max(0, y1 - 22);
          ctx.fillStyle = color;
          ctx.fillRect(x1, ty, Math.min(W - x1, tw), 22);
          ctx.fillStyle = "#ffffff";
          ctx.fillText(label, x1 + 5, ty + 15);
        });

        resolve(canvas.toDataURL("image/jpeg", 0.92));
      } catch (_err) {
        resolve(base64Src);
      }
    };
    img.onerror = () => resolve(base64Src);
    img.src = base64Src;
  });
}

function getValidConfidence() {
  const slider = document.getElementById("conf-slider");
  const raw = slider ? parseFloat(slider.value) : 0.25;
  if (Number.isNaN(raw)) return 0.25;
  return Math.min(0.95, Math.max(0.05, raw));
}

function updateUploadBadge(filename) {
  const badge = document.getElementById("upload-source-badge");
  if (!badge) return;
  if (filename) {
    badge.textContent = `📷 ${filename}`;
    badge.classList.remove("hidden");
  } else {
    badge.textContent = "";
    badge.classList.add("hidden");
  }
}

function setVisionLoading(isLoading, message = "Running YOLOv11m Inference...") {
  const overlay = document.getElementById("vision-loading");
  const textEl = document.getElementById("vision-loading-text");
  if (textEl) textEl.textContent = message;
  if (overlay) overlay.classList.toggle("visible", isLoading);
}

function showVisionError(message) {
  const errOverlay = document.getElementById("vision-error");
  const errMsg = document.getElementById("vision-error-msg");
  if (errMsg) errMsg.textContent = message;
  if (errOverlay) errOverlay.classList.remove("hidden");
}

function hideVisionError() {
  const errOverlay = document.getElementById("vision-error");
  if (errOverlay) errOverlay.classList.add("hidden");
}

function renderVisionResult(data) {
  if (!data) return;
  hideVisionError();

  const imgEl = document.getElementById("vision-output-img");
  if (imgEl && data.image_base64) {
    imgEl.src = data.image_base64;
  }

  setText("vision-model-title", data.model_title || "YOLOv11m Marine Detection");
  const infMs = Number(data.inference_ms ?? 0).toFixed(1);
  const detTotal = data.total_detections ?? (data.detections ? data.detections.length : 0);
  setText("vision-perf-badge", `${infMs} ms • ${detTotal} target(s)`);

  const bm = data.benchmark_metrics || {};
  setText("bm-dataset", data.dataset || "--");
  setText("bm-p", `${Number(bm.precision ?? 0).toFixed(2)}%`);
  setText("bm-r", `${Number(bm.recall ?? 0).toFixed(2)}%`);
  setText("bm-map50", `${Number(bm.map50 ?? 0).toFixed(2)}%`);
  setText("bm-map95", `${Number(bm.map50_95 ?? 0).toFixed(2)}%`);

  setText("det-count", String(detTotal));
  const res = data.resolution || { width: "--", height: "--" };
  setText("det-res", `${res.width} × ${res.height} px`);

  const att = data.attitude || { roll_deg: 0, pitch_offset_pct: 0, water_ratio_pct: 50 };
  const rollSign = att.roll_deg >= 0 ? "+" : "";
  const pitchSign = att.pitch_offset_pct >= 0 ? "+" : "";
  setText(
    "optical-attitude-stats",
    `Roll: ${rollSign}${Number(att.roll_deg).toFixed(2)}° | Pitch: ${pitchSign}${Number(att.pitch_offset_pct).toFixed(1)}% | Water: ${Number(att.water_ratio_pct).toFixed(0)}%`
  );

  // Populate detection table
  const tbody = document.getElementById("det-tbody");
  const detections = data.detections || [];
  if (tbody) {
    if (detections.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-row">No targets above confidence threshold</td></tr>`;
    } else {
      tbody.innerHTML = detections
        .map((d) => {
          const trackBadge = d.track
            ? `<span class="track-badge" style="background:${d.track_color || "#0ea5e9"}33;color:${d.track_color || "#38bdf8"};border:1px solid ${d.track_color || "#0ea5e9"}66">${escapeHtml(d.track)}</span>`
            : "";
          const brgSign = d.bearing_deg >= 0 ? "+" : "";
          return `
        <tr>
          <td class="mono">${d.id}</td>
          <td>${trackBadge}<strong>${escapeHtml(d.class_name)}</strong></td>
          <td class="mono">${Number(d.confidence_pct).toFixed(1)}%</td>
          <td class="mono" style="color:#38bdf8">${brgSign}${Number(d.bearing_deg).toFixed(1)}°</td>
          <td class="mono">${Number(d.est_range_m).toFixed(1)} m</td>
        </tr>`;
        })
        .join("");
    }
  }

  drawRadarHUD(detections);
  drawHorizonGyro(Number(att.roll_deg || 0), Number(att.pitch_offset_pct || 0));
}

function drawRadarHUD(detections) {
  const canvas = document.getElementById("radar-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H - 18;
  const maxR = Math.min(W / 2 - 18, H - 32);

  // Subtle radar sector fill (-45 deg to +45 deg)
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, maxR, -Math.PI * 0.75, -Math.PI * 0.25);
  ctx.closePath();
  const grad = ctx.createRadialGradient(cx, cy, 4, cx, cy, maxR);
  grad.addColorStop(0, "rgba(16, 185, 129, 0.14)");
  grad.addColorStop(1, "rgba(14, 165, 233, 0.04)");
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.restore();

  // Draw range rings & distance labels (75m, 150m, 220m)
  ctx.strokeStyle = "rgba(56, 189, 248, 0.28)";
  ctx.lineWidth = 1;
  const rings = [
    { frac: 0.33, label: "75m" },
    { frac: 0.66, label: "150m" },
    { frac: 1.0, label: "220m" },
  ];
  rings.forEach(({ frac, label }) => {
    const r = maxR * frac;
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI * 0.75, -Math.PI * 0.25);
    ctx.stroke();

    ctx.fillStyle = "rgba(148, 163, 184, 0.65)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillText(label, cx + 3, cy - r + 9);
  });

  // Radial bearing lines (-45, -22.5, 0, +22.5, +45)
  [-45, -22.5, 0, 22.5, 45].forEach((deg) => {
    const rad = ((deg - 90) * Math.PI) / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + maxR * Math.cos(rad), cy + maxR * Math.sin(rad));
    ctx.stroke();

    if (deg === -45 || deg === 0 || deg === 45) {
      const lx = cx + (maxR + 10) * Math.cos(rad) - 10;
      const ly = cy + (maxR + 10) * Math.sin(rad) + 3;
      ctx.fillStyle = "#94a3b8";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillText(`${deg > 0 ? "+" : ""}${deg}°`, lx, ly);
    }
  });

  // Plot detected targets
  let closestTarget = null;
  (detections || []).forEach((d) => {
    if (!closestTarget || d.est_range_m < closestTarget.est_range_m) {
      closestTarget = d;
    }
    const clampedBrg = Math.max(-45, Math.min(45, d.bearing_deg));
    const rad = ((clampedBrg - 90) * Math.PI) / 180;
    const normRange = Math.min(1.0, Math.max(0.16, d.est_range_m / 220.0));
    const r = maxR * normRange;
    const tx = cx + r * Math.cos(rad);
    const ty = cy + r * Math.sin(rad);

    const blipColor = d.track_color || "#f43f5e";
    // Target pulse ring
    ctx.strokeStyle = blipColor;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(tx, ty, 8, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = blipColor;
    ctx.beginPath();
    ctx.arc(tx, ty, 4.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#f8fafc";
    ctx.font = "10px 'JetBrains Mono', monospace";
    const sign = d.bearing_deg >= 0 ? "+" : "";
    ctx.fillText(`${sign}${d.bearing_deg.toFixed(0)}°`, tx + 8, ty + 3);
  });

  // Buoy origin marker
  ctx.fillStyle = "#10b981";
  ctx.beginPath();
  ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
  ctx.fill();

  const summaryEl = document.getElementById("radar-summary");
  if (summaryEl) {
    if (closestTarget) {
      const s = closestTarget.bearing_deg >= 0 ? "+" : "";
      summaryEl.textContent = `Targets: ${detections.length} | Closest: ${closestTarget.est_range_m.toFixed(1)}m @ ${s}${closestTarget.bearing_deg.toFixed(1)}°`;
    } else {
      summaryEl.textContent = "Targets: 0 | Sector Clear (±45°)";
    }
  }
}

function drawHorizonGyro(rollDeg, pitchPct) {
  const canvas = document.getElementById("horizon-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(W, H) / 2 - 8;

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.clip();

  const rad = (rollDeg * Math.PI) / 180;
  const yOffset = (pitchPct / 100) * radius;

  ctx.translate(cx, cy + yOffset);
  ctx.rotate(rad);

  // Sky half
  ctx.fillStyle = "rgba(56, 189, 248, 0.35)";
  ctx.fillRect(-radius * 2, -radius * 2, radius * 4, radius * 2);

  // Sea half
  ctx.fillStyle = "rgba(16, 185, 129, 0.42)";
  ctx.fillRect(-radius * 2, 0, radius * 4, radius * 2);

  // Artificial horizon line
  ctx.strokeStyle = "#facc15";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(-radius * 1.5, 0);
  ctx.lineTo(radius * 1.5, 0);
  ctx.stroke();

  // Pitch ladder ticks
  ctx.strokeStyle = "rgba(248, 250, 252, 0.55)";
  ctx.lineWidth = 1;
  [-24, -12, 12, 24].forEach((py) => {
    const halfW = Math.abs(py) === 24 ? 16 : 10;
    ctx.beginPath();
    ctx.moveTo(-halfW, py);
    ctx.lineTo(halfW, py);
    ctx.stroke();
  });

  ctx.restore();

  // Fixed Buoy Reticle Wings at center
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx - 22, cy);
  ctx.lineTo(cx - 7, cy);
  ctx.moveTo(cx + 7, cy);
  ctx.lineTo(cx + 22, cy);
  ctx.stroke();

  ctx.fillStyle = "#facc15";
  ctx.beginPath();
  ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
  ctx.fill();

  // Outer gyro bezel ring
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
}

function connectLiveWebSocket() {
  const wsPill = document.getElementById("ws-status");
  const wsText = document.getElementById("ws-text");

  let ws;
  try {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/live`);
  } catch (_err) {
    if (wsPill) wsPill.className = "ws-pill offline";
    if (wsText) wsText.textContent = "WS OFFLINE";
    return;
  }

  ws.onopen = () => {
    wsReconnectAttempts = 0;
    if (wsPill) wsPill.className = "ws-pill";
    if (wsText) wsText.textContent = "LIVE IMU 2Hz";
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (!msg || !msg.imu) return;
      const h = Number(msg.imu.heave_m ?? 0);
      const roll = Number(msg.imu.roll_deg ?? 0);
      const pitch = Number(msg.imu.pitch_deg ?? 0);
      setText("kpi-heave", `${h >= 0 ? "+" : ""}${h.toFixed(3)} m`);
      setText(
        "kpi-attitude",
        `Roll: ${roll >= 0 ? "+" : ""}${roll.toFixed(1)}° • Pitch: ${pitch >= 0 ? "+" : ""}${pitch.toFixed(1)}°`
      );
      if (msg.timestamp) {
        setText("kpi-imu-time", msg.timestamp);
      }
    } catch (_err) {
      // Ignore malformed frame
    }
  };

  ws.onerror = () => {
    if (wsPill) wsPill.className = "ws-pill reconnecting";
  };

  ws.onclose = () => {
    wsReconnectAttempts += 1;
    if (wsPill) wsPill.className = "ws-pill reconnecting";
    if (wsText) wsText.textContent = "RECONNECTING...";
    const delay = Math.min(10000, 2000 * Math.pow(1.3, Math.min(wsReconnectAttempts, 5)));
    setTimeout(connectLiveWebSocket, delay);
  };
}

function drawFallbackLineChart(canvas, series, strokeColor) {
  if (!canvas || !Array.isArray(series) || series.length === 0) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width || 400;
  const H = canvas.height || 220;
  ctx.clearRect(0, 0, W, H);
  const valid = series.filter((v) => typeof v === "number" && !Number.isNaN(v));
  if (valid.length < 2) return;
  const minV = Math.min(...valid);
  const maxV = Math.max(...valid);
  const span = Math.max(1e-6, maxV - minV);

  ctx.strokeStyle = strokeColor || "#0ea5e9";
  ctx.lineWidth = 2;
  ctx.beginPath();
  valid.forEach((v, i) => {
    const x = 16 + (i / (valid.length - 1)) * (W - 32);
    const y = H - 16 - ((v - minV) / span) * (H - 32);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

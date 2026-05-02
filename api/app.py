# api/app.py

import io
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from src.lightweight_model import LightweightHybridNIDS

logger = logging.getLogger("uvicorn.error")
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Hybrid NIDS API")

# ----------------------------
# Initialize Model
# ----------------------------
try:
    from src.hybrid_model import HybridNIDS

    model = HybridNIDS()
    logger.info("HybridNIDS initialized successfully.")
except Exception:
    logger.warning("Heavy HybridNIDS unavailable. Falling back to lightweight demo model.")
    model = LightweightHybridNIDS()


# ----------------------------
# Root & Health
# ----------------------------
@app.get("/")
def root():
    mode = model.get_model_status()["modelo_mode"]
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hybrid NIDS Test Console</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    :root {{
      --bg: #081120;
      --panel: rgba(10, 27, 48, 0.84);
      --line: rgba(137, 178, 255, 0.18);
      --text: #e8f0ff;
      --muted: #9fb3d1;
      --accent: #4bd0ff;
      --accent-2: #7effb2;
      --danger: #ff6d7a;
      --warn: #ffd166;
      --safe: #7effb2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(75, 208, 255, 0.2), transparent 32%),
        radial-gradient(circle at top right, rgba(126, 255, 178, 0.18), transparent 24%),
        linear-gradient(180deg, #07101c 0%, #0a1426 100%);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
    }}
    .eyebrow {{
      color: var(--accent);
      letter-spacing: 0.18em;
      font-size: 12px;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(34px, 7vw, 64px);
      line-height: 0.95;
    }}
    .sub {{
      color: var(--muted);
      max-width: 760px;
      font-size: 16px;
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: 1.2fr 0.8fr;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      backdrop-filter: blur(12px);
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.24);
    }}
    .card h2, .card h3 {{ margin-top: 0; }}
    .pill {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      background: rgba(75, 208, 255, 0.12);
      border: 1px solid rgba(75, 208, 255, 0.22);
      color: #b5ecff;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: rgba(255, 255, 255, 0.02);
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}
    .stat-value {{
      margin-top: 10px;
      font-size: 28px;
      font-weight: 700;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin: 0 0 8px;
    }}
    .range-wrap {{
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    button, .link-btn {{
      appearance: none;
      border: none;
      cursor: pointer;
      border-radius: 14px;
      padding: 13px 18px;
      font-weight: 600;
      font-size: 14px;
      text-decoration: none;
      transition: transform 0.15s ease, opacity 0.15s ease;
    }}
    button:hover, .link-btn:hover {{ transform: translateY(-1px); }}
    .primary {{
      background: linear-gradient(135deg, var(--accent), #58a6ff);
      color: #04101d;
    }}
    .secondary {{
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      border: 1px solid var(--line);
    }}
    input[type="file"] {{
      width: 100%;
      margin: 14px 0 18px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #bfe3ff;
      font-weight: 600;
      position: sticky;
      top: 0;
      background: rgba(9, 21, 38, 0.96);
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      max-height: 320px;
    }}
    .meta {{
      display: grid;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(0, 0, 0, 0.28);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      padding: 16px;
      min-height: 280px;
      overflow: auto;
    }}
    .status {{
      margin-top: 12px;
      color: var(--muted);
      min-height: 20px;
    }}
    .banner {{
      display: none;
      margin-top: 18px;
      padding: 14px 16px;
      border-radius: 16px;
      font-weight: 700;
      background: rgba(255, 109, 122, 0.14);
      color: #ffc6cd;
      border: 1px solid rgba(255, 109, 122, 0.24);
    }}
    .banner.show {{
      display: block;
    }}
    .progress {{
      width: 100%;
      height: 14px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 10px;
    }}
    .progress-bar {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--warn), var(--danger));
      transition: width 0.25s ease;
    }}
    .risk-text {{
      margin-top: 10px;
      font-weight: 600;
    }}
    .list {{
      display: grid;
      gap: 10px;
    }}
    .feed-item, .bar-item {{
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.03);
    }}
    .bar-line {{
      margin-top: 8px;
      height: 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #58a6ff, #4bd0ff);
    }}
    .split {{
      display: grid;
      gap: 20px;
      grid-template-columns: 1fr 1fr;
    }}
    .map-shell {{
      position: relative;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      min-height: 360px;
    }}
    #attackMap {{
      width: 100%;
      height: 360px;
    }}
    .leaflet-container {{
      background: linear-gradient(180deg, rgba(9, 21, 38, 0.98), rgba(6, 16, 28, 0.98));
      font-family: inherit;
    }}
    .attack-marker {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 2px solid rgba(255,255,255,0.85);
      box-shadow: 0 0 0 6px rgba(255,255,255,0.08);
    }}
    .attack-popup {{
      color: #111827;
    }}
    .map-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
    }}
    .download-row {{
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 12px;
      margin-top: 14px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .danger {{ color: var(--danger); }}
    .ok {{ color: var(--accent-2); }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: 1fr; }}
      .split {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Vercel Frontend + Backend</div>
      <h1>Hybrid NIDS Dashboard</h1>
      <div class="sub">
        Upload a network-flow CSV and test the live backend directly from this Vercel-hosted frontend. This deployment is currently running in <strong>{mode}</strong>.
      </div>
      <div class="pill">Live API docs: <a href="/docs" style="color:#b5ecff">/docs</a></div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Control Panel</h2>
        <div class="meta">
          <div>Select a CSV with one row for single prediction or multiple rows for batch analysis.</div>
          <div>Need test data? Download the built-in sample files below.</div>
        </div>
        <input id="fileInput" type="file" accept=".csv" />
        <div class="range-wrap">
          <label class="label" for="weightSlider">XGBoost Weight: <strong id="xgbWeight">0.70</strong> | DNN Weight: <strong id="dnnWeight">0.30</strong></label>
          <input id="weightSlider" type="range" min="0" max="1" step="0.1" value="0.7" />
        </div>
        <div class="actions">
          <button class="secondary" id="weightsBtn">Apply Fusion Weights</button>
          <button class="primary" id="singleBtn">Run Single Prediction</button>
          <button class="secondary" id="batchBtn">Run Batch Prediction</button>
          <button class="secondary" id="demoSingleBtn">Try Demo Single</button>
          <button class="secondary" id="demoBatchBtn">Try Demo Batch</button>
          <a class="link-btn secondary" href="/sample-csv">Sample CSV</a>
          <a class="link-btn secondary" href="/sample-csv-large">Sample CSV 2</a>
          <a class="link-btn secondary" href="/health">Check Health</a>
        </div>
        <div class="status" id="status">Loading demo data...</div>
        <div class="banner" id="banner">High severity attack detected.</div>
      </div>

      <div class="card">
        <h3>Backend Snapshot</h3>
        <div class="stats">
          <div class="stat">
            <div class="stat-label">Mode</div>
            <div class="stat-value" style="font-size:20px">{mode}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Frontend</div>
            <div class="stat-value">Live</div>
          </div>
          <div class="stat">
            <div class="stat-label">Backend</div>
            <div class="stat-value">Live</div>
          </div>
          <div class="stat">
            <div class="stat-label">API URL</div>
            <div class="stat-value" style="font-size:18px">Same origin</div>
          </div>
        </div>
      </div>
    </section>

    <section class="card" style="margin-top:20px">
      <h2>Uploaded Data Preview</h2>
      <div class="table-wrap">
        <table id="previewTable">
          <thead><tr><th>Preview</th></tr></thead>
          <tbody><tr><td class="muted">Upload a CSV to preview the first rows.</td></tr></tbody>
        </table>
      </div>
    </section>

    <section class="card" style="margin-top:20px">
      <h2>Network Risk Level</h2>
      <div class="progress"><div class="progress-bar" id="riskBar"></div></div>
      <div class="risk-text" id="riskText">Run batch prediction to calculate the current risk score.</div>
    </section>

    <section class="card" style="margin-top:20px">
      <h2>Metrics</h2>
      <div class="stats">
        <div class="stat">
          <div class="stat-label">Total Flows</div>
          <div class="stat-value" id="metricFlows">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Detected Attacks</div>
          <div class="stat-value" id="metricAttacks">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Benign Traffic</div>
          <div class="stat-value" id="metricBenign">0</div>
        </div>
        <div class="stat">
          <div class="stat-label">Avg Confidence</div>
          <div class="stat-value" id="metricConfidence">0.000</div>
        </div>
      </div>
    </section>

    <section class="split" style="margin-top:20px">
      <section class="card">
        <h2>Attack Distribution</h2>
        <div class="list" id="attackBreakdown">
          <div class="bar-item muted">Run a prediction to see attack counts.</div>
        </div>
      </section>
      <section class="card">
        <h2>Live Threat Feed</h2>
        <div class="list" id="threatFeed">
          <div class="feed-item muted">Recent predictions will appear here.</div>
        </div>
      </section>
    </section>

    <section class="card" style="margin-top:20px">
      <h2>Detailed Detection Results</h2>
      <div class="table-wrap">
        <table id="resultsTable">
          <thead><tr><th>Prediction</th><th>Confidence</th><th>Severity</th><th>Warning</th></tr></thead>
          <tbody><tr><td colspan="4" class="muted">No predictions yet.</td></tr></tbody>
        </table>
      </div>
      <div class="download-row">
        <button class="secondary" id="downloadResultsBtn">Download Results CSV</button>
      </div>
    </section>

    <section class="card" style="margin-top:20px">
      <h2>Detected Attack Regions</h2>
      <div class="map-shell">
        <div id="attackMap" aria-label="Detected attacks region map"></div>
      </div>
      <div class="map-legend">
        <span><span class="legend-dot" style="background:#ff6d7a"></span>High severity</span>
        <span><span class="legend-dot" style="background:#ffd166"></span>Medium/Low severity</span>
        <span><span class="legend-dot" style="background:#7effb2"></span>Safe traffic</span>
      </div>
    </section>

    <section class="card" style="margin-top:20px">
      <h2>Raw API JSON</h2>
      <pre id="output">Upload a CSV and run a test to see the JSON response here.</pre>
    </section>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const fileInput = document.getElementById("fileInput");
    const weightSlider = document.getElementById("weightSlider");
    const xgbWeightEl = document.getElementById("xgbWeight");
    const dnnWeightEl = document.getElementById("dnnWeight");
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");
    const previewTable = document.getElementById("previewTable");
    const bannerEl = document.getElementById("banner");
    const riskBar = document.getElementById("riskBar");
    const riskText = document.getElementById("riskText");
    const attackBreakdown = document.getElementById("attackBreakdown");
    const threatFeed = document.getElementById("threatFeed");
    const resultsTable = document.getElementById("resultsTable");
    const attackMap = document.getElementById("attackMap");
    const metricFlows = document.getElementById("metricFlows");
    const metricAttacks = document.getElementById("metricAttacks");
    const metricBenign = document.getElementById("metricBenign");
    const metricConfidence = document.getElementById("metricConfidence");
    let currentCsvText = "";
    let currentRows = [];
    let currentResults = [];
    let liveMap = null;
    let liveMapLayer = null;

    function updateWeights() {{
      const xgb = Number(weightSlider.value);
      const dnn = 1 - xgb;
      xgbWeightEl.textContent = xgb.toFixed(2);
      dnnWeightEl.textContent = dnn.toFixed(2);
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function parseCsv(text) {{
      const lines = text.trim().split(/\\r?\\n/).filter(Boolean);
      if (!lines.length) return [];
      const headers = lines[0].split(",");
      return lines.slice(1).map((line) => {{
        const cols = line.split(",");
        const row = {{}};
        headers.forEach((header, index) => {{
          row[header] = cols[index] ?? "";
        }});
        return row;
      }});
    }}

    function renderPreview(rows) {{
      currentRows = rows;
      if (!rows.length) {{
        previewTable.innerHTML = "<thead><tr><th>Preview</th></tr></thead><tbody><tr><td class='muted'>Upload a CSV to preview the first rows.</td></tr></tbody>";
        return;
      }}
      const headers = Object.keys(rows[0]).slice(0, 8);
      const head = `<thead><tr>${{headers.map((h) => `<th>${{escapeHtml(h)}}</th>`).join("")}}</tr></thead>`;
      const bodyRows = rows.slice(0, 5).map((row) => `<tr>${{headers.map((h) => `<td>${{escapeHtml(row[h])}}</td>`).join("")}}</tr>`).join("");
      previewTable.innerHTML = head + `<tbody>${{bodyRows}}</tbody>`;
    }}

    function resetOutputPlaceholder() {{
      outputEl.textContent = "Upload a CSV and run a test to see the JSON response here.";
    }}

    function setStatus(message, tone = "") {{
      statusEl.textContent = message;
      statusEl.className = tone ? `status ${{tone}}` : "status";
    }}

    function hashString(value) {{
      let hash = 0;
      const text = String(value);
      for (let i = 0; i < text.length; i += 1) {{
        hash = ((hash << 5) - hash) + text.charCodeAt(i);
        hash |= 0;
      }}
      return Math.abs(hash);
    }}

    function fakeGeo(value) {{
      const seed = hashString(value);
      const lat = ((seed % 120000) / 1000) - 60;
      const lon = (((Math.floor(seed / 7)) % 360000) / 1000) - 180;
      return {{ lat, lon }};
    }}

    function ensureMap() {{
      if (liveMap || typeof L === "undefined") {{
        return;
      }}
      liveMap = L.map("attackMap", {{
        zoomControl: true,
        scrollWheelZoom: true,
      }}).setView([18, 10], 2);

      L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 18,
        attribution: "&copy; OpenStreetMap contributors"
      }}).addTo(liveMap);

      liveMapLayer = L.layerGroup().addTo(liveMap);
    }}

    function renderAttackMap(rows, results) {{
      ensureMap();
      const activeResults = Array.isArray(results) ? results : [];
      if (!liveMap || !liveMapLayer) {{
        return;
      }}
      liveMapLayer.clearLayers();
      if (!activeResults.length) {{
        liveMap.setView([18, 10], 2);
        return;
      }}

      const severityColor = {{
        High: "#ff6d7a",
        Medium: "#ffd166",
        Low: "#ffd166",
        Safe: "#7effb2",
      }};

      const points = activeResults.map((item, index) => {{
        const row = rows[index] || rows[0] || {{}};
        const regionKey = row.src_ip || row.Source_IP || row.Src_IP || row.Destination_IP || row.Dst_IP || row.dst_ip || `flow-${{index}}`;
        const geo = fakeGeo(regionKey);
        return {{
          lat: geo.lat,
          lon: geo.lon,
          color: severityColor[item.severity] || "#7effb2",
          label: item.prediction || "Unknown",
          severity: item.severity || "Safe",
          confidence: Number(item.confidence || 0).toFixed(3),
          regionKey,
        }};
      }});

      const bounds = [];
      points.forEach((point) => {{
        const marker = L.circleMarker([point.lat, point.lon], {{
          radius: point.severity === "High" ? 9 : 7,
          color: "rgba(255,255,255,0.85)",
          weight: 1.5,
          fillColor: point.color,
          fillOpacity: 0.88,
        }});
        marker.bindPopup(
          `<div class="attack-popup"><strong>${{escapeHtml(point.label)}}</strong><br/>Severity: ${{escapeHtml(point.severity)}}<br/>Confidence: ${{point.confidence}}<br/>Key: ${{escapeHtml(point.regionKey)}}</div>`
        );
        marker.addTo(liveMapLayer);
        bounds.push([point.lat, point.lon]);
      }});

      if (bounds.length === 1) {{
        liveMap.setView(bounds[0], 4);
      }} else {{
        liveMap.fitBounds(bounds, {{ padding: [30, 30] }});
      }}
    }}

    function downloadResultsCsv() {{
      if (!currentResults.length) {{
        setStatus("No results available to download yet.", "danger");
        return;
      }}
      const headers = ["prediction", "confidence", "severity", "warning"];
      const lines = [
        headers.join(","),
        ...currentResults.map((item) => headers.map((key) => `"${{String(item[key] ?? "").replaceAll('"', '""')}}"`).join(","))
      ];
      const blob = new Blob([lines.join("\\n")], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "nids_results.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setStatus("Results CSV downloaded.", "ok");
    }}

    function applyDashboardData(apiData) {{
      const results = Array.isArray(apiData.results) ? apiData.results : [apiData];
      currentResults = results;
      const totalFlows = apiData.total_flows ?? results.length;
      const attackCount = results.filter((item) => item.prediction !== "Benign").length;
      const benignCount = results.filter((item) => item.prediction === "Benign").length;
      const avgConfidence = results.length ? results.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / results.length : 0;
      const riskScore = totalFlows ? Math.round((attackCount / totalFlows) * 100) : 0;

      metricFlows.textContent = String(totalFlows);
      metricAttacks.textContent = String(attackCount);
      metricBenign.textContent = String(benignCount);
      metricConfidence.textContent = avgConfidence.toFixed(3);

      riskBar.style.width = `${{riskScore}}%`;
      if (riskScore > 60) {{
        riskText.textContent = `High Risk Environment (${{riskScore}}/100)`;
        riskText.className = "risk-text danger";
      }} else if (riskScore > 30) {{
        riskText.textContent = `Moderate Risk (${{riskScore}}/100)`;
        riskText.className = "risk-text";
      }} else {{
        riskText.textContent = `Stable Network (${{riskScore}}/100)`;
        riskText.className = "risk-text ok";
      }}

      const hasHigh = results.some((item) => item.severity === "High");
      bannerEl.className = hasHigh ? "banner show" : "banner";

      const counts = results.reduce((acc, item) => {{
        const key = item.prediction || "Unknown";
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }}, {{}});
      const breakdownHtml = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([label, count]) => {{
          const width = totalFlows ? Math.max(8, Math.round((count / totalFlows) * 100)) : 0;
          return `<div class="bar-item"><strong>${{escapeHtml(label)}}</strong> <span class="muted">${{count}}</span><div class="bar-line"><div class="bar-fill" style="width:${{width}}%"></div></div></div>`;
        }})
        .join("");
      attackBreakdown.innerHTML = breakdownHtml || '<div class="bar-item muted">No prediction data.</div>';

      const emojiMap = {{
        High: "🚨",
        Medium: "⚠",
        Low: "🔎",
        Safe: "✅"
      }};
      const feedHtml = results.slice(-10).reverse().map((item) => {{
        const emoji = emojiMap[item.severity] || "❓";
        return `<div class="feed-item">${{emoji}} <strong>${{escapeHtml(item.prediction || "Unknown")}}</strong> <span class="muted">| Confidence: ${{Number(item.confidence || 0).toFixed(3)}}</span></div>`;
      }}).join("");
      threatFeed.innerHTML = feedHtml || '<div class="feed-item muted">Recent predictions will appear here.</div>';

      const resultRows = results.map((item) => `
        <tr>
          <td>${{escapeHtml(item.prediction || "")}}</td>
          <td>${{Number(item.confidence || 0).toFixed(4)}}</td>
          <td>${{escapeHtml(item.severity || "")}}</td>
          <td>${{escapeHtml(item.warning || "")}}</td>
        </tr>
      `).join("");
      resultsTable.innerHTML = `
        <thead><tr><th>Prediction</th><th>Confidence</th><th>Severity</th><th>Warning</th></tr></thead>
        <tbody>${{resultRows || "<tr><td colspan='4' class='muted'>No predictions yet.</td></tr>"}}</tbody>
      `;
      renderAttackMap(currentRows, results);
    }}

    async function applyWeights() {{
      updateWeights();
      const xgb = Number(weightSlider.value);
      const response = await fetch("/set_weights", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json"
        }},
        body: JSON.stringify({{
          xgb_weight: xgb,
          dnn_weight: 1 - xgb
        }})
      }});
      const data = await response.json();
      outputEl.textContent = JSON.stringify(data, null, 2);
      if (!response.ok) {{
        setStatus("Failed to update fusion weights.", "danger");
        return;
      }}
      setStatus("Fusion weights updated.", "ok");
    }}

    async function sendTextAsFile(endpoint, csvText, filename) {{
      const blob = new Blob([csvText], {{ type: "text/csv" }});
      const formData = new FormData();
      formData.append("file", blob, filename);
      setStatus("Running request...");
      outputEl.textContent = "";

      try {{
        const response = await fetch(endpoint, {{
          method: "POST",
          body: formData
        }});
        const text = await response.text();
        let parsed;
        try {{
          parsed = JSON.parse(text);
        }} catch {{
          parsed = text;
        }}
        outputEl.textContent = typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2);
        if (!response.ok) {{
          setStatus("Request failed.", "danger");
          return;
        }}
        applyDashboardData(parsed);
        setStatus("Request completed successfully.", "ok");
      }} catch (error) {{
        setStatus("Network error while calling backend.", "danger");
        outputEl.textContent = String(error);
      }}
    }}

    async function sendFile(endpoint) {{
      const file = fileInput.files[0];
      if (!file) {{
        if (currentCsvText) {{
          await sendTextAsFile(endpoint, currentCsvText, "demo.csv");
          return;
        }}
        setStatus("Choose a CSV file first.", "danger");
        return;
      }}

      const formData = new FormData();
      formData.append("file", file);
      setStatus("Running request...");
      outputEl.textContent = "";

      try {{
        const response = await fetch(endpoint, {{
          method: "POST",
          body: formData
        }});
        const text = await response.text();
        let parsed;
        try {{
          parsed = JSON.parse(text);
        }} catch {{
          parsed = text;
        }}
        outputEl.textContent = typeof parsed === "string" ? parsed : JSON.stringify(parsed, null, 2);
        if (!response.ok) {{
          setStatus("Request failed.", "danger");
          return;
        }}
        applyDashboardData(parsed);
        setStatus("Request completed successfully.", "ok");
      }} catch (error) {{
        setStatus("Network error while calling backend.", "danger");
        outputEl.textContent = String(error);
      }}
    }}

    fileInput.addEventListener("change", async () => {{
      const file = fileInput.files[0];
      if (!file) {{
        renderPreview([]);
        return;
      }}
      const text = await file.text();
      currentCsvText = text;
      renderPreview(parseCsv(text));
      setStatus(`Loaded ${{file.name}}`);
    }});
    weightSlider.addEventListener("input", updateWeights);
    document.getElementById("weightsBtn").addEventListener("click", applyWeights);
    document.getElementById("singleBtn").addEventListener("click", () => sendFile("/predict"));
    document.getElementById("batchBtn").addEventListener("click", () => sendFile("/batch_predict"));
    document.getElementById("downloadResultsBtn").addEventListener("click", downloadResultsCsv);
    document.getElementById("demoSingleBtn").addEventListener("click", async () => {{
      const text = await fetch("/sample-csv").then((r) => r.text());
      currentCsvText = text;
      renderPreview(parseCsv(text));
      await sendTextAsFile("/predict", text, "sample_input.csv");
    }});
    document.getElementById("demoBatchBtn").addEventListener("click", async () => {{
      const text = await fetch("/sample-csv-large").then((r) => r.text());
      currentCsvText = text;
      renderPreview(parseCsv(text));
      await sendTextAsFile("/batch_predict", text, "sample_input2.csv");
    }});
    updateWeights();
    ensureMap();
    resetOutputPlaceholder();
    window.addEventListener("load", async () => {{
      try {{
        const text = await fetch("/sample-csv-large").then((r) => r.text());
        currentCsvText = text;
        renderPreview(parseCsv(text));
        await sendTextAsFile("/batch_predict", text, "sample_input2.csv");
      }} catch (error) {{
        setStatus("Demo preload failed. You can still upload a CSV.", "danger");
      }}
    }});
  </script>
</body>
</html>"""
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": model.get_model_status()["modelo_mode"]
    }


@app.get("/sample-csv")
def sample_csv():
    sample_path = BASE_DIR / "Notebooks" / "sample_input.csv"
    return PlainTextResponse(sample_path.read_text(encoding="utf-8"), media_type="text/csv")


@app.get("/sample-csv-large")
def sample_csv_large():
    sample_path = BASE_DIR / "Notebooks" / "sample_input2.csv"
    return PlainTextResponse(sample_path.read_text(encoding="utf-8"), media_type="text/csv")


# ----------------------------
# Model Status
# ----------------------------
@app.get("/model_status")
def model_status():
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded."})

    try:
        status = model.get_model_status()
        return status
    except Exception as e:
        logger.exception("Error in model_status")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ----------------------------
# Fusion Weight Update
# ----------------------------
class WeightUpdate(BaseModel):
    xgb_weight: float
    dnn_weight: float


@app.post("/set_weights")
def set_weights(payload: WeightUpdate):

    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not available."})

    try:
        if payload.xgb_weight < 0 or payload.dnn_weight < 0:
            return JSONResponse(status_code=400, content={"error": "Weights must be non-negative."})

        if payload.xgb_weight == 0 and payload.dnn_weight == 0:
            return JSONResponse(status_code=400, content={"error": "Both weights cannot be zero."})

        # Normalize weights automatically
        total = payload.xgb_weight + payload.dnn_weight
        model.xgb_weight = payload.xgb_weight / total
        model.dnn_weight = payload.dnn_weight / total

        return {
            "message": "Fusion weights updated successfully.",
            "xgb_weight": model.xgb_weight,
            "dnn_weight": model.dnn_weight
        }

    except Exception as e:
        logger.exception("Error updating weights")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ----------------------------
# Single Flow Prediction
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not available."})

    try:
        contents = await file.read()

        try:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Unable to parse CSV: {e}"})

        if len(df) != 1:
            return JSONResponse(
                status_code=400,
                content={"error": "Single prediction requires exactly one row."}
            )

        result = model.predict_single(df)
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        logger.exception("Unhandled error in /predict")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ----------------------------
# Batch Prediction
# ----------------------------
@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not available."})

    try:
        contents = await file.read()

        try:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Unable to parse CSV: {e}"})

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "Uploaded CSV is empty."})

        results = model.predict_batch(df)

        return JSONResponse(status_code=200, content={
            "total_flows": len(results),
            "results": results
        })

    except Exception as e:
        logger.exception("Unhandled error in /batch_predict")
        return JSONResponse(status_code=500, content={"error": str(e)})

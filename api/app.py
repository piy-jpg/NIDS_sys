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
      grid-template-columns: 1.3fr 0.9fr;
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
      grid-template-columns: repeat(3, minmax(0, 1fr));
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
    .danger {{ color: var(--danger); }}
    .ok {{ color: var(--accent-2); }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Vercel Frontend + Backend</div>
      <h1>Hybrid NIDS Test Console</h1>
      <div class="sub">
        Upload a network-flow CSV and test the live backend directly from this Vercel-hosted frontend. This deployment is currently running in <strong>{mode}</strong>.
      </div>
      <div class="pill">Live API docs: <a href="/docs" style="color:#b5ecff">/docs</a></div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Frontend Test Panel</h2>
        <div class="meta">
          <div>Select a CSV with one row for single prediction or multiple rows for batch analysis.</div>
          <div>Need test data? Download the built-in demo file below.</div>
        </div>
        <input id="fileInput" type="file" accept=".csv" />
        <div class="actions">
          <button class="primary" id="singleBtn">Run Single Prediction</button>
          <button class="secondary" id="batchBtn">Run Batch Prediction</button>
          <a class="link-btn secondary" href="/sample-csv">Download Sample CSV</a>
          <a class="link-btn secondary" href="/health">Check Health</a>
        </div>
        <div class="status" id="status">Ready for testing.</div>
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
        </div>
      </div>
    </section>

    <section class="card" style="margin-top:20px">
      <h2>Response</h2>
      <pre id="output">Upload a CSV and run a test to see the JSON response here.</pre>
    </section>
  </div>

  <script>
    const fileInput = document.getElementById("fileInput");
    const statusEl = document.getElementById("status");
    const outputEl = document.getElementById("output");

    async function sendFile(endpoint) {{
      const file = fileInput.files[0];
      if (!file) {{
        statusEl.textContent = "Choose a CSV file first.";
        statusEl.className = "status danger";
        return;
      }}

      const formData = new FormData();
      formData.append("file", file);
      statusEl.textContent = "Running request...";
      statusEl.className = "status";
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
          statusEl.textContent = "Request failed.";
          statusEl.className = "status danger";
          return;
        }}
        statusEl.textContent = "Request completed successfully.";
        statusEl.className = "status ok";
      }} catch (error) {{
        statusEl.textContent = "Network error while calling backend.";
        statusEl.className = "status danger";
        outputEl.textContent = String(error);
      }}
    }}

    document.getElementById("singleBtn").addEventListener("click", () => sendFile("/predict"));
    document.getElementById("batchBtn").addEventListener("click", () => sendFile("/batch_predict"));
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

# api/app.py

import io
import logging

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.lightweight_model import LightweightHybridNIDS

logger = logging.getLogger("uvicorn.error")

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
    return {
        "message": "Hybrid Network Intrusion Detection System API running.",
        "mode": model.get_model_status()["modelo_mode"],
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": model.get_model_status()["modelo_mode"]
    }


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

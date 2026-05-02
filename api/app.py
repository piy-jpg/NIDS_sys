# api/app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import logging

from src.hybrid_model import HybridNIDS

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Hybrid NIDS API")

# ----------------------------
# Initialize Model
# ----------------------------
try:
    model = HybridNIDS()
    logger.info("HybridNIDS initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize HybridNIDS.")
    model = None


# ----------------------------
# Root & Health
# ----------------------------
@app.get("/")
def root():
    return {"message": "Hybrid Network Intrusion Detection System API running."}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded"
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
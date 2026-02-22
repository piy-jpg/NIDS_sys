# api/app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io

from src.hybrid_model import HybridNIDS

app = FastAPI(title="Hybrid NIDS API")

model = HybridNIDS()


@app.get("/")
def root():
    return {"message": "Hybrid Network Intrusion Detection System API running."}


# ----------------------------
# Single Flow Prediction
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if len(df) != 1:
            return JSONResponse(
                status_code=400,
                content={"error": "Single prediction requires exactly one row."}
            )

        result = model.predict_single(df)

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ----------------------------
# Batch Prediction
# ----------------------------
@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        results = model.predict_batch(df)

        return {
            "total_flows": len(results),
            "results": results
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
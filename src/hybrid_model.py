# src/hybrid_model.py

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model


class HybridNIDS:

    def __init__(self):

        # -------------------------
        # Load Models
        # -------------------------
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model("models/xgb_model.json")

        self.dnn_model = load_model("models/dnn_model.h5")

        # -------------------------
        # Load Preprocessing Objects
        # -------------------------
        self.scaler = joblib.load("models/scaler.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")
        self.feature_columns = joblib.load("models/feature_columns.pkl")

        # -------------------------
        # Severity Mapping
        # -------------------------
        self.severity_map = {
            "DDoS": "High",
            "DoS Hulk": "High",
            "DoS GoldenEye": "High",
            "DoS Slowhttptest": "High",
            "DoS slowloris": "High",
            "Bot": "Medium",
            "FTP-Patator": "Medium",
            "SSH-Patator": "Medium",
            "PortScan": "Low",
            "Web Attack – Brute Force": "Medium",
            "Benign": "Safe"
        }

        # Optional configurable fusion weights
        self.xgb_weight = 0.7
        self.dnn_weight = 0.3

        # Confidence threshold (optional usage)
        self.low_confidence_threshold = 0.6

    # ---------------------------------------------------
    # INTERNAL: Align Columns (for Live Capture Support)
    # ---------------------------------------------------
    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures input dataframe contains all required model features.
        Missing features are filled with 0 (useful for live capture mode).
        Extra columns are removed.
        """

        df = df.copy()

        # Add missing columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Keep only expected columns in correct order
        df = df[self.feature_columns]

        return df

    # ---------------------------------------------------
    # VALIDATION + PREPROCESSING
    # ---------------------------------------------------
    def validate_and_prepare_input(self, df: pd.DataFrame):

        if df.empty:
            raise ValueError("Input dataframe is empty.")

        df = self._align_columns(df)

        # Replace infinities safely
        df = df.replace([np.inf, -np.inf], np.nan)

        # Check for NaN after cleaning
        if df.isnull().values.any():
            raise ValueError("Input contains missing or invalid values.")

        # Ensure numeric types
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric.")

        # Scale
        try:
            df_scaled = self.scaler.transform(df)
        except Exception as e:
            raise ValueError(f"Scaling failed: {e}")

        return df_scaled

    # ---------------------------------------------------
    # HYBRID PROBABILITY FUSION
    # ---------------------------------------------------
    def hybrid_predict_proba(self, X_input):

        xgb_probs = self.xgb_model.predict_proba(X_input)
        dnn_probs = self.dnn_model.predict(X_input, verbose=0)

        # Weighted fusion
        final_probs = (
            self.xgb_weight * xgb_probs +
            self.dnn_weight * dnn_probs
        )

        return final_probs

    # ---------------------------------------------------
    # SINGLE FLOW PREDICTION
    # ---------------------------------------------------
    def predict_single(self, df: pd.DataFrame):

        X_scaled = self.validate_and_prepare_input(df)

        probs = self.hybrid_predict_proba(X_scaled)

        pred_index = int(np.argmax(probs, axis=1)[0])
        label = self.label_encoder.inverse_transform([pred_index])[0]
        confidence = float(np.max(probs))

        severity = self.severity_map.get(label, "Unknown")

        result = {
            "prediction": label,
            "confidence": confidence,
            "severity": severity
        }

        if confidence < self.low_confidence_threshold:
            result["warning"] = "Low confidence prediction"

        return result

    # ---------------------------------------------------
    # BATCH PREDICTION
    # ---------------------------------------------------
    def predict_batch(self, df: pd.DataFrame):

        X_scaled = self.validate_and_prepare_input(df)

        probs = self.hybrid_predict_proba(X_scaled)

        pred_indices = np.argmax(probs, axis=1)
        labels = self.label_encoder.inverse_transform(pred_indices)
        confidences = np.max(probs, axis=1)

        results = []

        for label, conf in zip(labels, confidences):

            entry = {
                "prediction": label,
                "confidence": float(conf),
                "severity": self.severity_map.get(label, "Unknown")
            }

            if conf < self.low_confidence_threshold:
                entry["warning"] = "Low confidence"

            results.append(entry)

        return results

    # ---------------------------------------------------
    # OPTIONAL: Live Packet Prediction Interface
    # ---------------------------------------------------
    def predict_live_dataframe(self, df: pd.DataFrame):
        """
        Used for live packet capture integration.
        Automatically aligns missing features.
        """
        return self.predict_batch(df)
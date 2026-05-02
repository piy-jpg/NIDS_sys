# src/hybrid_model.py
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import os
from typing import List, Dict

# Try to import Keras, but make it optional
try:
    from tensorflow.keras.models import load_model
    HAS_TF_KERAS = True
except ImportError:
    HAS_TF_KERAS = False
    load_model = None

logger = logging.getLogger("HybridNIDS")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


class HybridNIDS:
    def __init__(self, models_dir: str = "models"):
        """
        Loads models and preprocessing artifacts from models_dir.
        """
        self.models_dir = models_dir

        # ---------- load XGBoost ----------
        xgb_path_json = os.path.join(self.models_dir, "xgb_model.json")
        try:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path_json)
            logger.info(f"Loaded XGBoost model from {xgb_path_json}")
        except Exception as e:
            logger.exception(f"Failed to load XGBoost model: {e}")
            raise

        # ---------- load DNN (optional) ----------
        self.dnn_model = None
        self.dnn_loaded = False
        if HAS_TF_KERAS:
            dnn_path = os.path.join(self.models_dir, "dnn_model.h5")
            try:
                self.dnn_model = load_model(dnn_path)
                self.dnn_loaded = True
                logger.info(f"Loaded DNN model from {dnn_path}")
            except Exception as e:
                logger.warning(f"Failed to load DNN model, will use XGBoost only: {e}")
        else:
            logger.warning("TensorFlow/Keras not available, will use XGBoost only for predictions")

        # ---------- load preprocessing artifacts ----------
        try:
            self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
            self.label_encoder = joblib.load(os.path.join(self.models_dir, "label_encoder.pkl"))
            self.feature_columns = joblib.load(os.path.join(self.models_dir, "feature_columns.pkl"))
            logger.info("Loaded scaler, label_encoder, and feature_columns.")
        except Exception as e:
            logger.exception(f"Failed to load preprocessing artifacts: {e}")
            raise

        # severity map (keeps original mapping)
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

        # Fusion weights (exposed for experimentation)
        # If DNN is not available, use 100% XGBoost
        if self.dnn_loaded:
            self.xgb_weight = 0.7
            self.dnn_weight = 0.3
        else:
            self.xgb_weight = 1.0
            self.dnn_weight = 0.0

        # Confidence threshold for low-confidence warnings
        self.low_confidence_threshold = 0.6

    # -------------------------
    # Internal util: align columns
    # -------------------------
    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures input dataframe contains all required model features.
        Missing features are created and set to 0. Extra columns are dropped.
        """
        df = df.copy()

        # Add missing columns with zeros
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Keep only expected columns in the correct order
        df = df[self.feature_columns].copy()

        return df

    # -------------------------
    # Validation + preprocessing
    # -------------------------
    def validate_and_prepare_input(self, df: pd.DataFrame, impute_missing: bool = True) -> np.ndarray:
        """
        Aligns features, replaces inf, coerces numerics, imputes missing values (default),
        and scales using saved scaler. Returns scaled numpy array ready for models.
        """
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty.")

        # align columns to model feature list
        df_aligned = self._align_columns(df)

        # coerce numeric types (non-parsable values become NaN)
        for col in df_aligned.columns:
            df_aligned[col] = pd.to_numeric(df_aligned[col], errors="coerce")

        # Replace infinite values with NaN
        df_aligned.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Handle NaNs:
        if df_aligned.isnull().values.any():
            if impute_missing:
                # conservative imputation: fill numeric NaNs with 0
                logger.warning("Missing values found in input; imputing with 0 (configurable).")
                df_aligned.fillna(0, inplace=True)
            else:
                missing_count = int(df_aligned.isnull().sum().sum())
                raise ValueError(f"Input contains {missing_count} missing values; set impute_missing=True to auto-fill.")

        # Final numeric check
        if not all(pd.api.types.is_numeric_dtype(df_aligned[c]) for c in df_aligned.columns):
            raise ValueError("All aligned columns must be numeric after coercion and imputation.")

        # Scale
        try:
            X_scaled = self.scaler.transform(df_aligned.values)
        except Exception as e:
            logger.exception("Scaling failed.")
            raise ValueError(f"Scaling failed: {e}")

        return X_scaled

    # -------------------------
    # Hybrid fusion of probabilities
    # -------------------------
    def hybrid_predict_proba(self, X_input: np.ndarray) -> np.ndarray:
        """
        X_input: numpy array (n_samples, n_features)
        Returns: final_probs (n_samples, n_classes)
        """
        # XGBoost probabilities
        xgb_probs = self.xgb_model.predict_proba(X_input)

        # If DNN is available, fuse; otherwise use XGBoost only
        if self.dnn_loaded and self.dnn_model is not None:
            # DNN probabilities (keras returns numpy)
            dnn_probs = np.asarray(self.dnn_model.predict(X_input, verbose=0))
            # Fuse probabilities with weights
            final_probs = self.xgb_weight * xgb_probs + self.dnn_weight * dnn_probs
        else:
            # Use XGBoost only
            final_probs = xgb_probs

        return final_probs

    # -------------------------
    # Single flow prediction
    # -------------------------
    def predict_single(self, df: pd.DataFrame) -> Dict:
        """
        Accepts a dataframe with single row, returns a dict:
        { prediction, confidence, severity, (optional) warning }
        """
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

    # -------------------------
    # Batch prediction
    # -------------------------
    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        """
        Accepts dataframe with N rows; returns list of result dicts.
        """
        X_scaled = self.validate_and_prepare_input(df)

        probs = self.hybrid_predict_proba(X_scaled)

        pred_indices = np.argmax(probs, axis=1)
        labels = self.label_encoder.inverse_transform(pred_indices)
        confidences = np.max(probs, axis=1)

        results = []
        for label, conf in zip(labels, confidences):
            entry = {
                "prediction": str(label),
                "confidence": float(conf),
                "severity": self.severity_map.get(label, "Unknown")
            }
            if conf < self.low_confidence_threshold:
                entry["warning"] = "Low confidence"
            results.append(entry)

        return results

    # -------------------------
    # Live dataframe helper (same as batch)
    # -------------------------
    def predict_live_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """
        For live capture: returns same structure as predict_batch.
        """
        return self.predict_batch(df)

    # -------------------------
    # Model / artifact info (for dashboard)
    # -------------------------
    def get_model_status(self) -> Dict:
        """
        Returns basic metadata for the frontend (feature list, model names).
        """
        return {
            "n_features": len(self.feature_columns),
            "feature_columns": list(self.feature_columns),
            "xgb_loaded": True,
            "dnn_loaded": self.dnn_loaded,
            "xgb_weight": self.xgb_weight,
            "dnn_weight": self.dnn_weight,
            "modelo_mode": "Hybrid (XGBoost + DNN)" if self.dnn_loaded else "XGBoost only (DNN unavailable)"
        }
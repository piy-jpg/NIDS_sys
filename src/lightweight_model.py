import math
from typing import Dict, List

import pandas as pd


class LightweightHybridNIDS:
    """Vercel-safe heuristic predictor used when the full ML stack is unavailable."""

    def __init__(self) -> None:
        self.xgb_weight = 0.5
        self.dnn_weight = 0.5
        self.low_confidence_threshold = 0.6

    def _num(self, row: pd.Series, column: str, default: float = 0.0) -> float:
        value = row.get(column, default)
        try:
            parsed = float(value)
            if math.isnan(parsed) or math.isinf(parsed):
                return default
            return parsed
        except (TypeError, ValueError):
            return default

    def _score_row(self, row: pd.Series) -> Dict:
        flow_bytes = self._num(row, "Flow Bytes/s")
        flow_packets = self._num(row, "Flow Packets/s")
        syn_flags = self._num(row, "SYN Flag Count")
        rst_flags = self._num(row, "RST Flag Count")
        psh_flags = self._num(row, "PSH Flag Count")
        ack_flags = self._num(row, "ACK Flag Count")
        duration = self._num(row, "Flow Duration")
        packet_variance = self._num(row, "Packet Length Variance")
        init_fwd_win = self._num(row, "Init Fwd Win Bytes")
        protocol = str(row.get("Protocol", ""))

        score = 0.0
        if flow_packets > 10000:
            score += 0.35
        elif flow_packets > 1000:
            score += 0.2

        if flow_bytes > 1_000_000:
            score += 0.25
        elif flow_bytes > 100_000:
            score += 0.15

        if syn_flags >= 1 and ack_flags == 0:
            score += 0.25
        if rst_flags >= 1 or psh_flags >= 3:
            score += 0.1
        if duration > 1_000_000:
            score += 0.1
        if packet_variance > 10_000:
            score += 0.1
        if init_fwd_win in (-1.0, 0.0):
            score += 0.05
        if protocol == "17":
            score += 0.05

        confidence = max(0.52, min(0.98, 0.52 + score / 2.0))

        if score >= 0.75:
            prediction = "DDoS"
            severity = "High"
        elif score >= 0.45:
            prediction = "PortScan"
            severity = "Low"
        else:
            prediction = "Benign"
            severity = "Safe"

        result = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "severity": severity,
        }
        if confidence < self.low_confidence_threshold:
            result["warning"] = "Low confidence heuristic prediction"
        return result

    def predict_single(self, df: pd.DataFrame) -> Dict:
        if df is None or df.empty or len(df) != 1:
            raise ValueError("Single prediction requires exactly one row.")
        return self._score_row(df.iloc[0])

    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty.")
        return [self._score_row(row) for _, row in df.iterrows()]

    def predict_live_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        return self.predict_batch(df)

    def get_model_status(self) -> Dict:
        return {
            "n_features": None,
            "feature_columns": [],
            "xgb_loaded": False,
            "dnn_loaded": False,
            "xgb_weight": self.xgb_weight,
            "dnn_weight": self.dnn_weight,
            "modelo_mode": "Lightweight heuristic demo mode"
        }

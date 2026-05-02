# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import io
import time
import json
import hashlib
import random

st.set_page_config(page_title="Hybrid NIDS Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.metric-box { background-color:#111827; padding:15px; border-radius:12px; text-align:center; }
.metric-label { font-size:13px; color:#9ca3af; }
.metric-value { font-size:24px; font-weight:600; }
.alert-banner { background-color:#7f1d1d; color:white; padding:10px; border-radius:8px; text-align:center; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.title("🔐 Hybrid Network Intrusion Detection System")
st.caption("Real-Time Cyber Threat Monitoring Dashboard")

# ===============================
# Sidebar Configuration
# ===============================

API_URL = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")

st.sidebar.markdown("### ⚙ Fusion Control")

xgb_weight = st.sidebar.slider("XGBoost Weight", 0.0, 1.0, 0.7)
dnn_weight = 1.0 - xgb_weight

if st.sidebar.button("Apply Fusion Weights"):
    try:
        resp = requests.post(f"{API_URL}/set_weights", json={
            "xgb_weight": xgb_weight,
            "dnn_weight": dnn_weight
        })
        if resp.status_code == 200:
            st.sidebar.success("Fusion weights updated")
        else:
            st.sidebar.error("Failed to update weights")
    except:
        st.sidebar.error("Backend not reachable")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Network Traffic CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to begin monitoring.")
    st.stop()

# ===============================
# Load & Preview CSV
# ===============================

try:
    raw_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
except Exception as e:
    st.error(f"CSV Error: {e}")
    st.stop()

st.subheader("Uploaded Data Preview")
st.write(raw_df.head())

# ===============================
# Fetch Predictions
# ===============================

def fetch_predictions(file_bytes):
    files = {"file": ("data.csv", file_bytes, "text/csv")}
    response = requests.post(f"{API_URL}/batch_predict", files=files)
    if response.status_code != 200:
        raise RuntimeError(response.json().get("error"))
    return response.json()

try:
    with st.spinner("Running Hybrid Detection Engine..."):
        api_data = fetch_predictions(uploaded_file.getvalue())
        results = pd.DataFrame(api_data["results"])
        total_flows = api_data["total_flows"]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ===============================
# High Severity Flash Banner
# ===============================

if "High" in results["severity"].values:
    st.markdown(
        "<div class='alert-banner'>🚨 HIGH SEVERITY ATTACK DETECTED</div>",
        unsafe_allow_html=True
    )

# ===============================
# Risk Score Meter
# ===============================

attack_count = len(results[results["prediction"] != "Benign"])
risk_score = int((attack_count / len(results)) * 100)

st.markdown("### 🧠 Network Risk Level")
st.progress(risk_score / 100)

if risk_score > 60:
    st.error(f"High Risk Environment ({risk_score}/100)")
elif risk_score > 30:
    st.warning(f"Moderate Risk ({risk_score}/100)")
else:
    st.success(f"Stable Network ({risk_score}/100)")

# ===============================
# Metrics
# ===============================

col1, col2, col3, col4 = st.columns(4)

def metric(col, label, value):
    with col:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

metric(col1, "Total Flows", total_flows)
metric(col2, "Detected Attacks", attack_count)
metric(col3, "Benign Traffic", len(results[results["prediction"] == "Benign"]))
metric(col4, "Avg Confidence", round(results["confidence"].mean(), 3))

st.markdown("---")

# ===============================
# Charts
# ===============================

left, right = st.columns(2)

with left:
    attack_counts = results["prediction"].value_counts().reset_index()
    attack_counts.columns = ["Attack", "Count"]
    fig_pie = px.pie(attack_counts, names="Attack", values="Count", hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    fig_conf = px.histogram(results, x="confidence", nbins=20)
    st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# ===============================
# Live Threat Feed
# ===============================

st.markdown("### 🔴 Live Threat Feed")

for _, row in results.tail(10).iterrows():
    emoji = {
        "High": "🚨",
        "Medium": "⚠",
        "Low": "🔎",
        "Safe": "✅"
    }.get(row["severity"], "❓")

    st.write(f"{emoji} {row['prediction']} | Confidence: {row['confidence']:.3f}")

st.markdown("---")

# ===============================
# Simulated Attack Geo Map
# ===============================

def fake_geo(value):
    h = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
    random.seed(h)
    return random.uniform(-60, 60), random.uniform(-180, 180)

st.markdown("### 🌍 Attack Distribution Map")

geo_data = []

if "src_ip" in raw_df.columns:
    for ip in raw_df["src_ip"].astype(str):
        lat, lon = fake_geo(ip)
        geo_data.append((lat, lon))
else:
    for idx in range(len(results)):
        lat, lon = fake_geo(idx)
        geo_data.append((lat, lon))

geo_df = pd.DataFrame(geo_data, columns=["lat", "lon"])
st.map(geo_df)

st.markdown("---")

# ===============================
# Detailed Results
# ===============================

st.subheader("Detailed Detection Results")
st.dataframe(results, height=300)

st.download_button(
    "Download Results CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="nids_results.csv",
    mime="text/csv"
)

if st.checkbox("Show Raw API JSON"):
    st.code(json.dumps(api_data, indent=2), language="json")
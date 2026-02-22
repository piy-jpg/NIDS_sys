import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import io
import time

API_URL = "http://127.0.0.1:8000"

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Hybrid NIDS Enterprise Dashboard",
    layout="wide"
)

# ----------------------------
# CUSTOM STYLING
# ----------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.metric-box {
    background-color:#111827;
    padding:15px;
    border-radius:12px;
    text-align:center;
}
.metric-label {
    font-size:13px;
    color:#9ca3af;
}
.metric-value {
    font-size:26px;
    font-weight:600;
}
.alert-banner {
    background-color:#7f1d1d;
    color:white;
    padding:10px;
    border-radius:8px;
    text-align:center;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.title("🔐 Hybrid Network Intrusion Detection System")
st.caption("Enterprise Cyber Threat Monitoring Interface")

# ----------------------------
# SESSION STATE
# ----------------------------
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin monitoring.")
    st.stop()

# ----------------------------
# FETCH PREDICTIONS
# ----------------------------
def fetch_predictions(file_bytes):
    files = {"file": file_bytes}
    response = requests.post(f"{API_URL}/batch_predict", files=files)
    if response.status_code != 200:
        raise Exception(response.json().get("error", "API Error"))
    return response.json()

try:
    api_data = fetch_predictions(uploaded_file.getvalue())
    results = pd.DataFrame(api_data["results"])

    raw_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

    if "timestamp" in raw_df.columns:
        results["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    else:
        results["row_index"] = np.arange(len(results))

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.markdown("### Filters")

attack_filter = st.sidebar.multiselect(
    "Attack Type",
    sorted(results["prediction"].unique()),
    default=sorted(results["prediction"].unique())
)

severity_filter = st.sidebar.multiselect(
    "Severity",
    sorted(results["severity"].unique()),
    default=sorted(results["severity"].unique())
)

conf_range = st.sidebar.slider(
    "Confidence Range",
    0.0, 1.0,
    (0.0, 1.0),
    step=0.01
)

# Start / Stop Monitoring
if st.sidebar.button("▶ Start Monitoring"):
    st.session_state.live_mode = True

if st.sidebar.button("⏹ Stop Monitoring"):
    st.session_state.live_mode = False

# ----------------------------
# APPLY FILTERS
# ----------------------------
df_view = results.copy()
df_view = df_view[df_view["prediction"].isin(attack_filter)]
df_view = df_view[df_view["severity"].isin(severity_filter)]
df_view = df_view[
    (df_view["confidence"] >= conf_range[0]) &
    (df_view["confidence"] <= conf_range[1])
]

# ----------------------------
# ALERT BANNER
# ----------------------------
if st.session_state.live_mode:
    new_attacks = len(df_view[df_view["prediction"] != "Benign"])
    if new_attacks > 0:
        st.markdown(
            f"<div class='alert-banner'>⚠ {new_attacks} Potential Threats Detected</div>",
            unsafe_allow_html=True
        )

# ----------------------------
# METRICS ROW
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

def metric(column, label, value):
    with column:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

metric(col1, "Total Flows", len(results))
metric(col2, "Filtered Flows", len(df_view))
metric(col3, "Detected Attacks", len(results[results["prediction"] != "Benign"]))
metric(col4, "Benign Traffic", len(results[results["prediction"] == "Benign"]))

st.markdown("---")

# ----------------------------
# ROW 1
# ----------------------------
left, right = st.columns(2)

with left:
    attack_counts = df_view["prediction"].value_counts().reset_index()
    attack_counts.columns = ["Attack", "Count"]

    if not attack_counts.empty:
        fig_pie = px.pie(
            attack_counts,
            names="Attack",
            values="Count",
            hole=0.5
        )
        fig_pie.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, width="stretch")

with right:
    severity_counts = df_view["severity"].value_counts().reset_index()
    severity_counts.columns = ["Severity", "Count"]

    if not severity_counts.empty:
        fig_bar = px.bar(
            severity_counts,
            x="Severity",
            y="Count",
            color="Severity"
        )
        fig_bar.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, width="stretch")

st.markdown("---")

# ----------------------------
# ROW 2
# ----------------------------
colA, colB = st.columns(2)

with colA:
    if not df_view.empty:
        fig_conf = px.histogram(df_view, x="confidence", nbins=20)
        fig_conf.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_conf, width="stretch")

with colB:
    if not df_view.empty:
        timeline_df = df_view.copy()

        if "timestamp" in timeline_df.columns:
            fig_time = px.line(timeline_df, x="timestamp", y="confidence")
        else:
            if "row_index" not in timeline_df.columns:
                timeline_df["row_index"] = np.arange(len(timeline_df))
            fig_time = px.scatter(timeline_df, x="row_index", y="confidence")

        fig_time.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_time, width="stretch")

st.markdown("---")

# ----------------------------
# TABLE
# ----------------------------
st.subheader("Detailed Results")
st.dataframe(df_view, height=300)

st.download_button(
    "Download Filtered Results",
    data=df_view.to_csv(index=False).encode("utf-8"),
    file_name="nids_filtered_results.csv",
    mime="text/csv"
)

# ----------------------------
# LIVE SIMULATION LOOP
# ----------------------------
if st.session_state.live_mode:
    time.sleep(2)
    st.rerun()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.settings import DATA_DIR, SMART_CSV
from analytics.smart_features import SmartViolationProcessor
import os

st.set_page_config(page_title="IntelliTraffic AI", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.metric-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    text-align: center;
    border: 1px solid #333;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #4CAF50;
    margin: 10px 0;
}
.metric-label {
    font-size: 1rem;
    color: #AAAAAA;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if SMART_CSV.exists():
        return pd.read_csv(SMART_CSV)
    return pd.DataFrame()

st.title("🚦 IntelliTraffic AI Dashboard")
st.markdown("Advanced AI-powered traffic monitoring, violation detection, and predictive analytics.")

if st.sidebar.button("Run Smart Analysis", type="primary"):
    with st.spinner("Processing violation evidence..."):
        SmartViolationProcessor().run()
        st.cache_data.clear()
        st.success("Analysis complete!")

df = load_data()

if df.empty:
    st.info("No smart violation data found. Please run the detectors and then click 'Run Smart Analysis'.")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Violations</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
    with col2:
        high_severity = len(df[df.get("severity_score", 0) >= 4])
        st.markdown(f'<div class="metric-card"><div class="metric-label">High Severity Cases</div><div class="metric-value" style="color: #F44336;">{high_severity}</div></div>', unsafe_allow_html=True)
    with col3:
        total_fines = df.get("fine_amount", 0).sum()
        st.markdown(f'<div class="metric-card"><div class="metric-label">Estimated Fines (Rs)</div><div class="metric-value" style="color: #FFC107;">{total_fines:,}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Active Hotspots</div><div class="metric-value" style="color: #2196F3;">{df.get("hotspot", pd.Series()).nunique()}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Violations by Type")
        if "violation_type" in df.columns:
            counts = df["violation_type"].value_counts().reset_index()
            counts.columns = ["Violation", "Count"]
            fig = px.pie(counts, values="Count", names="Violation", hole=0.4, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Agbnl)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.subheader("Severity Distribution")
        if "severity_score" in df.columns:
            counts = df["severity_score"].value_counts().sort_index().reset_index()
            counts.columns = ["Severity Score", "Count"]
            fig = px.bar(counts, x="Severity Score", y="Count", template="plotly_dark", color="Severity Score", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Smart Alerts")
    if "alert_message" in df.columns:
        for idx, row in df.head(10).iterrows():
            severity = row.get("severity_score", 1)
            color = "red" if severity >= 4 else "orange" if severity == 3 else "green"
            icon = "🚨" if severity >= 4 else "⚠️" if severity == 3 else "ℹ️"
            st.markdown(f"> {icon} **[{row.get('violation_type', 'Unknown')}]** {row.get('alert_message', '')} *(Score: {severity})*")
    
    st.markdown("---")
    st.subheader("Violation Evidence Database")
    st.dataframe(df, use_container_width=True)

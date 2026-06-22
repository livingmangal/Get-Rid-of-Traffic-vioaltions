import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="API Explorer | IntelliTraffic", page_icon="🔌", layout="wide")
st.title("🔌 IntelliTraffic API Explorer")
st.markdown("Interact with the IntelliTraffic Backend API.")

API_BASE_URL = "http://localhost:8000"

st.sidebar.header("API Controls")
api_status = st.sidebar.empty()

try:
    response = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if response.status_code == 200:
        api_status.success("🟢 API is Online")
        health_data = response.json()
        st.sidebar.markdown(f"**Database**: {'MongoDB' if health_data.get('mongo_connected') else 'CSV Fallback'}")
        st.sidebar.markdown(f"**Records**: {health_data.get('records_available', 0)}")
    else:
        api_status.error("🔴 API Returned Error")
except Exception:
    api_status.error("🔴 API is Offline (Start with `uvicorn api.backend:app --reload`)")
    st.warning("Please start the backend API server. `python -m uvicorn api.backend:app --reload`")
    st.stop()

tabs = st.tabs(["Endpoints", "Summary", "Violations Search"])

with tabs[0]:
    st.subheader("Available Endpoints")
    endpoints = [
        ("GET", "/health", "Check API health and DB connection"),
        ("GET", "/violations", "List all violations (supports filtering)"),
        ("POST", "/violations", "Create a new violation record"),
        ("GET", "/analytics/summary", "Get high-level summary metrics"),
        ("GET", "/analytics/by-type", "Aggregate violations by type")
    ]
    for method, path, desc in endpoints:
        st.markdown(f"- **`{method}`** `{path}` - *{desc}*")

with tabs[1]:
    st.subheader("Analytics Summary")
    if st.button("Fetch Summary"):
        res = requests.get(f"{API_BASE_URL}/analytics/summary")
        if res.status_code == 200:
            st.json(res.json())
            
    st.subheader("By Type")
    if st.button("Fetch By Type"):
        res = requests.get(f"{API_BASE_URL}/analytics/by-type")
        if res.status_code == 200:
            data = res.json()
            if data:
                st.dataframe(pd.DataFrame(data))
            else:
                st.info("No data available")

with tabs[2]:
    st.subheader("Search Violations")
    col1, col2 = st.columns(2)
    with col1:
        v_type = st.text_input("Violation Type (optional)")
    with col2:
        min_sev = st.slider("Minimum Severity", 1, 5, 1)
        
    if st.button("Search"):
        params = {"min_severity": min_sev}
        if v_type:
            params["violation_type"] = v_type
            
        res = requests.get(f"{API_BASE_URL}/violations", params=params)
        if res.status_code == 200:
            data = res.json()
            st.success(f"Found {data.get('count', 0)} records")
            if data.get("records"):
                st.dataframe(pd.DataFrame(data.get("records")))

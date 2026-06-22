import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
from config.settings import SMART_CSV

st.set_page_config(page_title="Advanced Analytics | IntelliTraffic", page_icon="📊", layout="wide")
st.title("📊 Advanced Analytics & Geography")

@st.cache_data
def load_data():
    if SMART_CSV.exists():
        return pd.read_csv(SMART_CSV)
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data available for analytics.")
else:
    st.markdown("### Geographical Hotspots (Simulated)")
    if {"latitude", "longitude", "violation_type"}.issubset(df.columns):
        # Adding some jitter to coordinates for visualization if they are too identical
        map_df = df.copy()
        map_df["latitude"] = map_df["latitude"].astype(float) + (pd.Series([x % 100 for x in range(len(map_df))]) / 10000.0)
        map_df["longitude"] = map_df["longitude"].astype(float) + (pd.Series([(x * 3) % 100 for x in range(len(map_df))]) / 10000.0)
        
        fig = px.scatter_mapbox(
            map_df, 
            lat="latitude", 
            lon="longitude", 
            color="violation_type",
            size="severity_score",
            hover_name="hotspot",
            hover_data=["violation_type", "fine_amount"],
            zoom=13,
            mapbox_style="carto-darkmatter",
            title="Violation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Location data not available for mapping.")

    st.markdown("### Financial Impact")
    if {"violation_type", "fine_amount"}.issubset(df.columns):
        fine_df = df.groupby("violation_type")["fine_amount"].sum().reset_index()
        fig = px.bar(
            fine_df, 
            x="violation_type", 
            y="fine_amount", 
            title="Revenue Generation by Violation Type",
            labels={"violation_type": "Violation Type", "fine_amount": "Total Fines (Rs)"},
            template="plotly_dark",
            color="fine_amount",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

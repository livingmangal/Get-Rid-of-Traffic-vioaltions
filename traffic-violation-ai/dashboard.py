import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IntelliTraffic AI", layout="wide")

st.title("🚦 IntelliTraffic AI Dashboard")
st.subheader("Smart Traffic Violation & Road Safety Command Center")

DATA_DIR = "data"

reports = {
    "Night Traffic": "traffic_report.csv",
    "Pedestrian Safety": "pedestrian_report.csv",
    "Wrong Side": "wrong_side_report.csv",
    "Helmet": "helmet_report.csv",
    "Triple Riding": "triple_riding_report.csv",
    "Waterlogging": "waterlogging_report.csv",
}

summary = {}

for name, file in reports.items():
    path = os.path.join(DATA_DIR, file)

    st.markdown("---")
    st.header(name)

    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True)

        for col in df.columns:
            if "Violation" in col or "Alert" in col or "Status" in col or "Density" in col:
                summary[name] = str(df[col].iloc[-1])
    else:
        st.warning(f"{file} not found. Run its detector first.")

st.markdown("---")
st.header("🤖 AI Traffic Police Assistant")

recommendations = []

if os.path.exists("data/helmet_report.csv"):
    recommendations.append("Helmet violations found: issue warning/e-challan and increase helmet checking.")

if os.path.exists("data/wrong_side_report.csv"):
    recommendations.append("Wrong-side driving detected: deploy officer near hotspot and add road divider/signage.")

if os.path.exists("data/triple_riding_report.csv"):
    recommendations.append("Triple riding risk: enforce two-rider rule and safety awareness.")

if os.path.exists("data/waterlogging_report.csv"):
    recommendations.append("Waterlogging detected: reroute traffic and alert municipal drainage team.")

if os.path.exists("data/pedestrian_report.csv"):
    recommendations.append("Pedestrian activity detected: improve crossing safety and slow vehicle movement.")

if recommendations:
    for rec in recommendations:
        st.success(rec)
else:
    st.info("No reports found yet. Run detector scripts first.")

st.markdown("---")
st.header(" Final System Outcome")

st.write("""
This prototype automatically processes traffic videos, enhances low-light footage,
detects vehicles and pedestrians, identifies selected traffic violations, generates
evidence videos, stores CSV reports, and provides AI-assisted traffic recommendations.
""")
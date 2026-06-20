import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="IntelliTraffic API Dashboard",
    page_icon="🌐",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 55%, #020617 100%);
        color: white;
    }
    section[data-testid="stSidebar"] {
        background: #020617;
    }
    .title {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8, #22c55e, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 25px;
    }
    .card {
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 14px 38px rgba(0,0,0,0.35);
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_json(endpoint, params=None):
    try:
        response = requests.get(
            API_BASE + endpoint,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:
        st.error(f"API error: {error}")
        st.info("Start backend first: python -m uvicorn backend_api:app --reload --port 8000")
        st.stop()


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.25)",
        font_color="white",
        margin=dict(l=20, r=20, t=55, b=30),
    )
    return fig


st.markdown('<div class="title">IntelliTraffic API Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Dashboard fetching violation records and analytics from FastAPI backend.</div>',
    unsafe_allow_html=True,
)

health = get_json("/health")
summary = get_json("/analytics/summary")

st.sidebar.title("🌐 Backend Status")
st.sidebar.success("FastAPI connected")
st.sidebar.write("Database Mode:", health.get("database_mode"))
st.sidebar.write("Records:", health.get("records_available"))
st.sidebar.write("MongoDB:", health.get("mongo_connected"))

st.sidebar.markdown("---")
st.sidebar.title("Filters")

violation_type = st.sidebar.text_input("Violation type contains")
hotspot = st.sidebar.text_input("Hotspot contains")
vehicle_number = st.sidebar.text_input("Vehicle number contains")
min_severity = st.sidebar.slider("Minimum severity", 1, 5, 1)

params = {
    "min_severity": min_severity,
}

if violation_type:
    params["violation_type"] = violation_type

if hotspot:
    params["hotspot"] = hotspot

if vehicle_number:
    params["vehicle_number"] = vehicle_number

violations_response = get_json("/violations", params=params)
records = violations_response.get("records", [])
df = pd.DataFrame(records)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Records", summary.get("total_records", 0))
c2.metric("Total Demo Fine", f"₹{summary.get('total_fine', 0)}")
c3.metric("Average Severity", summary.get("average_severity", 0))
c4.metric("High Severity", summary.get("high_severity_count", 0))
c5.metric("Unique Vehicles", summary.get("unique_vehicles", 0))

st.markdown("## 📊 API Analytics")

type_data = get_json("/analytics/by-type")
module_data = get_json("/analytics/by-module")
hotspot_data = get_json("/analytics/hotspots")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    type_df = pd.DataFrame(type_data)

    if not type_df.empty:
        fig = px.bar(
            type_df,
            x="violation_type",
            y="count",
            text="count",
            title="Violations by Type from API",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
    else:
        st.warning("No violation type analytics found.")

with chart_col2:
    module_df = pd.DataFrame(module_data)

    if not module_df.empty:
        fig = px.bar(
            module_df,
            x="module",
            y="count",
            text="count",
            title="Records by Module from API",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
    else:
        st.warning("No module analytics found.")

st.markdown("## 🗺️ Hotspot Map from API")

hotspot_df = pd.DataFrame(hotspot_data)

if not hotspot_df.empty:
    fig = px.scatter_mapbox(
        hotspot_df,
        lat="latitude",
        lon="longitude",
        size="violations",
        color="average_severity",
        hover_name="hotspot",
        hover_data=["violations", "total_fine", "average_severity"],
        zoom=10,
        height=520,
        title="Violation Hotspots from FastAPI",
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=45, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hotspot data found.")

st.markdown("## 📁 Violation Records from API")

if df.empty:
    st.warning("No records returned from API.")
else:
    st.dataframe(df, use_container_width=True, height=420)

    st.markdown("## 📲 Alert Preview from API")

    selected_evidence = st.selectbox(
        "Select Evidence ID",
        df["evidence_id"].astype(str).tolist(),
    )

    alert_data = get_json(f"/alerts/preview/{selected_evidence}")

    st.markdown(
        f"""
        <div class="card">
            <h3>Alert Message</h3>
            <p><b>Evidence:</b> {alert_data.get("evidence_id")}</p>
            <p><b>Vehicle:</b> {alert_data.get("vehicle_number")}</p>
            <p><b>Violation:</b> {alert_data.get("violation_type")}</p>
            <p><b>Message:</b> {alert_data.get("message")}</p>
            <p><b>Status:</b> {alert_data.get("status")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("IntelliTraffic AI API Dashboard | FastAPI + MongoDB-ready + Streamlit")

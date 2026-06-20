from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SMART_CSV = DATA_DIR / "smart_violations.csv"

st.set_page_config(
    page_title="Smart Violation Intelligence",
    page_icon="🚦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56,189,248,0.22), transparent 35%),
            linear-gradient(135deg, #020617 0%, #0f172a 55%, #020617 100%);
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

    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(56,189,248,0.35);
        border-radius: 18px;
        padding: 22px;
        min-height: 125px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: white;
        font-size: 36px;
        font-weight: 900;
        margin-top: 8px;
    }

    .metric-note {
        color: #38bdf8;
        font-size: 13px;
        margin-top: 5px;
    }

    .section {
        font-size: 28px;
        font-weight: 850;
        margin-top: 28px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.20)",
        font_color="white",
        margin=dict(l=20, r=20, t=55, b=30),
    )
    return fig


def simple_natural_language_search(query, df):
    query = query.lower().strip()
    result = df.copy()
    explanation = []

    if "helmet" in query:
        result = result[result["violation_type"].astype(str).str.contains("HELMET", case=False, na=False)]
        explanation.append("helmet violations")

    if "triple" in query:
        result = result[result["violation_type"].astype(str).str.contains("TRIPLE", case=False, na=False)]
        explanation.append("triple riding violations")

    if "wrong" in query:
        result = result[result["violation_type"].astype(str).str.contains("WRONG", case=False, na=False)]
        explanation.append("wrong-side driving violations")

    if "red" in query:
        result = result[result["violation_type"].astype(str).str.contains("RED", case=False, na=False)]
        explanation.append("red-light violations")

    if "water" in query:
        result = result[result["violation_type"].astype(str).str.contains("WATER", case=False, na=False)]
        explanation.append("waterlogging events")

    if "high" in query:
        result = result[result["severity_score"] >= 4]
        explanation.append("high severity records")

    if "fine" in query:
        result = result[result["fine_amount"] > 0]
        explanation.append("records with fines")

    words = query.replace(",", " ").replace(".", " ").split()

    for word in words:
        if len(word) >= 6 and any(ch.isdigit() for ch in word):
            plate_mask = result["vehicle_number"].astype(str).str.contains(word.upper(), case=False, na=False)
            if plate_mask.any():
                result = result[plate_mask]
                explanation.append(f"vehicle number containing {word.upper()}")

    if not explanation:
        explanation.append("all records because no specific filter was detected")

    answer = f"Showing {len(result)} record(s) for: " + ", ".join(explanation) + "."

    return answer, result


st.markdown('<div class="title">Smart Violation Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Severity scoring, fine generation, heatmap, alert messages, and natural language search for IntelliTraffic AI.</div>',
    unsafe_allow_html=True,
)

if not SMART_CSV.exists():
    st.error("smart_violations.csv not found.")
    st.info("Run this first: python smart_features.py")
    st.stop()

df = pd.read_csv(SMART_CSV)

if df.empty:
    st.error("smart_violations.csv is empty.")
    st.stop()

st.sidebar.title("🚦 Smart Filters")

violation_types = sorted(df["violation_type"].dropna().unique().tolist())
hotspots = sorted(df["hotspot"].dropna().unique().tolist())

selected_types = st.sidebar.multiselect(
    "Violation Type",
    violation_types,
    default=violation_types,
)

selected_hotspots = st.sidebar.multiselect(
    "Hotspots",
    hotspots,
    default=hotspots,
)

min_severity = st.sidebar.slider(
    "Minimum Severity Score",
    min_value=1,
    max_value=5,
    value=1,
)

filtered = df[
    df["violation_type"].isin(selected_types)
    & df["hotspot"].isin(selected_hotspots)
    & (df["severity_score"] >= min_severity)
].copy()

total_records = len(filtered)
total_fine = int(filtered["fine_amount"].sum())
avg_severity = round(filtered["severity_score"].mean(), 2) if not filtered.empty else 0
alerts_ready = len(filtered[filtered["alert_status"] == "DEMO_READY"])

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("Total Violations", total_records, "Filtered smart evidence")

with c2:
    metric_card("Total Demo Fine", f"₹{total_fine}", "Prototype fine generator")

with c3:
    metric_card("Average Severity", avg_severity, "Risk scoring from 1 to 5")

with c4:
    metric_card("Alerts Ready", alerts_ready, "SMS/WhatsApp demo messages")

st.markdown('<div class="section">1. Violation Severity Score</div>', unsafe_allow_html=True)

severity_summary = (
    filtered.groupby("violation_type")["severity_score"]
    .mean()
    .reset_index()
    .sort_values("severity_score", ascending=False)
)

fig = px.bar(
    severity_summary,
    x="violation_type",
    y="severity_score",
    text="severity_score",
    title="Average Severity by Violation Type",
)
st.plotly_chart(style_fig(fig), use_container_width=True)

st.markdown('<div class="section">2. Smart Fine Generator</div>', unsafe_allow_html=True)

fine_summary = (
    filtered.groupby("violation_type")["fine_amount"]
    .sum()
    .reset_index()
    .sort_values("fine_amount", ascending=False)
)

fig = px.bar(
    fine_summary,
    x="violation_type",
    y="fine_amount",
    text="fine_amount",
    title="Generated Demo Fine by Violation Type",
)
st.plotly_chart(style_fig(fig), use_container_width=True)

st.dataframe(
    filtered[
        [
            "evidence_id",
            "timestamp",
            "vehicle_number",
            "violation_type",
            "severity_score",
            "fine_amount",
            "hotspot",
            "confidence",
        ]
    ],
    use_container_width=True,
    height=320,
)

st.markdown('<div class="section">3. Violation Heatmap</div>', unsafe_allow_html=True)

heat_df = (
    filtered.groupby(["hotspot", "latitude", "longitude"])
    .agg(
        violations=("evidence_id", "count"),
        total_fine=("fine_amount", "sum"),
        avg_severity=("severity_score", "mean"),
    )
    .reset_index()
)

fig = px.scatter_mapbox(
    heat_df,
    lat="latitude",
    lon="longitude",
    size="violations",
    color="avg_severity",
    hover_name="hotspot",
    hover_data=["violations", "total_fine", "avg_severity"],
    zoom=10,
    height=520,
    title="City Violation Hotspots",
)

fig.update_layout(
    mapbox_style="open-street-map",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    margin=dict(l=0, r=0, t=45, b=0),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section">4. WhatsApp/SMS Alert Generator</div>', unsafe_allow_html=True)

selected_evidence = st.selectbox(
    "Select evidence record",
    filtered["evidence_id"].tolist(),
)

selected_row = filtered[filtered["evidence_id"] == selected_evidence].iloc[0]

st.markdown(
    f"""
    <div class="card">
    <h3>Alert Preview</h3>
    <p><b>Vehicle:</b> {selected_row['vehicle_number']}</p>
    <p><b>Violation:</b> {selected_row['violation_type']}</p>
    <p><b>Fine:</b> ₹{selected_row['fine_amount']}</p>
    <p><b>Message:</b></p>
    <p>{selected_row['alert_message']}</p>
    <p><b>Status:</b> DEMO MODE - no real SMS sent</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Generate Demo Alert"):
    st.success("Demo alert generated successfully. Twilio integration can be connected in production.")

st.markdown('<div class="section">5. Natural Language Search</div>', unsafe_allow_html=True)

query = st.text_input(
    "Ask a question",
    placeholder="Example: Show all helmet violations today",
)

if query:
    answer, search_df = simple_natural_language_search(query, filtered)

    st.markdown(
        f"""
        <div class="card">
        <h3>AI Search Answer</h3>
        <p>{answer}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(search_df, use_container_width=True, height=320)
else:
    st.info("Try: Show all helmet violations today | Show high severity fines | Show triple riding")

st.markdown("---")
st.caption("Smart Violation Intelligence Prototype | Severity + Fine + Heatmap + Alert + NL Search")

from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="IntelliTraffic AI",
    page_icon="🚦",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PLATE_DIR = OUTPUT_DIR / "plate_evidence"

REPORT_FILES = {
    "Traffic Density": DATA_DIR / "traffic_report.csv",
    "Pedestrian Safety": DATA_DIR / "pedestrian_report.csv",
    "Wrong Side Driving": DATA_DIR / "wrong_side_report.csv",
    "Helmet Compliance": DATA_DIR / "helmet_report.csv",
    "Triple Riding": DATA_DIR / "triple_riding_report.csv",
    "Waterlogging": DATA_DIR / "waterlogging_report.csv",
    "License Plate OCR": DATA_DIR / "license_plate_ocr_report.csv",
    "Improved License Plate OCR": DATA_DIR / "license_plate_ocr_improved_report.csv",
}

VIDEO_FILES = {
    "Night Vision Detection": OUTPUT_DIR / "night_processed.mp4",
    "Pedestrian Safety Detection": OUTPUT_DIR / "pedestrian_processed.mp4",
    "Wrong Side Detection": OUTPUT_DIR / "wrong_side_processed.mp4",
    "Helmet Detection": OUTPUT_DIR / "no_helmet_processed.mp4",
    "Triple Riding Detection": OUTPUT_DIR / "triple_riding_processed.mp4",
    "Waterlogging Detection": OUTPUT_DIR / "waterlogging_processed.mp4",
    "License Plate OCR": OUTPUT_DIR / "license_plate_ocr.mp4",
    "Improved License Plate OCR": OUTPUT_DIR / "license_plate_ocr_improved.mp4",
}

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%);
    color: white;
}
section[data-testid="stSidebar"] {
    background: #020617;
}
.main-title {
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #22c55e, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 25px;
}
.card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0px 10px 35px rgba(0,0,0,0.35);
}
.metric-box {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 18px;
    padding: 22px;
    min-height: 125px;
}
.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 700;
}
.metric-value {
    color: white;
    font-size: 36px;
    font-weight: 900;
}
.metric-note {
    color: #38bdf8;
    font-size: 13px;
}
.section-title {
    font-size: 28px;
    font-weight: 850;
    margin-top: 30px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

def read_report(name, path):
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df["module"] = name

        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "status" not in df.columns:
            if "violation" in df.columns:
                df["status"] = df["violation"].astype(str)
            elif "risk" in df.columns:
                df["status"] = df["risk"].astype(str)
            elif "risk_level" in df.columns:
                df["status"] = df["risk_level"].astype(str)
            elif "plate_text" in df.columns:
                df["status"] = "PLATE_READ"
            else:
                df["status"] = name

        if "confidence" not in df.columns:
            if "ocr_confidence" in df.columns:
                df["confidence"] = df["ocr_confidence"]
            else:
                df["confidence"] = 0.85

        return df

    except Exception:
        return pd.DataFrame()

def load_data():
    reports = {}
    all_reports = []

    for name, path in REPORT_FILES.items():
        df = read_report(name, path)
        if not df.empty:
            reports[name] = df
            all_reports.append(df)

    if all_reports:
        combined = pd.concat(all_reports, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return reports, combined

def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def estimate_violations(df):
    if df.empty or "status" not in df.columns:
        return 0

    keywords = [
        "violation", "wrong", "helmet", "triple",
        "risk", "danger", "water", "plate", "read"
    ]

    count = 0

    for value in df["status"].astype(str):
        text = value.lower()
        if any(word in text for word in keywords):
            count += 1

    if count == 0:
        count = len(df)

    return count

def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.25)",
        font_color="white",
        margin=dict(l=20, r=20, t=55, b=30),
    )
    return fig

reports, all_df = load_data()

st.sidebar.title("🚦 IntelliTraffic AI")
st.sidebar.write("Traffic Violation Intelligence")

st.sidebar.markdown("---")

available_modules = list(reports.keys())

selected_modules = st.sidebar.multiselect(
    "Select modules",
    available_modules,
    default=available_modules
)

st.sidebar.markdown("---")
st.sidebar.subheader("Evidence Videos")

for name, path in VIDEO_FILES.items():
    if path.exists():
        st.sidebar.write("✅ " + name)
    else:
        st.sidebar.write("⚪ " + name)

st.markdown('<div class="main-title">IntelliTraffic AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered traffic violation detection, evidence generation, OCR, and smart city analytics.</div>',
    unsafe_allow_html=True
)

if all_df.empty:
    st.error("No CSV reports found inside the data folder.")
    st.info("Run your detector scripts first, then open this dashboard again.")
    st.stop()

if selected_modules:
    df = all_df[all_df["module"].isin(selected_modules)].copy()
else:
    df = all_df.copy()

total_records = len(df)
total_modules = df["module"].nunique()
total_violations = estimate_violations(df)

plate_reads = 0
if "plate_text" in df.columns:
    plate_reads = df["plate_text"].fillna("").astype(str).str.len().gt(3).sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("Total Evidence Records", total_records, "Generated CSV records")

with c2:
    metric_card("Active AI Modules", total_modules, "Connected detection systems")

with c3:
    metric_card("Violation Signals", total_violations, "Risk and violation events")

with c4:
    metric_card("Plate Reads", plate_reads, "OCR-based identification")

st.markdown('<div class="section-title">📌 Executive Summary</div>', unsafe_allow_html=True)

if total_violations >= 50:
    risk = "High"
    action = "Immediate enforcement and hotspot monitoring recommended."
elif total_violations >= 15:
    risk = "Medium"
    action = "Moderate risk detected. Deploy monitoring during peak hours."
else:
    risk = "Low"
    action = "System is stable. Continue periodic monitoring."

st.markdown(
    f"""
    <div class="card">
    <h3>City Traffic Risk Level: {risk}</h3>
    <p>
    IntelliTraffic AI processed traffic modules including night vision, pedestrian safety,
    wrong-side driving, helmet compliance, triple riding, waterlogging detection, and license plate OCR.
    </p>
    <p><b>Recommended Action:</b> {action}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section-title">📊 Analytics</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

module_counts = df.groupby("module").size().reset_index(name="records")

with chart_col1:
    fig = px.bar(
        module_counts.sort_values("records"),
        x="records",
        y="module",
        orientation="h",
        text="records",
        title="Evidence Records by Module"
    )
    st.plotly_chart(style_plot(fig), use_container_width=True)

with chart_col2:
    status_counts = df.groupby("status").size().reset_index(name="count")
    status_counts = status_counts.sort_values("count", ascending=False).head(10)

    fig = px.bar(
        status_counts,
        x="status",
        y="count",
        text="count",
        title="Top Detection Status"
    )
    st.plotly_chart(style_plot(fig), use_container_width=True)

st.markdown('<div class="section-title">🧠 Evidence Center</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Reports",
    "🎥 Videos",
    "🔍 License Plate OCR",
    "🖼 Plate Evidence",
    "🤖 Recommendation"
])

with tab1:
    st.subheader("Module Reports")

    for name, report_df in reports.items():
        if selected_modules and name not in selected_modules:
            continue

        with st.expander(f"{name} - {len(report_df)} records"):
            st.dataframe(report_df, use_container_width=True, height=300)

    st.subheader("Combined Report")
    st.dataframe(df, use_container_width=True, height=420)

with tab2:
    available_videos = {}

    for name, path in VIDEO_FILES.items():
        if path.exists():
            available_videos[name] = path

    if not available_videos:
        st.warning("No processed videos found inside outputs folder.")
    else:
        selected_video = st.selectbox("Select video evidence", list(available_videos.keys()))
        st.video(str(available_videos[selected_video]))

with tab3:
    ocr_frames = []

    for name in ["License Plate OCR", "Improved License Plate OCR"]:
        if name in reports:
            ocr_frames.append(reports[name])

    if not ocr_frames:
        st.warning("No OCR reports found.")
    else:
        ocr_df = pd.concat(ocr_frames, ignore_index=True)
        st.dataframe(ocr_df, use_container_width=True, height=350)

        if "plate_text" in ocr_df.columns:
            readable = ocr_df[ocr_df["plate_text"].fillna("").astype(str).str.len() > 3]

            if not readable.empty:
                plate_counts = readable.groupby("plate_text").size().reset_index(name="detections")
                plate_counts = plate_counts.sort_values("detections", ascending=False).head(10)

                fig = px.bar(
                    plate_counts,
                    x="plate_text",
                    y="detections",
                    text="detections",
                    title="Most Detected Plates"
                )
                st.plotly_chart(style_plot(fig), use_container_width=True)

with tab4:
    if not PLATE_DIR.exists():
        st.warning("No plate evidence folder found.")
    else:
        images = list(PLATE_DIR.glob("*.jpg")) + list(PLATE_DIR.glob("*.png"))

        if not images:
            st.warning("No plate crop images found yet.")
        else:
            st.success(f"{len(images)} plate crop images found.")
            cols = st.columns(4)

            for i, image_path in enumerate(images[:24]):
                with cols[i % 4]:
                    st.image(str(image_path), caption=image_path.name, use_container_width=True)

with tab5:
    recommendation_file = DATA_DIR / "ai_recommendations.txt"

    if recommendation_file.exists():
        st.text(recommendation_file.read_text(encoding="utf-8"))
    else:
        st.markdown(
            """
            <div class="card">
            <h3>Automated Recommendation</h3>
            <ul>
                <li>Deploy police near wrong-side driving hotspots.</li>
                <li>Use ANPR evidence for repeat offender tracking.</li>
                <li>Increase helmet compliance monitoring near two-wheeler routes.</li>
                <li>Alert municipal teams for waterlogging-prone roads.</li>
                <li>Use dashboard analytics for smart traffic decisions.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("IntelliTraffic AI | YOLOv8 + OpenCV + EasyOCR + Streamlit")

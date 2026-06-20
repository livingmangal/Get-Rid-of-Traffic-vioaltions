from pathlib import Path
from datetime import datetime
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

REPORT_FILES = {
    "Helmet Compliance": DATA_DIR / "helmet_report.csv",
    "Triple Riding": DATA_DIR / "triple_riding_report.csv",
    "Wrong Side Driving": DATA_DIR / "wrong_side_report.csv",
    "Waterlogging": DATA_DIR / "waterlogging_report.csv",
    "Pedestrian Safety": DATA_DIR / "pedestrian_report.csv",
    "License Plate OCR": DATA_DIR / "license_plate_ocr_report.csv",
    "Improved License Plate OCR": DATA_DIR / "license_plate_ocr_improved_report.csv",
}

OUTPUT_CSV = DATA_DIR / "smart_violations.csv"

SEVERITY_MAP = {
    "HELMET": 2,
    "TRIPLE_RIDING": 3,
    "WRONG_SIDE": 4,
    "RED_LIGHT": 5,
    "WATERLOGGING": 3,
    "PEDESTRIAN_RISK": 4,
    "PLATE_READ": 1,
    "UNKNOWN": 1,
}

FINE_MAP = {
    "HELMET": 500,
    "TRIPLE_RIDING": 1000,
    "WRONG_SIDE": 2000,
    "RED_LIGHT": 2000,
    "WATERLOGGING": 0,
    "PEDESTRIAN_RISK": 1500,
    "PLATE_READ": 0,
    "UNKNOWN": 0,
}

HOTSPOTS = [
    {
        "name": "Silk Board Junction",
        "lat": 12.9177,
        "lon": 77.6238,
    },
    {
        "name": "Marathahalli Bridge",
        "lat": 12.9569,
        "lon": 77.7011,
    },
    {
        "name": "Hebbal Flyover",
        "lat": 13.0358,
        "lon": 77.5970,
    },
    {
        "name": "KR Puram Junction",
        "lat": 13.0005,
        "lon": 77.6754,
    },
    {
        "name": "Electronic City Toll",
        "lat": 12.8452,
        "lon": 77.6602,
    },
    {
        "name": "MG Road Signal",
        "lat": 12.9756,
        "lon": 77.6068,
    },
]


def clean_plate(text):
    if pd.isna(text):
        return ""

    text = str(text).upper()
    allowed = []

    for ch in text:
        if ch.isalnum():
            allowed.append(ch)

    return "".join(allowed)


def detect_violation_type(module_name, row):
    combined_text = " ".join([str(x) for x in row.values]).lower()
    module_text = module_name.lower()

    if "helmet" in module_text or "helmet" in combined_text:
        return "HELMET"

    if "triple" in module_text or "triple" in combined_text:
        return "TRIPLE_RIDING"

    if "wrong" in module_text or "wrong" in combined_text:
        return "WRONG_SIDE"

    if "red" in combined_text and "light" in combined_text:
        return "RED_LIGHT"

    if "water" in module_text or "water" in combined_text:
        return "WATERLOGGING"

    if "pedestrian" in module_text or "pedestrian" in combined_text:
        return "PEDESTRIAN_RISK"

    if "plate" in module_text or "plate_text" in row.index:
        return "PLATE_READ"

    return "UNKNOWN"


def get_timestamp(row):
    for col in ["timestamp", "time", "date_time"]:
        if col in row.index and not pd.isna(row[col]):
            return str(row[col])

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_vehicle_number(row, fallback_index):
    plate_cols = [
        "plate_text",
        "vehicle_number",
        "number_plate",
        "plate_number",
        "registration_number",
    ]

    for col in plate_cols:
        if col in row.index:
            plate = clean_plate(row[col])
            if len(plate) >= 4:
                return plate

    demo_numbers = [
        "GJ01AB1234",
        "KA05MN9087",
        "DL08CA4421",
        "BR01AB7777",
        "MH12XY9090",
        "RJ14CD2026",
    ]

    return demo_numbers[fallback_index % len(demo_numbers)]


def get_confidence(row):
    for col in ["confidence", "ocr_confidence", "conf", "score"]:
        if col in row.index:
            try:
                return round(float(row[col]), 2)
            except Exception:
                pass

    return 0.85


def build_alert_message(vehicle_number, violation_type, fine_amount, hotspot):
    if fine_amount > 0:
        return (
            f"Dear Owner, your vehicle {vehicle_number} was detected violating "
            f"traffic rules at {hotspot}. Violation: {violation_type}. "
            f"Demo fine amount: ₹{fine_amount}. Please follow traffic rules."
        )

    return (
        f"Traffic Alert: vehicle {vehicle_number} was detected at {hotspot}. "
        f"Event type: {violation_type}. Please drive safely."
    )


def main():
    smart_rows = []
    evidence_counter = 1

    for module_name, path in REPORT_FILES.items():
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if df.empty:
            continue

        for index, row in df.iterrows():
            violation_type = detect_violation_type(module_name, row)
            severity_score = SEVERITY_MAP.get(violation_type, 1)
            fine_amount = FINE_MAP.get(violation_type, 0)

            vehicle_number = get_vehicle_number(row, evidence_counter)
            confidence = get_confidence(row)
            timestamp = get_timestamp(row)

            hotspot = HOTSPOTS[evidence_counter % len(HOTSPOTS)]

            evidence_id = f"EVT_{evidence_counter:05d}"

            alert_message = build_alert_message(
                vehicle_number,
                violation_type,
                fine_amount,
                hotspot["name"],
            )

            smart_rows.append(
                {
                    "evidence_id": evidence_id,
                    "timestamp": timestamp,
                    "module": module_name,
                    "violation_type": violation_type,
                    "vehicle_number": vehicle_number,
                    "severity_score": severity_score,
                    "fine_amount": fine_amount,
                    "hotspot": hotspot["name"],
                    "latitude": hotspot["lat"],
                    "longitude": hotspot["lon"],
                    "confidence": confidence,
                    "alert_message": alert_message,
                    "source_file": path.name,
                    "alert_status": "DEMO_READY",
                }
            )

            evidence_counter += 1

    if not smart_rows:
        print("No existing reports found.")
        print("Run detector files first, then run this again.")
        return

    smart_df = pd.DataFrame(smart_rows)
    smart_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("Smart feature generation completed.")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total smart records: {len(smart_df)}")
    print(f"Total demo fine amount: ₹{smart_df['fine_amount'].sum()}")


if __name__ == "__main__":
    main()

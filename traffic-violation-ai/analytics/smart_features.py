import os
import pandas as pd
from datetime import datetime
from config.settings import DATA_DIR, SMART_CSV
from utils.helpers import evidence_id

class SmartViolationProcessor:
    def __init__(self):
        self.reports = {
            "Helmet": DATA_DIR / "helmet_report.csv",
            "Triple Riding": DATA_DIR / "triple_riding_report.csv",
            "Wrong Side": DATA_DIR / "wrong_side_report.csv",
            "Pedestrian": DATA_DIR / "pedestrian_report.csv",
            "Waterlogging": DATA_DIR / "waterlogging_report.csv",
            "License Plate": DATA_DIR / "license_plate_ocr_improved_report.csv"
        }

    def run(self):
        print("Running Smart Violation Processor...")
        all_violations = []

        for module, path in self.reports.items():
            if not path.exists():
                continue
                
            df = pd.read_csv(path)
            
            # Simple conversion of raw reports into smart violations
            if module == "Wrong Side" and "evidence_id" in df.columns:
                for _, row in df.iterrows():
                    if row.get("status") == "WRONG_DIRECTION_TOWARDS_CAMERA":
                        all_violations.append({
                            "evidence_id": row.get("evidence_id"),
                            "timestamp": row.get("timestamp"),
                            "module": module,
                            "violation_type": "Wrong-Side Driving",
                            "vehicle_number": "UNKNOWN",
                            "severity_score": row.get("severity_score", 4),
                            "fine_amount": 2000,
                            "hotspot": "Flyover Road",
                            "latitude": 28.6139,
                            "longitude": 77.2090,
                            "confidence": row.get("confidence", 0.85),
                            "alert_message": f"Demo Alert: Wrong-side driving detected on Flyover Road. Fine: Rs 2000."
                        })
            elif module == "Helmet" and "Violations Detected" in df.columns:
                for _, row in df.iterrows():
                    count = row.get("Violations Detected", 0)
                    if count > 0:
                        all_violations.append({
                            "evidence_id": evidence_id("HLMT"),
                            "timestamp": row.get("Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "module": module,
                            "violation_type": "No Helmet",
                            "vehicle_number": "UNKNOWN",
                            "severity_score": 3,
                            "fine_amount": 1000,
                            "hotspot": "Traffic Road",
                            "latitude": 28.6140,
                            "longitude": 77.2095,
                            "confidence": 0.82,
                            "alert_message": f"Demo Alert: No-helmet violation detected. Fine: Rs 1000."
                        })
            elif module == "Triple Riding" and "Triple Riding Violations" in df.columns:
                for _, row in df.iterrows():
                    count = row.get("Triple Riding Violations", 0)
                    if count > 0:
                        all_violations.append({
                            "evidence_id": evidence_id("TRPL"),
                            "timestamp": row.get("Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "module": module,
                            "violation_type": "Triple Riding",
                            "vehicle_number": "UNKNOWN",
                            "severity_score": 5,
                            "fine_amount": 1500,
                            "hotspot": "Urban Road",
                            "latitude": 28.6145,
                            "longitude": 77.2100,
                            "confidence": 0.78,
                            "alert_message": f"Demo Alert: Triple riding violation detected. Fine: Rs 1500."
                        })
            elif module == "Waterlogging" and "Final Status" in df.columns:
                for _, row in df.iterrows():
                    if "ALERT" in str(row.get("Final Status", "")):
                        all_violations.append({
                            "evidence_id": evidence_id("WTR"),
                            "timestamp": row.get("Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "module": module,
                            "violation_type": "Waterlogging Risk",
                            "vehicle_number": "N/A",
                            "severity_score": 4,
                            "fine_amount": 0,
                            "hotspot": "Rain-Affected Road",
                            "latitude": 28.6150,
                            "longitude": 77.2110,
                            "confidence": 0.74,
                            "alert_message": f"Demo Alert: Waterlogging risk detected. Reroute traffic."
                        })
            elif module == "License Plate" and "evidence_id" in df.columns:
                for _, row in df.iterrows():
                    all_violations.append({
                        "evidence_id": row.get("evidence_id"),
                        "timestamp": row.get("timestamp"),
                        "module": module,
                        "violation_type": "License Plate Read",
                        "vehicle_number": row.get("plate_text", "UNKNOWN"),
                        "severity_score": 1,
                        "fine_amount": 0,
                        "hotspot": "Checkpoint A",
                        "latitude": 28.6155,
                        "longitude": 77.2120,
                        "confidence": row.get("ocr_confidence", 0.85),
                        "alert_message": f"Demo Alert: Vehicle {row.get('plate_text')} recorded."
                    })

        if all_violations:
            df_smart = pd.DataFrame(all_violations)
            df_smart.to_csv(SMART_CSV, index=False)
            print(f"Generated {len(all_violations)} smart violation records -> {SMART_CSV.name}")
        else:
            print("No violations found to process.")

if __name__ == "__main__":
    processor = SmartViolationProcessor()
    processor.run()

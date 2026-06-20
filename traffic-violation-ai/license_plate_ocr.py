import os
import cv2
import csv
import re
import easyocr
from datetime import datetime
from ultralytics import YOLO

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

VIDEO_PATH = "videos/night.mp4.mp4"
OUTPUT_PATH = "outputs/license_plate_ocr.mp4"
CSV_PATH = "data/license_plate_ocr_report.csv"

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(["en"], gpu=False)

vehicle_classes = ["car", "truck", "bus", "motorcycle"]

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video not found:", VIDEO_PATH)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

evidence_id = "LP-" + datetime.now().strftime("%Y%m%d-%H%M%S")

frame_no = 0
vehicles_detected = 0
plates_detected = 0
plate_records = []

def clean_plate(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_no += 1

    # Process OCR only every 10th frame for speed
    if frame_no % 10 != 0:
        out.write(frame)
        continue

    enhanced = cv2.convertScaleAbs(frame, alpha=1.6, beta=40)

    results = model(enhanced, conf=0.35, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name not in vehicle_classes:
                continue

            vehicles_detected += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            vehicle_h = y2 - y1
            vehicle_w = x2 - x1

            if vehicle_h <= 0 or vehicle_w <= 0:
                continue

            # Approximate number plate area: lower-middle vehicle region
            px1 = x1 + int(vehicle_w * 0.25)
            px2 = x1 + int(vehicle_w * 0.75)
            py1 = y1 + int(vehicle_h * 0.65)
            py2 = y1 + int(vehicle_h * 0.90)

            px1 = max(0, px1)
            py1 = max(0, py1)
            px2 = min(width, px2)
            py2 = min(height, py2)

            plate_roi = enhanced[py1:py2, px1:px2]

            plate_text = ""
            ocr_conf = 0.0

            if plate_roi.size > 0:
                gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)

                ocr_results = reader.readtext(gray)

                if len(ocr_results) > 0:
                    best = max(ocr_results, key=lambda x: x[2])
                    raw_text = best[1]
                    ocr_conf = float(best[2])
                    plate_text = clean_plate(raw_text)

            cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                enhanced,
                f"{cls_name} {conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.rectangle(enhanced, (px1, py1), (px2, py2), (0, 0, 255), 2)

            if plate_text and len(plate_text) >= 4:
                plates_detected += 1

                plate_records.append([
                    evidence_id,
                    datetime.now(),
                    cls_name,
                    plate_text,
                    round(ocr_conf, 2),
                    frame_no
                ])

                cv2.putText(
                    enhanced,
                    f"Plate: {plate_text} ({ocr_conf:.2f})",
                    (px1, max(py1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    enhanced,
                    "Plate ROI",
                    (px1, max(py1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    cv2.rectangle(enhanced, (20, 20), (760, 150), (0, 0, 0), -1)

    cv2.putText(
        enhanced,
        "AI License Plate OCR - Night Mode",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        enhanced,
        f"Evidence ID: {evidence_id}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        enhanced,
        f"Vehicles: {vehicles_detected} | Plates Read: {plates_detected}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    out.write(enhanced)

    if frame_no % 100 == 0:
        print(f"Processed {frame_no} frames...")

cap.release()
out.release()

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Evidence ID",
        "Timestamp",
        "Vehicle Type",
        "Plate Text",
        "OCR Confidence",
        "Frame Number"
    ])

    if plate_records:
        writer.writerows(plate_records)
    else:
        writer.writerow([
            evidence_id,
            datetime.now(),
            "N/A",
            "No reliable plate read",
            0.0,
            "N/A"
        ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
print("Vehicles Detected:", vehicles_detected)
print("Plates Read:", plates_detected)
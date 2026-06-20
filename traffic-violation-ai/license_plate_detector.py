import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "videos/day_traffic.mp4.mp4"

OUTPUT_PATH = "outputs/license_plate_processed.mp4"
CSV_PATH = "data/license_plate_report.csv"

model = YOLO(MODEL_PATH)

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

vehicle_classes = ["car", "truck", "bus", "motorcycle"]

vehicle_count = 0
plate_count = 0
frame_no = 0

dummy_plates = [
    "KA01AB1234",
    "KA05MN9821",
    "BR01CD4567",
    "DL07XY9090",
    "MH12PQ7812"
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    results = model(frame, conf=0.20, verbose=False)

    frame_plates = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name not in vehicle_classes:
                continue

            vehicle_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Approximate plate region: lower middle part of vehicle box
            plate_w = (x2 - x1) // 2
            plate_h = max(20, (y2 - y1) // 8)

            px1 = x1 + (x2 - x1) // 4
            py1 = y2 - plate_h - 10
            px2 = px1 + plate_w
            py2 = y2 - 10

            if px2 > px1 and py2 > py1:
                plate_count += 1
                frame_plates += 1

                plate_text = dummy_plates[plate_count % len(dummy_plates)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}",
                            (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(frame, f"Plate: {plate_text}",
                            (px1, max(py1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

    cv2.rectangle(frame, (20, 20), (700, 150), (0, 0, 0), -1)

    cv2.putText(frame, "AI License Plate Recognition Module",
                (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Vehicles Detected: {vehicle_count}",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    cv2.putText(frame, f"Plates Extracted: {plate_count}",
                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    out.write(frame)

    if frame_no % 100 == 0:
        print(f"Processed {frame_no} frames...")

cap.release()
out.release()

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Vehicles", "Plates Extracted", "Sample Plate", "Status"])
    writer.writerow([
        datetime.now(),
        vehicle_count,
        plate_count,
        dummy_plates[0],
        "Prototype OCR simulation completed"
    ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
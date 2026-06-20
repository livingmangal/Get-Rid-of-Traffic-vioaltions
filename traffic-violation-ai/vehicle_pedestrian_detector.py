import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "videos/pedestrian.mp4"

OUTPUT_PATH = "outputs/pedestrian_processed.mp4"
CSV_PATH = "data/pedestrian_report.csv"

model = YOLO(MODEL_PATH)

counts = {
    "person": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0
}

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

frame_no = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_no += 1

    results = model(frame, conf=0.20, verbose=False)

    frame_counts = {key: 0 for key in counts}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name not in counts:
                continue

            frame_counts[cls_name] += 1
            counts[cls_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "person":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    total_vehicles = (
        frame_counts["car"]
        + frame_counts["truck"]
        + frame_counts["bus"]
        + frame_counts["motorcycle"]
    )

    pedestrians = frame_counts["person"]

    if pedestrians == 0:
        pedestrian_safety = "SAFE"
    elif pedestrians <= 5 and total_vehicles <= 5:
        pedestrian_safety = "LOW RISK"
    elif pedestrians <= 10:
        pedestrian_safety = "MODERATE RISK"
    else:
        pedestrian_safety = "HIGH RISK"

    if total_vehicles < 5:
        traffic_density = "LOW"
    elif total_vehicles < 15:
        traffic_density = "MEDIUM"
    else:
        traffic_density = "HIGH"

    cv2.rectangle(frame, (20, 20), (620, 230), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "AI Traffic Monitoring - Pedestrian Safety",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {frame_counts['car']} | Bikes: {frame_counts['motorcycle']}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {frame_counts['bus']} | Trucks: {frame_counts['truck']}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Pedestrians: {pedestrians}",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Density: {traffic_density}",
        (30, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Pedestrian Safety: {pedestrian_safety}",
        (30, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    out.write(frame)

    if frame_no % 100 == 0:
        print(f"Processed {frame_no} frames...")

cap.release()
out.release()

total_vehicles = (
    counts["car"]
    + counts["truck"]
    + counts["bus"]
    + counts["motorcycle"]
)

if counts["person"] == 0:
    final_safety = "SAFE"
elif counts["person"] < 100:
    final_safety = "LOW RISK"
elif counts["person"] < 500:
    final_safety = "MODERATE RISK"
else:
    final_safety = "HIGH RISK"

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time",
        "Cars",
        "Motorcycles",
        "Buses",
        "Trucks",
        "Pedestrians",
        "Total Vehicle Detections",
        "Pedestrian Safety"
    ])

    writer.writerow([
        datetime.now(),
        counts["car"],
        counts["motorcycle"],
        counts["bus"],
        counts["truck"],
        counts["person"],
        total_vehicles,
        final_safety
    ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
print("Counts:", counts)
print("Pedestrian Safety:", final_safety)
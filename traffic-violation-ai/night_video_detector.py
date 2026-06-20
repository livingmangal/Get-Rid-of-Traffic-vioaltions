import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

# ---------- folders ----------
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------- paths ----------
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "videos/night.mp4.mp4"
OUTPUT_PATH = "outputs/night_processed.mp4"
CSV_PATH = "data/traffic_report.csv"

# ---------- model ----------
model = YOLO(MODEL_PATH)

counts = {
    "person": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0
}

# ---------- video ----------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video not found:", VIDEO_PATH)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

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

    # ---------- night enhancement ----------
    enhanced = cv2.convertScaleAbs(frame, alpha=2.2, beta=70)

    # ---------- YOLO detection ----------
    results = model(enhanced, conf=0.15, verbose=False)

    frame_counts = {
        "person": 0,
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorcycle": 0
    }

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

    total_vehicles = (
        frame_counts["car"]
        + frame_counts["truck"]
        + frame_counts["bus"]
        + frame_counts["motorcycle"]
    )

    if total_vehicles < 5:
        density = "LOW"
    elif total_vehicles < 15:
        density = "MEDIUM"
    else:
        density = "HIGH"

    # ---------- overlay ----------
    cv2.rectangle(enhanced, (20, 20), (520, 190), (0, 0, 0), -1)

    cv2.putText(enhanced, "AI Traffic Monitoring - Night Mode", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(enhanced, f"Cars: {frame_counts['car']}  Bikes: {frame_counts['motorcycle']}", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(enhanced, f"Buses: {frame_counts['bus']}  Trucks: {frame_counts['truck']}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(enhanced, f"Pedestrians: {frame_counts['person']}", (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(enhanced, f"Traffic Density: {density}", (30, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(enhanced)

    if frame_no % 100 == 0:
        print(f"Processed {frame_no} frames...")

cap.release()
out.release()

final_total_vehicles = (
    counts["car"] + counts["truck"] + counts["bus"] + counts["motorcycle"]
)

if final_total_vehicles < 100:
    final_density = "LOW"
elif final_total_vehicles < 500:
    final_density = "MEDIUM"
else:
    final_density = "HIGH"

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
        "Final Traffic Density"
    ])
    writer.writerow([
        datetime.now(),
        counts["car"],
        counts["motorcycle"],
        counts["bus"],
        counts["truck"],
        counts["person"],
        final_total_vehicles,
        final_density
    ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
print("Counts:", counts)
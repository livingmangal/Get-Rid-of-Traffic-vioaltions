import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "videos/wrong_direction.mp4.mp4"

OUTPUT_PATH = "outputs/wrong_side_processed.mp4"
CSV_PATH = "data/wrong_side_report.csv"

model = YOLO(MODEL_PATH)

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

vehicle_count = 0
wrong_side_count = 0
frame_no = 0

# For prototype:
# Vehicles on LEFT half moving/appearing in opposite lane are marked suspicious.
# Adjust this line position according to your video angle.
lane_middle_x = width // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    results = model(frame, conf=0.20, verbose=False)

    frame_vehicle_count = 0
    frame_wrong_count = 0

    # Draw lane divider
    cv2.line(frame, (lane_middle_x, 0), (lane_middle_x, height), (255, 0, 0), 3)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2

            frame_vehicle_count += 1
            vehicle_count += 1

            # Prototype rule:
            # If vehicle center is on restricted/opposite side, mark as wrong direction.
            is_wrong_side = center_x < lane_middle_x

            if is_wrong_side:
                frame_wrong_count += 1
                wrong_side_count += 1
                color = (0, 0, 255)
                label = f"WRONG SIDE {conf:.2f}"
            else:
                color = (0, 255, 0)
                label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (center_x, (y1 + y2) // 2), 5, color, -1)

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    if frame_wrong_count == 0:
        status = "NORMAL"
    elif frame_wrong_count <= 2:
        status = "WARNING"
    else:
        status = "HIGH RISK"

    cv2.rectangle(frame, (20, 20), (650, 180), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "AI Traffic Monitoring - Wrong Side Detection",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles Detected: {frame_vehicle_count}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Wrong Side Violations: {frame_wrong_count}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Risk Level: {status}",
        (30, 160),
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

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time",
        "Total Vehicle Detections",
        "Wrong Side Violations",
        "Status",
        "Video"
    ])
    writer.writerow([
        datetime.now(),
        vehicle_count,
        wrong_side_count,
        "HIGH RISK" if wrong_side_count > 20 else "WARNING" if wrong_side_count > 0 else "NORMAL",
        VIDEO_PATH
    ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
print("Total Vehicles:", vehicle_count)
print("Wrong Side Violations:", wrong_side_count)
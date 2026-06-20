import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "videos/tripleriding.mp4"

OUTPUT_PATH = "outputs/triple_riding_processed.mp4"
CSV_PATH = "data/triple_riding_report.csv"

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

total_motorcycles = 0
total_no_helmet = 0
total_triple_riding = 0
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    results = model(frame, conf=0.05, verbose=False)

    frame_motorcycles = 0
    frame_no_helmet = 0
    frame_triple = 0

    for result in results:
        motorcycles = []
        persons = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "motorcycle":
                motorcycles.append((x1, y1, x2, y2, conf))
            elif cls_name == "person":
                persons.append((x1, y1, x2, y2, conf))

        frame_motorcycles = len(motorcycles)

        for mx1, my1, mx2, my2, mconf in motorcycles:
            riders_near_bike = 0

            for px1, py1, px2, py2, pconf in persons:
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2

                if mx1 - 100 <= pcx <= mx2 + 100 and my1 - 220 <= pcy <= my2 + 100:
                    riders_near_bike += 1
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
                    cv2.putText(frame, "Rider", (px1, max(py1 - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            total_motorcycles += 1

            no_helmet = True
            triple_riding = riders_near_bike >= 3

            safety_score = 100
            violations = []

            if no_helmet:
                frame_no_helmet += 1
                total_no_helmet += 1
                safety_score -= 40
                violations.append("NO HELMET")

            if triple_riding:
                frame_triple += 1
                total_triple_riding += 1
                safety_score -= 50
                violations.append("TRIPLE RIDING")

            if safety_score <= 20:
                risk = "CRITICAL"
            elif safety_score <= 50:
                risk = "HIGH"
            elif safety_score <= 75:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            color = (0, 0, 255) if violations else (0, 255, 0)
            label = " + ".join(violations) if violations else "SAFE MOTORCYCLE"

            cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, 3)
            cv2.putText(frame, label, (mx1, max(my1 - 12, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.putText(frame, f"Riders: {riders_near_bike} | Risk: {risk}",
                        (mx1, min(my2 + 25, height - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if frame_triple > 0:
        frame_status = "CRITICAL"
    elif frame_no_helmet > 0:
        frame_status = "HIGH RISK"
    else:
        frame_status = "NORMAL"

    cv2.rectangle(frame, (20, 20), (760, 250), (0, 0, 0), -1)

    cv2.putText(frame, "AI Motorcycle Safety Monitoring",
                (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.putText(frame, f"Motorcycles: {frame_motorcycles}",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.putText(frame, f"No Helmet Violations: {frame_no_helmet}",
                (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, f"Triple Riding Violations: {frame_triple}",
                (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, f"Safety Status: {frame_status}",
                (30, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    out.write(frame)

    if frame_no % 100 == 0:
        print(f"Processed {frame_no} frames...")

cap.release()
out.release()

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time",
        "Motorcycle Detections",
        "No Helmet Violations",
        "Triple Riding Violations",
        "Risk Status",
        "Safety Recommendation"
    ])

    final_status = "NORMAL"
    if total_no_helmet > 0:
        final_status = "HIGH RISK"
    if total_triple_riding > 0:
        final_status = "CRITICAL"

    writer.writerow([
        datetime.now(),
        total_motorcycles,
        total_no_helmet,
        total_triple_riding,
        final_status,
        "Wear helmet, avoid triple riding, follow traffic rules"
    ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
print("Motorcycles:", total_motorcycles)
print("No Helmet:", total_no_helmet)
print("Triple Riding:", total_triple_riding)
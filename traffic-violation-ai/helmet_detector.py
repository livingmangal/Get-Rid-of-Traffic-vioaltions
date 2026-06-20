import os
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "videos/no_helmet.mp4.mp4"

OUTPUT_PATH = "outputs/no_helmet_processed.mp4"
CSV_PATH = "data/helmet_report.csv"

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

evidence_id = "NH-" + datetime.now().strftime("%Y%m%d-%H%M%S")

total_motorcycles = 0
helmet_violations = 0
max_confidence = 0.0
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    results = model(frame, conf=0.20, verbose=False)

    frame_motorcycles = 0
    frame_persons = 0
    frame_violations = 0
    frame_confidence = 0.0

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
        frame_persons = len(persons)

        for mx1, my1, mx2, my2, mconf in motorcycles:
            total_motorcycles += 1

            rider_found = False
            rider_conf = 0.0

            for px1, py1, px2, py2, pconf in persons:
                person_center_x = (px1 + px2) // 2
                person_center_y = (py1 + py2) // 2

                if (
                    mx1 - 80 <= person_center_x <= mx2 + 80
                    and my1 - 180 <= person_center_y <= my2 + 80
                ):
                    rider_found = True
                    rider_conf = pconf

                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Rider {pconf:.2f}",
                        (px1, max(py1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                    break

            if not rider_found:
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Motorcycle {mconf:.2f}",
                    (mx1, max(my1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                continue

            violation_confidence = min(0.95, max(0.75, (mconf + rider_conf) / 2))

            frame_confidence = max(frame_confidence, violation_confidence)
            max_confidence = max(max_confidence, violation_confidence)

            frame_violations += 1
            helmet_violations += 1

            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 3)

            cv2.putText(
                frame,
                f"POSSIBLE NO_HELMET {violation_confidence:.2f}",
                (mx1, max(my1 - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Safety: Wear ISI helmet",
                (mx1, min(my2 + 25, height - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    if frame_violations == 0:
        risk = "NORMAL"
        safety_score = 100
    elif frame_violations <= 2:
        risk = "WARNING"
        safety_score = 70
    elif frame_violations <= 5:
        risk = "HIGH"
        safety_score = 45
    else:
        risk = "CRITICAL"
        safety_score = 20

    cv2.rectangle(frame, (20, 20), (800, 235), (0, 0, 0), -1)

    cv2.putText(frame, "AI Traffic Monitoring - Helmet Compliance",
                (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Evidence ID: {evidence_id}",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Motorcycles: {frame_motorcycles} | Riders: {frame_persons}",
                (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    cv2.putText(frame, f"Violation: POSSIBLE_NO_HELMET | Count: {frame_violations} | Conf: {frame_confidence:.2f}",
                (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 0, 255), 2)

    cv2.putText(frame, f"Risk: {risk} | Safety Score: {safety_score}/100",
                (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

    cv2.putText(frame, "Action: Warning/e-challan + helmet awareness",
                (30, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    out.write(frame)

    if frame_no % 100 == 0:
        print(f"Processed {frame_no} frames...")

cap.release()
out.release()

if helmet_violations == 0:
    final_risk = "NORMAL"
    final_score = 100
elif helmet_violations <= 10:
    final_risk = "WARNING"
    final_score = 70
elif helmet_violations <= 50:
    final_risk = "HIGH"
    final_score = 45
else:
    final_risk = "CRITICAL"
    final_score = 20

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Evidence ID",
        "Timestamp",
        "Violation Type",
        "Motorcycle Detections",
        "Violation Count",
        "Confidence",
        "Risk Level",
        "Safety Score",
        "Recommended Action",
        "Video"
    ])

    writer.writerow([
        evidence_id,
        datetime.now(),
        "POSSIBLE_NO_HELMET",
        total_motorcycles,
        helmet_violations,
        round(max_confidence, 2),
        final_risk,
        final_score,
        "Issue warning/e-challan and increase helmet checking",
        VIDEO_PATH
    ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
print("Evidence ID:", evidence_id)
print("Motorcycles:", total_motorcycles)
print("Possible No Helmet Violations:", helmet_violations)
print("Risk:", final_risk)
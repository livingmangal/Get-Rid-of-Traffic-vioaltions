import os
import cv2
import csv
import numpy as np
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

VIDEO_PATH = "videos/waterlogging.mp4"
OUTPUT_PATH = "outputs/waterlogging_processed.mp4"
CSV_PATH = "data/waterlogging_report.csv"

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
water_alert_frames = 0
max_water_percent = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    # Resize-safe road ROI: bottom half of frame
    roi_y1 = height // 2
    road_roi = frame[roi_y1:height, 0:width]

    # Convert to HSV for water-like surface detection
    hsv = cv2.cvtColor(road_roi, cv2.COLOR_BGR2HSV)

    # Waterlogged areas often look gray/blue/low saturation with reflection
    lower_water = np.array([0, 0, 70])
    upper_water = np.array([180, 80, 255])

    mask = cv2.inRange(hsv, lower_water, upper_water)

    # Clean mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    water_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    water_percent = (water_pixels / total_pixels) * 100

    max_water_percent = max(max_water_percent, water_percent)

    if water_percent < 15:
        risk = "LOW"
        alert = "NORMAL ROAD"
        color = (0, 255, 0)
    elif water_percent < 35:
        risk = "MEDIUM"
        alert = "POSSIBLE WATERLOGGING"
        color = (0, 255, 255)
        water_alert_frames += 1
    else:
        risk = "HIGH"
        alert = "WATERLOGGING ALERT"
        color = (0, 0, 255)
        water_alert_frames += 1

    # Create colored overlay for detected water area
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colored_mask[np.where((colored_mask == [255, 255, 255]).all(axis=2))] = color

    overlay_roi = cv2.addWeighted(road_roi, 0.75, colored_mask, 0.25, 0)
    frame[roi_y1:height, 0:width] = overlay_roi

    # Draw ROI line
    cv2.line(frame, (0, roi_y1), (width, roi_y1), (255, 0, 0), 3)

    # Overlay panel
    cv2.rectangle(frame, (20, 20), (760, 210), (0, 0, 0), -1)

    cv2.putText(frame, "AI Road Safety - Waterlogging Detection",
                (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.putText(frame, f"Water Coverage: {water_percent:.2f}%",
                (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)

    cv2.putText(frame, f"Risk Level: {risk}",
                (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)

    cv2.putText(frame, f"Status: {alert}",
                (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)

    cv2.putText(frame, "Action: Reroute traffic / Alert municipal team",
                (30, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    out.write(frame)

    if frame_no % 100 == 0:
        print(f"Processed {frame_no} frames...")

cap.release()
out.release()

final_status = "NORMAL"
if max_water_percent >= 15:
    final_status = "POSSIBLE WATERLOGGING"
if max_water_percent >= 35:
    final_status = "WATERLOGGING ALERT"

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Time",
        "Video",
        "Total Frames",
        "Water Alert Frames",
        "Max Water Coverage %",
        "Final Status",
        "Recommended Action"
    ])
    writer.writerow([
        datetime.now(),
        VIDEO_PATH,
        frame_no,
        water_alert_frames,
        round(max_water_percent, 2),
        final_status,
        "Reroute traffic, alert municipal team, deploy water pump"
    ])

print("Done!")
print("Output:", OUTPUT_PATH)
print("Report:", CSV_PATH)
print("Max Water Coverage:", round(max_water_percent, 2), "%")
print("Final Status:", final_status)
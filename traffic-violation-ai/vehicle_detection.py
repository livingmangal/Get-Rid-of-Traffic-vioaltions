from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model(
    "videos/day_traffic.mp4.mp4",
    save=True
)

print("Video Detection Completed")
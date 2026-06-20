from ultralytics import YOLO

model = YOLO("../runs/detect/train-3/weights/best.pt")

results = model.predict(
    source="day_traffic.mp4.mp4",
    save=True,
    conf=0.4
)

print("Video processed!")
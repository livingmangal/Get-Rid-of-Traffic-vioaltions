import cv2
import numpy as np
import os
from pathlib import Path

VIDEO_DIR = Path("traffic-violation-ai/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

videos = [
    "no_helmet.mp4",
    "tripleriding.mp4",
    "wrong_direction.mp4",
    "night.mp4",
    "pedestrian.mp4",
    "waterlogging.mp4",
    "day_traffic.mp4"
]

print("Generating synthetic test videos...")

for video_name in videos:
    path = str(VIDEO_DIR / video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    
    for i in range(90):  # 3 seconds at 30 fps
        # Create a basic gray background
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        # Add some text and simple moving shapes so it looks like a video
        cv2.putText(frame, f"SYNTHETIC TEST VIDEO", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Module: {video_name}", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Moving block
        x = 100 + (i * 5) % 400
        cv2.rectangle(frame, (x, 200), (x+100, 300), (0, 0, 255), -1)
        
        out.write(frame)
    out.release()
    print(f"Created: {path}")

print("\nAll dummy videos generated successfully. You can now run the pipeline.")

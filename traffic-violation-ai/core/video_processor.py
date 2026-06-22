import cv2
import os
from ultralytics import YOLO
from config.settings import YOLO_MODEL

class VideoProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"Video not found: {video_path}")
            self.valid = False
            return
            
        self.valid = True
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.out = self.create_writer()

    def create_writer(self):
        return cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )

    def process(self, callback):
        if not self.valid:
            return
        frame_no = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_no += 1
            processed_frame = callback(frame, frame_no)
            if processed_frame is not None:
                self.out.write(processed_frame)
            if frame_no % 100 == 0:
                print(f"Processed {frame_no} frames...")
                
    def release(self):
        if self.valid:
            self.cap.release()
            self.out.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class YOLOProcessor(VideoProcessor):
    def __init__(self, video_path, output_path, model_path=YOLO_MODEL):
        super().__init__(video_path, output_path)
        self.model = YOLO(model_path) if self.valid else None
        
    def detect(self, frame, confidence=0.25):
        if self.model is None:
            return []
        return self.model(frame, conf=confidence, verbose=False)
        
    def get_detections_by_class(self, results):
        detections = {}
        if not results:
            return detections
            
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_name not in detections:
                detections[cls_name] = []
            detections[cls_name].append((x1, y1, x2, y2, conf))
            
        return detections

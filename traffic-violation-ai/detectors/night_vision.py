import cv2
import csv
from datetime import datetime
from config.settings import NightVisionConfig, VEHICLE_CLASSES
from core.video_processor import YOLOProcessor
from utils.helpers import classify_traffic_density

class NightVisionDetector(YOLOProcessor):
    def __init__(self):
        super().__init__(NightVisionConfig.video_path, NightVisionConfig.output_path)
        self.csv_path = NightVisionConfig.csv_path
        self.confidence_threshold = NightVisionConfig.confidence
        self.counts = {"person": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0}

    def process_frame(self, frame, frame_no):
        enhanced = cv2.convertScaleAbs(frame, alpha=2.2, beta=70)
        results = self.detect(enhanced, self.confidence_threshold)
        detections = self.get_detections_by_class(results)

        frame_counts = {"person": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        
        for cls_name in self.counts.keys():
            for x1, y1, x2, y2, conf in detections.get(cls_name, []):
                frame_counts[cls_name] += 1
                self.counts[cls_name] += 1
                cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(enhanced, f"{cls_name} {conf:.2f}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_vehicles = sum(frame_counts[c] for c in VEHICLE_CLASSES)
        density = classify_traffic_density(total_vehicles)

        cv2.rectangle(enhanced, (20, 20), (520, 190), (0, 0, 0), -1)
        cv2.putText(enhanced, "AI Traffic Monitoring - Night Mode", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(enhanced, f"Cars: {frame_counts['car']}  Bikes: {frame_counts['motorcycle']}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(enhanced, f"Buses: {frame_counts['bus']}  Trucks: {frame_counts['truck']}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(enhanced, f"Pedestrians: {frame_counts['person']}", (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(enhanced, f"Traffic Density: {density}", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return enhanced

    def run(self):
        print("Night vision detector started...")
        self.process(self.process_frame)
        print("\nNight vision detection completed.")
        
        final_total_vehicles = sum(self.counts[c] for c in VEHICLE_CLASSES)
        final_density = "LOW" if final_total_vehicles < 100 else "MEDIUM" if final_total_vehicles < 500 else "HIGH"

        with open(self.csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Cars", "Motorcycles", "Buses", "Trucks", "Pedestrians", "Total Vehicle Detections", "Final Traffic Density"])
            writer.writerow([datetime.now(), self.counts["car"], self.counts["motorcycle"], self.counts["bus"], self.counts["truck"], self.counts["person"], final_total_vehicles, final_density])

if __name__ == "__main__":
    detector = NightVisionDetector()
    detector.run()

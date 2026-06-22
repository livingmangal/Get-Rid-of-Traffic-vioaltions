import cv2
import csv
from datetime import datetime
from config.settings import VehiclePedestrianConfig, VEHICLE_CLASSES
from core.video_processor import YOLOProcessor
from utils.helpers import classify_traffic_density, classify_pedestrian_safety

class VehiclePedestrianDetector(YOLOProcessor):
    def __init__(self):
        super().__init__(VehiclePedestrianConfig.video_path, VehiclePedestrianConfig.output_path)
        self.csv_path = VehiclePedestrianConfig.csv_path
        self.confidence_threshold = VehiclePedestrianConfig.confidence
        self.counts = {"person": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0}

    def process_frame(self, frame, frame_no):
        results = self.detect(frame, self.confidence_threshold)
        detections = self.get_detections_by_class(results)

        frame_counts = {"person": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        
        for cls_name in self.counts.keys():
            for x1, y1, x2, y2, conf in detections.get(cls_name, []):
                frame_counts[cls_name] += 1
                self.counts[cls_name] += 1
                color = (0, 0, 255) if cls_name == "person" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total_vehicles = sum(frame_counts[c] for c in VEHICLE_CLASSES)
        pedestrians = frame_counts["person"]
        
        pedestrian_safety = classify_pedestrian_safety(pedestrians, total_vehicles)
        traffic_density = classify_traffic_density(total_vehicles)

        cv2.rectangle(frame, (20, 20), (620, 230), (0, 0, 0), -1)
        cv2.putText(frame, "AI Traffic Monitoring - Pedestrian Safety", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Cars: {frame_counts['car']} | Bikes: {frame_counts['motorcycle']}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Buses: {frame_counts['bus']} | Trucks: {frame_counts['truck']}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Pedestrians: {pedestrians}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Traffic Density: {traffic_density}", (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Pedestrian Safety: {pedestrian_safety}", (30, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run(self):
        print("Vehicle & Pedestrian detector started...")
        self.process(self.process_frame)
        print("\nVehicle & Pedestrian detection completed.")
        
        total_vehicles = sum(self.counts[c] for c in VEHICLE_CLASSES)
        final_safety = "SAFE" if self.counts["person"] == 0 else "LOW RISK" if self.counts["person"] < 100 else "MODERATE RISK" if self.counts["person"] < 500 else "HIGH RISK"

        with open(self.csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Cars", "Motorcycles", "Buses", "Trucks", "Pedestrians", "Total Vehicle Detections", "Pedestrian Safety"])
            writer.writerow([datetime.now(), self.counts["car"], self.counts["motorcycle"], self.counts["bus"], self.counts["truck"], self.counts["person"], total_vehicles, final_safety])

if __name__ == "__main__":
    detector = VehiclePedestrianDetector()
    detector.run()

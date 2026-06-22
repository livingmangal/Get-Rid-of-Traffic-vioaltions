import cv2
import csv
from datetime import datetime
from config.settings import TripleRidingConfig
from core.video_processor import YOLOProcessor
from utils.helpers import risk_level

class TripleRidingDetector(YOLOProcessor):
    def __init__(self):
        super().__init__(TripleRidingConfig.video_path, TripleRidingConfig.output_path)
        self.csv_path = TripleRidingConfig.csv_path
        self.confidence_threshold = TripleRidingConfig.confidence
        self.total_motorcycles = 0
        self.total_no_helmet = 0
        self.total_triple_riding = 0

    def process_frame(self, frame, frame_no):
        results = self.detect(frame, self.confidence_threshold)
        detections = self.get_detections_by_class(results)
        
        persons = detections.get("person", [])
        motorcycles = detections.get("motorcycle", [])
        
        frame_motorcycles = len(motorcycles)
        frame_no_helmet = 0
        frame_triple = 0

        for mx1, my1, mx2, my2, mconf in motorcycles:
            riders_near_bike = 0
            for px1, py1, px2, py2, pconf in persons:
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2
                if mx1 - 100 <= pcx <= mx2 + 100 and my1 - 220 <= pcy <= my2 + 100:
                    riders_near_bike += 1
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
                    cv2.putText(frame, "Rider", (px1, max(py1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            self.total_motorcycles += 1
            no_helmet = True  # Simplified logic from original
            triple_riding = riders_near_bike >= 3

            safety_score = 100
            violations = []

            if no_helmet:
                frame_no_helmet += 1
                self.total_no_helmet += 1
                safety_score -= 40
                violations.append("NO HELMET")

            if triple_riding:
                frame_triple += 1
                self.total_triple_riding += 1
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
            cv2.putText(frame, label, (mx1, max(my1 - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"Riders: {riders_near_bike} | Risk: {risk}", (mx1, min(my2 + 25, self.height - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if frame_triple > 0:
            frame_status = "CRITICAL"
        elif frame_no_helmet > 0:
            frame_status = "HIGH RISK"
        else:
            frame_status = "NORMAL"

        cv2.rectangle(frame, (20, 20), (760, 250), (0, 0, 0), -1)
        cv2.putText(frame, "AI Motorcycle Safety Monitoring", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, f"Motorcycles: {frame_motorcycles}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"No Helmet Violations: {frame_no_helmet}", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, f"Triple Riding Violations: {frame_triple}", (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, f"Safety Status: {frame_status}", (30, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        return frame

    def run(self):
        print("Triple riding detector started...")
        self.process(self.process_frame)
        print("\nTriple riding detection completed.")
        
        with open(self.csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Motorcycle Detections", "No Helmet Violations", "Triple Riding Violations", "Risk Status", "Safety Recommendation"])
            final_status = "NORMAL"
            if self.total_no_helmet > 0: final_status = "HIGH RISK"
            if self.total_triple_riding > 0: final_status = "CRITICAL"
            writer.writerow([datetime.now(), self.total_motorcycles, self.total_no_helmet, self.total_triple_riding, final_status, "Wear helmet, avoid triple riding, follow traffic rules"])

if __name__ == "__main__":
    detector = TripleRidingDetector()
    detector.run()

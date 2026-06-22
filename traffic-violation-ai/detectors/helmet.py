import cv2
import csv
from datetime import datetime
from config.settings import HelmetConfig, DETECTION_CLASSES
from core.video_processor import YOLOProcessor
from core.evidence import draw_violation_overlay, save_evidence_frame, print_violation_report

class HelmetDetector(YOLOProcessor):
    def __init__(self):
        super().__init__(HelmetConfig.video_path, HelmetConfig.output_path)
        self.csv_path = HelmetConfig.csv_path
        self.confidence_threshold = HelmetConfig.confidence
        self.evidence_saved = False
        self.total_violations = 0

    def box_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def is_person_near_bike(self, person_box, bike_box):
        px, py = self.box_center(person_box)
        bx, by = self.box_center(bike_box)
        bike_x1, bike_y1, bike_x2, bike_y2 = bike_box
        bike_width = bike_x2 - bike_x1
        bike_height = bike_y2 - bike_y1
        return abs(px - bx) < bike_width and abs(py - by) < bike_height * 1.5

    def process_frame(self, frame, frame_no):
        results = self.detect(frame, self.confidence_threshold)
        detections = self.get_detections_by_class(results)
        
        persons = detections.get("person", [])
        motorcycles = detections.get("motorcycle", [])
        
        violation_detected = False
        best_confidence = None

        for px1, py1, px2, py2, p_conf in persons:
            for bx1, by1, bx2, by2, b_conf in motorcycles:
                if self.is_person_near_bike((px1, py1, px2, py2), (bx1, by1, bx2, by2)):
                    violation_detected = True
                    best_confidence = max(p_conf, b_conf)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.putText(frame, "Rider / Possible No Helmet", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    cv2.putText(frame, "Motorcycle", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if violation_detected:
            self.total_violations += 1
            frame = draw_violation_overlay(frame, "No Helmet", best_confidence, extra_info="Rider detected without visible helmet. Evidence frame generated.")
            if not self.evidence_saved:
                evidence_path = save_evidence_frame(frame, "No Helmet")
                print_violation_report("No Helmet", best_confidence, evidence_path)
                self.evidence_saved = True
        else:
            cv2.putText(frame, "Monitoring for helmet violation...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

    def run(self):
        print("Helmet violation detector started...")
        self.process(self.process_frame)
        print("\nHelmet detection completed.")
        
        with open(self.csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Violations Detected"])
            writer.writerow([datetime.now(), self.total_violations])

if __name__ == "__main__":
    detector = HelmetDetector()
    detector.run()

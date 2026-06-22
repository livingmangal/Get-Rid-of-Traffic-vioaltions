import cv2
import csv
import numpy as np
from datetime import datetime
from config.settings import WaterloggingConfig
from core.video_processor import VideoProcessor

class WaterloggingDetector(VideoProcessor):
    def __init__(self):
        super().__init__(WaterloggingConfig.video_path, WaterloggingConfig.output_path)
        self.csv_path = WaterloggingConfig.csv_path
        self.water_alert_frames = 0
        self.max_water_percent = 0

    def process_frame(self, frame, frame_no):
        roi_y1 = self.height // 2
        road_roi = frame[roi_y1:self.height, 0:self.width]
        hsv = cv2.cvtColor(road_roi, cv2.COLOR_BGR2HSV)
        
        lower_water = np.array([0, 0, 70])
        upper_water = np.array([180, 80, 255])
        mask = cv2.inRange(hsv, lower_water, upper_water)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        water_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        water_percent = (water_pixels / total_pixels) * 100
        self.max_water_percent = max(self.max_water_percent, water_percent)

        if water_percent < 15:
            risk, alert, color = "LOW", "NORMAL ROAD", (0, 255, 0)
        elif water_percent < 35:
            risk, alert, color = "MEDIUM", "POSSIBLE WATERLOGGING", (0, 255, 255)
            self.water_alert_frames += 1
        else:
            risk, alert, color = "HIGH", "WATERLOGGING ALERT", (0, 0, 255)
            self.water_alert_frames += 1

        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask[np.where((colored_mask == [255, 255, 255]).all(axis=2))] = color

        overlay_roi = cv2.addWeighted(road_roi, 0.75, colored_mask, 0.25, 0)
        frame[roi_y1:self.height, 0:self.width] = overlay_roi

        cv2.line(frame, (0, roi_y1), (self.width, roi_y1), (255, 0, 0), 3)
        cv2.rectangle(frame, (20, 20), (760, 210), (0, 0, 0), -1)
        cv2.putText(frame, "AI Road Safety - Waterlogging Detection", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Water Coverage: {water_percent:.2f}%", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"Risk Level: {risk}", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"Status: {alert}", (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, "Action: Reroute traffic / Alert municipal team", (30, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return frame

    def run(self):
        print("Waterlogging detector started...")
        self.process(self.process_frame)
        print("\nWaterlogging detection completed.")
        
        final_status = "NORMAL"
        if self.max_water_percent >= 15: final_status = "POSSIBLE WATERLOGGING"
        if self.max_water_percent >= 35: final_status = "WATERLOGGING ALERT"

        with open(self.csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Video", "Total Frames", "Water Alert Frames", "Max Water Coverage %", "Final Status", "Recommended Action"])
            writer.writerow([datetime.now(), self.video_path, self.cap.get(cv2.CAP_PROP_FRAME_COUNT), self.water_alert_frames, round(self.max_water_percent, 2), final_status, "Reroute traffic, alert municipal team, deploy water pump"])

if __name__ == "__main__":
    detector = WaterloggingDetector()
    detector.run()

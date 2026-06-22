import cv2
import csv
import math
import numpy as np
from datetime import datetime
from config.settings import WrongSideConfig, VEHICLE_CLASSES
from core.video_processor import YOLOProcessor

class WrongSideDetector(YOLOProcessor):
    def __init__(self):
        super().__init__(WrongSideConfig.video_path, WrongSideConfig.output_path)
        self.csv_path = WrongSideConfig.csv_path
        self.confidence_threshold = WrongSideConfig.confidence
        self.expected_valid_flow = "AWAY"
        self.min_y_movement = 14
        self.min_area_growth_ratio = 0.18
        self.max_track_distance = 140
        self.track_memory_frames = 18
        
        self.road_zones_ratio = {
            "FLYOVER_STRAIGHT_ROAD": [(0.00, 0.30), (0.34, 0.24), (0.43, 0.52), (0.00, 0.55)],
            "BELOW_FLYOVER_STRAIGHT_ROAD": [(0.00, 0.50), (0.40, 0.47), (0.76, 1.00), (0.00, 1.00)],
            "CENTER_WRONG_DIRECTION_APPROACH_ZONE": [(0.28, 0.43), (0.70, 0.42), (0.88, 1.00), (0.22, 1.00)]
        }
        self.road_zones = {}
        if self.valid:
            for zone_name, points in self.road_zones_ratio.items():
                self.road_zones[zone_name] = self.make_polygon(points, self.width, self.height)

        self.tracks = {}
        self.next_track_id = 1
        self.saved_wrong_tracks = set()
        self.evidence_count = 0
        self.csv_data = []

    def make_polygon(self, points_ratio, width, height):
        points = [[int(xr * width), int(yr * height)] for xr, yr in points_ratio]
        return np.array(points, dtype=np.int32)

    def point_inside_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def point_inside_any_zone(self, point):
        for zone_name, polygon in self.road_zones.items():
            if self.point_inside_polygon(point, polygon):
                return True, zone_name
        return False, "OUTSIDE_ROAD_ZONE"

    def assign_track(self, point):
        best_track_id = None
        best_distance = 10**9
        for track_id, track in self.tracks.items():
            last_point = track["points"][-1]
            d = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            if d < best_distance:
                best_distance = d
                best_track_id = track_id
        if best_track_id is not None and best_distance <= self.max_track_distance:
            return best_track_id
        new_track_id = self.next_track_id
        self.next_track_id += 1
        return new_track_id

    def direction_decision(self, track):
        points = track["points"]
        areas = track["areas"]
        if len(points) < 4:
            return "TRACKING_DIRECTION", 0, 0
        old_y = points[0][1]
        new_y = points[-1][1]
        old_area = areas[0]
        new_area = areas[-1]
        movement_y = new_y - old_y
        area_growth_ratio = (new_area - old_area) / max(old_area, 1)

        if self.expected_valid_flow == "AWAY":
            if movement_y > self.min_y_movement or area_growth_ratio > self.min_area_growth_ratio:
                return "WRONG_DIRECTION_TOWARDS_CAMERA", movement_y, area_growth_ratio
            if movement_y < -self.min_y_movement or area_growth_ratio < -self.min_area_growth_ratio:
                return "VALID_DIRECTION_AWAY", movement_y, area_growth_ratio
        return "TRACKING_DIRECTION", movement_y, area_growth_ratio

    def get_severity(self, vehicle_type):
        if vehicle_type in ["bus", "truck"]: return 5, "HIGH"
        if vehicle_type == "car": return 4, "HIGH"
        if vehicle_type == "motorcycle": return 3, "MEDIUM"
        return 2, "LOW"

    def process_frame(self, frame, frame_no):
        display_frame = frame.copy()
        overlay = display_frame.copy()
        zone_colors = {
            "FLYOVER_STRAIGHT_ROAD": (0, 255, 0),
            "BELOW_FLYOVER_STRAIGHT_ROAD": (255, 140, 0),
            "CENTER_WRONG_DIRECTION_APPROACH_ZONE": (255, 0, 0),
        }
        for zone_name, polygon in self.road_zones.items():
            color = zone_colors.get(zone_name, (255, 255, 0))
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(display_frame, [polygon], True, color, 3)
        display_frame = cv2.addWeighted(overlay, 0.12, display_frame, 0.88, 0)

        arrow_x = int(self.width * 0.73)
        cv2.arrowedLine(display_frame, (arrow_x, int(self.height * 0.82)), (arrow_x, int(self.height * 0.45)), (0, 255, 0), 6, tipLength=0.18)
        cv2.putText(display_frame, "VALID FLOW: STRAIGHT / AWAY", (arrow_x - 240, int(self.height * 0.42)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.arrowedLine(display_frame, (int(self.width * 0.47), int(self.height * 0.46)), (int(self.width * 0.47), int(self.height * 0.82)), (0, 0, 255), 6, tipLength=0.18)
        cv2.putText(display_frame, "WRONG FLOW: TOWARDS CAMERA", (int(self.width * 0.36), int(self.height * 0.88)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)

        results = self.detect(frame, self.confidence_threshold)
        detections = self.get_detections_by_class(results)

        current_track_ids = set()
        vehicles_detected = 0
        wrong_count = 0

        for v_type in VEHICLE_CLASSES:
            for x1, y1, x2, y2, conf in detections.get(v_type, []):
                cx, cy = (x1 + x2) // 2, y2
                area = max(1, (x2 - x1) * (y2 - y1))
                inside_zone, road_zone = self.point_inside_any_zone((cx, cy))
                if not inside_zone:
                    continue

                vehicles_detected += 1
                track_id = self.assign_track((cx, cy))
                current_track_ids.add(track_id)

                if track_id not in self.tracks:
                    self.tracks[track_id] = {"points": [], "areas": [], "missing": 0, "vehicle_type": v_type, "road_zone": road_zone, "confidence": conf}

                self.tracks[track_id]["points"].append((cx, cy))
                self.tracks[track_id]["points"] = self.tracks[track_id]["points"][-10:]
                self.tracks[track_id]["areas"].append(area)
                self.tracks[track_id]["areas"] = self.tracks[track_id]["areas"][-10:]
                self.tracks[track_id]["missing"] = 0

                status, movement_y, area_growth_ratio = self.direction_decision(self.tracks[track_id])
                
                if status == "WRONG_DIRECTION_TOWARDS_CAMERA":
                    wrong_count += 1
                    severity_score, severity_level = self.get_severity(v_type)
                    action = "Flag opposite-direction vehicle for police review"
                    color = (0, 0, 255)
                elif status == "VALID_DIRECTION_AWAY":
                    severity_score, severity_level = 1, "LOW"
                    action = "Vehicle moving with valid traffic flow"
                    color = (0, 255, 0)
                else:
                    severity_score, severity_level = 1, "LOW"
                    action = "Tracking vehicle movement direction"
                    color = (255, 255, 0)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                cv2.circle(display_frame, (cx, cy), 7, color, -1)
                cv2.putText(display_frame, f"ID {track_id} | {v_type.upper()} | {status}", (x1, max(35, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                cv2.putText(display_frame, f"dy={int(movement_y)} area={area_growth_ratio:.2f}", (x1, min(self.height - 20, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if status == "WRONG_DIRECTION_TOWARDS_CAMERA":
                    evidence_id = f"WRONG_DIR_{track_id:04d}"
                    cv2.putText(display_frame, f"Evidence: {evidence_id}", (x1, min(self.height - 20, y2 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
                    if track_id not in self.saved_wrong_tracks:
                        self.saved_wrong_tracks.add(track_id)
                        self.evidence_count += 1
                        self.csv_data.append([evidence_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_no, track_id, v_type, road_zone, status, round(movement_y, 2), round(area_growth_ratio, 3), severity_score, severity_level, round(conf, 2), action, "Wrong-direction detection"])

        for track_id in list(self.tracks.keys()):
            if track_id not in current_track_ids:
                self.tracks[track_id]["missing"] += 1
            if self.tracks[track_id]["missing"] > self.track_memory_frames:
                del self.tracks[track_id]

        cv2.rectangle(display_frame, (0, 0), (self.width, 88), (8, 8, 8), -1)
        cv2.putText(display_frame, "Wrong-Direction Detection | Flyover + Below-Flyover Same Direction Road", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Vehicles: {vehicles_detected} | Wrong Direction: {wrong_count} | Evidence Saved: {self.evidence_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        return display_frame

    def run(self):
        print("Wrong-direction detection started...")
        self.process(self.process_frame)
        print("\nWrong-direction detection completed.")
        
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["evidence_id", "timestamp", "frame_number", "track_id", "vehicle_type", "road_zone", "status", "movement_y", "area_growth_ratio", "severity_score", "severity_level", "confidence", "action_recommendation", "note"])
            writer.writerows(self.csv_data)

if __name__ == "__main__":
    detector = WrongSideDetector()
    detector.run()

import cv2
import os
import re
import csv
import numpy as np
import easyocr
from datetime import datetime
from config.settings import LicensePlateConfig, VEHICLE_CLASSES, PLATE_DIR
from core.video_processor import YOLOProcessor

class LicensePlateDetector(YOLOProcessor):
    def __init__(self):
        super().__init__(LicensePlateConfig.video_path, LicensePlateConfig.output_path)
        self.csv_path = LicensePlateConfig.csv_path
        self.confidence_threshold = LicensePlateConfig.confidence
        self.process_every_n_frames = 30
        self.ocr_min_text_length = 5
        self.evidence_count = 0
        self.reader = easyocr.Reader(["en"], gpu=False) if self.valid else None
        self.csv_data = []

    def clean_plate_text(self, text):
        text = text.upper()
        return re.sub(r"[^A-Z0-9]", "", text)

    def looks_like_indian_plate(self, text):
        pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$"
        return re.match(pattern, text) is not None

    def enhance_plate_for_ocr(self, plate_img):
        enhanced_images = []
        if plate_img is None or plate_img.size == 0:
            return enhanced_images

        zoomed = cv2.resize(plate_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        enhanced_images.append(zoomed)
        gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        enhanced_images.append(contrast)

        denoised = cv2.bilateralFilter(contrast, 11, 17, 17)
        enhanced_images.append(denoised)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        enhanced_images.append(sharpened)

        _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images.append(otsu)

        adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
        enhanced_images.append(adaptive)

        return enhanced_images

    def run_best_ocr(self, plate_img):
        best_text = ""
        best_conf = 0
        enhanced_versions = self.enhance_plate_for_ocr(plate_img)

        for img in enhanced_versions:
            try:
                results = self.reader.readtext(img, detail=1, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                for result in results:
                    text = self.clean_plate_text(result[1])
                    conf = float(result[2])
                    if len(text) < self.ocr_min_text_length: continue
                    score = conf + (0.40 if self.looks_like_indian_plate(text) else 0)
                    if score > best_conf:
                        best_conf = score
                        best_text = text
            except Exception:
                continue
        return best_text, round(min(best_conf, 1.0), 2)

    def find_plate_candidates(self, vehicle_crop):
        candidates = []
        if vehicle_crop is None or vehicle_crop.size == 0: return candidates
        h, w = vehicle_crop.shape[:2]
        lower_half = vehicle_crop[int(h * 0.45):h, :]
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if ch == 0: continue
            aspect_ratio = cw / ch
            area = cw * ch
            if 2.0 <= aspect_ratio <= 6.5 and area > 700:
                absolute_y = y + int(h * 0.45)
                px1 = max(0, x - 5)
                py1 = max(0, absolute_y - 5)
                px2 = min(w, x + cw + 5)
                py2 = min(h, absolute_y + ch + 5)
                candidate = vehicle_crop[py1:py2, px1:px2]
                if candidate.size > 0: candidates.append((px1, py1, px2, py2, candidate))

        if not candidates:
            px1, px2 = int(w * 0.20), int(w * 0.80)
            py1, py2 = int(h * 0.55), int(h * 0.90)
            fallback = vehicle_crop[py1:py2, px1:px2]
            if fallback.size > 0: candidates.append((px1, py1, px2, py2, fallback))
        return candidates

    def process_frame(self, frame, frame_no):
        display_frame = frame.copy()
        if frame_no % self.process_every_n_frames != 0:
            return display_frame

        results = self.detect(frame, self.confidence_threshold)
        detections = self.get_detections_by_class(results)

        for v_type in VEHICLE_CLASSES:
            for x1, y1, x2, y2, conf in detections.get(v_type, []):
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.width, x2), min(self.height, y2)
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0: continue

                plate_candidates = self.find_plate_candidates(vehicle_crop)
                best_plate_text, best_ocr_conf, best_plate_crop, best_plate_box = "", 0, None, None

                for px1, py1, px2, py2, plate_crop in plate_candidates:
                    plate_text, ocr_conf = self.run_best_ocr(plate_crop)
                    if ocr_conf > best_ocr_conf:
                        best_ocr_conf, best_plate_text, best_plate_crop, best_plate_box = ocr_conf, plate_text, plate_crop, (px1, py1, px2, py2)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{v_type} {conf:.2f}", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if best_plate_text:
                    self.evidence_count += 1
                    evidence_id = f"PLATE_{self.evidence_count:04d}"
                    plate_crop_path = os.path.join(PLATE_DIR, f"{evidence_id}_{best_plate_text}.jpg")
                    zoomed_crop = cv2.resize(best_plate_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(plate_crop_path, zoomed_crop)

                    if best_plate_box:
                        px1, py1, px2, py2 = best_plate_box
                        cv2.rectangle(display_frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (255, 255, 0), 2)
                    cv2.putText(display_frame, f"PLATE: {best_plate_text}", (x1, min(self.height - 20, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    self.csv_data.append([evidence_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_no, v_type, best_plate_text, best_ocr_conf, plate_crop_path, "PLATE_READ"])
                else:
                    cv2.putText(display_frame, "Plate not readable", (x1, min(self.height - 20, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display_frame, "Improved ANPR: Vehicle + Plate Region + Zoom OCR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return display_frame

    def run(self):
        print("License plate detector started...")
        self.process(self.process_frame)
        print("\nLicense plate detection completed.")
        
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["evidence_id", "timestamp", "frame_number", "vehicle_type", "plate_text", "ocr_confidence", "plate_crop_path", "status"])
            writer.writerows(self.csv_data)

if __name__ == "__main__":
    detector = LicensePlateDetector()
    detector.run()

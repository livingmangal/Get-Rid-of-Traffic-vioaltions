import cv2
import os
import re
import csv
import numpy as np
import easyocr
from ultralytics import YOLO
from datetime import datetime


# =========================
# CONFIG
# =========================

VIDEO_CANDIDATES = [
    "videos/day_traffic.mp4.mp4",
    "videos/day_traffic.mp4",
    "videos/night.mp4.mp4",
    "videos/night.mp4",
    "videos/no_helmet.mp4.mp4",
    "videos/no_helmet.mp4"
]

OUTPUT_VIDEO = "outputs/license_plate_ocr_improved.mp4"
OUTPUT_CSV = "data/license_plate_ocr_improved_report.csv"
PLATE_EVIDENCE_DIR = "outputs/plate_evidence"

PROCESS_EVERY_N_FRAMES = 30
MAX_SECONDS_TO_PROCESS = 35
OCR_MIN_TEXT_LENGTH = 5

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(PLATE_EVIDENCE_DIR, exist_ok=True)


# =========================
# SELECT VIDEO
# =========================

VIDEO_PATH = None

for path in VIDEO_CANDIDATES:
    if os.path.exists(path):
        VIDEO_PATH = path
        break

if VIDEO_PATH is None:
    print("❌ No video found inside videos folder.")
    print("Put your video inside traffic-violation-ai/videos/")
    exit()

print(f"✅ Using video: {VIDEO_PATH}")


# =========================
# LOAD MODELS
# =========================

print("Loading YOLO vehicle model...")
vehicle_model = YOLO("yolov8n.pt")

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)


# =========================
# HELPER FUNCTIONS
# =========================

def clean_plate_text(text):
    """
    Cleans OCR output and keeps only uppercase letters/numbers.
    Example: 'BR 01 AB 1234' -> 'BR01AB1234'
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def looks_like_indian_plate(text):
    """
    Indian plate examples:
    BR01AB1234
    DL8CAF5030
    KA05MN1234
    """
    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$"
    return re.match(pattern, text) is not None


def enhance_plate_for_ocr(plate_img):
    """
    Creates multiple enhanced versions of the plate.
    OCR is tried on all versions and best result is selected.
    """

    enhanced_images = []

    if plate_img is None or plate_img.size == 0:
        return enhanced_images

    # 1. Zoom 4x
    zoomed = cv2.resize(
        plate_img,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    enhanced_images.append(zoomed)

    # 2. Grayscale
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )
    contrast = clahe.apply(gray)
    enhanced_images.append(contrast)

    # 4. Denoise
    denoised = cv2.bilateralFilter(
        contrast,
        11,
        17,
        17
    )
    enhanced_images.append(denoised)

    # 5. Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    enhanced_images.append(sharpened)

    # 6. Otsu threshold
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    enhanced_images.append(otsu)

    # 7. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )
    enhanced_images.append(adaptive)

    return enhanced_images


def run_best_ocr(plate_img):
    """
    Runs OCR on multiple enhanced versions and returns best text.
    """

    best_text = ""
    best_conf = 0

    enhanced_versions = enhance_plate_for_ocr(plate_img)

    for img in enhanced_versions:
        try:
            results = reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            for result in results:
                text = clean_plate_text(result[1])
                conf = float(result[2])

                if len(text) < OCR_MIN_TEXT_LENGTH:
                    continue

                score = conf

                # Bonus if text looks like Indian plate
                if looks_like_indian_plate(text):
                    score += 0.40

                if score > best_conf:
                    best_conf = score
                    best_text = text

        except Exception:
            continue

    return best_text, round(best_conf, 2)


def find_plate_candidates(vehicle_crop):
    """
    Finds possible plate-like rectangular regions inside a vehicle crop.
    This is a fallback plate detector when no custom plate model is used.
    """

    candidates = []

    if vehicle_crop is None or vehicle_crop.size == 0:
        return candidates

    h, w = vehicle_crop.shape[:2]

    # Number plates are usually in lower half of vehicle
    lower_half = vehicle_crop[int(h * 0.45):h, :]

    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edges = cv2.Canny(gray, 80, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)

        if ch == 0:
            continue

        aspect_ratio = cw / ch
        area = cw * ch

        # Plate-like rectangle conditions
        if 2.0 <= aspect_ratio <= 6.5 and area > 700:
            absolute_y = y + int(h * 0.45)

            px1 = max(0, x - 5)
            py1 = max(0, absolute_y - 5)
            px2 = min(w, x + cw + 5)
            py2 = min(h, absolute_y + ch + 5)

            candidate = vehicle_crop[py1:py2, px1:px2]

            if candidate.size > 0:
                candidates.append((px1, py1, px2, py2, candidate))

    # Fallback: lower-middle crop if contour fails
    if len(candidates) == 0:
        px1 = int(w * 0.20)
        px2 = int(w * 0.80)
        py1 = int(h * 0.55)
        py2 = int(h * 0.90)

        fallback = vehicle_crop[py1:py2, px1:px2]

        if fallback.size > 0:
            candidates.append((px1, py1, px2, py2, fallback))

    return candidates


# =========================
# VIDEO SETUP
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames_to_process = fps * MAX_SECONDS_TO_PROCESS
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

writer.writerow([
    "evidence_id",
    "timestamp",
    "frame_number",
    "vehicle_type",
    "plate_text",
    "ocr_confidence",
    "plate_crop_path",
    "status"
])


# =========================
# PROCESS VIDEO
# =========================

frame_count = 0
evidence_count = 0

print("Processing video... Please wait.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    if frame_count > max_frames_to_process:
        break

    display_frame = frame.copy()

    # Process only selected frames for speed
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        out.write(display_frame)
        continue

    results = vehicle_model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = vehicle_model.names[cls_id]
        conf = float(box.conf[0])

        if label not in VEHICLE_CLASSES:
            continue

        if conf < 0.35:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        vehicle_crop = frame[y1:y2, x1:x2]

        if vehicle_crop.size == 0:
            continue

        plate_candidates = find_plate_candidates(vehicle_crop)

        best_plate_text = ""
        best_ocr_conf = 0
        best_plate_crop = None
        best_plate_box = None

        for px1, py1, px2, py2, plate_crop in plate_candidates:
            plate_text, ocr_conf = run_best_ocr(plate_crop)

            if ocr_conf > best_ocr_conf:
                best_ocr_conf = ocr_conf
                best_plate_text = plate_text
                best_plate_crop = plate_crop
                best_plate_box = (px1, py1, px2, py2)

        # Draw vehicle box
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"{label} {conf:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        if best_plate_text:
            evidence_count += 1
            evidence_id = f"PLATE_{evidence_count:04d}"

            plate_crop_path = os.path.join(
                PLATE_EVIDENCE_DIR,
                f"{evidence_id}_{best_plate_text}.jpg"
            )

            # Save zoomed enhanced crop as proof
            zoomed_crop = cv2.resize(
                best_plate_crop,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            cv2.imwrite(plate_crop_path, zoomed_crop)

            if best_plate_box:
                px1, py1, px2, py2 = best_plate_box

                # Convert vehicle crop coordinates to full-frame coordinates
                fx1 = x1 + px1
                fy1 = y1 + py1
                fx2 = x1 + px2
                fy2 = y1 + py2

                cv2.rectangle(
                    display_frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    (255, 255, 0),
                    2
                )

            cv2.putText(
                display_frame,
                f"PLATE: {best_plate_text}",
                (x1, min(height - 20, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            writer.writerow([
                evidence_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_count,
                label,
                best_plate_text,
                best_ocr_conf,
                plate_crop_path,
                "PLATE_READ"
            ])

        else:
            cv2.putText(
                display_frame,
                "Plate not readable",
                (x1, min(height - 20, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.putText(
        display_frame,
        "Improved ANPR: Vehicle + Plate Region + Zoom OCR",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(display_frame)

cap.release()
out.release()
csv_file.close()

print("✅ Improved license plate OCR completed.")
print(f"🎥 Output video saved at: {OUTPUT_VIDEO}")
print(f"📄 CSV report saved at: {OUTPUT_CSV}")
print(f"🖼 Plate evidence crops saved in: {PLATE_EVIDENCE_DIR}")
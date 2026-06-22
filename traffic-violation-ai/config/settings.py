import os
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
VIDEO_DIR = BASE_DIR / "videos"
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
PLATE_DIR = OUTPUT_DIR / "plate_evidence"
SMART_CSV = DATA_DIR / "smart_violations.csv"

YOLO_MODEL = "yolov8n.pt"

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
DETECTION_CLASSES = VEHICLE_CLASSES + ["person"]

@dataclass
class DetectorConfig:
    video_path: str
    output_path: str
    csv_path: str
    confidence: float

HelmetConfig = DetectorConfig(
    video_path=str(VIDEO_DIR / "no_helmet.mp4"),
    output_path=str(OUTPUT_DIR / "helmet_output.mp4"),
    csv_path=str(DATA_DIR / "helmet_report.csv"),
    confidence=0.35
)

TripleRidingConfig = DetectorConfig(
    video_path=str(VIDEO_DIR / "tripleriding.mp4"),
    output_path=str(OUTPUT_DIR / "triple_riding_processed.mp4"),
    csv_path=str(DATA_DIR / "triple_riding_report.csv"),
    confidence=0.05
)

WrongSideConfig = DetectorConfig(
    video_path=str(VIDEO_DIR / "wrong_direction.mp4"),
    output_path=str(OUTPUT_DIR / "wrong_side_processed.mp4"),
    csv_path=str(DATA_DIR / "wrong_side_report.csv"),
    confidence=0.35
)

NightVisionConfig = DetectorConfig(
    video_path=str(VIDEO_DIR / "night.mp4"),
    output_path=str(OUTPUT_DIR / "night_processed.mp4"),
    csv_path=str(DATA_DIR / "traffic_report.csv"),
    confidence=0.15
)

VehiclePedestrianConfig = DetectorConfig(
    video_path=str(VIDEO_DIR / "pedestrian.mp4"),
    output_path=str(OUTPUT_DIR / "pedestrian_processed.mp4"),
    csv_path=str(DATA_DIR / "pedestrian_report.csv"),
    confidence=0.20
)

WaterloggingConfig = DetectorConfig(
    video_path=str(VIDEO_DIR / "waterlogging.mp4"),
    output_path=str(OUTPUT_DIR / "waterlogging_processed.mp4"),
    csv_path=str(DATA_DIR / "waterlogging_report.csv"),
    confidence=0.0
)

LicensePlateConfig = DetectorConfig(
    video_path=str(VIDEO_DIR / "day_traffic.mp4"),
    output_path=str(OUTPUT_DIR / "license_plate_ocr_improved.mp4"),
    csv_path=str(DATA_DIR / "license_plate_ocr_improved_report.csv"),
    confidence=0.35
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "intellitraffic_ai")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "violations")

def ensure_directories():
    for d in [DATA_DIR, OUTPUT_DIR, EVIDENCE_DIR, PLATE_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_directories()

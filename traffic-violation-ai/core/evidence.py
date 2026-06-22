import cv2
import os
from datetime import datetime
from config.settings import EVIDENCE_DIR

def draw_violation_overlay(
    frame,
    violation_type,
    confidence=None,
    status="VIOLATION DETECTED",
    extra_info="Evidence generated from traffic video"
):
    h, w = frame.shape[:2]
    panel_x1, panel_y1 = 20, 20
    panel_x2, panel_y2 = min(w - 20, 760), 165

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y1 + 40), (0, 0, 255), -1)

    cv2.putText(frame, status, (panel_x1 + 15, panel_y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Violation Type: {violation_type}", (panel_x1 + 15, panel_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if confidence is not None:
        try:
            confidence_text = f"Confidence: {float(confidence):.2f}"
        except Exception:
            confidence_text = f"Confidence: {confidence}"
        cv2.putText(frame, confidence_text, (panel_x1 + 15, panel_y1 + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Timestamp: {timestamp}", (panel_x1 + 15, panel_y1 + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
    cv2.putText(frame, extra_info, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame

def save_evidence_frame(frame, violation_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = violation_type.lower().replace(" ", "_").replace("-", "_")
    file_path = os.path.join(EVIDENCE_DIR, f"{clean_name}_evidence_{timestamp}.jpg")
    cv2.imwrite(file_path, frame)
    return file_path

def print_violation_report(violation_type, confidence=None, evidence_path=None):
    print("\n" + "=" * 60)
    print("TRAFFIC VIOLATION REPORT")
    print("=" * 60)
    print(f"Violation Type : {violation_type}")
    if confidence is not None:
        try:
            print(f"Confidence     : {float(confidence):.2f}")
        except Exception:
            print(f"Confidence     : {confidence}")
    print(f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if evidence_path:
        print(f"Evidence Saved : {evidence_path}")
    print("Status         : Evidence generated successfully")
    print("=" * 60 + "\n")

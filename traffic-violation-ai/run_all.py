import os
import subprocess
import time
from analytics.smart_features import SmartViolationProcessor
from analytics.recommendations import RecommendationEngine

def print_header(title):
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

def run_detectors():
    print_header("RUNNING AI TRAFFIC DETECTORS")
    detectors = [
        ("Helmet & Motorcycle", "detectors.helmet"),
        ("Triple Riding", "detectors.triple_riding"),
        ("Wrong Side Driving", "detectors.wrong_side"),
        ("Night Vision Traffic", "detectors.night_vision"),
        ("Pedestrian Safety", "detectors.vehicle_pedestrian"),
        ("Waterlogging", "detectors.waterlogging"),
        ("License Plate OCR", "detectors.license_plate")
    ]
    
    for name, module in detectors:
        print(f"\n---> Starting: {name}...")
        subprocess.run(["python", "-m", module])
        time.sleep(1)

def run_analytics():
    print_header("GENERATING AI ANALYTICS & SMART FEATURES")
    SmartViolationProcessor().run()
    RecommendationEngine().generate()

def start_services():
    print_header("STARTING DASHBOARD & API SERVICES")
    print("Starting FastAPI Backend (in background)...")
    api_process = subprocess.Popen(["python", "-m", "uvicorn", "api.backend:app", "--port", "8000"])
    
    print("Starting Streamlit Dashboard...")
    try:
        subprocess.run(["python", "-m", "streamlit", "run", "dashboards/main.py"])
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_process.terminate()

if __name__ == "__main__":
    print_header("INTELLITRAFFIC AI - FULL PIPELINE RUNNER")
    print("1. Run Full Pipeline (Detectors -> Analytics -> Dashboard)")
    print("2. Run Analytics & Dashboard Only (if detectors already ran)")
    print("3. Run Dashboard Only")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_detectors()
        run_analytics()
        start_services()
    elif choice == "2":
        run_analytics()
        start_services()
    elif choice == "3":
        start_services()
    else:
        print("Invalid choice. Exiting.")

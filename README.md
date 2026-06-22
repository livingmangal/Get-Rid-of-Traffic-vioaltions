# IntelliTraffic AI

## AI-Powered Traffic Violation Detection and Smart-City Enforcement Intelligence

IntelliTraffic AI is a computer-vision-based smart traffic monitoring prototype that converts traffic videos into actionable enforcement intelligence. The system detects traffic violations, generates evidence, performs license plate OCR, calculates severity scores, generates demo fines, maps violation hotspots, and provides dashboard-based analytics through Streamlit and FastAPI.

---

## Project Objective

Modern cities face traffic problems such as wrong-side driving, helmet non-compliance, triple riding, pedestrian safety risks, waterlogging, and manual traffic surveillance overload.

The objective of IntelliTraffic AI is to build an AI-powered prototype that helps traffic authorities automatically analyze video footage, detect violations, generate evidence, and visualize insights for better enforcement decisions.

---

## Key Features

* Vehicle and pedestrian detection
* Night / low-light traffic video enhancement
* Helmet violation risk detection prototype
* Triple riding detection
* Wrong-side driving detection
* Waterlogging detection
* License plate OCR / ANPR prototype
* CSV-based evidence generation
* Premium Streamlit dashboard
* Smart violation intelligence dashboard
* Severity score calculation
* Demo fine generation
* Violation hotspot heatmap
* Alert message generator
* Natural language search
* FastAPI backend
* MongoDB-ready architecture with CSV fallback
* API dashboard fetching backend data

---

## Tech Stack

| Area                 | Technology                      |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Computer Vision      | OpenCV                          |
| Object Detection     | YOLOv8                          |
| OCR                  | EasyOCR                         |
| Dashboard            | Streamlit                       |
| Charts and Heatmap   | Plotly                          |
| Backend API          | FastAPI                         |
| Database Layer       | MongoDB-ready with CSV fallback |
| Data Handling        | Pandas                          |
| Version Control      | Git and GitHub                  |

---

## System Architecture

Traffic Video Input
↓
Image Preprocessing
↓
YOLOv8 Object Detection
↓
Violation Logic Engine
↓
License Plate OCR
↓
Evidence Generation
↓
CSV / FastAPI Backend
↓
Dashboard Analytics
↓
Smart Enforcement Recommendation

---

## Modules

### 1. Vehicle and Pedestrian Detection

This module detects vehicles and pedestrians from traffic videos. It identifies cars, motorcycles, buses, trucks, and people. The output is an annotated video and a CSV report.

### 2. Night Vision Enhancement

This module improves low-light video frames using brightness and contrast enhancement so that detection works better in poor lighting conditions.

### 3. Helmet Violation Detection Prototype

This module detects motorcycles and nearby riders to identify possible helmet-risk situations. In the prototype, this is rule-based. In production, it can be upgraded with a custom helmet/no-helmet trained model.

### 4. Triple Riding Detection

This module detects motorcycles and counts nearby riders. If three or more people are detected near one motorcycle, the system marks it as a possible triple-riding violation.

### 5. Wrong-Side Driving Detection

This module uses a virtual lane or road boundary to identify vehicles appearing on the wrong side of the road.

### 6. Waterlogging Detection

This module analyzes the road region to detect possible waterlogging using image processing. It assigns a risk level and generates a report.

### 7. License Plate OCR / ANPR

This module detects vehicles, crops the possible number plate area, zooms and enhances the crop, and applies OCR to extract possible registration numbers.

### 8. Smart Violation Intelligence

This module converts violation records into decision intelligence by adding severity scores, demo fine values, hotspot locations, alert-ready messages, and searchable records.

### 9. FastAPI Backend

The backend exposes violation and analytics data through API endpoints. It supports MongoDB-ready storage and CSV fallback mode for prototype execution.

---

## API Endpoints

| Endpoint                        | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| `/`                             | Backend status                                  |
| `/health`                       | API health and database mode                    |
| `/violations`                   | Fetch violation records                         |
| `/violations/{evidence_id}`     | Fetch one violation by evidence ID              |
| `/analytics/summary`            | Get total records, fine, severity, and vehicles |
| `/analytics/by-type`            | Get violation analytics by type                 |
| `/analytics/by-module`          | Get records by detection module                 |
| `/analytics/hotspots`           | Get hotspot heatmap data                        |
| `/alerts/preview/{evidence_id}` | Generate demo alert message                     |

---

## Evaluation Metrics

The system supports evaluation using the following metrics:

| Metric               | Meaning                                                    |
| -------------------- | ---------------------------------------------------------- |
| Accuracy             | Overall correct predictions out of total predictions       |
| Precision            | Out of detected violations, how many were actually correct |
| Recall               | Out of real violations, how many were detected             |
| F1-Score             | Balanced score between precision and recall                |
| mAP@0.5              | Object detection quality at IoU threshold 0.5              |
| mAP@0.5:0.95         | Object detection quality across multiple IoU thresholds    |
| FPS                  | Processing speed of video frames                           |
| OCR Readable Rate    | Percentage of plates successfully read                     |
| Plate-Level Accuracy | Percentage of full plates correctly recognized             |
| Character Accuracy   | Accuracy of individual plate characters                    |

### Important Evaluation Note

This is a prototype system. Final numeric accuracy, precision, recall, F1-score, and mAP values require manually labeled validation videos. The current version demonstrates the complete working pipeline from video input to smart enforcement analytics.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r traffic-violation-ai/requirements.txt
```

### 2. Run the Full Pipeline

The project has been refactored into a modular architecture. You can now run the entire pipeline (detectors, analytics, and dashboards) using a single command from the `traffic-violation-ai` directory:

```bash
cd traffic-violation-ai
python run_all.py
```

You will be prompted to choose:
1. **Run Full Pipeline**: Runs all detectors sequentially, generates analytics, starts the backend API, and launches the Streamlit dashboard.
2. **Run Analytics & Dashboard Only**: Skips the time-consuming detection phase if you already have the outputs generated.
3. **Run Dashboard Only**: Only starts the FastAPI backend and Streamlit dashboard.

### 3. Run FastAPI backend (Manual)

If you wish to run the backend separately:

```bash
cd traffic-violation-ai
python -m uvicorn api.backend:app --reload --port 8000
```

Open API docs: `http://127.0.0.1:8000/docs`

### 4. Run Streamlit Dashboards (Manual)

The unified dashboard can be launched manually:

```bash
cd traffic-violation-ai
python -m streamlit run dashboards/main.py
```

---

## Demo Flow

1. Run the full pipeline via `python run_all.py`.
2. Show processed violation videos and evidence frames from the `outputs/` folder.
3. Open the unified Streamlit Dashboard (`http://localhost:8501`).
4. Navigate through the **Main Dashboard** to see metrics and recent smart alerts.
5. Navigate to the **Advanced Analytics** tab to view the geographic hotspot heatmap and fine revenue distributions.
6. Navigate to the **API Explorer** tab to interact directly with the FastAPI backend.
7. Open FastAPI `/docs` at `http://127.0.0.1:8000/docs` to show the Swagger UI.

---

## Limitations

* License plate OCR depends heavily on video quality.
* Helmet detection is rule-based in the prototype.
* Wrong-side detection uses virtual lane logic.
* MongoDB is optional in prototype mode.
* Real-world deployment requires more labeled traffic datasets.
* OCR works best with daylight, high-resolution, stable videos.

---

## Future Scope

* Custom helmet/no-helmet detection model
* Dedicated Indian number plate detection model
* Red-light violation detection
* Seatbelt violation detection
* Illegal parking detection
* Real-time CCTV stream support
* MongoDB cloud deployment
* E-challan integration
* Mobile app for traffic officers
* Smart signal optimization using congestion analytics
* Automatic repeat offender tracking

---

## Final Impact

IntelliTraffic AI does not only detect traffic violations. It transforms raw traffic footage into smart-city enforcement intelligence by combining detection, evidence generation, OCR, analytics, backend APIs, hotspot mapping, alert messages, and decision support dashboards.

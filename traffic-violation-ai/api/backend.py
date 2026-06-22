from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import math
from pymongo import MongoClient

from config.settings import MONGO_URI, MONGO_DB, MONGO_COLLECTION, SMART_CSV, DATA_DIR

app = FastAPI(
    title="IntelliTraffic AI Backend",
    description="FastAPI backend for traffic violation records, analytics, alerts, and MongoDB storage.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_client = None
mongo_collection = None
mongo_connected = False

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1200)
    mongo_client.admin.command("ping")
    mongo_collection = mongo_client[MONGO_DB][MONGO_COLLECTION]
    mongo_connected = True
except Exception:
    pass

class ViolationCreate(BaseModel):
    evidence_id: Optional[str] = None
    timestamp: Optional[str] = None
    module: str
    violation_type: str
    vehicle_number: Optional[str] = "UNKNOWN"
    severity_score: int = 1
    fine_amount: int = 0
    hotspot: Optional[str] = "Unknown Location"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: Optional[float] = 0.85
    alert_message: Optional[str] = None
    source_file: Optional[str] = "api"
    alert_status: Optional[str] = "DEMO_READY"

def clean_record(record):
    clean = {}
    for k, v in record.items():
        if v is None:
            clean[k] = None
        elif isinstance(v, float) and math.isnan(v):
            clean[k] = None
        elif hasattr(v, "item"):
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean

def get_all_records() -> List[Dict[str, Any]]:
    if mongo_connected and mongo_collection is not None:
        return [clean_record(r) for r in mongo_collection.find({}, {"_id": 0})]
    
    if SMART_CSV.exists():
        df = pd.read_csv(SMART_CSV).where(pd.notnull(pd.read_csv(SMART_CSV)), None)
        return [clean_record(r) for r in df.to_dict(orient="records")]
    return []

@app.get("/")
def root():
    return {"message": "IntelliTraffic AI FastAPI Backend is running", "database": "MongoDB" if mongo_connected else "CSV fallback"}

@app.get("/health")
def health():
    return {"api_status": "running", "mongo_connected": mongo_connected, "records_available": len(get_all_records())}

@app.get("/violations")
def get_violations(
    violation_type: Optional[str] = Query(default=None),
    hotspot: Optional[str] = Query(default=None),
    min_severity: int = Query(default=1, ge=1, le=5),
    vehicle_number: Optional[str] = Query(default=None),
):
    filtered = []
    for record in get_all_records():
        if violation_type and violation_type.lower() not in str(record.get("violation_type", "")).lower(): continue
        if hotspot and hotspot.lower() not in str(record.get("hotspot", "")).lower(): continue
        if int(record.get("severity_score", 0)) < min_severity: continue
        if vehicle_number and vehicle_number.lower() not in str(record.get("vehicle_number", "")).lower(): continue
        filtered.append(record)
    return {"count": len(filtered), "records": filtered}

@app.get("/violations/{evidence_id}")
def get_violation_by_id(evidence_id: str):
    for record in get_all_records():
        if str(record.get("evidence_id")) == evidence_id:
            return record
    raise HTTPException(status_code=404, detail="Evidence ID not found")

@app.post("/violations")
def create_violation(violation: ViolationCreate):
    record = violation.model_dump()
    if not record.get("evidence_id"): record["evidence_id"] = "API_" + datetime.now().strftime("%Y%m%d%H%M%S")
    if not record.get("timestamp"): record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = clean_record(record)
    
    if mongo_connected and mongo_collection is not None:
        mongo_collection.insert_one(record)
        return {"message": "Violation inserted into MongoDB", "record": record}
    
    existing = get_all_records()
    existing.append(record)
    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(existing).to_csv(SMART_CSV, index=False, encoding="utf-8")
    return {"message": "Violation saved to CSV fallback", "record": record}

@app.get("/analytics/summary")
def analytics_summary():
    df = pd.DataFrame(get_all_records())
    if df.empty: return {"total_records": 0, "total_fine": 0, "average_severity": 0, "high_severity_count": 0}
    return {
        "total_records": len(df),
        "total_fine": int(pd.to_numeric(df.get("fine_amount", 0), errors="coerce").fillna(0).sum()),
        "average_severity": round(pd.to_numeric(df.get("severity_score", 0), errors="coerce").fillna(0).mean(), 2),
        "high_severity_count": int((pd.to_numeric(df.get("severity_score", 0), errors="coerce").fillna(0) >= 4).sum())
    }

@app.get("/analytics/by-module")
def analytics_by_module():
    df = pd.DataFrame(get_all_records())
    if df.empty or "module" not in df.columns: return []
    return df.groupby("module").size().reset_index(name="count").sort_values("count", ascending=False).to_dict(orient="records")

@app.get("/analytics/by-type")
def analytics_by_type():
    df = pd.DataFrame(get_all_records())
    if df.empty or "violation_type" not in df.columns: return []
    res = df.groupby("violation_type").agg(count=("evidence_id", "count"), total_fine=("fine_amount", "sum"), average_severity=("severity_score", "mean")).reset_index()
    res["average_severity"] = res["average_severity"].round(2)
    return res.to_dict(orient="records")

@app.get("/analytics/hotspots")
def analytics_hotspots():
    df = pd.DataFrame(get_all_records())
    if df.empty or not {"hotspot", "latitude", "longitude"}.issubset(set(df.columns)): return []
    res = df.groupby(["hotspot", "latitude", "longitude"]).agg(violations=("evidence_id", "count"), total_fine=("fine_amount", "sum"), average_severity=("severity_score", "mean")).reset_index()
    res["average_severity"] = res["average_severity"].round(2)
    return res.to_dict(orient="records")

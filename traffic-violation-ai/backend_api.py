from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import math

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SMART_CSV = DATA_DIR / "smart_violations.csv"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "intellitraffic_ai")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "violations")

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


def connect_mongo():
    global mongo_client, mongo_collection, mongo_connected

    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1200)
        mongo_client.admin.command("ping")
        mongo_collection = mongo_client[DB_NAME][COLLECTION_NAME]
        mongo_connected = True
    except Exception:
        mongo_client = None
        mongo_collection = None
        mongo_connected = False


connect_mongo()


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


def clean_value(value):
    if value is None:
        return None

    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass

    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass

    return value


def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: clean_value(value) for key, value in record.items()}


def load_csv_records() -> List[Dict[str, Any]]:
    if not SMART_CSV.exists():
        return []

    try:
        df = pd.read_csv(SMART_CSV)
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")
        return [clean_record(record) for record in records]
    except Exception:
        return []


def get_mongo_records() -> List[Dict[str, Any]]:
    if not mongo_connected or mongo_collection is None:
        return []

    records = list(mongo_collection.find({}, {"_id": 0}))
    return [clean_record(record) for record in records]


def get_all_records() -> List[Dict[str, Any]]:
    mongo_records = get_mongo_records()

    if mongo_records:
        return mongo_records

    return load_csv_records()


def filter_records(
    records: List[Dict[str, Any]],
    violation_type: Optional[str],
    hotspot: Optional[str],
    min_severity: int,
    vehicle_number: Optional[str],
) -> List[Dict[str, Any]]:

    filtered = []

    for record in records:
        if violation_type:
            if violation_type.lower() not in str(record.get("violation_type", "")).lower():
                continue

        if hotspot:
            if hotspot.lower() not in str(record.get("hotspot", "")).lower():
                continue

        try:
            severity = int(record.get("severity_score", 0))
        except Exception:
            severity = 0

        if severity < min_severity:
            continue

        if vehicle_number:
            if vehicle_number.lower() not in str(record.get("vehicle_number", "")).lower():
                continue

        filtered.append(record)

    return filtered


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


@app.get("/")
def root():
    return {
        "message": "IntelliTraffic AI FastAPI Backend is running",
        "docs": "/docs",
        "database": "MongoDB" if mongo_connected else "CSV fallback",
    }


@app.get("/health")
def health():
    return {
        "api_status": "running",
        "mongo_connected": mongo_connected,
        "database_mode": "MongoDB" if mongo_connected else "CSV fallback",
        "csv_exists": SMART_CSV.exists(),
        "records_available": len(get_all_records()),
    }


@app.post("/seed-from-csv")
def seed_from_csv():
    if not mongo_connected or mongo_collection is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not connected. API is currently using CSV fallback.",
        )

    records = load_csv_records()

    if not records:
        raise HTTPException(
            status_code=404,
            detail="No smart_violations.csv records found. Run python smart_features.py first.",
        )

    mongo_collection.delete_many({})
    mongo_collection.insert_many(records)

    return {
        "message": "MongoDB seeded successfully from CSV",
        "inserted_records": len(records),
        "database": DB_NAME,
        "collection": COLLECTION_NAME,
    }


@app.get("/violations")
def get_violations(
    violation_type: Optional[str] = Query(default=None),
    hotspot: Optional[str] = Query(default=None),
    min_severity: int = Query(default=1, ge=1, le=5),
    vehicle_number: Optional[str] = Query(default=None),
):
    records = get_all_records()

    filtered = filter_records(
        records=records,
        violation_type=violation_type,
        hotspot=hotspot,
        min_severity=min_severity,
        vehicle_number=vehicle_number,
    )

    return {
        "count": len(filtered),
        "database_mode": "MongoDB" if mongo_connected and get_mongo_records() else "CSV fallback",
        "records": filtered,
    }


@app.get("/violations/{evidence_id}")
def get_violation_by_id(evidence_id: str):
    records = get_all_records()

    for record in records:
        if str(record.get("evidence_id")) == evidence_id:
            return record

    raise HTTPException(status_code=404, detail="Evidence ID not found")


@app.post("/violations")
def create_violation(violation: ViolationCreate):
    record = violation.model_dump()

    if not record.get("evidence_id"):
        record["evidence_id"] = "API_" + datetime.now().strftime("%Y%m%d%H%M%S")

    if not record.get("timestamp"):
        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not record.get("alert_message"):
        record["alert_message"] = (
            f"Demo Alert: Vehicle {record.get('vehicle_number')} detected for "
            f"{record.get('violation_type')} at {record.get('hotspot')}. "
            f"Demo fine: Rs {record.get('fine_amount')}."
        )

    record = clean_record(record)

    if mongo_connected and mongo_collection is not None:
        mongo_collection.insert_one(record)
        return {
            "message": "Violation inserted into MongoDB",
            "record": record,
        }

    existing = load_csv_records()
    existing.append(record)

    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(existing).to_csv(SMART_CSV, index=False, encoding="utf-8")

    return {
        "message": "Violation saved to CSV fallback",
        "record": record,
    }


@app.get("/analytics/summary")
def analytics_summary():
    records = get_all_records()
    df = records_to_dataframe(records)

    if df.empty:
        return {
            "total_records": 0,
            "total_fine": 0,
            "average_severity": 0,
            "high_severity_count": 0,
            "unique_vehicles": 0,
            "database_mode": "MongoDB" if mongo_connected else "CSV fallback",
        }

    total_fine = int(pd.to_numeric(df.get("fine_amount", 0), errors="coerce").fillna(0).sum())
    average_severity = round(pd.to_numeric(df.get("severity_score", 0), errors="coerce").fillna(0).mean(), 2)
    high_severity_count = int((pd.to_numeric(df.get("severity_score", 0), errors="coerce").fillna(0) >= 4).sum())

    unique_vehicles = 0
    if "vehicle_number" in df.columns:
        unique_vehicles = int(df["vehicle_number"].astype(str).nunique())

    return {
        "total_records": len(df),
        "total_fine": total_fine,
        "average_severity": average_severity,
        "high_severity_count": high_severity_count,
        "unique_vehicles": unique_vehicles,
        "database_mode": "MongoDB" if mongo_connected and get_mongo_records() else "CSV fallback",
    }


@app.get("/analytics/by-module")
def analytics_by_module():
    records = get_all_records()
    df = records_to_dataframe(records)

    if df.empty or "module" not in df.columns:
        return []

    result = (
        df.groupby("module")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )

    return result


@app.get("/analytics/by-type")
def analytics_by_type():
    records = get_all_records()
    df = records_to_dataframe(records)

    if df.empty or "violation_type" not in df.columns:
        return []

    result = (
        df.groupby("violation_type")
        .agg(
            count=("evidence_id", "count"),
            total_fine=("fine_amount", "sum"),
            average_severity=("severity_score", "mean"),
        )
        .reset_index()
    )

    result["average_severity"] = result["average_severity"].round(2)

    return result.to_dict(orient="records")


@app.get("/analytics/hotspots")
def analytics_hotspots():
    records = get_all_records()
    df = records_to_dataframe(records)

    required = {"hotspot", "latitude", "longitude"}

    if df.empty or not required.issubset(set(df.columns)):
        return []

    result = (
        df.groupby(["hotspot", "latitude", "longitude"])
        .agg(
            violations=("evidence_id", "count"),
            total_fine=("fine_amount", "sum"),
            average_severity=("severity_score", "mean"),
        )
        .reset_index()
    )

    result["average_severity"] = result["average_severity"].round(2)

    return result.to_dict(orient="records")


@app.get("/alerts/preview/{evidence_id}")
def alert_preview(evidence_id: str):
    records = get_all_records()

    for record in records:
        if str(record.get("evidence_id")) == evidence_id:
            message = record.get("alert_message")

            if not message:
                message = (
                    f"Demo Alert: Vehicle {record.get('vehicle_number')} detected for "
                    f"{record.get('violation_type')} at {record.get('hotspot')}."
                )

            return {
                "evidence_id": evidence_id,
                "vehicle_number": record.get("vehicle_number"),
                "violation_type": record.get("violation_type"),
                "message": message,
                "status": "DEMO_ONLY_NOT_SENT",
            }

    raise HTTPException(status_code=404, detail="Evidence ID not found")

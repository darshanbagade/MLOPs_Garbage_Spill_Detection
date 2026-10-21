from __future__ import annotations

import base64
import json
import os
import sqlite3
import importlib.util
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

from app.monitoring import summarise_detections, summarise_predictions

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "model_registry.json"
PAGE_PATH = ROOT / "app" / "templates" / "index.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.50"))

with REGISTRY_PATH.open("r", encoding="utf-8") as fh:
    REGISTRY: dict[str, Any] = json.load(fh)

app = FastAPI(
    title="Garbage & Spill Detection MLOps",
    description="YOLOv8 inference API with model metadata, SQLite prediction logs, and operational monitoring.",
    version="1.0.0",
)
_MODEL_CACHE: dict[str, Any] = {}


def get_db_path() -> Path:
    configured = Path(os.getenv("PREDICTIONS_DB", "data/predictions.db"))
    return configured if configured.is_absolute() else ROOT / configured


def connect_db() -> sqlite3.Connection:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db() -> None:
    with connect_db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                file_name TEXT NOT NULL,
                detection_count INTEGER NOT NULL,
                avg_confidence REAL,
                max_confidence REAL,
                class_counts_json TEXT NOT NULL,
                elapsed_ms REAL NOT NULL
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)")


def model_metadata(model_type: str) -> dict[str, Any]:
    if model_type not in REGISTRY["models"]:
        raise HTTPException(status_code=400, detail="model_type must be 'garbage' or 'spill'.")
    raw = REGISTRY["models"][model_type]
    weight_path = ROOT / raw["weights"]
    return {
        **raw,
        "model_type": model_type,
        "weights_available": weight_path.is_file(),
        "weights_size_mb": round(weight_path.stat().st_size / (1024 * 1024), 2) if weight_path.is_file() else None,
        "loaded_in_memory": model_type in _MODEL_CACHE,
    }


def get_model(model_type: str) -> Any:
    """Load a model only on first use to keep health/docs endpoints lightweight."""
    if model_type in _MODEL_CACHE:
        return _MODEL_CACHE[model_type]
    metadata = model_metadata(model_type)
    if not metadata["weights_available"]:
        raise RuntimeError(f"Weights not found: {REGISTRY['models'][model_type]['weights']}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is missing. Run: python -m pip install -r requirements.txt") from exc
    model = YOLO(str(ROOT / REGISTRY["models"][model_type]["weights"]))
    _MODEL_CACHE[model_type] = model
    return model


def insert_prediction(row: dict[str, Any]) -> int:
    ensure_db()
    with connect_db() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
            (created_at, model_type, model_version, file_name, detection_count, avg_confidence,
             max_confidence, class_counts_json, elapsed_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row["created_at"], row["model_type"], row["model_version"], row["file_name"],
             row["detection_count"], row["avg_confidence"], row["max_confidence"],
             json.dumps(row["class_counts"]), row["elapsed_ms"]),
        )
        return int(cur.lastrowid)


def read_history(limit: int = 20) -> list[dict[str, Any]]:
    ensure_db()
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item["class_counts"] = json.loads(item.pop("class_counts_json"))
        output.append(item)
    return output


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PAGE_PATH)


@app.get("/api/health")
def health() -> dict[str, Any]:
    models = {key: model_metadata(key) for key in REGISTRY["models"]}
    missing = [key for key, val in models.items() if not val["weights_available"]]
    runtime_ready = importlib.util.find_spec("ultralytics") is not None and importlib.util.find_spec("torch") is not None
    return {
        "status": "healthy" if not missing and runtime_ready else "degraded",
        "service": "garbage-spill-detection-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_configured": len(models),
        "models_loaded": sum(1 for value in models.values() if value["loaded_in_memory"]),
        "runtime_dependencies_available": runtime_ready,
        "missing_models": missing,
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"project_name": REGISTRY["project_name"], "models": [model_metadata(key) for key in REGISTRY["models"]]}


@app.post("/api/predict")
async def predict(
    model_type: str = Query("garbage", description="garbage or spill"),
    confidence: float = Query(0.25, ge=0.05, le=0.95, description="Minimum detection confidence"),
    image: UploadFile = File(...),
) -> dict[str, Any]:
    metadata = model_metadata(model_type)
    if not metadata["weights_available"]:
        raise HTTPException(status_code=503, detail=f"Model weights are missing for '{model_type}'.")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum upload size is 10 MB.")
    try:
        source_image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=415, detail="Upload a valid JPG, PNG, or WEBP image.") from exc

    start = time.perf_counter()
    try:
        model = get_model(model_type)
        results = model.predict(source=source_image, conf=confidence, verbose=False)
        result = results[0]
        names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                coords = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                class_name = str(names.get(class_id, names.get(str(class_id), class_id))) if isinstance(names, dict) else str(class_id)
                detections.append({"class_id": class_id, "class_name": class_name, "confidence": round(score, 4), "xyxy": coords})
        summary = summarise_detections(detections)
        plotted = result.plot()  # Ultralytics returns a BGR numpy image.
        rendered = Image.fromarray(plotted[:, :, ::-1].copy())
        buffer = BytesIO()
        rendered.save(buffer, format="JPEG", quality=88, optimize=True)
        annotated_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    except HTTPException:
        raise
    except Exception as exc:
        message = str(exc)
        if "ultralytics" in message.lower() or "torch" in message.lower():
            raise HTTPException(status_code=503, detail=f"Model runtime is not ready: {message}") from exc
        raise HTTPException(status_code=500, detail=f"Inference failed: {message[:300]}") from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    created_at = datetime.now(timezone.utc).isoformat()
    filename = Path(image.filename or "upload.jpg").name[:180]
    log_id = insert_prediction({
        "created_at": created_at,
        "model_type": model_type,
        "model_version": metadata["version"],
        "file_name": filename,
        **summary,
        "elapsed_ms": elapsed_ms,
    })
    return {
        "prediction_id": log_id,
        "created_at": created_at,
        "model_type": model_type,
        "model_name": metadata["display_name"],
        "model_version": metadata["version"],
        "file_name": filename,
        "confidence_threshold": confidence,
        "inference_ms": elapsed_ms,
        **summary,
        "detections": detections,
        "annotated_image_base64": annotated_b64,
    }


@app.get("/api/history")
def history(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    rows = read_history(limit)
    return {"count": len(rows), "items": rows}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    # The dashboard intentionally reports the latest 100 inference events as a simple monitoring window.
    rows = read_history(100)
    summary_rows = [{
        "model_type": row["model_type"],
        "detection_count": row["detection_count"],
        "avg_confidence": row["avg_confidence"],
    } for row in rows]
    response = summarise_predictions(summary_rows, LOW_CONFIDENCE_THRESHOLD)
    response["window_size"] = 100
    return response

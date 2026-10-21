# Garbage & Spill Detection — MLOps Demo

A local-first MLOps showcase that serves **your two supplied YOLOv8 checkpoints** through one browser dashboard and API.

- `models/best.pt` — garbage detection checkpoint (copied from the supplied file)
- `models/best_spill2.pt` — spill detection checkpoint (copied from the supplied file)
- FastAPI inference endpoint and interactive Swagger docs
- Single-page dashboard for image upload, annotated detections, model metadata, and recent activity
- SQLite metadata logging and confidence-based operational monitoring
- Training/evaluation scripts, optional MLflow experiment tracking, Docker, and GitHub Actions tests

> **Guidelines PDF:** The referenced guidelines PDF was not present in the attachments I received. This package follows common MLOps project practices, but a line-by-line compliance check against your course's PDF remains pending until it is supplied.

## 1. Run the demo on Windows (recommended)

**Requirements:** Windows 10/11, Python 3.11 recommended (to match the Docker runtime), and an internet connection for first-time package installation. The installer falls back to the Python version on PATH if 3.11 is not installed. CPU inference works; the first prediction can take longer while the model initializes. A GPU is not required for the demo.

1. Extract the ZIP to a simple path, for example `C:\MLOps\Garbage_Spill_MLOps`.
2. Open that folder in VS Code.
3. Double-click `setup_windows.bat` once. Wait for package installation to finish.
4. Double-click `run_windows.bat`.
5. Open **http://127.0.0.1:8000** in your browser.
6. Upload a clear photo, choose **Garbage Detection** or **Spill Detection**, and click **Run detection**.
7. For the API walkthrough, open **http://127.0.0.1:8000/docs**. The health endpoint is **http://127.0.0.1:8000/api/health**.

Alternative terminal setup (run from this project folder):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, use `setup_windows.bat` / `run_windows.bat` instead. The app logs prediction metadata to `data/predictions.db`; it does not persist the uploaded image.

## 2. Quick demo flow for the teacher

1. Show `config/model_registry.json`: two versioned checkpoints and model metadata.
2. Show the dashboard and upload a sample image for garbage detection.
3. Change to spill detection and run a second prediction.
4. Point to annotated bounding boxes, class label, confidence, and latency.
5. Show the Recent prediction activity table: each run creates a SQLite record.
6. Show the monitoring cards and open `/api/health`, `/api/models`, and `/docs`.
7. Explain that GitHub Actions runs unit tests and Docker provides a reproducible deployment option.

Use a genuine sample image that fits each model's training domain. Do not assume a photo must contain an object; an empty prediction is also a valid model output.

## 3. MLOps components

| Area | Implemented in this project | Purpose |
|---|---|---|
| Model packaging/registry | `models/`, `config/model_registry.json` | Keeps each checkpoint, version, label list, and source together |
| Inference service | `app/main.py` | Reproducible HTTP API and health/readiness endpoints |
| User interface | `app/templates/index.html` | Live prediction demo without a separate frontend build |
| Operational logging | `data/predictions.db` | Persists model version, prediction count, confidence summary, and latency |
| Monitoring | `/api/metrics` and dashboard | Summarises last 100 runs and low-confidence rate |
| Training lifecycle | `scripts/train.py` | Retrains and optionally registers a new checkpoint version |
| Offline evaluation | `scripts/evaluate.py` | Computes precision, recall, mAP50 and mAP50–95 on labeled data |
| Experiment tracking (optional) | `requirements-mlflow.txt`, `--mlflow` flags | Logs training/evaluation parameters, metrics, and artifacts to MLflow |
| Automated checks | `tests/`, `.github/workflows/ci.yml` | Tests helper logic and registry integrity on pushes/PRs |
| Container deployment | `Dockerfile`, `docker-compose.yml` | Reproduces the local serving environment |

**Monitoring caveat:** The current monitoring reports confidence and request aggregates, not verified statistical data drift. Confidence is also not accuracy. To make accuracy claims, evaluate a labeled, held-out test split and report the resulting metrics. To claim drift detection, add a reference dataset and validated image/feature distribution comparison.

## 4. API endpoints

- `GET /api/health` — service status and presence of both checkpoint files
- `GET /api/models` — registered model metadata and class names
- `POST /api/predict?model_type=garbage&confidence=0.25` — multipart image inference
- `GET /api/history?limit=20` — latest logged inference events
- `GET /api/metrics` — operational aggregates for the latest 100 prediction events
- `GET /docs` — interactive OpenAPI / Swagger UI

Example PowerShell request after the server starts:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/predict?model_type=garbage&confidence=0.25" -F "image=@C:\path\to\sample.jpg"
```

Choose `model_type=spill` to use the spill checkpoint.

## 5. Optional MLflow experiment UI

MLflow is optional; the app itself does not require it. Once `requirements.txt` is installed, install the optional dependency:

```powershell
python -m pip install -r requirements-mlflow.txt
```

After running a training/evaluation command with `--mlflow`, view experiments in a second terminal:

```powershell
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open **http://127.0.0.1:5000**. Do not present training metrics until training/evaluation has actually run and produced a report.

## 6. Retrain or evaluate (not required to demo inference)

The supplied notebooks establish the YOLOv8 training approach, but the original training datasets are not included in this ZIP. Export/download each dataset in **YOLOv8 detection format** from its Roboflow project, then update a `data.yaml` so `train`, `val`, and (if available) `test` point to real image directories.

Dataset references from your README:

- Spill: https://universe.roboflow.com/mateo-forero-uao1f/spill-detection-sh5mc/dataset/1
- Garbage: https://universe.roboflow.com/proje-nkf76/atik-ayristirma

Example commands (replace paths with your actual exported dataset YAML paths):

```powershell
python scripts/train.py --task garbage --data "C:\datasets\garbage\data.yaml" --epochs 30
python scripts/evaluate.py --task garbage --data "C:\datasets\garbage\data.yaml" --split test
```

To replace the active checkpoint and increment its registry version, add `--register`. Keep a copy of the current checkpoint before doing so. Use `--mlflow` after installing the optional dependency to log run parameters, metrics, and artifacts.

Starter notebook templates are in `notebooks/`. They intentionally do not contain the credential-bearing download URLs from the original Colab notebooks.

## 7. Docker (optional)

Install Docker Desktop and run from the extracted project folder:

```powershell
docker compose up --build
```

Then browse to **http://127.0.0.1:8000**. The image is larger because it installs PyTorch/Ultralytics and includes the checkpoints.

## 8. Troubleshooting

- **`python` is not found:** install Python 3.11 and enable “Add Python to PATH”, then reopen the terminal.
- **PowerShell virtual-environment activation is blocked:** run the supplied `.bat` files, which activate the environment through `cmd.exe`.
- **`python-multipart` error:** activate `.venv` and run `python -m pip install -r requirements.txt` again.
- **Model runtime / PyTorch error:** run `python -m pip install --upgrade -r requirements.txt`; keep the model files under `models/` with their original names.
- **Port 8000 already in use:** stop the other process or run `python -m uvicorn app.main:app --host 127.0.0.1 --port 8001`, then open port 8001.
- **No boxes shown:** try a domain-relevant, clear image and lower the confidence setting slightly. This does not establish that the detector is accurate or inaccurate.

## 9. Security note

The supplied Colab notebooks contained Roboflow download URLs with API credentials embedded in them. Those URLs are deliberately **not copied** into this ZIP. Revoke/rotate those credentials in Roboflow if they are still active, and use Colab Secrets or environment variables for future work.

## 10. Limitations to state honestly

- The ZIP includes trained weight files but not the full training datasets.
- Training notebooks did not provide exported validation/test metric output, so this project does not claim a numerical accuracy score.
- Spill class labels are numeric (`0`–`5`) in the supplied notebook; their meanings must be verified in the dataset's original `data.yaml`.
- It is a classroom/demo MLOps baseline, not a production safety system; human review is required before taking real-world environmental action.

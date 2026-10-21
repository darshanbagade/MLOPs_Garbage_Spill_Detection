"""Re-train one detector and optionally register its best checkpoint.

Example:
  python scripts/train.py --task garbage --data path/to/garbage/data.yaml --epochs 30 --mlflow
  python scripts/train.py --task spill --data path/to/spill/data.yaml --epochs 25 --register
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REGISTRY_PATH = ROOT / "config" / "model_registry.json"


def optional_mlflow(enabled: bool):
    if not enabled:
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("MLflow was requested but is not installed. Run: python -m pip install -r requirements-mlflow.txt") from exc
    mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
    mlflow.set_experiment("garbage-spill-detection")
    return mlflow


def last_metrics(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    last = rows[-1]
    metrics: dict[str, float] = {}
    for key, value in last.items():
        if value is None or not value.strip():
            continue
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def bump_version(version: str) -> str:
    match = re.search(r"-v(\d+)$", version)
    return f"{version[:match.start()]}-v{int(match.group(1)) + 1}" if match else version + "-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the garbage or spill YOLO detector.")
    parser.add_argument("--task", required=True, choices=["garbage", "spill"])
    parser.add_argument("--data", required=True, help="Path to a YOLO data.yaml file for the relevant dataset.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="Optional device, e.g. 0 or cpu. Default lets Ultralytics choose.")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Base checkpoint used for fine-tuning.")
    parser.add_argument("--register", action="store_true", help="Copy the best checkpoint to models/ and update the model registry.")
    parser.add_argument("--mlflow", action="store_true", help="Log training params/metrics/artifacts to local MLflow tracking.")
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_path}\nDownload/export the dataset and update the --data path.")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise SystemExit("epochs, imgsz, and batch must be positive values (imgsz >= 32).")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is missing. Run: python -m pip install -r requirements.txt") from exc

    mlflow = optional_mlflow(args.mlflow)
    run_context = mlflow.start_run(run_name=f"{args.task}-{date.today().isoformat()}") if mlflow else _NullContext()
    with run_context as run:
        if mlflow:
            mlflow.log_params({"task": args.task, "base_model": args.base_model, "data_yaml": str(data_path), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "device": args.device or "auto"})
        model = YOLO(args.base_model)
        train_kwargs: dict[str, Any] = {
            "data": str(data_path), "epochs": args.epochs, "imgsz": args.imgsz,
            "batch": args.batch, "project": str(ROOT / "runs" / "train"),
            "name": f"{args.task}-{date.today().isoformat()}", "exist_ok": True,
        }
        if args.device:
            train_kwargs["device"] = args.device
        results = model.train(**train_kwargs)
        save_dir = Path(results.save_dir).resolve()
        best_path = save_dir / "weights" / "best.pt"
        results_csv = save_dir / "results.csv"
        if not best_path.is_file():
            raise SystemExit(f"Training finished, but best checkpoint was not found: {best_path}")
        metrics = last_metrics(results_csv)
        report_path = ROOT / "reports" / f"{args.task}-training-{date.today().isoformat()}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"task": args.task, "data_yaml": str(data_path), "save_dir": str(save_dir), "best_checkpoint": str(best_path), "metrics_from_final_csv_row": metrics}, indent=2), encoding="utf-8")
        print(f"\nTraining output: {save_dir}")
        print(f"Best weights:    {best_path}")
        print(f"Metrics report:  {report_path}")
        if args.register:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            entry = registry["models"][args.task]
            target = ROOT / entry["weights"]
            shutil.copy2(best_path, target)
            entry["version"] = bump_version(entry["version"])
            entry["checkpoint_date"] = date.today().isoformat()
            entry["training_source"] = f"scripts/train.py ({data_path.name})"
            names = getattr(model, "names", None)
            if isinstance(names, dict):
                entry["classes_from_training_notebook"] = [str(names[k]) for k in sorted(names)]
            entry["notes"] = "Updated by scripts/train.py; evaluate on a held-out labelled dataset before deployment claims."
            registry["registry_version"] = int(registry.get("registry_version", 1)) + 1
            REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
            print(f"Registered as {entry['version']} at {target.relative_to(ROOT)}")
        else:
            print("Not registered. Use --register to replace the active model and increment its registry version.")
        if mlflow:
            for key, value in metrics.items():
                if any(term in key.lower() for term in ("precision", "recall", "map", "fitness", "loss")):
                    mlflow.log_metric(key.replace(" ", "_"), value)
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            if results_csv.exists():
                mlflow.log_artifact(str(results_csv), artifact_path="training")
            mlflow.log_artifact(str(best_path), artifact_path="checkpoint")
            print(f"MLflow run: {run.info.run_id}")


class _NullContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()

"""Evaluate a registered YOLO detector on a labelled validation/test split."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one packaged detector against a YOLO data.yaml.")
    parser.add_argument("--task", choices=["garbage", "spill"], required=True)
    parser.add_argument("--data", required=True, help="YOLO data.yaml containing a labelled val/test split.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mlflow", action="store_true", help="Log evaluation metrics to local MLflow tracking.")
    args = parser.parse_args()
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML not found: {data}")
    registry = json.loads((ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))
    weights = ROOT / registry["models"][args.task]["weights"]
    if not weights.is_file():
        raise SystemExit(f"Model weights not found: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is missing. Run: python -m pip install -r requirements.txt") from exc
    mlflow = None
    run_context = None
    if args.mlflow:
        try:
            import mlflow as mlflow_module
        except ImportError as exc:
            raise SystemExit("MLflow was requested but is not installed. Run: python -m pip install -r requirements-mlflow.txt") from exc
        mlflow = mlflow_module
        mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
        mlflow.set_experiment("garbage-spill-detection")
        run_context = mlflow.start_run(run_name=f"eval-{args.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    else:
        from contextlib import nullcontext
        run_context = nullcontext()
    with run_context:
        if mlflow:
            mlflow.log_params({"task": args.task, "model_version": registry["models"][args.task]["version"], "data_yaml": str(data), "split": args.split, "imgsz": args.imgsz})
        model = YOLO(str(weights))
        results = model.val(data=str(data), split=args.split, imgsz=args.imgsz, project=str(ROOT / "runs" / "eval"), name=args.task, exist_ok=True)
        box = getattr(results, "box", None)
        metrics = {}
        if box is not None:
            for key, attr in [("precision", "mp"), ("recall", "mr"), ("mAP50", "map50"), ("mAP50_95", "map")]:
                value = getattr(box, attr, None)
                if value is not None:
                    metrics[key] = float(value)
        report = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "task": args.task, "model_version": registry["models"][args.task]["version"], "weights": str(weights.relative_to(ROOT)), "data_yaml": str(data), "split": args.split, "metrics": metrics, "note": "Metrics only apply to the supplied labelled split; do not report them as general real-world accuracy."}
        out = ROOT / "reports" / f"{args.task}-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Saved: {out}")
        if mlflow and metrics:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out), artifact_path="evaluation")

if __name__ == "__main__":
    main()

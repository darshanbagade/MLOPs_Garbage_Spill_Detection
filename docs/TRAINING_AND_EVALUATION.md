# Training, evaluation and model updates

## Training data

The original uploaded README names these datasets:

- Spill: https://universe.roboflow.com/mateo-forero-uao1f/spill-detection-sh5mc/dataset/1
- Garbage: https://universe.roboflow.com/proje-nkf76/atik-ayristirma

The original notebooks used Ultralytics YOLOv8 and training durations of 25 epochs (spill) and 30 epochs (garbage). Their output cells with usable evaluation metrics were not provided, so no performance figure is inferred here. The dataset downloads are not included in the deliverable.

## Train without registering

Export the dataset in YOLOv8 detection format. Ensure `data.yaml` points to valid image/label directories and the class names align with the dataset. Then run from the project root:

```powershell
python scripts/train.py --task garbage --data "C:\datasets\garbage\data.yaml" --epochs 30 --imgsz 640
python scripts/train.py --task spill --data "C:\datasets\spill\data.yaml" --epochs 25 --imgsz 640
```

A training run is saved under `runs/train/`; the final-row training metrics are written to `reports/`. Training can take significant time and GPU resources. It is not required for the demo because the current trained checkpoints are included.

## Evaluate on a labeled test split

```powershell
python scripts/evaluate.py --task garbage --data "C:\datasets\garbage\data.yaml" --split test
python scripts/evaluate.py --task spill --data "C:\datasets\spill\data.yaml" --split test
```

The script reports precision, recall, mAP50 and mAP50–95 when exposed by the installed Ultralytics version. Use test data that was not used for training/model selection. A good report should also record the dataset version, class mapping, sample count and limitations.

## Register a new model version

Back up the existing `models/*.pt` file, then run training with `--register`:

```powershell
python scripts/train.py --task garbage --data "C:\datasets\garbage\data.yaml" --epochs 30 --register
```

This explicitly copies the new best checkpoint to the configured model path, increments the registry version and refreshes the class list from training metadata. Commit the registry update and associate the new model weights with the same source-control release. For team-scale workflows, store larger weights in an artifact store or Git LFS.

## Optional MLflow

Install `requirements-mlflow.txt`, then add `--mlflow` to a training or evaluation command. Runs log parameters and any available metrics/artifacts to the local `mlruns/` directory. Start the UI with:

```powershell
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Visit `http://127.0.0.1:5000`. MLflow is optional for inference.

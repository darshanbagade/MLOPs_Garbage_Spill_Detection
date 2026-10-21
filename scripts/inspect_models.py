"""Print the project model registry and whether bundled checkpoints exist."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
registry = json.loads((ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))
print(registry["project_name"])
for key, model in registry["models"].items():
    path = ROOT / model["weights"]
    size = f"{path.stat().st_size / 1024 / 1024:.2f} MB" if path.is_file() else "MISSING"
    print(f"- {key}: {model['version']} | {model['display_name']} | {size} | {'READY' if path.is_file() else 'NOT READY'}")
    print(f"  Labels: {', '.join(model['classes_from_training_notebook'])}")

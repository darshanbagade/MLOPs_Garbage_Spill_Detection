import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_both_checkpoints_and_six_labels():
    data = json.loads((ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))
    assert set(data["models"]) == {"garbage", "spill"}
    for model in data["models"].values():
        assert (ROOT / model["weights"]).is_file()
        assert len(model["classes_from_training_notebook"]) == 6

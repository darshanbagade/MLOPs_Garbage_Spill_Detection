from fastapi.testclient import TestClient

from app.main import app


def test_health_lists_bundled_models():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["models_configured"] == 2
    assert body["missing_models"] == []
    assert body["status"] in {"healthy", "degraded"}


def test_model_registry_endpoint_has_both_models():
    with TestClient(app) as client:
        response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert {model["model_type"] for model in body["models"]} == {"garbage", "spill"}


def test_invalid_model_type_is_rejected_before_inference():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict?model_type=unknown",
            files={"image": ("sample.jpg", b"not-an-image", "image/jpeg")},
        )
    assert response.status_code == 400
    assert "model_type" in response.json()["detail"]

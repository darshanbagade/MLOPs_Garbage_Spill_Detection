# Guideline mapping status

The assignment guideline PDF referenced in the request was not attached in the files available for this build, so this document is a provisional mapping to common MLOps expectations—not a claim of compliance with the unseen rubric.

| Common expectation | Project item | Evidence/demo |
|---|---|---|
| Reuse trained models | `models/*.pt` | Registry page and two live inferences |
| Serving API | `app/main.py` | `/docs`, `/api/health`, `/api/predict` |
| Simple UI | `app/templates/index.html` | Local dashboard |
| Reproducible dependencies | `requirements.txt`, setup scripts, Dockerfile | Fresh environment setup |
| Model version metadata | `config/model_registry.json` | Registry panel / `/api/models` |
| Prediction logs and monitoring | SQLite + `/api/history` + `/api/metrics` | Run two predictions, refresh activity |
| Training and evaluation | `scripts/train.py`, `scripts/evaluate.py` | Explain commands; execute only with dataset YAML |
| Experiment tracking | Optional MLflow files/flags | Runs if installed and invoked |
| Automated testing | `tests/`, GitHub Actions workflow | `python -m pytest -q` / workflow tab after push |
| Container option | `Dockerfile`, `docker-compose.yml` | Optional `docker compose up --build` |

Once the course PDF is available, compare every mandatory item against it and adjust the project before claiming full rubric compliance. In particular, a course may require specific tools, screenshots, diagrams, report chapters, or a particular deployment target not inferable from the attached notebooks.

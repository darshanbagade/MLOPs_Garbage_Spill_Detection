# Architecture and request lifecycle

```mermaid
flowchart TD
    A[Browser dashboard] -->|Upload image| B[FastAPI service]
    B --> C{Requested model}
    C -->|garbage| D[models/best.pt]
    C -->|spill| E[models/best_spill2.pt]
    D --> F[YOLOv8 inference]
    E --> F
    F --> G[Boxes, labels, confidence]
    G --> H[Annotated image response]
    G --> I[SQLite prediction metadata]
    I --> J[History and confidence monitoring]
    K[Training script] --> L[Training run and metrics]
    L --> M{Explicit --register?}
    M -->|yes| N[Replace checkpoint + increment registry version]
    M -->|no| O[Keep current production/demo checkpoint]
    N --> D
    N --> E
    L -. optional .-> P[MLflow experiment tracking]
    B -. tests and container build .-> Q[GitHub Actions / Docker]
```

## Boundaries

- The browser handles image selection and visualizes API responses.
- FastAPI validates the upload, loads the selected YOLO model lazily, runs prediction, and returns boxes plus an annotated JPEG.
- SQLite stores metadata only, not uploaded source images.
- Model registry JSON is a lightweight project-level registry; it is not a managed enterprise model registry.
- Monitoring in this version is an operational baseline (counts, response latency, confidence distribution), not a certified drift detector.

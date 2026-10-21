# Presentation checklist (5–7 minutes)

## Before class

- [ ] Extract the ZIP to a writable local folder.
- [ ] Run `setup_windows.bat` and wait for dependencies to install.
- [ ] Run `run_windows.bat`; keep that terminal open.
- [ ] Verify `http://127.0.0.1:8000/api/health` reports healthy.
- [ ] Prepare one garbage image and one spill-related image saved locally. Use images allowed by the course rules.
- [ ] Keep `http://127.0.0.1:8000` and `/docs` ready in separate browser tabs.

## Suggested explanation

1. **Problem:** environmental monitoring needs faster identification of visible garbage/spill events.
2. **Models:** two already-trained YOLOv8 detection checkpoints are packaged locally.
3. **Serving:** FastAPI exposes model selection through a stable HTTP endpoint; the frontend calls that endpoint.
4. **Monitoring:** each request logs timestamp, model version, object count, confidence summary and latency into SQLite.
5. **Lifecycle:** the training script writes a training run and report; `--register` is an explicit action that updates the model artifact/version.
6. **Reproducibility and quality:** requirements, tests, Docker packaging and GitHub Actions form the basic delivery workflow; evaluation uses a labeled held-out set.

## Live steps

- [ ] Show model registry metadata.
- [ ] Run garbage detection and point at annotated boxes and confidence values.
- [ ] Run spill detection.
- [ ] Show logged activity and monitoring values changing.
- [ ] Open `/api/health`, `/api/models`, and `/docs`.
- [ ] State limitations: no reported accuracy until evaluation; numeric spill labels need mapping; confidence is not accuracy.

## Demo fallback

If first-time installation or PyTorch import fails, do not pretend inference succeeded. Show the API/docs, architecture, training/evaluation scripts, and registry code. Use the original runtime error to troubleshoot rather than fabricating detections.

# Validation Report

Date: 2026-06-22

## Docker End-To-End

Status: blocked by Docker Desktop engine/proxy health.

Evidence:

- `docker --version`: Docker CLI available.
- `docker compose config --quiet`: compose syntax valid.
- `docker compose up --build -d`: initially started containers inside the Docker Desktop WSL VM, but Windows Docker API later returned HTTP 500 for `/version`, `/containers/json`, and `/images/json`.
- Docker Desktop backend logs reported waiting for the VM init control API and `backend is not running`.
- Windows port checks for `localhost:80`, `localhost:8000`, and `localhost:5432` failed because Docker Desktop port publishing was not functioning.
- Inside the Docker Desktop VM, running processes were observed for Redis, Postgres, Uvicorn, Celery worker, and nginx.
- Alembic migration was applied inside the backend container namespace using the Postgres container IP.

Conclusion: app containers can run, but Docker Desktop must be restarted/repaired until `docker ps` and published localhost ports work before the formal nginx-based gate can pass.

## Real Data

Source: `PHOTOALBUM/`.

- 40 images copied to `data/raw_album_pages/`.
- Blur scoring: 12 images at or above threshold `100.0`; 28 below threshold.
- Provisional split created:
  - 25 train images
  - 5 validation images
  - 10 golden fixture images
- This split is not production-valid yet because most images fail the blur threshold and labels are not annotated.

## Local Pipeline Smoke

Command:

```bash
python scripts/run_real_data_smoke.py
```

Result:

- 40 pages processed
- 40 completed
- 0 failed
- 85 photos extracted

This validates deterministic local pipeline behavior on the supplied real pages, using classical segmentation fallback.

Segmentation-only smoke (`SMOKE_LIMIT=3`, `SMOKE_SEGMENTATION_ONLY=true`) completed quickly with 3 pages, 9 extracted photos, and 0 failed pages.

## Load Dataset

Command:

```bash
python scripts/generate_load_dataset.py --source data/raw_album_pages --output data/load_dataset --variants 10
```

Result: 400 JPEG variants generated.

## Security And Tests

- `python -m pytest -q`: 44 passed, 2 skipped pending golden labels.
- `pytest --cov=backend/app --cov-report=term-missing`: 74% coverage.
- `python -m bandit -r backend/app`: no issues.
- `npm audit --audit-level=high`: 0 vulnerabilities.

## Remaining Gates

- Docker Desktop API/port publishing must be healthy enough for `docker compose`, `docker exec`, and nginx `http://localhost` validation.
- Label Studio annotation is required for YOLO training and golden IoU regression. `scripts/export_label_studio_tasks.py` now creates import tasks with optional classical preannotations, `scripts/convert_label_studio_to_yolo.py` converts reviewed Label Studio JSON exports to YOLO labels, and `scripts/export_reviewed_yolo_labels.py` exports masks corrected inside PhoSca to `data/label_exports/yolo/`. Then run `scripts/prepare_yolo_dataset.py` to recreate train/val/golden splits. `python scripts/validate_yolo_dataset.py --data data/data.yaml` currently reports 25 train images, 5 validation images, 0 train labels, 0 validation labels, and missing golden labels.
- PaddleOCR model files are not present under `models/paddleocr/`.
- `models/yolov8-seg-album.onnx` is not present, so ONNX segmentation quality cannot be measured yet. `scripts/train_segmentation_model.py` now provides the reproducible train/export path once labels exist.
- `models/photo-orientation.onnx` is not present. `scripts/train_orientation_model.py` now provides the reproducible train/export path once curated upright crops exist under `data/orientation_photos/`.
- Golden OCR transcripts in `data/golden_fixtures/ocr_ground_truth.json` are placeholders and need manual text.

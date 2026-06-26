# Photo Album Digitization Platform

Local-first photo album digitization stack for high-resolution album page ingestion, segmentation, OCR review, enhancement, metadata persistence, and manual correction.

## Run With Docker Compose

1. Copy `.env.example` to `.env`.
2. Set `HOST_STORAGE_PATH` to the absolute path of this repository's `storage` directory.
3. Start the stack:

```bash
docker compose up --build
```

Services:

- Frontend dashboard: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Metrics: `http://localhost:8000/metrics`

The backend and worker run as UID `1000`, containers use `no-new-privileges`, and the storage volume is configured with `noexec,nosuid,nodev`.

## Database Migrations

Inside the backend container:

```bash
alembic upgrade head
```

The initial migration creates albums, pages, extracted photos, OCR results, audit logs, indexes, and a PostgreSQL full-text index over OCR text.

## Local Development

For a lightweight SQLite-backed dev run:

```powershell
$env:DATABASE_URL='sqlite+aiosqlite:///./dev.db'
$env:REDIS_URL='redis://localhost:6379/0'
$env:STORAGE_PATH='C:\Users\gabri\Documents\Projects\PhoSca\storage'
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm run dev
```

## ML Models

Place local-only model files under `models/`:

- `models/yolov8-seg-album.onnx`
- `models/photo-orientation.onnx` (optional 0/90/180/270 classifier)
- `models/paddleocr/`

The segmentation ONNX parser supports YOLO-style box outputs and segmentation prototype masks, including channels-first and channels-last prototype tensors. If ONNX weights are absent, the pipeline records that fact and uses the classical contour fallback when enabled. Photo orientation correction is enabled by default; it uses `models/photo-orientation.onnx` when present, otherwise it falls back to a conservative face/detail heuristic. Train the optional orientation model from manually reviewed upright crops with `scripts/train_orientation_model.py`. PaddleOCR is loaded lazily; OCR can also be tested with `.ocr.json` sidecars next to uploaded page images.

## Tests And Scans

```bash
python -m pytest
cd frontend && npm run build && npm audit
python -m bandit -r backend/app
```

Current validation snapshot:

- Supplied dataset copied from `PHOTOALBUM/` to `data/raw_album_pages/`: 40 album-page JPEGs.
- Blur gate at `100.0`: 12 keep, 28 discard. See `data/blur_scores.csv`.
- Local real-data smoke run: 40 pages processed, 0 failures, 85 photos extracted with classical segmentation fallback.
- Synthetic load dataset generated: 400 images in `data/load_dataset/`.
- Test coverage: 74% line coverage with `pytest --cov=backend/app --cov-report=term-missing`.
- Security scans: `bandit -r backend/app` clean; `npm audit --audit-level=high` clean.

Docker Desktop note: containers were observed running inside the `docker-desktop` WSL VM, but the Windows Docker API/port proxy returned HTTP 500 and did not publish ports to `localhost`. Logs showed Docker Desktop waiting on the VM init control API. Restart Docker Desktop fully if `docker ps` returns API 500, then rerun `docker compose up --build -d`.

Fast segmentation-only smoke check:

```powershell
$env:SMOKE_LIMIT='3'
$env:SMOKE_SEGMENTATION_ONLY='true'
python scripts/run_real_data_smoke.py
```

Load-test scaffold:

```bash
locust -f locustfile.py --host http://localhost:8000
```

## Backups

See `docs/OPERATIONS.md` and `scripts/backup.sh`. Backups include PostgreSQL dumps and storage syncs, encrypted with GPG.

## Annotation And Regression

See `docs/SEGMENTATION_STRATEGY.md` for the failure analysis, researched options, and selected segmentation/orientation architecture. Use `docs/LABEL_STUDIO.md` to create YOLO polygon annotations. The golden regression tests remain skipped until `data/golden_fixtures/labels/*.txt` contains 10 locked labels. After processing the golden album, run:

```bash
python scripts/export_label_studio_tasks.py --source-images data/raw_album_pages --output data/label_studio_tasks.json --preannotate
python scripts/convert_label_studio_to_yolo.py --input data/label_studio_export.json --output data/label_exports/yolo
# Or export masks reviewed inside PhoSca directly:
python scripts/export_reviewed_yolo_labels.py --album-id <album-id> --output data/label_exports/yolo --manual-only --require-complete
python scripts/prepare_yolo_dataset.py --source-images data/raw_album_pages --source-labels data/label_exports/yolo
python scripts/validate_yolo_dataset.py --data data/data.yaml
python scripts/train_segmentation_model.py --data data/data.yaml --export models/yolov8-seg-album.onnx
python scripts/train_orientation_model.py --images data/orientation_photos --output models/photo-orientation.onnx
python scripts/validate_segmentation.py --album-id <album-id>
python scripts/validate_ocr.py --album-id <album-id>
```

`scripts/export_label_studio_tasks.py` creates Label Studio import JSON and can seed editable polygon predictions from the classical detector. `scripts/convert_label_studio_to_yolo.py` converts reviewed Label Studio JSON exports into YOLO segmentation labels. `scripts/export_reviewed_yolo_labels.py` exports masks corrected in the PhoSca review canvas into the same YOLO label directory. `scripts/prepare_yolo_dataset.py` recreates deterministic train/val/golden splits from raw images plus exported YOLO labels. `scripts/validate_yolo_dataset.py` checks label completeness, polygon validity, and golden leakage before training. `scripts/train_segmentation_model.py` lazily imports Ultralytics; install it only in the training environment if it is not already available. `scripts/train_orientation_model.py` requires PyTorch plus `onnx`, uses manually upright crops under `data/orientation_photos`, and exports a 0/90/180/270 correction classifier. `scripts/validate_segmentation.py` measures polygon IoU from saved segmentation masks and falls back to boxes only for legacy rows without masks. Production sign-off requires mean IoU `> 0.85` and mean CER `< 0.10`.

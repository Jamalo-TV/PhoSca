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
- CPU worker queue: decode, geometry, hashing, persistence
- Analysis queue: segmentation/OCR, one resident process per accelerator
- Restoration queue: optional enhancement, one resident process per accelerator

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
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd frontend
npm run dev
```

The development servers listen on all network interfaces. Open
`https://<PC-IP>:5173/mobile` on a phone in the same WLAN (for example,
`https://192.168.1.20:5173/mobile`). HTTPS is required for the live in-browser
camera. Trust the local certificate on the phone once; otherwise the browser
will block `getUserMedia`. The desktop interface remains available at the root
URL. Windows Firewall may ask once for permission on private networks.

Original mobile captures are stored unchanged below
`storage/uploads/<album-id>/`. The database keeps the original filename,
SHA-256 hash and the generated page ID. Derived photos are stored separately.

## ML Models

Place local-only model files under `models/`:

- `models/yolov8-seg-album.onnx`
- `models/photo-orientation.onnx` (optional 0/90/180/270 classifier)
- `models/paddleocr/`

The segmentation ONNX parser supports YOLO-style box outputs and segmentation prototype masks, including channels-first and channels-last prototype tensors. Classical boundary analysis is enabled as risk-triggered geometric evidence: it runs when the model is missing, empty, or produces a suspicious instance set, and is fused without suppressing legitimate overlaps. Photo orientation correction uses `models/photo-orientation.onnx` when present; without a calibrated model, face/detail evidence is advisory only and pixels are preserved for review. Train the optional orientation model from manually reviewed upright crops with `scripts/train_orientation_model.py`. PaddleOCR is loaded lazily; OCR can also be tested with `.ocr.json` sidecars next to uploaded page images.

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

See `docs/SEGMENTATION_STRATEGY.md` for the failure analysis, researched options, and selected segmentation/orientation architecture. The shortest path for YOLO polygon annotations is the workspace under `data/segmentation_training/`:

```bash
python scripts/setup_segmentation_training.py bootstrap
python scripts/convert_label_studio_to_yolo.py --input data/segmentation_training/exports/label_studio_export.json --output data/segmentation_training/labels
python scripts/setup_segmentation_training.py train
```

`bootstrap` copies the existing example pages from `data/raw_album_pages`, creates editable Label Studio preannotations, and writes `data/segmentation_training/label_studio_tasks.json`. Review complete printed photos as `photo` polygons, export the reviewed JSON into `data/segmentation_training/exports/label_studio_export.json`, then run the conversion and training commands. The training command prepares `data/yolo_dataset`, validates labels, fine-tunes the YOLO segmentation model, and exports `models/yolov8-seg-album.onnx`.

Use `docs/LABEL_STUDIO.md` for the labeling UI details. Ten locked golden pages and their verified polygon labels are available as a same-domain regression holdout; they are not an album-independent consumer benchmark. The lower-level commands remain available:

```bash
python scripts/export_label_studio_tasks.py --source-images data/raw_album_pages --output data/label_studio_tasks.json --preannotate
python scripts/convert_label_studio_to_yolo.py --input data/label_studio_export.json --output data/label_exports/yolo
# Or export masks reviewed inside PhoSca directly:
python scripts/export_reviewed_dataset.py --database-url $DATABASE_URL --output data/reviewed_dataset
python scripts/split_reviewed_dataset.py --input data/reviewed_dataset/manifest.jsonl --output data/reviewed_dataset/splits
# Derive legacy YOLO labels only from explicitly approved, valid geometry:
python scripts/export_reviewed_yolo_labels.py --album-id <album-id> --output data/label_exports/yolo --manual-only --require-complete
python scripts/prepare_yolo_dataset.py --source-images data/raw_album_pages --source-labels data/label_exports/yolo
python scripts/validate_yolo_dataset.py --data data/data.yaml
python scripts/train_segmentation_model.py --data data/data.yaml --export models/yolov8-seg-album.onnx
python scripts/train_orientation_model.py --images data/orientation_photos --output models/photo-orientation.onnx
python scripts/validate_segmentation.py --album-id <album-id>
python scripts/validate_ocr.py --album-id <album-id>
```

Safety and operations evidence:

```bash
python scripts/evaluate_onnx_segmentation.py --include-classical-fallback --output runs/cropping-current.json
python scripts/evaluate_enhancement.py --output-dir runs/enhancement-current
python scripts/fit_confidence_calibration.py labelled-predictions.json --holdout-id <independent-holdout-id>
python scripts/evaluate_shadow_drift.py stable.json shadow.json --output evidence/shadow-drift.json
python scripts/run_retention.py
python scripts/verify_launch_gates.py evidence/launch-gates.json
```

`scripts/setup_segmentation_training.py` orchestrates the normal bootstrap/prepare/train loop. `scripts/export_label_studio_tasks.py` creates Label Studio import JSON and can seed editable polygon predictions from the classical detector. `scripts/convert_label_studio_to_yolo.py` converts reviewed Label Studio JSON exports into YOLO segmentation labels. `scripts/export_reviewed_yolo_labels.py` exports masks corrected in the PhoSca review canvas into the same YOLO label directory. `scripts/prepare_yolo_dataset.py` recreates deterministic train/val/golden splits from raw images plus exported YOLO labels. `scripts/validate_yolo_dataset.py` checks label completeness, polygon validity, and golden leakage before training. `scripts/train_segmentation_model.py` lazily imports Ultralytics; install it only in the training environment if it is not already available. `scripts/train_orientation_model.py` requires PyTorch plus `onnx`, uses manually upright crops under `data/orientation_photos`, and exports a 0/90/180/270 correction classifier. `scripts/validate_segmentation.py` measures polygon IoU from saved segmentation masks and falls back to boxes only for legacy rows without masks. Production sign-off requires mean IoU `> 0.85` and mean CER `< 0.10`.

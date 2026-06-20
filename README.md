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
- `models/paddleocr/`

If ONNX weights are absent, the pipeline records that fact and uses the classical contour fallback when enabled. PaddleOCR is loaded lazily; OCR can also be tested with `.ocr.json` sidecars next to uploaded page images.

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

Load-test scaffold:

```bash
locust -f locustfile.py --host http://localhost:8000
```

## Backups

See `docs/OPERATIONS.md` and `scripts/backup.sh`. Backups include PostgreSQL dumps and storage syncs, encrypted with GPG.

## Annotation And Regression

Use `docs/LABEL_STUDIO.md` to create YOLO polygon annotations. The golden regression tests remain skipped until `data/golden_fixtures/labels/*.txt` contains 10 locked labels. After processing the golden album, run:

```bash
python scripts/validate_segmentation.py --album-id <album-id>
python scripts/validate_ocr.py --album-id <album-id>
```

Production sign-off requires mean IoU `> 0.85` and mean CER `< 0.10`.

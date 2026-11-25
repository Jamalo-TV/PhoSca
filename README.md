# PhoSca

Local-first photo album digitization stack for high-resolution album page ingestion, segmentation, OCR review, enhancement, metadata persistence, and manual correction.

## Wie PhoSca funktioniert

PhoSca speichert die Aufnahme einer Albumseite als Originaldatei und
erzeugt pro physischem Papierabzug eigene Bilddateien. Ein Kontaktabzug
mit mehreren Bildfeldern zählt als ein Abzug. Zwei benachbarte
Einzelabzüge sollen zwei getrennte Ergebnisse liefern. Die Desktop-Ansicht
enthält den vollständigen Zuschnitt-Editor. Unter `/mobile` lassen sich
Seiten aufnehmen oder hochladen, die Abzugsanzahl prüfen, Papierecken
korrigieren und verfügbare Bilder ansehen und herunterladen.

### 1. Aufnahme, Upload und Speicherung

Die React-Oberfläche legt ein Album an und sendet die Dateien an FastAPI.
Mobil kann sie grosse Dateien in Teilen übertragen und bei fehlender
Verbindung in IndexedDB zwischenspeichern. Der Server prüft Dateityp,
Lesbarkeit, Dateigrösse und Pixelzahl. Er speichert die unveränderten
Originalbytes mit SHA-256 unter `storage/uploads/<album-id>/`. Die
Datenbank verknüpft Album-ID, Seiten-ID, Dateiname und Status. Beim
Wiederholen eines mobilen Uploads kann der Server die Seite anhand ihres
Hashes wiederverwenden. Für die Arbeit mit dem Bild berücksichtigt er
die EXIF-Ausrichtung. Bei HDR-JPEGs bleibt die Gain-Map in der
Originaldatei erhalten.
[Upload](backend/app/routers/pages.py) · [Speicherung](backend/app/services/uploads.py) ·
[Offline-Warteschlange](frontend/src/offlineQueue.js)

### 2. Abzüge auf der Seite finden

Der Upload startet bei `AUTO_ANALYZE_UPLOADS=true` die
Hintergrundverarbeitung im API-Dienst. Für getrennt eingereihte
Analysen stehen Celery und Redis zur Verfügung. Die Verarbeitung
kontrolliert den Datei-Hash erneut und misst die Unschärfe. Der
konfigurierte YOLO-ONNX-Detektor liefert Vorschläge für einzelne
Abzüge und kann ihre sichtbaren Flächen als Masken schätzen.
Bei fehlenden, auffälligen oder schwierigen Vorschlägen ergänzt
OpenCV Konturen, Kontrastflächen oder Papierränder. Dichte Seiten kann
das System zusätzlich in überlappenden Teilbildern untersuchen.

PhoSca gleicht Vorschläge mit den Originalpixeln ab. Es prüft
überlappende Masken, Helligkeits- und Farbverläufe entlang möglicher
Papierkanten sowie Anzeichen für Albumfenster, Negativhüllen, getrennte
Abzüge oder ein gemeinsames Papier mit mehreren Bildfeldern. Das
System kann Teilvorschläge zusammenführen oder bei unklarer Zuordnung
eine zusätzliche Kontur zur Prüfung anzeigen. Die sichtbare Maske und
die vier Eckpunkte für die Entzerrung werden getrennt gespeichert.
[Verarbeitungsdienst](backend/app/services/pipeline.py) ·
[Erkennung und Zusammenführung](backend/app/pipeline/segmentation.py)

### 3. Zwei Zuschnitte aus denselben Originalpixeln bilden

| Modus | Ergebnis |
| --- | --- |
| **Ganzer Abzug** | Der entzerrte physische Abzug bis zur Papierkante. Sichtbarer Bildinhalt und der echte Papierrand bleiben erhalten. |
| **Nur Bildmotiv** | Der entzerrte Bildbereich innerhalb der Motivgrenze. Bei einem verifizierten randlosen Abzug können beide Modi dasselbe Bild verwenden. |

Für **Ganzer Abzug** verfeinert PhoSca die vermuteten Papierecken
mit Messungen am Original. Eine perspektivische Projektion bildet
das Viereck auf ein Rechteck ab. Für das Seitenverhältnis nutzt
PhoSca nach Möglichkeit eingegebene Papiermasse, eine zur Aufnahme
passende Kamerakalibrierung oder Hinweise auf die Papierebene.
Unzureichende Messpunkte, Wölbung, Verdeckung oder widersprüchliche
Kanten können eine Prüfung auslösen.

Für **Nur Bildmotiv** untersucht PhoSca zusätzlich die Übergänge
zwischen Papier und Bild. Der Renderer berechnet beide Ausgaben
direkt aus der Quellseite. Beim präzisen Sampling gehen in die
Interpolation nur Quellpixel ein, die zur jeweiligen Fläche gehören.
Die Randprüfung sucht auf den vorgeschlagenen hellen Abfallstreifen
nach dunklen Markierungen und bestätigt den Kontrast im Original.
Bei schwacher Schrift prüft sie zusätzlich die Strichform und die
Originalpixel auf beiden Seiten des Strichs. Beide Seiten müssen
zum selben sichtbaren Papierrand gehören.
Sie kann einen Ausschnitt zurückhalten, wenn er Bildinhalt oder
Beschriftung entfernen würde. Die Markierung zeigt eine Fundstelle.
Der Hinweis fordert dazu auf, den gesamten Rand zu prüfen. Wenn
Fotoecken Papier verdecken, kann **Ganzer Abzug** in Prüfung liegen,
während das vollständig sichtbare Motiv geometrisch verwendbar ist.
[Perspektive und Modi](backend/app/pipeline/perspective.py) ·
[Quellpixel-Sampling](backend/app/pipeline/crop_sampling.py) ·
[Motivrand und Beschriftung](backend/app/pipeline/motif_margin.py)

### 4. Prüfen, korrigieren und exportieren

Die Warteschlange führt offene Fragen zur Abzugsanzahl, Geometrie,
Texterkennung und Bildverbesserung. In der Originalansicht lassen sich
gemessene Kanten einblenden. Im Desktop-Editor können Nutzer Vorschläge
verwerfen, fehlende Abzüge mit vier Punkten ergänzen und Papier- sowie
Motivecken getrennt setzen. Gemessene Breite und Höhe in Millimetern
und die gewählte Drehung bleiben in der Korrektur gespeichert. Bei
unklarer Motivgrenze startet der Editor am Papierrand und wartet auf
eine bewusste Auswahl. Eine offene Motivprüfung bleibt auch nach einer
älteren allgemeinen Geometriebestätigung sichtbar.

Nach einer Änderung rendert die API die Varianten erneut vom
Albumoriginal. Ein Modus kann eine Vorschau mit Korrekturhinweisen
zeigen, während sein Download gesperrt bleibt. Die Galerie lädt die
gewählte verfügbare Variante. Manuelle Korrekturen durchlaufen
weiterhin Geometrie-, Nachbarschafts- und Pixelprüfungen. Rechenläufe,
Bilddateien, Prüfentscheide und Korrekturrevisionen bleiben mit der
Quellseite verknüpft.
[Review-API](backend/app/routers/review.py) · [Editor-API](backend/app/routers/photos.py) ·
[Varianten und Download](backend/app/pipeline/variants.py)

### 5. Text, Bildverbesserung und Betrieb

PaddleOCR kann Text auf der Seite erkennen und Abzügen zuordnen.
Die Oberfläche unterstützt Korrekturen und eine Textsuche. PhoSca
kann pro Abzug weitere Bildversionen erzeugen, etwa eine konservative
Farb- oder Rauschkorrektur sowie optionale Restaurierungsvarianten.
Die ursprüngliche perspektivisch korrigierte Version bleibt
erhalten. Eigene Qualitätsregeln steuern die Wahl der Standardversion.
Ein perzeptueller Hash weist auf ähnliche Abzüge hin.

Docker Compose verbindet Nginx und React mit FastAPI, PostgreSQL,
Redis, den Celery-Workern sowie dem Volume für Originale und
Derivate. Lokal kann SQLite als Entwicklungsdatenbank dienen.
Das Modellregister bindet die konfigurierten Gewichte an ihre Hashes.
Bei aktiviertem `AUTH_REQUIRED` prüft die API Rollen und Albumzugriff
über einen API-Schlüssel oder eine Browsersitzung.
[Verarbeitung und OCR](backend/app/services/pipeline.py) ·
[Freigabe von Bildversionen](backend/app/pipeline/release.py) · [Betrieb](docs/OPERATIONS.md)

### 6. Training und Produktionsfreigabe

Die Qualitätsmessung ordnet Vorschläge eindeutig den sichtbaren
Abzügen zu. Sie misst Erkennungspräzision, Trefferquote und die
Übereinstimmung von Masken und Zuschnitten per IoU. Für beide Modi
prüft sie Eckfehler in Originalpixeln, Seitenverhältnis, Landmarken,
Orientierung, Inhaltsverlust, fremde Interpolationspixel und Laufzeit.
Sie erfasst auch brauchbare Geometrien, die zur Prüfung zurückgehalten
werden. Für echte Freigabereferenzen toleriert die Prüfung höchstens
zwei Originalpixel Eckfehler und ein Prozent Abweichung im
Seitenverhältnis. Sie verlangt ausserhalb eines Ein-Pixel-Bands
vollständigen Inhalt und eine Ausgabe ohne fremde Quellpixel.

Die automatische Freigabe braucht einen bestandenen Nachweis für die
eingesetzte Konfiguration. Die Herkunft der Modellgewichte wird mit
Quellen, Album-IDs, Bild-Hashes und Splits dokumentiert. Die
Detektorkonfidenz wird auf einem separaten Datensatz für die
eingesetzte Konfiguration kalibriert. Testquellen dürfen sich weder
nach Album-ID noch nach Bild-Hash mit Trainings-, Entwicklungs- oder
Kalibrierungsquellen überschneiden. Beide Modi brauchen überprüfte
Referenzen aus mindestens zwei gemeinsamen physischen Albumgruppen.
Pro Modus müssen mindestens **90 %** der wiederherstellbaren
Zuschnitte automatisch freigabefähig sein, insgesamt und in jeder
erforderlichen Gruppe. Unwiederherstellbare Fälle müssen in Prüfung
bleiben.

Die automatische Freigabe bleibt gesperrt. Der eingefrorene
synthetische Vergleich mit dem zugehörigen Kontrollmodell erreicht
**18/18 erkannte Abzüge** und **23/23 bestandene wiederherstellbare
Zuschnitte**. Diese Messung prüft Regressionen. Die unabhängigen
Originalraster und die passende Kalibrierung für die Freigabe sind
noch unvollständig.
[Prüfverfahren und Belege](data/evaluation/README.md) ·
[Freigabeprüfung im Code](backend/app/pipeline/crop_validation.py)

## Beispiele und offene Fälle

![Drei generierte Fälle mit Eingabe und Ergebnissen. Eine Schrägaufnahme gelingt in beiden Modi. Bei einer verdeckten Papierecke bleibt der ganze Abzug in Prüfung. Ein blasser Datumsaufdruck wird vom Motivzuschnitt abgeschnitten.](docs/images/crop-examples.png)

- **Schrägaufnahme.** Beide Zuschnitte gelingen. PhoSca entzerrt den ganzen
  Abzug und entfernt im Motivmodus den hellen Papierrand. Die synthetische
  Probe hat bekannte Referenzgeometrie und einen leicht versetzten
  Startvorschlag.
- **Fotoecke auf Papier.** Für den ganzen Abzug erscheint eine
  Prüfvorschau. Das vollständig sichtbare Motiv lässt sich in dieser
  synthetischen Probe separat entzerren.
- **Blasser Datumsaufdruck.** Der Motivzuschnitt erhält in dieser Probe
  keinen der 727 vermessenen Schriftpunkte auf dem Rand. Blasse Schrift,
  einzelne Zeichen und durchgehende Schriftzüge
  erkennt die Randprüfung noch nicht zuverlässig. Der automatische
  Produktdownload bleibt gesperrt.
- **Echter Kontaktabzug.** PhoSca markierte einen Teil der Beschriftung
  am Original. Nach der manuellen Korrektur mit Test-Papiermassen
  enthielten beide Galerie-Downloads alle 35 Bildfelder. Im Motiv
  waren alle 1’938 vermessenen Beschriftungspunkte enthalten.
  [Kontaktabzug und Downloadpfad](data/evaluation/README.md#review-visible-markings-outside-the-proposed-motif)
- **Echter Abzug am Aufnahmebildrand.** Beim NerdScan-Entwicklungsbild
  traf die getestete Konfiguration drei erwartete Abzüge und lieferte
  zwei zusätzliche Vorschläge. Beide Modi blieben in Prüfung.
  Bei angeschnittenen Originalen hilft eine neue Aufnahme mit sichtbar
  belassenen Papierrändern. [Entwicklungsreferenz](data/evaluation/nerdscan-clipped-development.json) ·
  [Auswertung](data/evaluation/README.md)

Die Abbildung enthält generierte Testbilder und lässt sich lokal neu berechnen.

```bash
PYTHONPATH=backend python scripts/render_crop_examples.py
```

## Run With Docker Compose

1. Copy `.env.example` to `.env`.
2. Set `HOST_STORAGE_PATH` to the absolute path of this repository's `storage` directory.
3. Start the stack:

```bash
docker compose up --build
```

Services:

- Frontend dashboard: `http://localhost/` (port 5173 for local Vite development)
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
$env:YOLO_MODEL_PATH=(Join-Path $PWD 'models/yolov8-seg-album.onnx')
$env:MODEL_MANIFEST_PATH=(Join-Path $PWD 'models/manifest.json')
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
Uploads trigger photo detection and extraction automatically. In the photo
detail, **Ganzer Abzug** keeps the original print border and **Nur Bildmotiv**
uses the verified inner edge. The preview and download follow the selected
option. Review the source page when a boundary cannot be verified. Reprocess
an existing photo to generate the new variants from its stored album page.
For a strongly oblique photo of a flat print, measure the print and enter the
width and height in millimetres in the corner editor. Use numbered edges 1–2
for width and 2–3 for height. PhoSca uses those measurements for perspective
correction in both crop modes and retains them for subsequent processing.
In the crop editor, choose **Ausrichtung ab Albumseite** to keep the source
orientation or rotate the rectified photo in 90° steps. The selection applies
to both crop modes and is retained when the photo is reprocessed.
Use **Fehlenden Abzug ergänzen** in the desktop review and mark the four
corners to add a print that was missed during analysis. Saving the print
reopens the page's photo-count review and adds it to the crop editor.

`ENABLE_BORDER_TRIM=false` disables automatic inner-edge detection. Reviewers
can still provide inner corners. `ENABLE_PRECISE_PHOTO_CROPS=false` disables
the precise crop path for ordinary new processing. The configured detector
can still run. Automatic downloads of the two precise modes require a passing independent album report for the
deployed checkpoint and settings at `CROP_VALIDATION_REPORT_PATH`. The
preview and manual corner editor are available while automatic crops are in
review. See [crop evaluation](data/evaluation/README.md) for the report format.
For HDR JPEGs the app processes the primary image and keeps the original gain
map in the stored file.

## ML Models

Place local-only model files under `models/`:

- `models/yolov8-seg-album.onnx`
- `models/photo-orientation.onnx` (optional 0/90/180/270 classifier)
- `models/paddleocr/`

The segmentation ONNX parser supports YOLO-style box outputs and segmentation prototype masks, including channels-first and channels-last prototype tensors. Classical boundary analysis is enabled as risk-triggered geometric evidence: it runs when the model is missing, empty, or produces a suspicious instance set, and is fused with additional checks for overlapping prints. Photo orientation correction uses `models/photo-orientation.onnx` when present; without a calibrated model, face/detail evidence is advisory only and pixels are preserved for review. Train the optional orientation model from manually reviewed upright crops with `scripts/train_orientation_model.py`. PaddleOCR is loaded lazily; OCR can also be tested with `.ocr.json` sidecars next to uploaded page images.

## Tests And Scans

```bash
python -m pytest
cd frontend && npm run build && npm audit
python -m bandit -r backend/app
```

Earlier smoke and setup results:

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

Safety and operations evidence. See [crop evaluation data](data/evaluation/README.md) for exporting calibration scores from reviewed albums.

```bash
python scripts/evaluate_onnx_segmentation.py --include-classical-fallback --output runs/cropping-current.json
python scripts/evaluate_enhancement.py --output-dir runs/enhancement-current
python scripts/fit_confidence_calibration.py runs/crop-quality/calibration-scores.json --holdout-id <independent-holdout-id>
python scripts/evaluate_shadow_drift.py stable.json shadow.json --output evidence/shadow-drift.json
python scripts/run_retention.py
python scripts/verify_launch_gates.py evidence/launch-gates.json
```

`scripts/setup_segmentation_training.py` orchestrates the normal bootstrap/prepare/train loop. `scripts/export_label_studio_tasks.py` creates Label Studio import JSON and can seed editable polygon predictions from the classical detector. `scripts/convert_label_studio_to_yolo.py` converts reviewed Label Studio JSON exports into YOLO segmentation labels. `scripts/export_reviewed_yolo_labels.py` exports masks corrected in the PhoSca review canvas into the same YOLO label directory. `scripts/prepare_yolo_dataset.py` recreates deterministic train/val/golden splits from raw images plus exported YOLO labels. `scripts/validate_yolo_dataset.py` checks label completeness, polygon validity, and golden leakage before training. `scripts/train_segmentation_model.py` lazily imports Ultralytics; install it only in the training environment if it is not already available. `scripts/train_orientation_model.py` requires PyTorch plus `onnx`, uses manually upright crops under `data/orientation_photos`, and exports a 0/90/180/270 correction classifier. `scripts/validate_segmentation.py` measures polygon IoU from saved segmentation masks and falls back to boxes only for legacy rows without masks. These are diagnostic checks for segmentation and OCR. The precise crop release requirements are documented in [crop evaluation](data/evaluation/README.md).

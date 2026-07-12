# Operations

## Backup

Run `scripts/backup.sh` from the backend or a maintenance container with `pg_dump`, `rsync`, and `gpg` available.

Required environment:

- `DATABASE_URL`: PostgreSQL connection string.
- `STORAGE_PATH`: storage volume path, default `/app/storage`.
- `YOLO_MODEL_PATH` and `MODEL_MANIFEST_PATH`: mounted local segmentation
  artifact and its SHA-256/license manifest. The service refuses an
  unregistered or hash-mismatched model when `REQUIRE_REGISTERED_MODELS=true`.
- `BACKUP_ROOT`: external drive or NAS mount, default `/backups/album-digitizer`.

Optional:

- `GPG_RECIPIENT`: public-key recipient. If omitted, the script uses symmetric AES-256 encryption.

## Restore

1. Decrypt the backup archive with `gpg`.
2. Stop backend and worker containers.
3. Restore PostgreSQL with `psql "$DATABASE_URL" < database.sql`.
4. Sync `storage/` back into the configured storage volume.
5. Start services and run a small album retrieval smoke test.
6. Dokumentiere Datum, Operator, Archiv-SHA-256, wiederhergestellte DB-Zeilen,
   stichprobenartig verifizierte Asset-Hashes und Recovery-Zeit als Evidenz für
   `retention_and_backup_restore`. Ein erzeugtes Backup allein besteht das Gate
   nicht.

## Retention

Der sichere Standard ist ein Dry-Run mit gehashtem Report:

```text
python scripts/run_retention.py
python scripts/run_retention.py --execute
```

`page_original`, `corrected_print` und `released_variant` werden nie durch die
Derivative-Retention entfernt. Pfade ausserhalb von `STORAGE_PATH/processed`
werden als Fehler gemeldet und nicht gelöscht.

## Shadow, Canary und Launch-Gates

Der Default blockiert alle Derived Releases. Erst nach bestandenen externen
Gates werden Kill-Switch und Shadow-Modus bewusst geändert und ein Canary > 0
gesetzt. Die Zuordnung erfolgt stabil aus der Photo-ID.

```text
python scripts/evaluate_shadow_drift.py stable.json shadow.json --output evidence/shadow-drift.json
python scripts/verify_launch_gates.py evidence/launch-gates.json --report evidence/launch-gate-report.json
```

Das zweite Kommando endet mit Status 2, solange nur ein Gate blockiert ist,
Evidenz fehlt oder deren SHA-256 nicht übereinstimmt.

## Future Rust Extraction Points

Only consider Rust if profiling shows Python bottlenecks after load testing. Candidate boundaries are contour extraction, pHash batch computation, and EXIF batch rewrites. Keep the Python API contract unchanged.
## Tenant- und API-Key-Betrieb

Für produktive Deployments muss `AUTH_REQUIRED=true` gesetzt sein. Anschliessend
werden tenantgebundene Schlüssel erst nach `alembic upgrade head` erzeugt:

```text
python scripts/create_api_key.py --tenant <tenant-name> --label <operator> --role admin
```

Das Kommando zeigt das Geheimnis nur einmal. Clients senden es als
`X-API-Key`. `viewer` darf nur lesen, `editor` darf Alben und Fotos verändern,
und `admin` ist zusätzlich für Trainingsdaten-Endpunkte erforderlich. Jeder
Album-, Foto-, Seiten-, Review-, Such- und Run-Abruf wird bei aktivierter
Authentifizierung auf den Tenant des Schlüssels eingeschränkt und gibt für
fremde Ressourcen absichtlich `404` zurück.

`AUTH_REQUIRED=false` ist ausschliesslich für lokale Entwicklung und die
bestehenden Test-Fixtures vorgesehen. Es ist kein zulässiger Consumerbetrieb.

## Lokale Segmentierungsmodell-Prüfung

Nach einem Training wird der Runtime-Pfad (nicht nur Ultralytics selbst) gegen
die gesperrten Golden-Labels geprüft:

```text
python scripts/evaluate_onnx_segmentation.py --model models/yolov8-seg-album.onnx --manifest models/manifest.json
```

Das Kommando verifiziert zuerst Modellhash und Lizenzmanifest, nutzt die
konfigurierte ONNX-Runtime und meldet IoU, Precision, Recall sowie
page-perfect Ergebnisse inklusive False Positives und False Negatives.

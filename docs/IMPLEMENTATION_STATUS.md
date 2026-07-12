# Cropping & Enhancement – Implementierungsstatus

Stand: 11. Juli 2026

Die Anwendung ist nach den folgenden Containment-Änderungen bewusst auf
**archivtreue, konservative Ausgabe** gestellt. Ein Consumer-Auto-Release ist
weiterhin nicht freigegeben: Der erforderliche unabhängige Safety-Holdout,
annotierte Trainingsdaten, Modelllizenzen sowie Privacy-/Security-Freigaben
existieren im Repository nicht.

## Implementiert

- Zentraler serverseitiger Release-Selector: Original ist verpflichtend;
  `review_needed`, fehlgeschlagene Stages, Warnings und fehlgeschlagene Gates
  vetoen alle abgeleiteten Varianten.
- API und UI verwenden die serverseitige Freigabe statt `enhanced` als Default.
- Modellbasierte SR, GFPGAN und Scratch-Inpainting sind standardmässig aus;
  Modelle können keine Variante selbst freigeben.
- Robuste Quad-Prüfungen (eindeutige Ecken, Konvexität, Winkel, Kondition,
  Reprojection) und ein Protected-Region-Hard-Gate für bekannte Gesichtsregionen.
- Klassische und ONNX-Proposals werden erst nach Bewertung zusammengeführt;
  Proposal- und Release-Threshold sind getrennt.
- Uploads werden vor Persistenz wirklich dekodiert und gegen Pixel-/Frame-Limits
  geprüft. Die Arbeitsdekodierung berücksichtigt EXIF-Orientierung und ICC nach
  sRGB; mehrdeutige Mehrframe-Dateien werden abgelehnt.
- Lokaler Model-Manifest-Vertrag mit SHA-256- und Lizenzprüfung für aktivierte
  ONNX-Artefakte.
- Versionierte `ProcessingRun`-/`StageRun`-Provenienz inklusive Konfigurations-
  Fingerprint, Input-/Output-Hashes, Retry-Versuchen und idempotentem
  Page-Replay. Ein Immutable-Asset-Register erfasst Original, korrigierten
  Print und Release-Asset inklusive Elternbeziehung; die Pipeline schreibt
  Derivate pro Run und Stage in getrennte Pfade. Run-/Stage-/Asset-Lineage ist
  über `/api/v1/pages/{page_id}/runs` und `/api/v1/runs/{run_id}` lesbar.
- Ein unabhängiger deterministischer Quality Critic zeichnet Pixelintegrität,
  Stage-Fehler, Quality-Regression und Source-Delta als Gates auf.
- Alembic-Migrationen sind auf PostgreSQL und SQLite ausführbar; der
  Segmentierungsvalidator verwendet nun globales Hungarian One-to-one-Matching
  und Page-perfect-Metriken.
- Die Privacy-Control-Plane enthält eine explizite Album-Löschoperation. Sie
  kaskadiert Datenbankzeilen und entfernt nur verifizierte, albumgebundene
  Upload-, Derivat- und DLQ-Verzeichnisse innerhalb von `STORAGE_PATH`.
- Tenant- und API-Key-Rollen (`viewer`, `editor`, `admin`) sind implementiert.
  Bei `AUTH_REQUIRED=true` sind Kernressourcen tenant-scoped, mutierende
  Aufrufe für Viewer gesperrt und Trainingsendpunkte adminpflichtig.
- Dataset- und Segmentierungsvalidatoren zählen False Positives/Duplikate und
  lehnen fehlende Golden-Labels als Fehler ab.
- Protected Regions vereinigen nun Front-/Profilgesichter, hochkonfidente
  Personenvorschläge und externe Person-/Objekt-/Text-/Subject-Regionen. Die
  Union wird einmal je Seite ermittelt und bleibt ein Hard Veto am Crop-Rand.
- Confidence ist über ein versioniertes, monotones Holdout-Artefakt
  kalibrierbar. Fehlt oder verletzt es den Vertrag, erzwingt
  `REQUIRE_CONFIDENCE_CALIBRATION=true` manuelle Review.
- Shadow, deterministische Canary-Buckets und ein globaler Release-Kill-Switch
  sind serverseitige Freigabepolicies. Der sichere Default bleibt
  `SHADOW_MODE=true`, `CANARY_PERCENTAGE=0`, `RELEASE_KILL_SWITCH=true`.
- Celery routet CPU-, Analyse-GPU- und Restoration-GPU-Stages in getrennte
  Queues; die GPU-Pools laufen mit genau einem Prozess und Late-Ack/Redelivery.
- Prometheus exportiert bounded-cardinality Safety-/Qualitätsmetriken ohne
  Bild-, OCR-, Nutzer- oder Assetdaten. Ein Shadow-Drift-CLI erzeugt ein
  maschinenlesbares Gate.
- Retention besitzt Dry-Run und Execute-Modus, bewahrt Original-/Release-Assets,
  verweigert Pfade ausserhalb des Storage-Roots und erzeugt gehashte Evidenz.
  Backups schreiben einen SHA-256-/Grössen-Nachweis; ein Restore-Drill bleibt
  für die Freigabe weiterhin zwingend.
- `scripts/verify_launch_gates.py` prüft 13 harte Gates samt Verantwortlichem,
  Datum, Evidenzdatei und SHA-256. Die eingecheckte Vorlage ist absichtlich
  vollständig `blocked` und kann keine externe Freigabe vortäuschen.

## Aktuelle Trainings- und Evaluierungsevidenz

- 40 manuell annotierte Albumseiten wurden deterministisch in 25 Train-, 5
  Validierungs- und 10 gesperrte Golden-Seiten aufgeteilt; der Datensatz ist
  formell valide.
- YOLO11n-seg wurde auf der RTX 4080 trainiert und als hash- und
  lizenzregistriertes `models/yolov8-seg-album.onnx` exportiert.
- Der echte ONNX-Runtime-Pfad erzielt auf dem Golden-Holdout bei IoU 0,85 eine
  mittlere IoU von 0,973, Precision/Recall von je 0,966 und 9/10
  page-perfect Ergebnisse. `page_31` bleibt als bekannter Review-Fall.

## Harte, externe Launch-Blocker

Diese Punkte lassen sich nicht seriös durch Code oder automatische
Pseudo-Annotationen ersetzen:

1. Der aktuelle Datensatz umfasst nur 40 Seiten aus einem engen Aufnahme- und
   Albumkontext. Er ist für den lokalen Review-Pilot geeignet, aber kein
   belastbarer Consumer-Safety-Holdout.
2. Das Segmentierungsmodell ist lokal hash- und lizenzregistriert. Es fehlen
   weiterhin freigegebene Landmark-, Person-, Objekt- und Textmodellartefakte
   sowie eine unabhängige Lizenzfreigabe für einen Consumerbetrieb.
3. Es fehlen der im Masterplan geforderte Safety-Holdout, Human-Study,
   Privacy/DPIA-, Lizenz- und unabhängige Security-Freigaben.
4. Der lokale Immutable-Asset-Store und die Queue-Topologie sind implementiert.
   Ein produktiver Object Store, KMS/Secrets, GPU-Scheduler und Backup-Ziel
   erfordern weiterhin verbindliche Infrastrukturentscheidungen und reale
   Restore-/Chaos-Evidenz.

Bis diese Nachweise vorliegen, darf die Anwendung nur mit Original-Fallback und
manueller Review betrieben werden. Die im Masterplan genannten Auto-Release- und
Canary-Gates dürfen nicht als bestanden dargestellt werden.

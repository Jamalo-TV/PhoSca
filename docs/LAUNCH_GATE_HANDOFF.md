# Consumer-Launch-Gate-Handoff

Dieses Dokument beschreibt die externen Nachweise, die vor einem
Consumer-Auto-Release benötigt werden. Es ersetzt keine Freigabe.

## 1. Segmentierungsdaten

- 500+ repräsentative Albumseiten mit geprüften Printpolygonen/Quads.
- Album-/Nutzer-/Gerätetrennung zwischen Train, Val und Safety-Holdout.
- Alle 10 Dateien in `data/golden_fixtures/` erhalten gelockte YOLO-Polygonlabels.
- Zwei unabhängige Annotationen für Hard Cases; Adjudication und Reason Code
  dokumentieren.
- `python scripts/validate_yolo_dataset.py` muss ohne Fehler enden.

## 2. Modellfreigabe

- Lizenz, Herkunft, SHA-256 und Input-Contract jedes Gewichts prüfen.
- Eintrag in `models/manifest.json` anhand von `models/manifest.example.json`
  erstellen; Gewichte werden ausschliesslich lokal bereitgestellt.
- Export-/Inference-Parität auf dem gelockten Holdout prüfen.
- Sicherheitskritische Model-Slices: kleine/verdecke Gesichter, Gruppen,
  Texte, Randmotive, Hochformat, Panorama, Schatten und gewölbte Seiten.

## 3. Qualitäts- und Human-Gates

- Crop: Precision, Recall, Page-perfect, Protected-cut und Risk-Coverage auf
  einem unabhängigen Safety-Holdout berichten.
- Enhancement: No-op-Rate, Identity-/Text-/Halo-/Inpainting-Fehler und
  Naturalness mit blindem Human-A/B bewerten.
- Kein Hard Gate darf durch Mittelwerte kompensiert werden.

## 4. Security, Privacy und Betrieb

- `AUTH_REQUIRED=true`; Tenant-Admin-Key mit `scripts/create_api_key.py`
  provisionieren und Secret Management ausserhalb des Repositories nutzen.
- Datenschutz-/Lösch-/Exportprozess, Aufbewahrungsfristen, DPIA/DSFA und
  Modell-/Datenlizenzen schriftlich freigeben.
- Unabhängigen Penetrationstest, Last-/Soak-/Rollback-Drill sowie Canary-Plan
  abschliessen.

## Freigabeprotokoll

| Gate | Verantwortlich | Evidenz-Link | Datum | Freigegeben |
|---|---|---|---|---|
| Segmentierungsdaten | | | | |
| Modell-/Lizenzinventar | | | | |
| Crop Safety Holdout | | | | |
| Enhancement Human Study | | | | |
| Security/Pen-Test | | | | |
| Privacy/DPIA | | | | |
| SLO/Chaos/Rollback | | | | |

# PhoSca – Gesamtplan für zuverlässiges Cropping und Enhancement mit Mobile-Workflow

**Stand:** 11. Juli 2026  
**Primärziel:** Cropping und Enhancement mit realen Albumseiten so lange
verbessern, bis die lokale Pipeline zuverlässig arbeitet.  
**Arbeitsweise:** Viele kurze Arbeitseinheiten von ungefähr fünf Minuten auf
dem Smartphone, ergänzt durch automatische Trainings-, Evaluations- und
Reportingläufe auf dem PC.

---

## 1. Zielbild

PhoSca erhält drei eng verbundene lokale Anwendungen:

1. **Mobile Capture:** Eine sehr einfache Smartphone-Webseite öffnet die
   Kamera. Nach der Aufnahme wird das Original automatisch und zuverlässig auf
   den PC hochgeladen und dort in einem definierten Inbox-Ordner gespeichert.
2. **Mobile Review:** Eine zweite mobile Oberfläche liefert kurze Aufgaben für
   Crop-Korrektur und Enhancement-Bewertung. Jede Aufgabe soll üblicherweise in
   20–90 Sekunden lösbar sein.
3. **PC Training und Evaluation:** Der PC validiert das Feedback, versioniert
   die Daten, trainiert neue Kandidaten und vergleicht sie gegen ein gesperrtes
   Testset. Ein neues Modell wird niemals allein aufgrund neuer Handy-Labels
   aktiviert.

```text
Smartphone-Kamera
  → Upload-Queue
  → PC-Inbox
  → Qualitätsprüfung
  → Crop-Vorschläge
  → Mobile Crop-Review
  → versionierte Labels
  → Training/Calibration
  → Shadow-Evaluation

Bestätigte Crops
  → Degradation Planner
  → No-op/Enhancement-Kandidaten
  → Mobile Blind-A/B
  → versionierte Präferenzen und Fehlergründe
  → Planner-/Critic-Verbesserung
  → gesperrte Evaluation
```

Das System soll deine verfügbare Zeit ausnutzen, ohne vorauszusetzen, dass du
eine vollständige Seite oder einen ganzen Batch in einer Sitzung abschliesst.

---

## 2. Umfang und Abgrenzung

Dieser Plan umfasst:

- mobile Aufnahme von Albumseiten;
- automatischen Upload auf den eigenen PC;
- lokale Inbox und sichere Dateibenennung;
- Offline-/Retry-Queue auf dem Smartphone;
- Cropping-Erkennung, Korrektur und Annotation;
- Protected Regions für Gesichter, Personen, Objekte und Text;
- Enhancement-No-op, konservative Restaurierung und Qualitätskritik;
- mobile A/B-Bewertung;
- Active Learning und priorisierte Microtasks;
- lokale Modelltrainings und reproduzierbare Evaluation;
- Statusübersicht über Daten- und Modellfortschritt.

Vorerst nicht enthalten:

- öffentliche Kundenregistrierung und Bezahlung;
- allgemeiner Cloudbetrieb;
- öffentlicher Consumer-Launch;
- kreative generative Restaurierung, Diffusion oder Outpainting;
- automatische Modellfreigabe ohne gesperrte Evaluation.

---

## 3. Grundregeln

### 3.1 Originale bleiben unverändert

- Das hochgeladene Kamerabild wird unverändert gespeichert.
- Crops und Enhancements sind neue, abgeleitete Assets.
- Jede Datei erhält SHA-256, Aufnahmezeit, Batch-ID und technische Metadaten.
- Kein Trainingslauf überschreibt Originale oder frühere Labels.

### 3.2 Feedback ist nicht sofort Ground Truth

Eine Handy-Antwort wird zunächst als Review-Ereignis gespeichert. Sie wird erst
nach technischer Validierung und gegebenenfalls zweiter Prüfung zum
Trainingslabel. Dadurch kann ein versehentlicher Tap kein Modell unmittelbar
verschlechtern.

### 3.3 Testdaten bleiben gesperrt

- Daten werden albumweise in Train, Validation, Calibration und Test geteilt.
- Seiten desselben Albums dürfen nicht über Splits verteilt werden.
- Testbilder erscheinen nicht als normale Trainings-Review-Aufgaben.
- Testlabels werden nicht für Threshold-Tuning verwendet.

### 3.4 No-op ist ein Erfolg

Enhancement muss nicht jedes Foto verändern. Bei guten oder unsicheren Bildern
ist `Original behalten` die richtige Entscheidung.

### 3.5 Unsicherheit wird genutzt

Die Review-Queue priorisiert:

- unsichere Modellvorhersagen;
- starke Abweichungen mehrerer Detektoren;
- neue Album-/Gerätetypen;
- kleine oder randnahe Prints;
- Ergebnisse nahe einer Freigabeschwelle;
- Enhancement-Kandidaten, bei denen Critic und IQA widersprechen.

---

## 4. Mobile Capture – Albumseiten aufnehmen und automatisch auf den PC laden

## 4.1 Bedienablauf

Die Capture-Seite soll im Idealfall nur folgende Schritte benötigen:

1. Seite auf dem Smartphone öffnen.
2. Album beziehungsweise Aufnahme-Batch auswählen oder neu anlegen.
3. `Foto aufnehmen` drücken.
4. Albumseite fotografieren.
5. Vorschau kurz bestätigen oder erneut aufnehmen.
6. Upload startet automatisch.
7. Die Seite zeigt eindeutig `auf PC gespeichert` oder `wartet auf Upload`.
8. Sofort kann die nächste Seite fotografiert werden.

Kein manuelles Verschieben, Umbenennen oder Auswählen eines PC-Ordners pro Bild.

## 4.2 Technische Smartphone-Funktionen

- installierbare Progressive Web App;
- `<input type="file" accept="image/*" capture="environment">` als robuste
  Basis;
- optional `getUserMedia` für eine eigene Kameravorschau;
- Full-resolution Upload ohne Browser-Kompression, soweit das Gerät dies
  erlaubt;
- lokale IndexedDB-Queue;
- Uploadfortsetzung nach Netzunterbrechung;
- Chunked Upload für grosse Bilder;
- sichtbarer Status `queued`, `uploading`, `verified`, `failed`;
- Hintergrund-Wiederholung, soweit Mobile-Browser dies zulassen;
- Wake Lock während aktivem Upload;
- Erkennung doppelter Uploads über Datei-Hash;
- keine Löschung aus der lokalen Queue, bevor der Server den Hash bestätigt;
- optionaler Hinweis auf Bewegung, geringe Auflösung und extreme Unschärfe;
- keine automatische Beauty-, Filter- oder Enhancementfunktion des Webclients.

## 4.3 Verbindung zum PC

Für die lokale Entwicklungsphase werden zwei Modi vorgesehen:

### Modus A – gleiches WLAN

- PhoSca läuft auf dem PC;
- das Smartphone öffnet eine lokale HTTPS-Adresse;
- ein QR-Code auf dem PC enthält URL und einmaliges Pairing-Token;
- Uploads gehen direkt an den PC.

### Modus B – unterwegs

- private Verbindung über Tailscale/WireGuard oder einen vergleichbaren
  privaten Tunnel;
- keine öffentliche ungeschützte Portfreigabe am Router;
- wenn kein Netz vorhanden ist, bleiben Aufnahmen verschlüsselt beziehungsweise
  browserisoliert in der Upload-Queue und werden später übertragen.

Die erste Implementierung beginnt mit demselben WLAN. Der private Remote-Modus
folgt, sobald der lokale Ablauf zuverlässig ist.

## 4.4 PC-Inbox

Nach erfolgreichem Upload wird jede Datei atomar gespeichert:

```text
storage/inbox/<album-id>/<capture-batch-id>/
  ├─ originals/<capture-id>.<original-extension>
  ├─ metadata/<capture-id>.json
  └─ quarantine/...
```

Metadaten:

- Capture-ID und Batch-ID;
- ursprünglicher Dateiname;
- SHA-256 und Dateigrösse;
- MIME-Typ und tatsächliches Decoderformat;
- Pixelabmessungen und Frameanzahl;
- EXIF-Orientierung und Aufnahmezeit;
- Smartphone-/Browserklasse in datensparsamer Form;
- Uploadzeit und Uploadversuche;
- optionaler Benutzerhinweis wie `starke Reflexion` oder `Seite gewölbt`.

Nach dem atomaren Write und der Hashprüfung verschiebt der Importer die Datei
logisch von `inbox` nach `accepted` oder `quarantine`. Erst danach wird die
Aufnahme als auf dem PC gespeichert angezeigt.

## 4.5 Capture-Hilfe

Die Kameraansicht darf Hinweise geben, aber keine Aufnahme blockieren:

- gesamte Albumseite im Bild;
- etwas Sicherheitsrand um die Seite;
- Kamera möglichst parallel;
- Reflexionen vermeiden;
- ausreichende Helligkeit;
- Hinweis bei starker Bewegungsunschärfe;
- Hinweis, wenn die Auflösung zu gering ist;
- optional automatische Aufnahme bei stabiler Kamera und sichtbarem Seitenquad.

Fehlerhafte Aufnahmen können trotzdem hochgeladen und als Hard Cases markiert
werden. Gerade diese Daten sind für die Robustheit wertvoll.

## 4.6 Capture-Abnahmekriterien

- 100 aufeinanderfolgende Aufnahmen ohne verlorene Datei;
- Upload wird nach WLAN-Unterbruch fortgesetzt;
- Duplikate werden erkannt, nicht doppelt trainiert;
- Smartphone zeigt Erfolg erst nach serverseitiger Hashprüfung;
- Originaldatei auf dem PC entspricht bytegenau dem Upload;
- eine Aufnahme inklusive Bestätigung benötigt im Normalfall unter 15 Sekunden;
- Quarantänefälle besitzen einen verständlichen Grund.

---

## 5. Mobile Review – Grundkonzept für fünf Minuten freie Zeit

## 5.1 Startseite

Die mobile Review-Seite zeigt grosse, klar getrennte Aktionen:

- `Cropping prüfen`;
- `Enhancement vergleichen`;
- `Schnelle Ja/Nein-Aufgaben`;
- `Angefangene Aufgabe fortsetzen`;
- `Fortschritt`.

Eine Sitzung startet standardmässig mit fünf Aufgaben und kann jederzeit ohne
Datenverlust beendet werden.

## 5.2 Microtask-Prinzip

Jede Aufgabe besitzt:

- genau eine primäre Entscheidung;
- grosse Touch-Ziele;
- automatisches Speichern nach jedem Schritt;
- Undo für den letzten Schritt;
- keine langen Freitextfelder;
- feste Reason Codes;
- optional `unsicher/überspringen`;
- eine geschätzte Dauer;
- stabile Wiederaufnahme auf einem anderen Gerät.

Zielzeiten:

| Aufgabe | Zielzeit |
|---|---:|
| Anzahl Prints bestätigen | 10–20 s |
| Crop korrekt/falsch | 10–20 s |
| False Positive markieren | 10–30 s |
| fehlenden Print grob markieren | 20–45 s |
| Quad-Ecken korrigieren | 30–90 s |
| Enhancement A/B | 15–30 s |
| Enhancement-Fehlergrund | 10–25 s |

---

## 6. Mobile Cropping Review

## 6.1 Review-Stufen

Cropping wird nicht sofort mit einem komplexen Polygoneditor begonnen. Die
Aufgaben werden von einfach nach aufwendig getrennt.

### Stufe 1 – Printanzahl

Anzeige der Albumseite mit nummerierten Erkennungen.

Antworten:

- `alle Prints gefunden`;
- `Print fehlt`;
- `zu viele Erkennungen`;
- `unsicher`.

### Stufe 2 – Einzelcrop bestätigen

Anzeige von Albumseite und hervorgehobener Crop-Grenze.

Antworten:

- `korrekt und vollständig`;
- `zu eng`;
- `zu weit`;
- `falsches Objekt`;
- `Print geteilt`;
- `mehrere Prints zusammen`;
- `unsicher`.

### Stufe 3 – False Positive entfernen

Ein Tap auf eine Erkennung markiert sie als `not_a_print`. Vor dem endgültigen
Speichern gibt es Undo.

### Stufe 4 – fehlenden Print hinzufügen

Zunächst reicht ein grobes Rechteck durch Ziehen. Der PC kann daraus einen
Polygonvorschlag berechnen. Eine spätere Aufgabe bestätigt oder korrigiert ihn.

### Stufe 5 – Quad-/Polygonkorrektur

- vier grosse Eckpunkte;
- Zoom und Pan mit zwei Fingern;
- magnetisches Snapping an starke Printkanten;
- Sicherheitsrand-Overlay;
- Zurücksetzen auf Modellvorschlag;
- `Rand nicht sichtbar` als explizite Option;
- keine versehentliche Bestätigung bei sich schneidendem Polygon.

### Stufe 6 – Protected-Region-Prüfung

Anzeige erkannter Gesichter, Personen, Objekte und Textregionen.

Antworten:

- `relevanter Inhalt vollständig`;
- `Kopf/Gesicht abgeschnitten`;
- `Körperteil abgeschnitten`;
- `Tier/Objekt abgeschnitten`;
- `Text abgeschnitten`;
- `Schutzregion falsch`;
- `Schutzregion fehlt`.

## 6.2 Mobile Crop-UX

- Seite füllt die verfügbare Breite;
- Cropgrenzen haben hohen Kontrast;
- jede Erkennung erhält eine andere Farbe und Nummer;
- Ansicht `ganze Seite` und `Crop vergrössert`;
- langes Drücken zeigt Originalpixel ohne Overlay;
- Eckpunkte bleiben auch bei kleinen Prints gut greifbar;
- Landscape-Modus für Polygonkorrektur;
- Haptik bei Snap, soweit vom Browser unterstützt;
- keine automatische nächste Aufgabe nach einer riskanten Polygonänderung,
  bevor die Vorschau bestätigt wurde.

## 6.3 Labelstatus

Jedes Crop-Label besitzt einen Status:

- `model_proposal`;
- `quick_reviewed`;
- `manually_corrected`;
- `second_reviewed`;
- `adjudicated`;
- `locked_test_label`.

Nur ausreichend geprüfte Labels fliessen ins Training. Schnelle Ja/Nein-
Antworten können Confidence- und Active-Learning-Signale liefern, ohne als
perfektes Polygon behandelt zu werden.

## 6.4 Crop-Active-Learning-Queue

Priorität erhält eine Seite, wenn:

- Modellconfidence nahe der Reviewschwelle liegt;
- mehrere Modelle unterschiedliche Printanzahlen liefern;
- TTA/Skalierung instabile Polygone ergibt;
- ein Crop eine Protected Region berührt;
- Print sehr klein oder randnah ist;
- Album-/Geräteklasse im Datensatz selten ist;
- vorheriges schnelles Feedback einen Fehler signalisiert;
- die Seite besonders hohen Informationsgewinn erwarten lässt.

Nicht nur schwierige Fälle werden gezeigt. Ein fester Anteil zufälliger und
offensichtlich guter Beispiele verhindert ein verzerrtes Trainingsbild.

## 6.5 Crop-Qualitätsziele

Auf einem albumgetrennten, gesperrten Testset:

- Precision mindestens 98 %;
- Recall mindestens 98 %;
- Page-perfect mindestens 95 %;
- keine geometrisch ungültige automatische Ausgabe;
- kein bestätigter Kopf-, Körper-, Objekt- oder Textschnitt in automatisch
  akzeptierten Crops;
- alle unsicheren Fälle besitzen Reviewgrund und Originalfallback.

---

## 7. Mobile Enhancement Review

## 7.1 Reihenfolge

Nur bestätigte vollständige Crops dürfen Enhancement-Aufgaben erzeugen. Sonst
würde ein Cropfehler fälschlich als Enhancementfehler bewertet.

## 7.2 Aufgabentypen

### Typ 1 – Muss dieses Foto verändert werden?

Anzeige des Original-Crops.

Antworten:

- `Original ist gut – No-op`;
- `Rauschen`;
- `unscharf`;
- `Farbstich/vergilbt`;
- `verblasst`;
- `Kratzer/Fleck`;
- `Kompressionsartefakte`;
- `anderes/unsicher`.

Mehrfachauswahl ist möglich, aber der erste Tap speichert bereits einen
Zwischenstand.

### Typ 2 – Verblindeter A/B-Vergleich

- links und rechts zufällig angeordnet;
- Modellname und Variante bleiben verborgen;
- synchrones Zoom/Pan;
- schneller Wechsel beziehungsweise Split-View;
- Auswahl `links besser`, `rechts besser`, `gleich`, `beide schlecht`;
- `Original behalten` bleibt sichtbar, ohne zu verraten, welche Seite das
  Original ist.

### Typ 3 – Fehlergrund

Nach `beide schlecht` oder einer auffälligen Auswahl:

- `unnatürlich`;
- `zu glatt`;
- `überschärft/Halo`;
- `falsche Farben`;
- `Details erfunden`;
- `Gesicht verändert`;
- `Text verändert`;
- `Kratzer schlecht repariert`;
- `Kontrast/Clipping`;
- `Original wäre besser`;
- `unsicher`.

### Typ 4 – lokale Problemstelle

Optionaler Tap oder kleiner Brush auf die Stelle, an der der Fehler sichtbar
ist. Dies erzeugt eine lokale Fehlerregion für Critic-Training und spätere
Analyse. Komplexe Pixelmasken werden nicht als Pflichtaufgabe auf dem Handy
verlangt.

### Typ 5 – Varianten-Ranking

Für schwierige Bilder können Original, konservative Variante und ein oder zwei
Modellkandidaten als kleines Turnier verglichen werden. Es werden nie zu viele
Varianten gleichzeitig gezeigt.

## 7.3 Schutz gegen Bewertungsverzerrung

- Seitenposition wird randomisiert;
- identische Kontrollpaare messen Tap-Konsistenz;
- einige Paare werden später wiederholt;
- keine permanente Anzeige bisheriger Mehrheitsentscheidungen;
- nicht nur stark beschädigte Bilder, sondern auch gute No-op-Bilder;
- getrennte Auswertung nach Defektklasse;
- schnelle Antworten werden gespeichert, aber extrem kurze beziehungsweise
  inkonsistente Serien für Training niedriger gewichtet;
- ein einzelnes A/B-Votum darf kein Modell automatisch freigeben.

## 7.4 Enhancement-Active-Learning-Queue

Priorität erhalten:

- Planner unsicher zwischen No-op und Verarbeitung;
- Critic und IQA widersprechen sich;
- mehrere Kandidaten liegen nahe beieinander;
- Gesichter oder Text sind vorhanden;
- starke lokale Änderungen;
- seltene Defektklasse;
- neue Kamera-/Albumdomäne;
- Kandidat gewinnt automatisch, besitzt aber einen schwachen Gate-Margin.

## 7.5 Enhancement-Qualitätsziele

- No-op-FPR zunächst höchstens 5 %, Ziel unter 2 %;
- gute Bilder werden überwiegend unverändert gewählt;
- bei degradierter Teilmenge gewinnt die neue konservative Pipeline gegen
  Original oder einfache Baseline;
- kein freigegebener Kandidat mit bestätigter Identitäts- oder Textänderung;
- keine automatische kreative Rekonstruktion;
- jeder abgelehnte Kandidat besitzt Gate- oder Reviewgrund;
- Original bleibt jederzeit verfügbar.

---

## 8. Feedback-, Daten- und Lernkreislauf

## 8.1 Ereignismodell

Jede Aktion wird append-only gespeichert:

- Task-ID und Tasktyp;
- Asset-, Crop- und Variant-ID;
- Modell-/Pipelineversion;
- ursprünglicher Vorschlag;
- Nutzerentscheidung;
- Reason Codes;
- optionale Polygon- oder Problemregion;
- Geräteklasse und UI-Version;
- Start-/Endzeit und Undo;
- Reviewstatus;
- Herkunftssplit.

Keine Bildpixel, OCR-Texte oder Gesichtsmerkmale gehören in Telemetrielogs.

## 8.2 Vom Feedback zum Training

```text
Mobile Antwort
  → Schemaprüfung
  → Geometrie-/Plausibilitätsprüfung
  → Duplikatprüfung
  → Status quick_reviewed
  → bei Hard Case zweite Review
  → Konflikt-Adjudication
  → versionierter Dataset-Snapshot
  → Training eines Kandidaten
  → Evaluation auf gesperrtem Testset
  → Shadow-Vergleich
  → manuelle Modellentscheidung
```

## 8.3 Trainingstakt

Nicht nach jedem Tap neu trainieren. Vorgeschlagener Rhythmus:

- Feedback sofort speichern;
- täglich oder nach einer Mindestzahl neue Aufgaben aufbereiten;
- Dataset-Snapshot bei beispielsweise 100 neuen bestätigten Cropkorrekturen
  oder 250 Enhancementpräferenzen;
- Training nur auf dem PC und nur mit versioniertem Snapshot;
- Evaluation automatisch nach jedem Training;
- Modellaktivierung erst nach Vergleichsreport.

## 8.4 Modellversionierung

Jede Version bindet:

- Trainingsdaten-Snapshot und Hash;
- Code-Commit;
- Modellkonfiguration;
- Ausgangsgewichte;
- Exportparameter;
- Modellhash;
- Calibration-Artefakt;
- Testreport;
- bekannte Fehler und Freigabestatus.

## 8.5 Schutz vor Modellverschlechterung

- Testsplit bleibt unverändert;
- Champion-Challenger-Vergleich;
- keine Aktivierung bei schlechterem kritischem Slice;
- Page-perfect und Protected-cut sind wichtiger als mittlere IoU allein;
- Enhancement-Humanurteil ist wichtiger als isolierter IQA-Gewinn;
- alte Modelle und Originalassets bleiben verfügbar;
- Rollback ist lokal mit einem Konfigurationswechsel möglich.

---

## 9. Mobile Seiten und UX-Struktur

## 9.1 Capture-PWA

Routen beziehungsweise Ansichten:

- `/mobile/capture` – Kamera und letzter Upload;
- `/mobile/capture/batch` – Album/Batch auswählen;
- `/mobile/capture/queue` – wartende und fehlgeschlagene Uploads;
- `/mobile/capture/help` – kurze Aufnahmehilfe.

## 9.2 Review-PWA

- `/mobile/review` – Aufgabenauswahl und Fortschritt;
- `/mobile/review/crop/count` – Printanzahl;
- `/mobile/review/crop/quality` – korrekt/falsch;
- `/mobile/review/crop/edit` – Quad-/Polygoneditor;
- `/mobile/review/crop/protection` – Motivschutz;
- `/mobile/review/enhancement/need` – No-op/Defekt;
- `/mobile/review/enhancement/ab` – Blind-A/B;
- `/mobile/review/enhancement/problem` – Reason Code/lokale Stelle;
- `/mobile/review/progress` – persönliche Arbeitsstatistik.

Capture und Review dürfen technisch dieselbe PWA und Anmeldung verwenden. In
der Bedienung bleiben sie zwei klar getrennte Hauptaktionen.

## 9.3 Mobile Designanforderungen

- Touchflächen mindestens ungefähr 44 × 44 CSS-Pixel;
- keine Hover-only-Funktionen;
- sichere Darstellung im Hoch- und Querformat;
- Unterstützung von Pinch-Zoom und Pan ohne Seiten-Scrollkonflikt;
- dunkles, sonnenlichttaugliches und kontrastreiches Theme;
- sofort sichtbarer Offline-/Sync-Status;
- Text und Controls bleiben auch mit Handschuh- beziehungsweise Einhandbedienung
  möglichst einfach;
- Aufgaben können ohne Freitext beendet werden;
- nach jedem Tap persistierter Fortschritt;
- kein Datenverlust beim Sperren des Displays oder Schliessen des Tabs.

---

## 10. Backend- und Datenmodell-Erweiterungen

Benötigte Kernobjekte:

### CaptureBatch

- ID, Album, Name, Status;
- erstellt/geändert;
- erwartete optionale Seitenzahl;
- Geräte-/Aufnahmeserie;
- Syncfortschritt.

### CaptureAsset

- Originalpfad und Hash;
- Batch und Album;
- Uploadstatus;
- Decode-/Qualitätsmetadaten;
- Quarantänegrund;
- Importstatus.

### ReviewTask

- Tasktyp;
- Priorität und Active-Learning-Grund;
- Asset-/Photo-/Variant-Referenzen;
- Payloadschema;
- Status, Lease und Ablaufzeit;
- geschätzte Dauer;
- Split-Schutz.

### ReviewEvent

- append-only Antwort;
- Nutzer/Reviewer;
- Reason Codes;
- Geometrie oder lokale Region;
- UI-/Modellversion;
- Zeit und Undo-Bezug.

### DatasetSnapshot

- enthaltene Seiten/Labels;
- Splitmanifest;
- Hash und Erstellungsgrund;
- Labelstatusverteilung;
- Ausschlüsse und bekannte Grenzen.

### ModelEvaluation

- Modell-/Calibration-Hash;
- Crop- und Enhancementmetriken;
- Slice-Ergebnisse;
- Regressionen;
- Reportpfad;
- Status Champion, Challenger, rejected oder research.

---

## 11. API-Plan

### Capture

- Batch erstellen/auflisten;
- Upload initialisieren;
- Chunk hochladen;
- Upload abschliessen;
- serverseitigen Hashstatus abfragen;
- Queue/Fehler anzeigen;
- Aufnahme erneut einem Album zuordnen.

### Review

- nächste Aufgabe nach Typ und Zeitbudget reservieren;
- Taskpayload laden;
- Zwischenstand speichern;
- Antwort und Reason Codes abschliessen;
- Aufgabe überspringen;
- Undo-Ereignis schreiben;
- persönlichen Fortschritt abfragen.

### Training und Evaluation

- Dataset-Vorschau erzeugen;
- Snapshot einfrieren;
- Labelvalidator ausführen;
- Trainingsjob lokal starten;
- Evaluationsergebnis lesen;
- Modellvergleich anzeigen;
- Championwechsel bewusst bestätigen.

Alle mutierenden Requests sind idempotent. Mobile Wiederholungen nach einem
Netzfehler dürfen weder doppelte Uploads noch doppelte Reviewantworten erzeugen.

---

## 12. Entwicklungsphasen

## Phase M0 – Mobile Basis und Pairing

### Umsetzung

- responsive PWA-Shell;
- Smartphone-PC-Pairing per QR-Code und kurzlebigem Token;
- HTTPS im lokalen Netz;
- Geräte-/Sitzungsstatus;
- IndexedDB und allgemeine Sync-Queue;
- installierbares PWA-Manifest.

### Gate

- PWA funktioniert auf dem vorgesehenen Smartphone;
- erneutes Öffnen erhält Session und Queue;
- nicht gepairtes Gerät kann nicht hochladen oder reviewen;
- Offlinezustand wird korrekt angezeigt.

## Phase M1 – Capture und automatische PC-Inbox

### Umsetzung

- Kameraaufnahme;
- Vorschau/Wiederholen;
- Chunked Upload;
- Retry und Hashbestätigung;
- CaptureBatch;
- atomare PC-Inbox;
- Import-/Quarantäneworker;
- Uploadfortschritt.

### Gate

- Capture-Abnahmekriterien aus Abschnitt 4.6 bestanden;
- mindestens ein vollständiger realer Album-Batch wurde ohne manuelles
  Dateiverschieben aufgenommen.

## Phase M2 – schnelle Crop-Reviews

### Umsetzung

- Printanzahl;
- Crop korrekt/falsch;
- Reason Codes;
- False Positive entfernen;
- fehlenden Print grob markieren;
- ReviewTask/ReviewEvent;
- Task-Leasing und Wiederaufnahme.

### Gate

- fünf Aufgaben sind zuverlässig in einer kurzen Sitzung lösbar;
- Antworten erzeugen gültige, nachvollziehbare Events;
- keine Testdaten gelangen in die normale Trainingsqueue.

## Phase M3 – mobiler Polygoneditor und Protected Regions

### Umsetzung

- Touch-Quad-Editor;
- Zoom/Pan/Snap;
- Polygonvalidierung;
- Motivschutz-Overlay;
- zweite Review und Adjudication;
- Active-Learning-Priorisierung.

### Gate

- Polygonkorrekturen funktionieren für kleine, gedrehte und randnahe Prints;
- ungültige Polygone können nicht abgeschlossen werden;
- Korrekturen lassen sich verlustfrei nach YOLO/Benchmarkformat exportieren.

## Phase M4 – Crop-Trainingskreislauf

### Umsetzung

- Dataset-Snapshots;
- albumgetrennte Splits;
- automatisches Training;
- ONNX-Export und Parität;
- Calibration;
- Champion-Challenger-Report;
- visuelle Fehlergalerie.

### Gate

- ein neuer Kandidat kann aus bestätigten Mobile-Labels reproduzierbar
  trainiert und evaluiert werden;
- Aktivierung bleibt manuell und fail-closed;
- Crop-Qualitätsziele sind erreicht oder verbleibende Slices klar ausgewiesen.

## Phase M5 – mobile Enhancement-Diagnose und Blind-A/B

### Umsetzung

- No-op-/Defektaufgaben;
- synchroner A/B-Viewer;
- Randomisierung;
- Reason Codes;
- lokale Problemstelle;
- Kontroll- und Wiederholungsaufgaben;
- Präferenzexport.

### Gate

- Seitenposition bleibt wirklich verblindet;
- Zoom/Pan zeigt dieselbe Region beider Varianten;
- No-op ist gleichwertig auswählbar;
- Antworten sind konsistent versioniert und auswertbar.

## Phase M6 – Enhancement-Lernkreislauf

### Umsetzung

- Planner aus Defektlabels verbessern;
- No-op-Schwelle kalibrieren;
- Critic anhand lokaler Fehlergründe testen;
- Kandidatenranking aus A/B-Präferenzen auswerten;
- No-op-, Identitäts-, Text- und Artefaktreports;
- konservative Champion-Challenger-Auswahl.

### Gate

- Enhancement-Qualitätsziele aus Abschnitt 7.5 erreicht;
- gute Fotos werden zuverlässig nicht verändert;
- kritische Schäden vetoen Kandidaten;
- Human-A/B und automatische Gates widersprechen sich nicht systematisch.

## Phase M7 – vollständiger lokaler Dauerkreislauf

### Ziel

```text
5 Minuten frei
  → einige Seiten fotografieren oder fünf Reviews lösen
  → PC synchronisiert und validiert automatisch
  → neue bestätigte Daten sammeln sich an
  → geplanter Snapshot/Trainingslauf
  → Report und neue gezielte Aufgaben
```

### Gate

- mehrere reale Aufnahme- und Reviewzyklen ohne Datenverlust;
- Fortschritt nach Album, Fehlerklasse und Modellversion sichtbar;
- Modelle verbessern sich auf gesperrten Daten und nicht nur auf neuen Labels;
- jederzeitiger Rückgriff auf vorherigen Champion und Originale.

---

## 13. Empfohlene konkrete Reihenfolge

1. Mobile PWA-Shell und PC-Pairing bauen.
2. Kameraaufnahme, Offline-Queue und automatische PC-Inbox fertigstellen.
3. Bestehende Crops als einfache Ja/Nein-Aufgaben mobil anzeigen.
4. Reason Codes, fehlende Prints und False Positives ergänzen.
5. Touch-Polygoneditor implementieren.
6. Dataset-Snapshot und Crop-Trainingskreislauf anschliessen.
7. Protected Regions und Confidence-Kalibrierung mit Mobile-Hard-Cases
   verbessern.
8. Crop-Gates auf dem gesperrten Testset erfüllen.
9. No-op-/Defektklassifikation mobil anbieten.
10. Blind-A/B-Viewer für Enhancement bauen.
11. konservativen Enhancement Planner und Critic mit den Bewertungen
    verbessern.
12. vollständigen lokalen Dauerkreislauf betreiben und messen.

Diese Reihenfolge liefert früh einen nützlichen Capture-Workflow und danach
schrittweise bessere Labels. Enhancement beginnt erst ernsthaft, wenn Crops
verlässlich genug sind.

---

## 14. Fortschrittsanzeige

Die mobile und PC-Ansicht soll nicht nur eine rohe Anzahl Taps zeigen.

Sinnvolle Kennzahlen:

- aufgenommene und serverseitig bestätigte Seiten;
- noch nicht geprüfte Seiten;
- bestätigte, korrigierte und zweitgeprüfte Crops;
- häufigste Crop-Fehlergründe;
- Abdeckung nach Album, Gerät und Hard-Case-Klasse;
- No-op-Entscheidungen;
- Enhancement-A/B-Vergleiche;
- häufigste Enhancement-Schäden;
- neuer Feedbackstand seit letztem Dataset-Snapshot;
- aktueller Champion und letzter Challenger;
- Testmetriken und Regressionen;
- geschätzte Anzahl kurzer Aufgaben bis zum nächsten sinnvollen Trainingslauf.

Gamification bleibt sachlich: Fortschritt und hilfreiche Abdeckung werden
angezeigt, aber Geschwindigkeit darf nicht zu ungenauen Labels motivieren.

---

## 15. Was von dir benötigt wird

### Regelmässig auf dem Smartphone

- Albumseiten fotografieren;
- schnelle Crop-Bestätigungen;
- fehlende oder falsche Prints markieren;
- bei ausgewählten Hard Cases Ecken korrigieren;
- No-op/Defekt angeben;
- verblindete Enhancement-Paare vergleichen;
- `unsicher` verwenden, wenn eine Entscheidung nicht verlässlich möglich ist.

### Einmalig beziehungsweise gelegentlich

- PC und Smartphone für den lokalen Zugriff verbinden;
- Album-/Batchnamen festlegen;
- bestätigen, welche Aufnahmen zum selben Album gehören;
- besonders schwierige Konflikte adjudizieren;
- nach Möglichkeit Referenzscans einzelner Prints bereitstellen;
- bei subjektiven Farbfragen den gewünschten historischen beziehungsweise
  neutralen Look festlegen.

### Automatisierbare Arbeit auf dem PC

- Uploadannahme und Dateiablage;
- technische Qualitätsprüfung;
- Proposal-Erzeugung;
- Task-Priorisierung;
- Labelvalidierung;
- Dataset-Snapshots;
- Training und ONNX-Export;
- Calibration und Evaluation;
- Fehlergalerien und Reports;
- Backup der Originale, Labels und Modellartefakte.

---

## 16. Abschlussdefinition

Der Cropping- und Enhancement-Entwicklungsplan gilt als erfolgreich umgesetzt,
wenn:

1. du mit der mobilen Capture-Seite ein Foto aufnehmen kannst und es ohne
   manuelles Verschieben hashverifiziert im PC-Inbox-Ordner landet;
2. Uploads bei Unterbrechungen nicht verloren gehen;
3. du in ungefähr fünf Minuten mehrere sinnvolle Crop- oder Enhancement-
   Aufgaben abschliessen kannst;
4. Mobile-Feedback versioniert und kontrolliert in Dataset-Snapshots gelangt;
5. neue Modelle reproduzierbar trainiert, kalibriert und gegen gesperrte Daten
   verglichen werden;
6. Cropping die definierten Precision-, Recall-, Page-perfect- und
   Protected-cut-Ziele erreicht;
7. Enhancement gute Bilder überwiegend No-op lässt;
8. Enhancement keine automatisch akzeptierten Identitäts-, Text- oder
   Halluzinationsschäden erzeugt;
9. Originale, Labels und frühere Champions jederzeit wiederherstellbar sind;
10. ein vollständiger lokaler Report den Qualitätsstand und alle verbleibenden
    Fehlerklassen zeigt.

Erst danach wird ein eigener Plan für Kundenbetrieb und Produktisierung
gestartet.

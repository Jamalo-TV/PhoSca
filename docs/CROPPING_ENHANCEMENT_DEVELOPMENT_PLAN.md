# PhoSca – Entwicklungsplan nur für Cropping und Enhancement

**Stand:** 11. Juli 2026  
**Ziel:** Eine lokal ausführbare, messbar zuverlässige Bildpipeline entwickeln,
bevor Kundenbetrieb, Cloudplattform, Skalierung, Abrechnung, Consumer-Launch und
formale Unternehmensfreigaben betrachtet werden.

## 1. Klare Abgrenzung

Dieser Plan umfasst ausschliesslich:

1. Albumseite einlesen;
2. alle physischen Prints vollständig erkennen;
3. Perspektive sicher korrigieren;
4. unsichere Ergebnisse zur manuellen Prüfung markieren;
5. gute Prints unverändert lassen;
6. beschädigte Prints nur bei erkennbarem Bedarf konservativ verbessern;
7. Original, Crop und Enhancement direkt vergleichbar ausgeben;
8. Qualität auf den bereitgestellten Seiten reproduzierbar messen.

Dieser Plan umfasst vorerst ausdrücklich nicht:

- Kundenkonten, Bezahlung oder öffentlichen Betrieb;
- Cloud-, Kubernetes- oder Multi-Tenant-Infrastruktur;
- Consumer-SLOs, Canary-Rollout oder Supportprozesse;
- Marketing, Produktdesign oder App-Store-Veröffentlichung;
- formale DPIA/DSFA-, Pen-Test- oder Consumer-Launch-Freigaben;
- generatives Outpainting oder kreative Bildveränderung.

Die lokale Pipeline bleibt trotzdem sicher: Originale werden nie überschrieben,
unsichere Ergebnisse werden nicht automatisch als besser bezeichnet, und
generative Enhancementpfade bleiben deaktiviert.

## 2. Annahmen über die bereitgestellten Daten

Für die Planung wird angenommen, dass ausreichend Albumseiten bereitgestellt
werden. Zielgrösse für einen belastbaren Entwicklungsdatensatz:

- mindestens 500 Albumseiten;
- möglichst 20 oder mehr unterschiedliche Alben;
- unterschiedliche Kameras beziehungsweise Smartphones;
- unterschiedliche Beleuchtung, Schatten und Perspektiven;
- helle, dunkle, farbige und strukturierte Albumseiten;
- matte und glänzende Prints;
- einzelne grosse und viele kleine Prints;
- randnahe, gedrehte, überlappende und teilweise verdeckte Prints;
- Gruppenbilder, kleine Gesichter, Tiere, Gegenstände und Text im Foto;
- gute, unscharfe, vergilbte, verrauschte und beschädigte Fotos.

Wenn anfangs weniger Seiten verfügbar sind, wird derselbe Plan als Pilot
ausgeführt. Die erreichten Kennzahlen gelten dann nur für diesen Pilotdatensatz.

### Für Enhancement zusätzlich sehr hilfreich

Albumseiten allein zeigen, was restauriert werden soll, aber nicht objektiv,
wie das perfekte Ergebnis aussehen müsste. Für eine belastbare Enhancement-
Bewertung werden deshalb nach Möglichkeit benötigt:

- 50–100 hochwertige Referenzscans derselben physischen Prints;
- unveränderte gute Prints als No-op-Sicherheitsset;
- Beispiele realer Kratzer, Flecken, Risse, Rauschen und Unschärfe;
- optional professionell restaurierte Referenzen.

Ohne Referenzscans kann Enhancement trotzdem entwickelt werden. Die Beurteilung
stützt sich dann stärker auf No-op-Sicherheit, lokale Fehlerdetektoren und
verblindete menschliche A/B-Vergleiche.

## 3. Definition des fertigen lokalen Ergebnisses

Die Entwicklungsphase ist abgeschlossen, wenn ein lokaler Testlauf für jede
bereitgestellte Seite Folgendes erzeugt:

```text
Albumseite
  ├─ erkannte Print-Polygone mit Confidence
  ├─ vollständiger perspektivkorrigierter Crop je Print
  ├─ Crop-Gate: bestanden oder Review mit Reason Code
  ├─ Original-Crop als unveränderlicher Master
  ├─ Enhancement-Entscheidung: No-op oder konkrete Verarbeitung
  ├─ optionale konservative Enhancement-Variante
  ├─ Enhancement-Gates und Messwerte
  └─ visueller Vergleich: Seite / Crop / Enhancement
```

Ein technisch fehlerfreier Lauf genügt nicht. Richtig bedeutet:

- kein sichtbarer Print fehlt;
- keine Albumfläche wird als zusätzlicher Print ausgegeben;
- der vollständige physische Print bleibt erhalten;
- Köpfe, Körperteile, Tiere, Gegenstände und Text werden nicht abgeschnitten;
- Perspektivkorrektur erzeugt keine starke Verzerrung oder schwarzen Ränder;
- gute Fotos bleiben No-op;
- Enhancement erzeugt keine neuen Gesichter, Details, Texte oder Strukturen;
- bei Unsicherheit wird Original beziehungsweise Review gewählt.

## 4. Phase C0 – Dateneingang und reproduzierbare Baseline

### Arbeit

1. Alle bereitgestellten Seiten inventarisieren.
2. Dateiformat, Auflösung, EXIF, ICC, Orientierung und Mehrfachframes prüfen.
3. Exakte Duplikate und nahezu identische Aufnahmen erkennen.
4. Bilder nach Album, Gerät und Aufnahmeserie gruppieren.
5. Bestehende Pipeline unverändert auf allen Seiten ausführen.
6. Laufzeit, Detektionszahl, Confidence und Reviewgründe speichern.
7. Fehlergalerie mit Miss, False Positive, Fragment, falscher Perspektive und
   problematischem Enhancement erstellen.

### Ergebnis

- Dataset-Inventar;
- unveränderlicher Baseline-Report;
- visuelle Baseline-Galerie;
- Liste reproduzierbarer Hard Cases;
- deterministischer Train-/Val-/Testsplit ohne Album-Leakage.

### Gate C0

- jede Datei ist dekodierbar oder mit Grund abgelehnt;
- keine Seite liegt gleichzeitig in Training und Test;
- derselbe Pipeline-Lauf erzeugt reproduzierbare Ergebnisse;
- Originaldateien bleiben unverändert.

## 5. Phase C1 – Ground Truth für vollständige Prints

### Annotation

Für jeden sichtbaren physischen Print wird ein Polygon beziehungsweise Quad
annotiert. Wichtig ist die vollständige Printgrenze, nicht ein ästhetischer
Innenausschnitt.

Reason Codes für schwierige Fälle:

- `print_complete`;
- `print_partially_occluded`;
- `print_boundary_uncertain`;
- `overlapping_prints`;
- `page_edge_cut`;
- `reflection_or_glare`;
- `curved_page`;
- `damaged_print_edge`;
- `not_a_print`.

Mindestens alle Testseiten und Hard Cases werden doppelt geprüft. Konflikte
werden sichtbar adjudiziert statt durch Mittelung versteckt.

### Ergebnis

- versionierte Polygonlabels;
- dokumentierte Labelregeln;
- Annotation-QA-Report;
- gesperrter Testsplit;
- automatische Prüfungen auf leere, ungültige und sich selbst schneidende
  Polygone.

### Gate C1

- jede Testseite besitzt vollständige Labels;
- keine ungelösten Annotationkonflikte;
- Testlabels werden nicht für Training oder Threshold-Tuning verwendet;
- Labelvalidator endet ohne Fehler.

## 6. Phase C2 – Print Detection und sichere Geometrie

### Modell- und Algorithmusarbeit

1. Bestehenden ONNX-Segmentierer auf dem erweiterten Datensatz evaluieren.
2. Fehler nach Albumtyp, Gerät, Printanzahl und Printgrösse auswerten.
3. Modell auf dem neuen Trainingssplit nachtrainieren.
4. Niedrigen Proposal-Threshold und getrennten Release-/Review-Threshold
   beibehalten.
5. Modell- und klassische Kandidaten erst nach vollständiger Bewertung
   zusammenführen.
6. Kandidatenset global auswählen, damit ein grosser falscher Container nicht
   mehrere richtige Prints verdrängt.
7. Polygone in robuste Quads überführen.
8. Konvexität, Winkel, Eindeutigkeit, Fläche, Homographie-Kondition und
   Reprojection Error prüfen.
9. Safe Margin verwenden, wenn die Printgrenze unsicher ist.
10. Border Trim weiterhin deaktiviert lassen, bis ein eigener Test seine
    Sicherheit belegt.

### Primäre Metriken

- Precision und Recall bei strenger Polygon-IoU;
- mittlere und untere Perzentile der Polygon-IoU;
- Page-perfect Rate;
- False Positives je Seite;
- vollständig verpasste Seiten;
- Duplikate je physischem Print;
- geometrisch ungültige Warps;
- Laufzeit pro Megapixel.

### Zielwerte für den lokalen Testsplit

- Precision mindestens 98 %;
- Recall mindestens 98 %;
- Page-perfect mindestens 95 %;
- keine geometrisch ungültige automatische Ausgabe;
- bekannte Hard Cases dürfen Review auslösen, aber nicht still falsch
  ausgegeben werden.

Diese Werte sind Entwicklungsziele und werden nach Grösse und Schwierigkeit des
bereitgestellten Datensatzes überprüft. Sie dürfen nicht durch Training auf dem
Testsplit erreicht werden.

### Gate C2

- Zielwerte auf dem gesperrten lokalen Testsplit erreicht;
- keine Regression gegenüber der Baseline in einem relevanten Slice;
- jeder verworfene Crop besitzt einen stabilen Reason Code;
- visueller Report aller Fehler liegt vor.

## 7. Phase C3 – Schutz von Bildinhalten

Print Detection und semantischer Motivschutz werden getrennt betrachtet. Der
Printdetektor soll die physische Grenze finden. Die Protected Regions prüfen,
ob die vorgeschlagene Grenze plausibel ist oder relevante Inhalte schneidet.

### Detektoren

- Gesicht und Kopf;
- Person und Pose/Keypoints;
- Tiere und wichtige Objekte;
- Text innerhalb des Prints;
- allgemeines Hauptmotiv beziehungsweise Subject Mask.

Die konservative Union aller ausreichend sicheren Regionen wird verwendet.
Ein Schnitt durch eine bekannte Protected Region ist ein Hard Veto. Ein
fehlender oder widersprüchlicher Detektor reduziert die Freigabe-Confidence.

### Besondere Tests

- kleine und teilweise verdeckte Gesichter;
- Gruppen am Bildrand;
- Haare, Hände und Füsse nahe der Printgrenze;
- Tiere und Gegenstände am Rand;
- Text, Schilder und Datumsstempel;
- Albumbeschriftungen ausserhalb des Prints, die den Crop nicht fälschlich
  blockieren dürfen;
- sehr dunkle oder überbelichtete Personen.

### Gate C3

- kein bestätigter Protected-Cut wird automatisch akzeptiert;
- False-Veto-Rate ist gemessen und für Review handhabbar;
- Ergebnisse sind nach Regionstyp und Objektgrösse aufgeschlüsselt;
- fehlende Modellgewichte oder Detector-Fehler führen zu Review statt Freigabe.

## 8. Phase C4 – Confidence-Kalibrierung und Crop-Auswahl

Die rohe Modell-Confidence wird nicht als Wahrscheinlichkeit interpretiert.

### Arbeit

1. Vorhersagen auf einem nicht zum Training verwendeten Kalibrierungssplit
   sammeln.
2. Jede Vorhersage als korrekt oder falsch labeln.
3. Monotone Kalibrierung fitten.
4. Reliability Diagram, Brier Score und Calibration Error berichten.
5. Risk-Coverage-Kurve erstellen.
6. Schwellen für `auto_candidate`, `review` und `reject` auswählen.
7. Stabilität nach Gerät, Albumtyp, Printanzahl und Printgrösse prüfen.

### Gate C4

- Kalibrierungsartefakt ist versioniert und an Dataset-/Modellhash gebunden;
- niedrige Confidence korreliert tatsächlich mit höherem Fehlerrisiko;
- Ziel-Fehlerrate wird durch Abstention eingehalten;
- ohne gültiges Artefakt bleibt die Ausgabe im Reviewmodus.

## 9. Phase E0 – Enhancement-Datensatz und No-op-Baseline

Enhancement beginnt erst mit den bestandenen beziehungsweise manuell
bestätigten vollständigen Crops. Ein fehlerhafter Crop darf nicht durch
Enhancement kaschiert werden.

### Datensätze

1. **No-op-Set:** bereits gute Crops, die unverändert bleiben sollen.
2. **Defektset:** reale Beispiele für Rauschen, Unschärfe, Vergilbung,
   Verblassung, Kratzer, Flecken und Kompressionsartefakte.
3. **Paarset:** degradierter Crop plus hochwertiger Referenzscan, soweit
   vorhanden.
4. **Identitäts-/Textset:** Gesichter, Gruppen und Text für lokale Hard Gates.

### Baseline

- Original unverändert;
- Lanczos/Bicubic als nichtgenerative Skalierungsreferenz;
- aktuelle konservative Farb-/Denoise-Operationen;
- bestehende Modellkandidaten nur als nicht freigegebene Vergleichsvarianten.

### Gate E0

- No-op-Set ist explizit gelockt;
- Defektklassen und Schweregrade sind dokumentiert;
- Referenzpaare sind geometrisch und farblich ausgerichtet;
- Baseline-Report enthält globale und lokale Fehlerbilder.

## 10. Phase E1 – Degradation Planner mit echter No-op-Entscheidung

Der Planner entscheidet zuerst, ob überhaupt verarbeitet werden soll. Er gibt
für jede Operation Bedarf, Schweregrad und Unsicherheit aus.

Mögliche Entscheidungen:

- `noop`;
- `conservative_color`;
- `denoise`;
- `deblur_candidate`;
- `small_defect_repair`;
- `fidelity_super_resolution`;
- `review_only`.

Regeln:

- niedrige Diagnose-Confidence führt zu No-op oder Review;
- keine Operation wird allein wegen eines globalen Max-Severity-Werts
  aktiviert;
- Face Restoration, grosses Inpainting und generative SR bleiben deaktiviert;
- Kosten und Laufzeit sind nachrangig gegenüber Bildsicherheit.

### Gate E1

- No-op-FPR auf guten Bildern ist höchstens 5 %, Zielwert später unter 2 %;
- jede ausgeführte Operation hat einen dokumentierten Auslöser;
- Planner-Ausfälle ergeben Original/Review;
- Routing-Fehler sind je Defektklasse gemessen.

## 11. Phase E2 – konservative Enhancement-Kandidaten

### Reihenfolge

1. optionales, schwaches Denoising;
2. konservative Farb- und Dynamikkorrektur;
3. optionales fidelity-first Deblur;
4. kleine Defektreparatur mit präziser Maske;
5. optionale fidelity-first Super-Resolution;
6. Source Blend nur, wenn dadurch ein gemessener Sicherheitsgewinn entsteht.

Jede Stufe erzeugt einen Kandidaten. Der Eingang bleibt erhalten. Eine
zurückgesetzte Stufe beeinflusst spätere Stufen nicht.

### Verbotene automatische Standardpfade

- globales GFPGAN;
- kreative Face Reconstruction;
- grosses generatives Inpainting;
- Diffusion und Outpainting;
- pauschales SR für jedes Bild;
- automatische Kratzerentfernung ohne präzise Defektmaske.

### Gate E2

- jede Stufe ist einzeln abschaltbar;
- keine Modellgewichte werden zur Laufzeit heruntergeladen;
- Modell, Hash, Parameter und Input-/Output-Hash werden protokolliert;
- Fehler erzeugen kein als „enhanced“ bezeichnetes Originalduplikat;
- No-op erzeugt keine unnötige Datei.

## 12. Phase E3 – unabhängiger Enhancement Critic

Der Critic wird getrennt von den Enhancementmodellen implementiert.

### Hard Gates

- gültig dekodierbare Pixel;
- keine starke Highlight-/Shadow-Clipping-Zunahme;
- keine neuen Halos oder Ringing-Artefakte;
- keine unzulässige Text-/OCR-Änderung;
- keine lokale Gesichts-/Identitätsänderung;
- keine unplausiblen Inpaintingstrukturen;
- keine starke Detail- oder Texturvernichtung;
- Source Delta innerhalb der für die konkrete Operation erlaubten Grenze.

### Unterstützende Metriken

- Full-reference-Metriken auf Paaren;
- lokale Differenzkarten;
- Schärfe, Rauschen und Dynamik;
- mehrere perceptual/IQA-Signale;
- Vergleich mit einfacher nichtgenerativer Baseline.

Keine einzelne IQA-Zahl darf ein Hard Gate überstimmen.

### Gate E3

- alle bekannten schlechten Kandidaten werden mit hoher Rate abgelehnt;
- gute konservative Kandidaten werden nicht pauschal verworfen;
- Gate Recall und False-positive Rate sind dokumentiert;
- Original bleibt bei jedem Gatefehler ausgewählt.

## 13. Phase E4 – verblindete lokale A/B-Bewertung

Diese Phase bewertet nicht den Consumer-Launch, sondern nur, ob die lokale
Enhancementpipeline sichtbar besser als Original und Baseline ist.

### Ablauf

- zufällige linke/rechte Position;
- keine Anzeige von Modellnamen;
- Bewertung von Natürlichkeit, Detail, Farbe, Identität und sichtbaren Schäden;
- No-op als gültige bevorzugte Antwort;
- separate Kennzeichnung kritischer Fehler;
- mehrere Bewertungen pro Bild, soweit Reviewer verfügbar sind.

### Gate E4

- Enhancement gewinnt bei tatsächlich degradierten Bildern signifikant gegen
  die konservative Baseline;
- gute Bilder werden überwiegend als No-op bewertet;
- kritische Identitäts-, Text- oder Halluzinationsfehler bleiben unter einer
  vorab festgelegten sehr niedrigen Grenze;
- automatische Metriken und Humanurteil widersprechen sich nicht systematisch.

## 14. Phase I0 – integrierter lokaler End-to-End-Test

### Zielablauf

```text
Ordner mit Albumseiten auswählen
  → Seiten prüfen
  → Prints erkennen
  → Crops und Reviewgründe anzeigen
  → bestätigte Crops verarbeiten
  → No-op/Enhancement-Kandidaten erzeugen
  → Original/Crop/Enhancement vergleichen
  → Ergebnisse und Qualitätsreport exportieren
```

### Testarten

- kompletter Lauf über alle bereitgestellten Seiten;
- Wiederholung desselben Laufs;
- Abbruch und Fortsetzung;
- fehlendes Modell;
- korrupte Datei;
- sehr grosse Seite;
- Seite ohne Print;
- manuell korrigiertes Polygon;
- Enhancementfehler;
- Export und erneutes Öffnen der Resultate.

### Endgültiges lokales Abnahmegate

Cropping:

- C2-, C3- und C4-Gates bestanden;
- alle Fehlerfälle sichtbar und reproduzierbar;
- keine automatische Ausgabe eines geometrisch oder semantisch ungültigen
  Crops.

Enhancement:

- E1- bis E4-Gates bestanden;
- No-op funktioniert zuverlässig;
- kein kritischer Identitäts-, Text-, Inpainting- oder Halluzinationsfehler in
  der freigegebenen lokalen Testmenge;
- Original ist jederzeit wiederherstellbar.

Bedienung:

- ein lokaler Nutzer kann die Ergebnisse ohne Datenbankeingriffe prüfen und
  korrigieren;
- pro Ergebnis sind Modell, Parameter, Gates und Reason Codes sichtbar;
- vollständiger HTML-/JSON-Qualitätsreport wird erzeugt.

## 15. Konkrete Arbeitsreihenfolge

1. Seiten übernehmen, inventarisieren und Baseline laufen lassen.
2. Albumweise Train-/Val-/Testsplit erstellen.
3. vollständige Printpolygone annotieren und prüfen.
4. Printsegmentierer trainieren und auf dem gelockten Testsplit evaluieren.
5. Geometrie-, Protected-Region- und Post-Warp-Gates fertigstellen.
6. Confidence kalibrieren und Reviewschwellen festlegen.
7. Cropping-End-to-End-Gate bestehen.
8. aus bestätigten Crops No-op-, Defekt- und Paarsets aufbauen.
9. Degradation Planner und No-op optimieren.
10. konservative Enhancement-Kandidaten integrieren.
11. unabhängigen Critic kalibrieren.
12. lokale verblindete A/B-Bewertung durchführen.
13. vollständigen lokalen End-to-End-Lauf und Abschlussreport erzeugen.

## 16. Was für diesen Plan von dir benötigt wird

Vorausgesetzt werden die Albumseiten. Zusätzlich sind folgende Mitwirkungen
notwendig:

- Bestätigung, dass die bereitgestellten Bilder für diese lokale Entwicklung
  verwendet werden dürfen;
- nach Möglichkeit Information, welche Seiten zum selben Album beziehungsweise
  zur selben Aufnahmeserie gehören;
- menschliche Prüfung schwieriger Printgrenzen;
- nach Möglichkeit hochwertige Referenzscans einzelner Prints;
- mindestens eine Person für die verblindete Enhancementbewertung;
- Entscheidung bei subjektiven Restaurierungsfragen, etwa gewünschte
  Farbneutralität oder Erhalt eines historischen Looks.

Alle übrigen Arbeiten dieses Plans – Import, Splits, Annotationstools,
Training, Modellintegration, Cropping, Enhancement, Gates, Testautomatisierung,
Auswertung und Reports – können innerhalb des Projekts technisch umgesetzt
werden.

## 17. Was bewusst erst danach folgt

Erst wenn das lokale End-to-End-Gate bestanden ist, beginnt ein separater
Produktionsplan für:

- Kunden und Benutzerkonten;
- Hosting und Object Store;
- Multi-Tenancy und Abrechnung;
- formale Security-, Privacy- und Lizenzfreigaben;
- Last, SLO, Support und Betrieb;
- Consumer-Canary und öffentlichen Launch.

Damit bleibt die Reihenfolge klar: zuerst nachweisbar gute Crops und sichere
Enhancements, danach Produktisierung und Kundenbetrieb.

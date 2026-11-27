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

Bei der Live-Aufnahme zeigt PhoSca das ganze Videobild zur Ausrichtung.
Beim Auslösen fragt die App auf Desktop und Mobile ein Kamerastandbild
über `ImageCapture.takePhoto()` an. Liefert die Kamera ein JPEG oder PNG,
übernimmt die App diese Datei bytegleich. Die Kamera und der Browser
bestimmen ihre Auflösung. Kann die Kamera kein verwendbares Standbild
liefern, erfasst die App ein Videobild in dessen tatsächlicher Auflösung
und zeigt einen Hinweis zur Aufnahmequalität. Mobile bietet zusätzlich
**Mit Handy-Kamera aufnehmen**. Damit kann das Betriebssystem eine
Aufnahmedatei direkt an die App übergeben.
[Aufnahmepfad im Code](frontend/src/cameraCapture.js) ·
[Vergleich mit einer echten Albumaufnahme](data/evaluation/README.md#preserve-camera-stills-before-page-analysis)

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

`POST /api/v1/albums/{album_id}/pages/upload` akzeptiert JPEG und PNG
als Multipart-Felder `files` oder `file`. Die Standardlimits betragen
50 MiB pro Datei, 200 MiB pro Anfrage und 40 Megapixel. `201` liefert
Seiten-ID, Dateiname, Status und SHA-256. `verified: true` bestätigt
den geprüften Upload. Ein normales Duplikat desselben Hashes im selben
Album liefert `409`. Ein Aufruf mit `X-Upload-Retry: 1` kann die
vorhandene Seite zurückgeben. Der Chunk-Endpunkt `/upload/chunk` sammelt
Teile über Upload-ID und inklusive `Content-Range` und validiert die
zusammengesetzte Datei vor der Speicherung.

Die Datenbank trennt das Original `Page`, die Abzüge `ExtractedPhoto`,
die Ausführung `ProcessingRun` mit einzelnen `StageRun`-Einträgen
und die erzeugten Dateien `Asset`. Ein `ReviewEvent` dokumentiert
einen Entscheid mitsamt Prüfsnapshot. Der Seitenstatus
`review_needed` ist ein reguläres Verarbeitungsergebnis. Er kann vom
Status eines einzelnen Abzugs oder Zuschnittmodus abweichen.
[Upload](backend/app/routers/pages.py) · [Speicherung](backend/app/services/uploads.py) ·
[Datenmodell](backend/app/models/entities.py) · [Offline-Warteschlange](frontend/src/offlineQueue.js)

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

Bei zwei bis acht Vorschlägen aus einer unsicheren Ganzseitenanalyse
kann PhoSca vier überlappende Teilbilder auf weitere Abzüge prüfen.
Dafür müssen mindestens zwei mittelgrosse Vorschläge eine gültige
Geometrie haben und mindestens zwei Vorschläge unter der regulären
Konfidenzschwelle des Detektors liegen.
Ein zusätzlicher Vorschlag braucht auf allen vier Seiten Unterstützung
aus den Originalpixeln und darf keine vorhandene oder ungeklärte
Papierkontur schneiden. Er erscheint gestrichelt in der Anzahlprüfung.
Findet die Kachelsuche einen solchen Abzug, prüft PhoSca auch die um
90 und 270 Grad gedrehten Modellansichten. Der direkte Vergleich
braucht eine eindeutige Übereinstimmung dieser beiden Ansichten
und vier klar messbare Papierkanten in den Originalpixeln.
Bleibt genau eine brauchbare Maske aus einer gedrehten Ansicht frei,
kann PhoSca ihren Umriss am gerippten Albumkarton nachmessen.
Ein zusätzliches Teilbild muss diese Maske bestätigen. Die Messungen
müssen auf allen vier Seiten den Übergang vom Albumkarton zum Foto
und an zwei gegenüberliegenden Seiten das Ende der Albumrippen
zeigen. Ein umschliessender Papierträger oder ein anderer bereits
beanspruchter Umriss verhindert den neuen Prüfvorschlag.
[Abzug aus einer gedrehten Ansicht prüfen](data/evaluation/README.md#corroborate-one-rotated-mask-in-a-local-view)

Berührt eine Teilbildmaske genau einen Aufnahmerand, kann PhoSca
zwei weitere Ausschnitte vergleichen. Bei übereinstimmenden Masken
prüft es vom Original aus, wo sichtbare Papierkanten verlaufen und
wo der gerippte Albumkarton bis zum Aufnahmerand reicht. Die
Anzahlprüfung auf Desktop und Mobil zeigt die belegten Abschnitte
als offene Linien. Die Ecke am Rand bleibt offen. Der Hinweis
liefert weder einen Zuschnitt noch eine Bestätigung eines Abzugs.
[Offene Kanten am Aufnahmerand](data/evaluation/README.md#show-open-print-edges-near-the-capture-boundary)

Ein Kachelvorschlag mit schwächerer Modellkonfidenz und messbarem
Papierrand kann eine zusätzliche Modellansicht auslösen. PhoSca
zentriert einen engeren Ausschnitt um diesen Vorschlag. Für einen
neuen Eintrag in der Anzahlprüfung müssen beide Masken und ihre
nachgemessenen Papierecken übereinstimmen. Die zweite Ansicht muss
vier eindeutige Kanten im Original bestätigen.

Scheitert die erste Kantenmessung eines schwächeren Kachelvorschlags
und liefern die bisherigen Prüfungen keinen weiteren Abzug, versucht
PhoSca bis zu zwei unterschiedlich grosse Teilbilder. Beide Ansichten
müssen dieselbe Maske stützen. PhoSca misst die Ecken für jede Maske
separat an den Originalpixeln. Die vier Seiten brauchen starke
Unterstützung. Drei davon müssen eindeutig sein. Eine mehrdeutige
Kante bleibt Teil der Geometrieprüfung.
Die Anzahl erhöht sich erst, wenn du den eigenen Papierabzug am
Original bestätigst. Deckt eine manuell gesetzte Kontur mindestens 55 %
des offenen Kachelvorschlags ab und liegen mindestens 75 % ihrer Fläche
darin, verweist PhoSca beim Hinzufügen oder Verschieben eines Abzugs
auf die Anzahlprüfung. Danach prüfst du seine Ecken und beide Zuschnitte.
[Kachelprüfung](data/evaluation/README.md#missing-prints-in-count-review) ·
[Seite mit genau zwei Modellvorschlägen](data/evaluation/README.md#check-a-risky-page-with-two-model-proposals) ·
[Zweite Kachelansicht](data/evaluation/README.md#confirm-a-weak-tile-mask-from-a-centered-view) ·
[Wiederholte Kantenmessung](data/evaluation/README.md#refit-an-unmeasured-tile-from-two-closer-views)

Bei einem starken Kachelvorschlag mit fehlgeschlagener Kantenmessung
kann eine gerippte Albumseite zusätzliche Evidenz liefern. PhoSca
sucht ausserhalb der Maske nach einem gleichmässig gefärbten Streifen
mit wiederkehrenden Helligkeitsrippen. Die Farbe dieses Streifens
bestimmt, auf welcher Seite jeder gemessenen Kante der Albumkarton
liegen muss. Die vier Kanten brauchen verteilte Unterstützung und der
Umriss muss zur Detektormaske passen. Eine quer zum Referenzstreifen
verlaufende Rippe muss auch vor und nach der Nachbarschaft des Abzugs
messbar sein. Der Umriss erscheint in der Anzahlprüfung, wenn er sich
mit keinem bestehenden oder offenen Umriss überschneidet. Eine
abfotografierte Reproduktion kann dieselbe Textur zeigen. Deshalb
bleiben Papierzugehörigkeit, Ecken und beide Zuschnittmodi prüfpflichtig.
[Gerippter Albumkarton](data/evaluation/README.md#check-a-failed-tile-fit-against-ribbed-backing)

Werden bei der Kachelsuche bereits weitere Abzüge gefunden, kann
PhoSca eine Lücke neben einem fast quadratischen Abzug untersuchen.
Zwei längliche Vorschläge müssen dieselbe Achse stützen. PhoSca
projiziert den kleinen Umriss entlang dieser Achse in freie Bereiche
und prüft jede freie Position in zwei unterschiedlich grossen
Teilbildern. Beide Ansichten müssen dieselbe Papierfläche stützen.
Bei abweichenden Kanten prüft PhoSca den Farbwechsel zum Karton
und ob die Albumrippen vor der Kante enden. Ein gemessener
gemeinsamer Papierrand oder eine bereits beanspruchte Fläche
verhindert den Vorschlag. Die Anzahlprüfung zeigt einen möglichen
zusätzlichen Abzug. Die Bestätigung seiner Anzahl und die Freigabe
seiner Zuschnitte bleiben getrennte Schritte.
[Zusatzansichten und Originalmessung](data/evaluation/README.md#confirm-an-unclaimed-print-in-two-local-views)

Nach einem erfolgreichen Kachellauf kann PhoSca bis zu drei weitere
Teilbildmasken prüfen. Es wählt kleinere Fenster, die die ganze
vorläufige Kontur enthalten. Zwei Ansichten müssen denselben Abzug
finden. Stimmen beide an drei Seiten überein und bleibt eine vierte
Seite unklar, prüft PhoSca dort den Übergang vom Albumkarton ins
Foto. Dabei muss der Karton aus einem separaten Streifen im Original
bestimmbar sein. Konkurrierende Kanten oder ein gemeinsamer Papierrand
können den neuen Vorschlag verhindern. Bestätige einen zusätzlichen
Papierabzug in der Anzahlprüfung und prüfe danach beide Zuschnittmodi.
[Kleinere Kacheln und umstrittene Kanten](data/evaluation/README.md#recheck-an-unclaimed-tile-in-smaller-views)

Wenn die kleineren Ausschnitte einen zusätzlichen Umriss bestätigt
haben, können zwei bereits gemessene Vorschläge eine Reihe oder
Spalte erkennen lassen. PhoSca setzt den Abstand zwischen den beiden
Umrissen einmal fort. Genau eine freie Position darf übrig bleiben.
Zwei Ausschnitte müssen dort dieselbe Papierfläche finden. Im
Original müssen die vier Kanten übereinstimmen und die Albumrippen
an einer Flanke enden. Ein grösserer umschliessender Papierrand oder
eine bereits beanspruchte Fläche verhindert den Vorschlag. Die Anzahl
ändert sich erst mit der Bestätigung, dass es ein eigener Papierabzug
ist. Danach bleiben seine Zuschnitte separat prüfpflichtig.
[Freie Position in einer Reihe prüfen](data/evaluation/README.md#check-a-vacant-position-beside-measured-prints)

Deckt ein schwacher Ganzseitenvorschlag möglicherweise zwei Abzüge ab,
kann ein einzelner starker Kachelfund innerhalb dieser Kontur eine
weitere Prüfung auslösen. PhoSca sucht in der Mitte des gemeinsamen
Vorschlags nach einer durchgehenden Albumrippe. Die Messung prüft die
Textur in der Fuge und ihre Fortsetzung auf beiden Seiten. Stützen die
Originalpixel zwei disjunkte Papierumrisse, zeigt die Anzahlprüfung
beide Umrisse und die Rippe am vorhandenen Eintrag. Beide Exporte
bleiben gesperrt, solange die Papierzuordnung ungeklärt ist.

Stimmen die Konturen mit zwei Papierabzügen am Original überein, lädt
**Umriss 1 laden** oder **Umriss 2 laden** den jeweiligen Entwurf in den
Editor. Speichere einen Abzug im vorhandenen Eintrag und ergänze den
zweiten anschliessend. Bei einem gemeinsamen Papierabzug bestätige
**Beide Motive liegen auf einem gemeinsamen Papierabzug** und setze die
vier Ecken so, dass beide gemessenen Umrisse vollständig darin liegen.
PhoSca prüft die gewählte Kontur beim Speichern erneut und protokolliert
den Papierentscheid. Danach gelten die üblichen Zuschnittprüfungen für
beide Modi.
[Prüfung zweier möglicher Abzüge](data/evaluation/README.md#measure-a-possible-pair-with-one-separate-tile-mask)

Blockiert eine einzelne, am Aufnahmerand abgestützte Kontrastkontur
einen gemessenen Kachelumriss, prüft PhoSca beide am Original. Ist
der Papierverlauf des Kachelfunds belegt und führt eine Seite der
alten Kontur ohne belegte Papierkante quer durch sein Bild, kann
der Editor den Kachelfund als gestrichelte Alternative zeigen.
Unter **Ganzer Abzug** prüfst du die Alternative, lädst sie in
denselben Eintrag und korrigierst bei Bedarf die Ecken. PhoSca prüft
vor dem Speichern erneut, ob der Umriss andere Vorschläge schneidet.
Die Alternative entsteht bei der Abzugserkennung auf der Albumseite.
Eine Neuberechnung des einzelnen Zuschnitts nutzt die gespeicherte
Kontur dieses Eintrags.
[Vergleich auf der Spiralalbumseite](data/evaluation/README.md#review-a-tile-outline-blocked-by-a-capture-frame-contour)

Erfasst ein Vorschlag nur den oberen Teil eines möglichen Abzugs,
kann PhoSca unter **Ganzer Abzug** zwei gemessene seitliche
Fortsetzungen und zwei mögliche Endkanten zeigen. Dafür müssen
Bildstrukturen den bisherigen Schnitt durchqueren und die
seitlichen Übergänge sich im Original fortsetzen. Die offenen
Linien helfen dir zu prüfen, wie weit das Papier reicht und ob
ein zweiter Abzug aufliegt. Abzugzahl und gespeicherte Ecken folgen
weiterhin dem bestehenden Eintrag, bis du sie prüfst.
[Beispiel mit zwei Endkanten](data/evaluation/README.md#show-open-extent-guides-on-a-partial-print)

Ein einzelner Vorschlag kann zwei mögliche Abzüge umfassen. Sind
beide Umrisse am Original messbar und setzt sich die Struktur des
Albumträgers durch den Zwischenraum fort, zeigt die Anzahlprüfung
zwei gestrichelte Konturen und die gemessene Spur dazwischen. Prüfe
am Original, ob beide Motive auf einem oder auf zwei Papierträgern
liegen. Für die Anzahl zählt weiterhin der bestehende Eintrag.
[Beispiel mit zwei möglichen Abzügen](data/evaluation/README.md#compare-two-possible-prints-within-one-count-proposal)

Bei sehr schmalen Modellkonturen am Aufnahmerand vergleicht PhoSca
Farbe und Oberflächenstruktur mit der Umgebung ausserhalb der
Albumseite. Sichtbare Papierkanten schützen angeschnittene Abzüge.
Deutet die Messung auf Unterlage hin, zeigt die Anzahlprüfung die
Kontur und die gemessenen Stellen separat. Du kannst einen dort
erkennbaren Papierabzug zur Geometrieprüfung übernehmen oder die
Anzahl ohne den Stoffvorschlag bestätigen.
[Prüfung auf Stoff am Aufnahmerand](data/evaluation/README.md#review-fabric-along-the-capture-frame)

PhoSca gleicht Vorschläge mit den Originalpixeln ab. Es prüft
überlappende Masken, Helligkeits- und Farbverläufe entlang möglicher
Papierkanten sowie Anzeichen für Albumfenster, Negativhüllen, getrennte
Abzüge oder ein gemeinsames Papier mit mehreren Bildfeldern. Das
System kann Teilvorschläge zusammenführen oder bei unklarer Zuordnung
eine zusätzliche Kontur zur Prüfung anzeigen. Die sichtbare Maske und
die vier Eckpunkte für die Entzerrung werden getrennt gespeichert.

Bei hellen Abzügen auf dunklem Scannergrund sucht PhoSca einen am
Aufnahmerand abgeschnittenen Umriss zunächst im verkleinerten Bild.
Es misst die sichtbaren Seiten anschliessend im Original und prüft,
ob ein dünner Streifen Scannergrund quer zwischen der Papierkante
und dem gegenüberliegenden Aufnahmerand verläuft. Ein bestätigter
Streifen sperrt den gemeinsamen Umriss. Wenn zwei Modellvorschläge
an unterschiedlichen Aufnahmerändern enden und den gemeinsamen
Umriss weitgehend abdecken, kann die Anzahl offen bleiben. PhoSca
erhält dann beide Masken und zeigt die gemessenen Papierkanten als
gestrichelte Linien in der Anzahlprüfung. In der Einzelprüfung
zeichnet **Ganzer Abzug** dieselben Kanten ein. Beide Zuschnitte
bleiben in Prüfung, bis die physische Zuordnung und die vollständige
Aufnahme geklärt sind.
[Prüfung am Aufnahmerand](data/evaluation/README.md#review-a-shared-capture-corner-and-preserve-separate-prints)

Mit **Sichtbaren Bereich im Original vergleichen** öffnest du in der
Desktop- und Mobilprüfung einen vergrösserten Ausschnitt der Aufnahme.
Das Fenster erscheint, wenn zwei gemessene Linien an einer sichtbaren
Ecke zusammentreffen, an unterschiedlichen Bildrändern enden und
mindestens zwei dazu verknüpfte Vorschläge aktiv sind. Es zeigt den
Bereich zwischen den gemessenen Linien und den Aufnahmerändern samt
etwas Umgebung. Türkis markiert die zwei gemessenen Linien. Mit
**Vorschläge einblenden** siehst du zusätzlich die Konturen der
verknüpften Vorschläge in Violett und die übrigen aktiven Konturen in
Orange. So kannst du eine Überlappung getrennt von der gemessenen
Papierkante beurteilen. Der Vergleich verwendet die unverzerrte
Originalansicht und dient der Prüfung der Anzahl.
[Messung des Originalvergleichs](data/evaluation/README.md#compare-the-visible-capture-corner-in-the-original)

Bei einem strittigen Mehrbildabzug verknüpft PhoSca den gemessenen
Aussenumriss mit dem vorhandenen Gesamtvorschlag und markiert die
Schnittlinien der Teilvorschläge. **Aussenumriss prüfen** lädt die
gemessenen Ecken als Entwurf für diesen Abzug. Auf dem Desktop wird
der Abzug auf der Seite ausgewählt. Mobile lädt seine Geometrieaufgabe
direkt anhand der Abzug-ID. Beim Öffnen und Verschieben der Ecken
bleibt die gespeicherte Anzahl gleich. Erst Speichern übernimmt die
Ecken. Wenn die Kanten- und Nachbarprüfung es zulässt, zeigt der
Editor unter **Ganzer Abzug** die bisherige Vorschau neben dem
entzerrten **Möglichen Aussenumriss**. Gestrichelte Linien im Vergleich
zeigen die Kanten der Teilvorschläge. Der Vergleich setzt voraus,
dass diese Teile zum gleichen Papier gehören. Auf der Originalseite
lassen sich die Linien auf einen Spalt, eine zweite Papierkante oder
eine überlagerte Schicht prüfen. Die PNG-Vorschau ist ein eigenes
Prüfobjekt. Ein neu gespeicherter Nachbarumriss, der in diese Fläche
reicht, zieht sie zurück. Bei einem geschätzten Seitenverhältnis weist
der Editor darauf hin. Beide Zuschnitte bleiben in Prüfung, bis ihre
jeweiligen Freigabebedingungen erfüllt sind.
[Beispiel Kontaktabzug](data/evaluation/README.md#compare-the-disputed-whole-contact-print)

**Zusammengehörigkeit prüfen** markiert den Gesamtvorschlag und seine
Teilvorschläge auf derselben Originalseite. Prüfe an den gepunkteten
Linien, ob dort eine eigene Papierkante, ein Spalt oder eine aufgelegte
zweite Schicht zu sehen ist. Mit **Trennlinie im Original vergrössern**
siehst du beide Enden und die Mitte jeder vorgeschlagenen Trennlinie.
Die Markierungen stehen seitlich, damit die Stelle dazwischen sichtbar
bleibt. Wenn alle markierten Bereiche zu einem
Papierabzug gehören, lässt **Als einen Abzug zählen** den Gesamtvorschlag
in der Liste und entfernt nur seine Teilvorschläge aus der aktiven
Zählung. PhoSca speichert den Entscheid mit den vorherigen Konturen.
Bei Abbrechen bleiben alle Vorschläge einzeln prüfbar.

Nach der Bestätigung kannst du auf derselben Seite über **Aussenumriss
prüfen** die vier gemessenen Ecken des verbleibenden Abzugs laden.
Wenn du den gemessenen Aussenumriss zuerst speicherst, kannst du die
Zusammengehörigkeit anschliessend bestätigen. PhoSca prüft, dass die
gespeicherte sichtbare Fläche und die vier Ecken weiterhin dem
gemessenen Umriss entsprechen und beide Modi wegen der Überlappung in
Prüfung sind. Bei einer abweichenden Kontur müssen die betroffenen
Vorschläge einzeln geprüft werden. Der Anzahlentscheid bestätigt die
physische Zuordnung. Für **Ganzer Abzug** und **Nur Bildmotiv** prüfst
und speicherst du die jeweilige Geometrie separat.
[Gegenprobe mit einem zweiten Papier](data/evaluation/README.md#confirm-a-shared-paper-grid-in-count-review)

Bei drei verknüpften Vorschlägen für eine mögliche Negativhülle öffnet
**Negativhülle prüfen** eine Auswahl in der Anzahlprüfung. Desktop und
Mobile markieren die drei Konturen türkis auf der Originalseite.
Die gestrichelte Linie zeigt den gemessenen Zwischenraum im Filmraster.
Bei einer klaren Aussenkante sucht PhoSca zusätzlich nach einem
durchsichtigen Saum neben dem hellen Träger. Dafür müssen die Textur
des Untergrunds auf beiden Seiten des Saums zusammenpassen und ihre
Kontraste unter dem Saum schwächer werden. Ein erkannter Nachbar im
Messstreifen oder sichtbarer Untergrund zwischen den beiden Rasterteilen
verhindert diesen zusätzlichen Hinweis. Die gepunktete Linie markiert
den möglichen Saum. Eine gemeinsame Schutzfolie kann ebenso echte
Papierabzüge bedecken. Prüfe daher alle drei markierten Vorschläge am
Original und brich bei Unsicherheit ab. Beim Öffnen der Gruppenprüfung
zeigt PhoSca einen vergrösserten Originalausschnitt an der Rasterteilung.
Bei einem gemeinsam gemessenen Saum kommt ein zweiter Ausschnitt hinzu.
Kurze türkisfarbene Striche zeigen die Stelle der Messung.
Bestätigt der Reviewer, dass alle drei zur Hülle gehören, speichert
PhoSca einen gemeinsamen Entscheid und entfernt die drei Vorschläge
aus der aktiven Abzugsliste. Der Server verlangt die ausdrückliche
Bestätigung, die aktuelle Anzahlrevision und beide Zuschnittmodi aller
drei Vorschläge im Prüfstatus. Hat jemand die Anzahl auf dieser Revision
bereits bestätigt, muss die Anzahlprüfung zuerst wieder geöffnet werden.
Die vorherigen Masken und Quellvorschauen
bleiben in der Geometrieprüfung nachvollziehbar. Die Anzahlprüfung
bleibt auf derselben Seite offen. Für einen daneben liegenden
Papierabzug gelten weiterhin seine eigenen Konturen und Zuschnittregeln.
[Originalausschnitte und Gegenproben](data/evaluation/README.md#inspect-the-original-before-excluding-a-negative-sleeve)

**Gemeinsame Abbildung prüfen** erscheint auf Desktop und Mobile, wenn
genau vier automatische Vorschläge stark überlappen und ein gemeinsamer
Zierrahmen im Original sichtbar ist. PhoSca sucht einen schmalen hellen
Rand mit vier nach innen gewölbten Ecken. Die Helligkeit und die Farbe
müssen sich auf beiden Seiten jeder gestützten Stelle vom Rand
unterscheiden. Eine zweite gut belegte eigene Kante eines Vorschlags
verhindert den Gruppenhinweis. Die gelbe Linie zeigt den gemessenen
Zierrahmen. Prüfe am Original, ob jeder der vier Vorschläge denselben
aufgedruckten Deckelschmuck zeigt. Wenn ein eigener Papierabzug aufliegt
oder du dir unsicher bist, brich ab und prüfe die Konturen einzeln.

**Alle vier Vorschläge aus der Abzugsliste nehmen** speichert einen
gemeinsamen Entscheid für genau die vier aktuellen Vorschläge. Bei
geänderten Konturen, zusätzlichen Vorschlägen, freigegebenen Zuschnitten
oder einer inzwischen bestätigten Anzahl verlangt PhoSca eine neue
Prüfung. Die vorherigen Masken und unbeschnittenen Vorschauen stehen
weiterhin in der Geometrieprüfung zur Verfügung. Prüfe danach die Seite
und bestätige ihre Anzahl separat.
[Beispiel des Spiralalbumdeckels](data/evaluation/README.md#review-four-proposals-on-one-illustrated-cover)

Bei zwei strittigen Vorschlägen kann PhoSca kurze, am Original
gemessene Randabschnitte beidseits der vermuteten Teilung einzeichnen.
**Gemeinsame Papierkante prüfen** markiert beide Vorschläge. Der
Reviewer sieht so, wo der Materialwechsel am Aussenrand liegt, und
prüft am Original, wie viele Gegenstände dort liegen und welche davon
Fotoabzüge sind. Gehören beide Vorschläge zur selben Zeichnung oder zu
einem anderen Gegenstand ohne Fotoabzug, nimmt ein ausdrücklicher
Entscheid beide aus der aktiven Liste. Ein Foto lässt sich nach
Abbrechen im gewöhnlichen Editor korrigieren.
[Beispiel Willett 109](data/evaluation/README.md#review-a-disputed-paper-join)

Wenn der klassische Ersatzpfad einen Abzug aus Linienpaaren findet,
kann PhoSca am Original nach sichtbaren Kanten eines versetzten
Nachbarabzugs suchen. **Möglichen weiteren Abzug prüfen** blendet
die gestützten Abschnitte ein. Die Abzugsliste wächst erst, wenn
der Reviewer im Desktop-Editor einen eigenen Abzug speichert.
Eine beobachtete gewölbte Papierkante bleibt auch nach der Eingabe
von Breite und Höhe in der Geometrieprüfung.
[Beispiel Golden 11](data/evaluation/README.md#trace-the-visible-edges-of-a-missing-neighbour)

Wenn derselbe Ersatzpfad nach einer Neukompression leer bleibt,
kann PhoSca eine Kontur aus gestützten Linien und ihren Enden im
Original anbieten. Dazu müssen auch sichtbare Randabschnitte eines
möglichen Nachbarabzugs vorliegen. Die violette Kontur ist ein
Entwurf in der Anzahlprüfung. Beim Speichern zählt sie als manuell
erfasster Abzug. Beide Zuschnittmodi bleiben in Prüfung, bis die
vier Papierecken im Geometrie-Editor bestätigt sind und zu den
gemessenen Kanten passen.
[Beispiel Golden 11 nach JPEG-Neukompression](data/evaluation/README.md#review-source-edges-after-an-empty-line-pair-search)

Bei gedrehten Aufnahmen kann sich die verkleinerte Liniensuche an
der Albumkante orientieren. Liefert der erste Durchlauf keine
gestützten Spuren, sucht PhoSca im Original nach den Enden des
Materialwechsels an den beiden Querkanten. Eine kurze Verlängerung
in der Nähe des bekannten Abzugs ist nur zulässig, wenn der
Randverlauf im Original durchgehend gestützt ist und auf dem
freien Abschnitt der Wechsel von hellem Albumkarton zu Foto
sichtbar bleibt. PhoSca prüft danach die drei freien Seiten
erneut und zeichnet nur ihre gemessenen Abschnitte ein. Die
Punkte werden in die Koordinaten der hochgeladenen Seite
zurückgerechnet. Über die Abzugsliste und die Zuschnittfreigabe
entscheiden weiterhin die jeweiligen Prüfungen.
[Gedrehte JPEG75-Beispiele](data/evaluation/README.md#recover-source-traces-on-rotated-jpeg-uploads)

Bleibt eine ausreichend belegte Liniensuche ohne Umrisskandidaten,
kann eine zweite Verkleinerung andere zusammenhängende Linien
sichtbar machen. Ein dort gefundener, im Original geprüfter Umriss
dient nur als Bezug für die Kantensuche am Nachbarabzug. In der
Anzahlprüfung erscheinen die dort beobachteten offenen
Kantenabschnitte. Der Bezugsumriss bleibt intern. Das manuelle
Erfassen des Nachbarabzugs erzeugt einen Datensatz und behält eine
gemessene Papierwölbung als Prüfgrund bei.
[Beispiel JPEG55 bei 90°](data/evaluation/README.md#show-only-the-measured-neighbour-edges-at-jpeg55)

Wenn der Zuschnitt an einer Papierkante in Prüfung bleibt und zwei
Linien im Original ausreichend gestützt sind, zeichnet **Ganzer Abzug**
beide als gestrichelte Kanten ein. Die Linien folgen dem beobachteten
Abschnitt. Der Editor bittet den Reviewer, die Papierkante zu bestimmen.
Der gespeicherte Zuschnitt, die Anzahl der erkannten Abzüge und die
Downloadfreigabe bleiben beim Einblenden der Linien gleich. Ein bereits
vorhandener Umriss- oder Kantenhinweis hat Vorrang.
[Kantenvergleich](data/evaluation/README.md#show-competing-edges-in-review)

Bei überlappenden Abzügen kann zusätzlich ein türkis gestrichelter
Abschnitt ausserhalb des bisherigen Zuschnitts erscheinen. PhoSca
sucht dazu im Original nach einem Übergang vom gemessenen Papierton
des Abzugs zum dunkleren Untergrund. Der Hinweis endet vor der Fläche,
die ein Nachbarvorschlag beansprucht. Die violetten Alternativen an
den anderen Kanten bleiben sichtbar. Vergleiche die Markierungen mit
dem Original und prüfe, welcher Abzug oben liegt. Beide Zuschnittmodi
bleiben für diesen Fall in Prüfung.
[Überlappung und sichtbarer Papierübergang](data/evaluation/README.md#show-only-the-exposed-edge-of-an-overlapping-print)

Eine helle Albumkante kann knapp ausserhalb des Abzugs liegen und
über dessen Seitenkante hinaus weiterlaufen. Bleibt der Zuschnitt
in Prüfung, sucht PhoSca im Original nach einem weiter innen
liegenden Übergang vom hellen Karton zur dunkleren Fläche. Der
Übergang muss entlang der Linie sichtbar sein und an beiden
Querkanten enden. **Ganzer Abzug** zeigt den belegten Abschnitt
gestrichelt. Prüfe am Original, ob die Linie die Papierkante oder
eine Motivgrenze markiert, und verfolge den Papierrand rundum.
Die Markierung lässt sich auch bei eingeblendeter aktueller
Zuschnittlinie prüfen. Beide Zuschnittmodi bleiben in Prüfung.
[Beispiel Golden-Seite 26](data/evaluation/README.md#show-an-inward-source-transition-below-an-album-edge)

Bei einer leicht gebogenen Kante kann die gerade Zuschnittlinie
ein Stück sichtbares Foto abschneiden. PhoSca vergleicht dafür
gerade und gekrümmte Verläufe mit dichten Messungen im Original.
Es prüft das Material ausserhalb der sichtbaren Kante über mehrere
Pixel hinweg und berücksichtigt Nachbarabzüge. Ist ein Abschnitt
gut belegt, erscheint unter **Ganzer Abzug** eine zweite, transparente
Prüfvorschau. Sie ergänzt den beobachteten Abschnitt innerhalb einer
kleinen Sicherheitszone an der Kante. Der Vergleich zeigt dieselbe
Perspektive. Der vollständige Papierrand und das Seitenverhältnis
bleiben in Prüfung.

Eine neue überlappende Markierung blendet die zweite Vorschau aus.
Der gemessene Kantenverlauf bleibt für die Prüfung gespeichert.
Eine Bestätigung der bisherigen vier Ecken oder eine minimale
Verschiebung löst die Prüfung einer gebogenen Kante nicht auf.
Bei sichtbarer Wölbung hilft eine flache Neuaufnahme.
[Gemessener gebogener Rand auf Golden-Seite 26](data/evaluation/README.md#recover-an-observed-bowed-edge-in-a-transparent-review-preview)

Sind zwei benachbarte Zuschnittkanten in entgegengesetzte Richtungen
verschoben, misst PhoSca beide Übergänge im Original. Ein sichtbarer
Bildstreifen ausserhalb des bisherigen Zuschnitts kann in der
transparenten Prüfvorschau wieder sichtbar werden. An der anderen
Seite kann die Vorschau einen hellen Streifen teilweise ausblenden,
wenn sich der Untergrund über die Ecke hinaus fortsetzt. Schutzflächen
mit ihrem Sicherheitsabstand bleiben sichtbar. Eine zweite, klar
messbare Aussenkante im hellen Streifen oder ein Nachbarvorschlag im
Messbereich verhindert den transparenten Vergleich.

Die zwei Vorschauen unter **Ganzer Abzug** zeigen den Unterschied mit
derselben Perspektive. Der helle Streifen kann zum Fotopapier gehören.
Prüfe beide Übergänge am Original. Die offenen Markierungen zeigen,
wo die Übergänge messbar sind. Beide Zuschnittmodi warten weiterhin
auf die Prüfung ihrer jeweiligen Grenzen.
[Beispiel Golden-Seite 04](data/evaluation/README.md#compare-two-displaced-edges-on-golden-page-04)

Liegt eine unsichere Zuschnittkante nahe am Aufnahmerand, kann sie
einer Linie zwischen tieferen Albumlagen folgen. PhoSca sucht dann
im Original nach einem Übergang zum Fotomotiv weiter innen und nach
einer separaten Linie, die sich seitlich neben dem Foto fortsetzt.
Helligkeit, Farbwechsel und Schichtabfolge müssen zusammenpassen.
Die transparente Prüfvorschau blendet nur einen gemessenen Abschnitt
hinter dieser Linie aus. Der helle Streifen am Foto und geschützte
Bildbereiche bleiben sichtbar. Vergleiche den ausgeblendeten Streifen
mit dem Original. Ein ähnlich gefärbter Schatten oder eine gedruckte
Linie kann denselben Verlauf zeigen. Beide Modi bleiben in Prüfung.
[Beispiel gedrehte Golden-Seite 04](data/evaluation/README.md#compare-a-continuing-album-layer-near-the-capture-edge)

Endet der bisherige Zuschnitt an einer Seite mitten in dunklen
Bilddetails, kann PhoSca einen kurzen zusätzlichen Streifen zur
Prüfung zeigen. Die Suche beginnt am schwach gestützten Ende einer
Seite. Sie verfolgt den Übergang zum hellen Untergrund im Original
und prüft, ob sich feine Helligkeitsänderungen vom Bild bis zum
Übergang fortsetzen. Eine helle Unterbrechung begrenzt den Streifen.
Ein Nachbarumriss im Messbereich verhindert den Vorschlag.

Unter **Ganzer Abzug** erscheint der zusätzliche Streifen in einer
transparenten Vorschau mit derselben Perspektive. Die Markierung im
Original zeigt den geprüften Abschnitt. Vergleiche seine Struktur mit
dem Foto und prüfe besonders, ob ein Schatten oder ein benachbarter
Abzug die Kante bildet. Der Vergleich ist ein Angebot für die
manuelle Prüfung beider Zuschnittmodi.
[Beispiel mit dunklem Bildverlust am linken Rand](data/evaluation/README.md#compare-a-local-dark-print-edge)

Bei hellen Papierrändern direkt nebeneinander kann der Zuschnitt an
der inneren Bildkante enden. PhoSca prüft zusätzlich, ob sich der
helle Streifen ausserhalb der Schnittkante bis zum benachbarten
Vorschlag fortsetzt. Eine weiter aussen gefundene Linie kann zum
Nachbarabzug gehören, wenn sie nur an einem ihrer Enden gestützt ist.
PhoSca zeichnet in diesem Fall die **fragliche Schnittkante** ins
Original ein. Bei einer Neuberechnung bleibt die Stelle markiert.
Setze die beiden Ecken an den sichtbaren Papierrand. Beim Speichern
muss die Markierung innerhalb des korrigierten Abzugs liegen.
Bei unklarer Zuordnung helfen getrennte Aufnahmen. Bis zur Korrektur
bleiben beide Zuschnittmodi in Prüfung.
[Gedrehter Abzug mit gleichfarbigem Nachbarpapier](data/evaluation/README.md#hold-the-inner-cut-when-pale-stock-continues-into-a-neighbor)

Für einen schwachen Modellvorschlag auf dunklem Albumkarton prüft
PhoSca die Pixel des Originals auf beiden Seiten der vorgeschlagenen
Kontur. Dazu muss mindestens ein Nachbarabzug durch den klassischen
Kantenpfad und das Modell gestützt sein. PhoSca sucht nach ähnlich
dunklem Material beidseits der Kontur und nach einem hellen Rand des
Nachbarabzugs ausserhalb der vorgeschlagenen Fläche. Ein eigener,
durchgehender Materialwechsel kann den Hinweis unterdrücken.

In der **Anzahlprüfung** zählt PhoSca solche Flächen separat als offene
Vorschläge. Die gestrichelten Markierungen zeigen ungefähre Bereiche
entlang des Vorschlags. Türkis markiert ähnlich dunkles Material auf
beiden Seiten, Orange die Nähe zum hellen Rand eines Nachbarabzugs.
Im Hintergrund bleibt die ursprüngliche Maske bei der Prüfung der
benachbarten Zuschnitte berücksichtigt, solange der Vorschlag offen ist.
Ein dunkler Fotoabzug kann dieselben Messwerte liefern.

Die Bestätigung der erkannten Anzahl schliesst die übrigen offenen
Vorschläge, wenn der Reviewer deren Liste geprüft hat. Ist auf einer
Fläche ein eigener Abzug sichtbar, kann der Reviewer ihn öffnen, die
vier Ecken setzen und seine Existenz ausdrücklich bestätigen. Das
zählt den Abzug. Für **Ganzer Abzug** und **Nur Bildmotiv** folgen eine
separate Bestätigung und Messung der Papierkanten im Original sowie
die üblichen Prüfungen von Geometrie, Bildinhalt und Pixelquellen.
[Beispiel Willett 99 und Gegenproben](data/evaluation/README.md#review-a-dark-album-backing-proposal)

Bei der klassischen Kantensuche können Bilddetails zwei scheinbare
Abzüge innerhalb eines bereits erkannten Abzugs bilden. PhoSca prüft
solche Paare an den Originalpixeln. Jede Innenkontur muss zwei
gemessene Aussenkanten des vollständigen Abzugs teilen. An der
fraglichen Seite im Inneren muss eine eigenständige Papierkante
schwach gestützt sein. Eine durchgehende Trennlinie zwischen zwei
Abzügen oder ein belegter Rand eines aufgelegten Fotos lässt die
möglichen Einzelabzüge in der aktiven Liste. Die Suche prüft auch
schräge und leicht gewölbte Trennlinien.

In der **Anzahlprüfung** erscheinen die beiden fraglichen
Innenkonturen einzeln in Violett. Türkis zeigt den Abschnitt, an dem
der Vorschlag im Bild endet. Prüfe am Original, ob dort tatsächlich
ein eigener Papierabzug liegt. Du kannst jede Kontur einzeln öffnen
und nach bewusster Bestätigung mit ihren tatsächlichen Papierecken
erfassen. Sind alle physischen Abzüge bereits erfasst, schliesst die
Bestätigung der Anzahl diese Vorschläge. Beide Zuschnittmodi des
übergeordneten Abzugs brauchen weiterhin die eigene Zuschnittprüfung.
[Gedrehte Golden-Seite 26 und Gegenproben](data/evaluation/README.md#review-two-fallback-contours-inside-a-detected-print)

Liegt eine kleine Kantenkontur fast vollständig im erkannten Abzug,
prüft PhoSca zusätzlich die Papierkanten ausserhalb dieser Kontur.
Die Anzahlprüfung zeigt in Türkis zwei Abschnitte, über die
Motivdetails hinweggehen. Blau zeigt zwei gemessene Papierkanten,
die über die kleine Kontur hinauslaufen. Ein fast gleich
aufgelegter zweiter Abzug kann ähnliche Spuren erzeugen.

Öffne die **Innenkontur** und prüfe, ob der kleine Bereich seinen
eigenen Papierrand hat. Du kannst einen tatsächlich sichtbaren
weiteren Abzug ausdrücklich ergänzen. Bis zum Entscheid bleibt die
ursprüngliche Kontur als Nachbaranspruch bei den Zuschnitten
berücksichtigt. Der Anzahlentscheid speichert die geprüften Konturen
und Messabschnitte. Die beiden Zuschnittmodi haben ihre eigene Prüfung.
[Innenkontur auf Scan 67](data/evaluation/README.md#review-a-small-contour-sharing-two-paper-edges)

Ein schwacher Kontrastvorschlag am Aufnahmerand kann vom erkannten
Foto über den Albumkarton bis auf den Untergrund reichen. PhoSca
sucht im Original nach einem Materialwechsel, der beide seitlichen
Grenzen des Vorschlags kreuzt und ausserhalb davon weiterläuft. Die
Prüfung vergleicht ausserdem das Material beidseits der seitlichen
Grenzen, die sichtbare Albumfläche und die vom Foto übernommene
Innenkante. Ist der Befund ausreichend gestützt, zeigt die
**Anzahlprüfung** den gesamten offenen Vorschlag und markiert den
gemessenen Materialwechsel in Türkis.

Prüfe auch den Bereich zwischen der türkisen Linie und dem
Aufnahmerand. Dort kann ein eigener schmaler Abzug sichtbar sein.
Wenn du Papier erkennst, öffne den Vorschlag und setze die Ecken an
den sichtbaren Abzug. Die Markierung dient als Orientierung bei der
Anzahlprüfung. Die beiden Zuschnittmodi werden separat beurteilt.
[Beispiel mit Albumkante und Untergrund](data/evaluation/README.md#review-a-fallback-outline-crossing-the-album-edge)

Bei einer hellen **Albumtasche mit Prägung** kann das Modell die
Taschenkontur als Abzug vorschlagen. PhoSca untersucht einen schwachen
Modellvorschlag in dieser Lage, wenn ein klarer zweiter Abzug in
derselben Spalte liegt. Die Prüfung sucht zwei Wiederholungsrichtungen
der Struktur im Original. Entlang der beiden Seiten des Vorschlags
muss sich das Muster durchgehend über die Kontur fortsetzen. Wenn
die Prüfung Bildstruktur oder eine deutlich abweichende Farbe
misst, bleibt der Vorschlag in der Abzugsliste.

Der zweifelhafte Vorschlag bleibt als gestrichelte Kontur unter
**Albumtasche prüfen** offen. Türkis zeigt die beiden gemessenen
Abschnitte. Die gleiche Prägung kann über einem sehr blassen Foto
liegen. Ist die Tasche leer und sind alle Abzüge erfasst, bestätige
die Anzahl. Liegt dort ein Papierabzug, bestätige seine Existenz
und setze die Ecken ans sichtbare Papier in der Tasche. Bis zum
Entscheid berücksichtigt PhoSca die offene Kontur bei benachbarten
Zuschnitten. Ein ergänzter Abzug braucht die eigene Papierprüfung
und die Freigabe beider Zuschnittmodi.
[Scan 67 und Gegenproben](data/evaluation/README.md#review-continuous-embossing-across-an-album-pocket)

Umschliesst eine klassische Kontur einen bereits erkannten Abzug
und einen grossen Teil der Albumhülle, prüft PhoSca die beiden
Umrisse separat. Es sucht am kleineren Abzug nach einem Wechsel
zwischen Bild und Prägung und prüft an der grösseren Kontur, wo
die Prägung über deren Rand hinweg weiterläuft. Ein sichtbarer
weiterer Abzug oder ein anderer offener Vorschlag kann gegen diese
Zuordnung sprechen.

Unter **Kontur um Abzug prüfen** erscheint der grössere Vorschlag
mit den gemessenen Abschnitten. Der kleinere Abzug zählt weiterhin
mit. Schau am Original nach, ob um ihn herum ein eigener Abzug
sichtbar ist. Du kannst diesen mit seiner tatsächlichen Papierkontur
ergänzen. Die grössere offene Kontur fliesst bis zum Anzahlentscheid
weiter in die Zuschnittprüfung ein. [Kontur um den Abzug auf Scan 67](data/evaluation/README.md#review-an-embossed-carrier-around-a-detected-print)

Wenn der Abzug selbst in einer geprägten Albumhülle liegt, kann ein
zu weit gefasster Zuschnitt sichtbare Folie enthalten. Bei deutlich
verschobenen Kanten sucht PhoSca im Original nach einer möglichen
Papierkontur. Es prüft an mehreren Seiten, ob sich die gemessene
Prägung unmittelbar ausserhalb der Kante wiederholt und auf der
Innenseite abbricht. Nur die Abschnitte mit nachgewiesenem Übergang
erscheinen in der Zuschnittprüfung.

Auch Folie über einem blassen Bildrand kann an so einem Übergang
enden. Prüfe am Original, wo das Papier endet und ob es bis unter die
Folie reicht. Setze die Ecken an den sichtbaren Papierrand. Bei
unklaren oder verdeckten Ecken hilft eine Aufnahme des frei liegenden
Abzugs. PhoSca zeigt die offenen Messabschnitte neben der bestehenden
Kontur. Für beide Zuschnittmodi gilt weiterhin die eigene Prüfung.
[Messung der Papierkante unter Albumfolie](data/evaluation/README.md#trace-paper-under-an-embossed-carrier)

Berührt ein klassischer Kontrastvorschlag nur eine Aufnahmeecke,
kann auch ein schwach erkannter Nachbar den Ausgangspunkt für die
**Anzahlprüfung** liefern. Dazu müssen im Original mindestens drei
Seiten des erfassten Nachbarn klar gestützt sein. Eine davon muss
mit der einzigen klar gestützten Seite des Vorschlags übereinstimmen.
An einer weiteren Seite muss der Vorschlag innerhalb des Motivs
enden. An der exponierten Seite muss auf beiden Seiten ähnlich
helles Material sichtbar sein. Eine durchgehende schmale Farbspur
kann einen eigenen Papierrand anzeigen und verhindert diesen Hinweis.

Der Vorschlag erscheint als offene violette Kontur. Orange zeigt die
gemeinsame Kante, Türkis den Abschnitt im erfassten Abzug und Blau
den Abschnitt mit ähnlich hellem Material. Prüfe am Original, ob
ein eigener Abzug aufliegt. Wenn du dort Papier erkennst, kannst du
den Vorschlag öffnen, seine Existenz bestätigen und vier Papierecken
setzen. Bis zur Entscheidung berücksichtigt PhoSca die ursprüngliche
Fläche bei den Zuschnittprüfungen der Nachbarn. Die Anzahlbestätigung
entscheidet über die Vorschläge. Für den Export braucht jeder
Zuschnittmodus seine eigene Freigabe.
[Beispiel Golden 04 bei 270° und JPEG75](data/evaluation/README.md#review-a-fallback-contour-sharing-a-weakly-detected-print-edge)

Unter **Schrägen Vorschlag prüfen** zeigt die Anzahlprüfung einen
aktiven Vorschlag mit vier Messstrecken auf der Originalseite.
Orange markiert eine Kante, die mit einem erfassten Nachbarabzug
übereinstimmt, und eine zweite Linie, die ausserhalb des Vorschlags
mit der hellen Fläche auf derselben Seite weiterläuft. Türkis
markiert zwei Abschnitte, an denen Bilddetails quer durch den
vorgeschlagenen Umriss laufen. Auch ein tatsächlich verdeckter
Abzug kann solche Spuren ergeben. Prüfe, ob irgendwo ein eigener
Papierrand oder eigener Bildinhalt sichtbar ist.

Mit **Diesen Vorschlag aus der Abzugsliste nehmen** entfernst du
nur den hervorgehobenen Vorschlag aus der Anzahl. Bis dahin zählt
er mit und seine Fläche fliesst in die Zuschnittprüfung ein.
PhoSca speichert den Entscheid mit der geprüften Kontur und der
aktuellen Anzahlrevision. Die übrigen Abzüge können anschliessend
mit den noch offenen Nachbarmasken neu berechnet werden. Bei einem
eigenen Abzug oder bei Unsicherheit kannst du den Dialog abbrechen.
[Beispiel mit zwei querlaufenden Bildmotiven](data/evaluation/README.md#review-a-diagonal-fallback-contour-on-the-original)

Unter **Mögliche Albumseite** zeigt die Anzahlprüfung einen grossen
klassischen Vorschlag, wenn zwei Falzlinien im Original zu ihm passen
und genau ein plausibler kleinerer Fotoabzug darin liegt. Türkis
markiert den Falz und Violett den kleineren Vorschlag. Die vergrösserte
Originalansicht zeigt die Materialübergänge am Falz. Weitere
konkurrierende Vorschläge im grossen Umriss verhindern den Hinweis.

Die Anzahl ändert sich erst nach **Grossen Vorschlag aus der
Abzugsliste nehmen**. Wähle diese Aktion, wenn der grosse Umriss das
Albumblatt erfasst. Prüfe den kleineren Vorschlag getrennt. Die API
gleicht die angezeigte Anzahlrevision, den grossen Vorschlag und die
ursprüngliche Kontur des kleineren Vorschlags ab. Beide
Zuschnittmodi benötigen weiterhin ihre eigene Geometrieprüfung.
[Beispiel Rudge mit drei Vorschlägen für zwei Abzüge](data/evaluation/README.md#review-a-book-gutter-and-the-contained-print)

Wenn ein klassisch erkannter Umriss eine Ecke diagonal abschneidet,
kann **Ganzer Abzug** eine zweite Vorschau anbieten. PhoSca verlängert
die beiden betroffenen Seiten in einem begrenzten Suchbereich und
prüft im Original den Übergang vom hellen Album zum dunkleren Bild.
Beide Linien müssen bis zur vorgeschlagenen Ecke sichtbar sein und
dort enden. Andere Abzüge und geschützte Bildbereiche werden bei der
Vorschau berücksichtigt. Eine neue manuell erfasste Fläche, die die
ergänzte Ecke beansprucht, entfernt den Vorschlag. Strittige Kanten
an anderer Stelle bleiben im Editor sichtbar.

Der Reviewer prüft am Original, ob der ergänzte Bereich zum selben
Abzug gehört. **Aussenumriss in den Editor laden** übernimmt die
Punkte als Entwurf. Erst eine ausdrücklich bestätigte Papierkorrektur
speichert den Umriss. Das Seitenverhältnis und die Eignung für beide
Exporte werden separat geprüft.
[Beispiel Golden 05](data/evaluation/README.md#offer-a-reviewed-preview-for-the-inset-corner-on-golden-05)

In der **Anzahlprüfung** kann PhoSca einen gemeinsamen Papierumriss
für zwei klassische Teilvorschläge anzeigen. Ein Teilvorschlag muss
drei am Original belegte Papierkanten und eine schwache Innenkante
haben. Der zweite muss bis zum Aufnahmerand reichen. Kann PhoSca in
seinem sichtbaren Bereich eine eigene Papierkante messen, unterbleibt
der gemeinsame Vorschlag. Die Suche bleibt auf die Umgebung der
beiden Flächen begrenzt. Der Editor kennzeichnet die schwächere Seite
des gemeinsamen Umrisses mit kürzeren Strichen.

**Möglichen gemeinsamen Abzug prüfen** öffnet die vier Ecken am
Original. Die beiden bisherigen Flächen bleiben beim Prüfen gezählt.
Mit der Bestätigung ersetzt PhoSca genau diese beiden Flächen in
einem Schritt durch einen neuen Abzug. Beide Zuschnittmodi des neuen
Abzugs behalten die Prüfung der Papierkante. Eine neue überlappende
Markierung, eine veränderte Teilfläche oder deren Ablehnung macht
einen offenen Vorschlag ungültig.
[Beispiel Golden 05 unten rechts](data/evaluation/README.md#review-the-lower-right-fragments-together)

Bei zwei eng benachbarten Abzügen mit unterscheidbarem Papier
kann der Editor einen zusätzlichen Aussenumriss anbieten. PhoSca
schätzt den Papierton für jeden Abzug aus seinen gestützten Seiten.
Es prüft die Übergänge an beiden Umrissen im Original und verlangt
auch an den Ecken belegte Materialwechsel. Die bisherigen Ecken
dürfen höchstens 0,75 Originalpixel ausserhalb des Entwurfs liegen.
Der Vorschlag darf keine Fläche eines anderen Vorschlags
beanspruchen. Mit **Aussenumriss in den Editor
laden** lässt sich der zusätzliche Streifen direkt am Original
prüfen. Erst die ausdrückliche Bestätigung speichert die Korrektur.
Beide Abzüge behalten ihre eigene Anzahl und ihren eigenen Prüfstatus.
[Beispiel mit eng anliegenden Abzügen](data/evaluation/README.md#offer-editable-edges-for-distinct-adjacent-paper-stocks)

PhoSca kann unter **Nur Bildmotiv** schon eine Prüfvorschau zeigen,
während die Papiergrenze zum Nachbarabzug noch offen ist. Dafür
müssen alle vier Motivkanten im Original durchgehend und bis an die
Ecken messbar sein. Das gesamte Motiv muss innerhalb des bisherigen
Abzugs liegen. Ansprüche anderer Abzüge sowie geschützte Stellen
und Markierungen am Rand können die Vorschau verhindern. Vergleiche
die Vorschau mit dem Original und bestätige zuerst den Papierumriss.
[Messung beider Motive](data/evaluation/README.md#preview-an-inboard-motif-beside-a-disputed-paper-edge)

Sind Papiergrenze und Seitenverhältnis unklar, kann **Nur Bildmotiv**
drei gemessene Motivkanten als gestrichelte Linien anzeigen. Dazu muss
PhoSca bei zwei Suchtiefen dieselben hellen Ränder finden und alle drei
Übergänge im Original bestätigen. Berührt ein benachbarter Abzug den
Prüfbereich einer Linie, blendet PhoSca alle drei Linien aus. Das gilt
auch nach späteren Änderungen am Nachbarabzug. Die ungemessene vierte
Seite bleibt offen. Prüfe auch dort und ausserhalb der Linien, ob
Bildinhalt oder zugehörige Beschriftung liegt. Am Handy erscheinen die
Linien in der Geometrieprüfung. Im Desktop-Editor erscheinen sie unter
**Nur Bildmotiv**. Erst deine geprüften Ecken definieren den Zuschnitt.
[Weihnachtskartenseite und Messung](data/evaluation/README.md#show-three-open-motif-guides-on-the-souvenir-card)

Ist eine Seite neben hellem Untergrund nur abschnittsweise sichtbar,
kann **Ganzer Abzug** an beiden Enden gestrichelte Hinweise zeigen.
PhoSca prüft dazu die sichtbaren Materialwechsel, das Ende der
angrenzenden Kanten und die Fläche zum nächsten Abzug. Die unklare
Stelle bleibt zwischen den Linien offen. Die Ecken im Editor sind
weiterhin die bisher gespeicherten Ecken, bis sie bewusst geändert werden.
[Beispiel unscharfer Scan](data/evaluation/README.md#show-the-visible-spans-of-an-interrupted-paper-edge)

Ragt eine Ecke weit über den Abzug hinaus, prüft PhoSca, wo die
beiden angrenzenden Papierkanten im Original enden. Eine gestützte
innere Kante kann in der Geometrieprüfung eine zweite Vorschau geben.
Daneben bleibt der bisherige Zuschnitt sichtbar. Die gestrichelten
Abschnitte helfen bei der Entscheidung, ob der Streifen zum Papier
gehört. Schutzwarnungen gelten weiter und die Galerie zeigt die
bisherige Vorschau, solange die Geometrie in Prüfung ist.
[Beispiel Golden 33](data/evaluation/README.md#compare-an-overextended-corner-with-the-measured-edge)

Die gleiche Gegenüberstellung hilft, wenn eine schwach gemessene
Zuschnittseite schräg in die Albumseite hineinragt. PhoSca sucht am
Original einen Übergang, der an beiden Ecken endet und zu den
benachbarten Seiten passt. Es prüft den hellen Streifen dahinter auf
einen weiteren durchgehenden Rand und sichtbare Markierungen. Der
zusätzliche Zuschnitt steht als Prüfvorschau neben dem bisherigen Bild.
Prüfe am Original, ob der Streifen zum Papierabzug gehört.
[Beispiel Golden 05 mit Modell](data/evaluation/README.md#compare-the-lower-paper-edge-on-golden-05-with-the-model-enabled)

Bei genau einer strittigen automatischen Gesicht- oder Kopfmarkierung
kann der Editor die Stelle am Original vergrössert zeigen. Gehört sie
zur Albumseite, kann der Reviewer das beim Speichern der vier Ecken
bestätigen. Die Markierung wird nur für diesen Abzug und diese Ecken
aus der Prüfung genommen. Weitere geschützte Stellen bleiben geprüft.
Ändern sich die Originaldatei oder ihre Schutzbereiche, wird die
Freigabe erneut geprüft. Das Seitenverhältnis wird separat bestimmt.
[Beispiel Schutzmarkierung](data/evaluation/README.md#review-one-album-page-mark-during-a-corner-edit)

Liegt das Zentrum einer automatischen Schutzmarkierung bereits jenseits
einer messbaren Papierkante, kann der Editor bis zu vier solcher Stellen
getrennt am Original vergrössern. Jede Markierung muss überwiegend
ausserhalb des Papiers liegen. PhoSca verlangt einen durchgehend
messbaren Übergang am selben Rand. Sobald ein anderer aktiver oder
noch nicht geklärter Abzug die Stelle oder ihren Schutzabstand
beansprucht, bleibt der Schutz bestehen. Nach der Anzahlprüfung lässt
sich der Abzug auf Desktop oder Mobil neu berechnen.

Der Reviewer beurteilt jede Stelle einzeln. Erst wenn alle sichtbaren
Stellen ausdrücklich dem Hintergrund zugeordnet sind, kann PhoSca die
zugehörigen automatischen Markierungen beim Speichern für diesen einen
Abzug ausnehmen. Die vier Ecken bleiben auf der gemessenen Kontur.
Die Bestätigung gilt für diese Aufnahme, die gespeicherten Ecken und
die aktuelle Schutz- und Anzahlprüfung. Ein neuer überlappender
Anspruch hält die betroffenen Downloads erneut zurück. Ein ungeklärtes
Papierformat und eine strittige Motivgrenze bleiben eigene Prüfungen.
[Beispiel mit zwei Markierungen](data/evaluation/README.md#review-multiple-advisory-marks-at-a-measured-paper-edge)

Beim manuellen Ergänzen eines Abzugs prüft PhoSca einen einzelnen
automatischen Gesicht- oder Kopftreffer, dessen erkannter Bereich
vollständig jenseits der Papierkante liegt. Ragt nur sein
Sicherheitsabstand in den vorgeschlagenen Zuschnitt, sucht PhoSca im
Original nach zwei gegenüberliegenden Kanten und der dazwischen
durchgehenden Prägung. Die vier Ecken lassen sich zur Prüfung speichern.
Das gelbe Rechteck zeigt den erkannten Bereich und die türkise Linie
den gemessenen Kantenabschnitt. Weichen die Ecken dort um mehr als
zwei native Pixel ab, müssen sie vor der Bestätigung korrigiert werden.

Der Reviewer bestätigt für die gespeicherten Ecken, dass die gesamte
gelbe Markierung ausserhalb dieses Abzugs liegt. PhoSca prüft dabei
die Originalpixel, die Markierung und die Ansprüche der Nachbarabzüge
erneut. Für diesen Zuschnitt entfällt nur der zusätzliche Abstand.
Der erkannte Bereich bleibt geschützt. Die Entscheidung ist an die
Aufnahme, die Ecken und den Stand der Schutz- und Anzahlprüfung gebunden.
«Ganzer Abzug» und «Nur Bildmotiv» durchlaufen weiter ihre jeweiligen
Geometrie- und Inhaltsprüfungen.
[Beispiel U008](data/evaluation/README.md#review-a-protection-margin-outside-a-manually-added-print)

Bei einem manuell ergänzten Abzug kann PhoSca einzelne gerundete
Eckkonturen am Original zeigen. Es vergleicht die Farbwechsel am
möglichen Papierrand mit der Farbe und dem Helligkeitsmuster der
sichtbaren Albumprägung daneben. Wo beide Hinweise zusammenpassen,
blendet die zweite Prüfvorschau den gemessenen Eckbereich transparent
aus. Unklare Bereiche bleiben darin sichtbar. Die bisherige Vorschau
und die gespeicherten Ecken stehen weiter zum Vergleich bereit.

Die Eckprüfung betrifft beide Zuschnittmodi. Sie bleibt bei späteren
Änderungen der Ecken, eingetragenen Papiermassen und einer erneuten
Verarbeitung bestehen. Bei einer konkurrierenden Papierkontur oder
einer veralteten Schutzbestätigung zieht PhoSca die betroffene
Vergleichsvorschau zurück. Prüfe auch alle geraden Seiten am Original.
[Beispiel mit drei gemessenen Ecken](data/evaluation/README.md#compare-supported-corner-shapes-on-a-manually-added-print)

Bei einem Abzug mit gemessener Eckform kann PhoSca zusätzlich einen
kurzen Abschnitt knapp ausserhalb einer geraden Seite markieren.
Es vergleicht dort den Farbwechsel mit dem sichtbaren Albumkarton und
der Farbe weiter innen auf dem Foto. Die offene Linie kennzeichnet den
Abschnitt, den du am Original prüfen kannst. Eine vergrösserte Ansicht
zeigt den markierten Bereich mit seiner Umgebung. Die vier Ecken, die
Eckenvorschau und beide Export-Sperren gelten weiter. Ändert sich ein
Nachbarabzug so, dass die Messung oder eine Schutzbestätigung ihre
Grundlage verliert, zieht PhoSca die betroffenen Hinweise zurück.
[Beispiel an der Oberkante von U008](data/evaluation/README.md#inspect-a-local-edge-section-outside-a-manual-crop)

Bei einem zur Prüfung zurückgehaltenen Abzug kann PhoSca einen
weiteren sichtbaren Bildstreifen ausserhalb des bisherigen Umrisses
türkis markieren. Es sucht von einer gut belegten Seite aus, prüft
den hellen Zwischenraum, den Übergang zum strukturierten Streifen und
die dunkle Fläche dahinter. Zwei seitliche Kontraste stützen die
Lage. Die Markierung zeigt nur den beobachteten Streifen am Original.
Prüfe dort, ob er zum selben Abzug gehört. Falls Albumkarton
Bildinhalt verdeckt, braucht es eine unverdeckte Aufnahme.
Beansprucht ein anderer Abzug den Streifen, entfällt der Hinweis.
[Beispiel mit verdecktem Rudge-Abzug](data/evaluation/README.md#show-a-lower-exposed-strip-next-to-a-held-print)

Bei einer **möglichen Albumöffnung** zeigt die Geometrieprüfung zwei
violette gestrichelte Linien im Original. Der Hinweis kann erscheinen,
wenn beide Zuschnittmodi in Prüfung liegen, zwei Seiten des bisherigen
Vorschlags gut gestützt sind und PhoSca an den gegenüberliegenden
Seiten eine gebogene Kante und einen weiter innen liegenden geraden
Übergang misst. Vergleiche an diesen Linien, wo Fotopapier endet und
Albumkarton beginnt. Ist ein Teil des Bildes verdeckt, braucht es eine
Aufnahme mit freiem Blick auf den Abzug. Die Linien sind Orientierung
für die Prüfung. Die vier Griffe zeigen die aktuelle Zuschnittgeometrie.
[Beispiel und Gegenproben](data/evaluation/README.md#show-possible-album-opening-edges-in-review)

Vor dem Speichern einer Korrektur prüft PhoSca geschützte Stellen
in den Originalkoordinaten. Für **Ganzer Abzug** zählt der sichtbare
Teil der bisherigen Abzugsfläche. Für einen ausdrücklich gesetzten
Ausschnitt unter **Nur Bildmotiv** zählt der ganze Abzug als Grundlage.
Würde der neue Ausschnitt geschützte Bildfläche verlieren, markiert
der Editor die Stelle rot. Die zuletzt gespeicherten Ecken und Downloads
stehen weiter zur Verfügung. Sie werden erst durch eine gültige
Korrektur ersetzt.
[Beispiel Golden 10](data/evaluation/README.md#protect-visible-source-content-in-ordinary-edits)

Bei einer manuellen Korrektur vergleicht PhoSca die Fläche zwischen
den vier Ecken mit den aktiven Nachbarabzügen. Eine deutliche Überdeckung
hält beide Zuschnittmodi zurück. Die violette Kontur zeigt den
konkurrierenden Nachbarvorschlag im Original. Gehört beim Ausweichen
ein ausgelassener Streifen laut bisheriger sichtbarer Kontur zum
Abzug, markiert der Editor auch diesen möglichen Inhaltsverlust.
Unter «Markierte Stellen im Original ansehen» lässt sich die
markierte Fläche mit ihrem Umfeld prüfen. Eine ähnlich gefärbte
Albumseite liefert einen Hinweis für die Suche nach feinen Schatten
und Papierkanten. Die geometrische Prüfung entscheidet weiter über
die Downloads.
Bei tatsächlich verdeckten Papierkanten hilft eine zweite Aufnahme
mit getrennten Abzügen.
[Beispiel Golden 05](data/evaluation/README.md#check-ordinary-corner-edits-at-neighboring-prints)

Ein grosser Vorschlag, der wegen mehrerer innerer Konturen als
Albumfläche verworfen wurde, kann in der Anzahlprüfung wieder zur
Auswahl stehen. Dafür prüft PhoSca drei Aussenkanten am dunklen
Albumuntergrund und Motivstrukturen, die über die geschätzten
Innenkanten hinweg verlaufen. Der Editor zeigt die ursprüngliche
Kontur und lädt auf Wunsch einen separaten Entwurf mit vier Ecken.
Bei einem möglichen Ganzabzug markiert der Desktop-Editor die
zugeordneten Teilvorschläge violett. Gehören alle zu demselben Papier,
kann der Reviewer sie zusammen mit den vier geprüften Ecken als
einen Abzug speichern. PhoSca prüft, ob die markierten Vorschläge
noch aktuell sind und innerhalb der neuen Kontur liegen. Die
ursprünglichen Vorschläge bleiben in der Prüfungshistorie.
Anschliessend prüft PhoSca den Zuschnitt separat. Beim Speichern
der Ecken vergleicht es alle vier Papierseiten mit dem Original.
Verdeckte oder schwach belegte Papierkanten halten beide Downloads
in Prüfung.
[Beispiel Willett-Album](data/evaluation/README.md#compare-a-large-mounted-print-with-album-container-proposals)

Die ONNX-Koordinaten werden nach dem Entfernen des Letterbox-Paddings
auf das Quellbild zurückgerechnet. Eine auf eine Linie kollabierte
Vorhersage kann verworfen werden, während die übrigen Vorschläge einer
Inferenz erhalten bleiben. Das Modellregister prüft die Gewichtsdatei
gegen ihren SHA-256. Die Standardschwellen
`YOLO_PROPOSAL_THRESHOLD=0.25` und `YOLO_CONFIDENCE_THRESHOLD=0.70`
haben verschiedene Aufgaben. Die erste begrenzt die Modellvorschläge.
Die zweite beeinflusst, ob ein Vorschlag mit seiner verfügbaren
Konfidenz zur Prüfung muss. Ein hoher Score allein klärt weder die
Pixelzugehörigkeit zwischen Nachbarn noch die physische Papierkante.
[Verarbeitungsdienst](backend/app/services/pipeline.py) ·
[Erkennung und Zusammenführung](backend/app/pipeline/segmentation.py) ·
[Konfiguration](backend/app/config.py)

### 3. Zwei Zuschnitte aus denselben Originalpixeln bilden

| Modus | Ergebnis |
| --- | --- |
| **Ganzer Abzug** | Der entzerrte physische Abzug bis zur Papierkante. Sichtbarer Bildinhalt und der echte Papierrand bleiben erhalten. |
| **Nur Bildmotiv** | Der entzerrte Bildbereich innerhalb der Motivgrenze. Bei einem verifizierten randlosen Abzug können beide Modi dasselbe Bild verwenden. |

| Geometriefeld | Bedeutung |
| --- | --- |
| `bounding_box` | Achsenparalleler Such- und Anzeigeausschnitt. |
| `visible_polygon` | Schätzung der sichtbaren Abzugsfläche, gegebenenfalls mit mehr als vier Punkten. Grundlage für Instanz- und Überlappungsprüfung. |
| `warp_quad` | Vier Eckpunkte der verwendeten Projektion. Sichtbare Kontur und Projektionsviereck können voneinander abweichen. |
| `image_quad` | Vier ausdrücklich gesetzte Motivecken im selben Seitenkoordinatensystem. Beim Speichern müssen sie innerhalb des Papier-Vierecks liegen. |
| `print_size_mm` | Gemessene physische Breite und Höhe. Die Reihenfolge der Ecken bestimmt, welche Kante zu welcher Dimension gehört. |
| `page_to_output`, `shape` | 3×3-Homographie von Quellpixelzentren zu Ausgabepixelzentren und Rastergrösse pro Modus. Die Homographie enthält die gewählte Ausgabeausrichtung. |

Die präzisen Polygonkoordinaten verwenden `(0, 0)` bis `(1, 1)`
für die äussersten Pixelzentren. Ein Punkt wird im Raster mit
`x · (Breite − 1)` und `y · (Höhe − 1)` abgebildet. Das Bezugsbild
ist das dekodierte, EXIF-ausgerichtete Quellraster. Bei zur Aufnahme
passender geometrischer Kamerakalibrierung kann es vor der Projektion
entzerrt werden.

Für **Ganzer Abzug** verfeinert PhoSca die vermuteten Papierecken
mit Messungen am Original. Eine perspektivische Projektion bildet
das Viereck auf ein Rechteck ab. Für das Seitenverhältnis nutzt
PhoSca nach Möglichkeit eingegebene Papiermasse, eine zur Aufnahme
passende Kamerakalibrierung oder Hinweise auf die Papierebene.
Unzureichende Messpunkte, Wölbung, Verdeckung oder widersprüchliche
Kanten können eine Prüfung auslösen.

Gemessene Papiermasse haben Vorrang. Bei vorhandener Kamera-Matrix
liefert eine tragfähige Schätzung der Papierebene oder eine orthogonale
Rekonstruktion das Verhältnis. Ohne diese Daten verwendet PhoSca
nahezu frontale Seitenlängen oder prüft, ob die perspektivische
Schätzung bei leicht veränderten Eingaben stabil bleibt. Eine
schräg gesehene Kante allein kann bei unbekannter Kamera mehrere
physische Seitenverhältnisse erlauben. In diesem Fall ist
`PRINT_ASPECT_UNCERTAIN` mit Papiermassen oder einer besseren Aufnahme
auflösbar. Aussagen zur Kamerakalibrierung sind an den SHA-256 der
Aufnahme gebunden. Solche Ergebnisse einer automatischen Erkennung
benötigen weiterhin Prüfung.

Wenn ein manuell gespeicherter Zuschnitt bei **Ganzer Abzug** nur noch
wegen des Seitenverhältnisses in Prüfung ist, zeigt PhoSca **Gespeicherter
Zuschnitt zur Prüfung**. Die Vorschau zeigt die gespeicherten Bildpixel
mit dem geschätzten Seitenverhältnis. Miss Breite und Höhe am physischen
Abzug und speichere beide Masse. Während du Ecken, Masse oder Drehung
bearbeitest, blendet der Editor die gespeicherte Vorschau aus. Die
Exportentscheidung erfolgt weiterhin für beide Modi separat.
[Gespeicherte Prüfvorschau](data/evaluation/README.md#show-the-saved-raster-while-the-physical-aspect-remains-open)

Bei einem gezackten Abzug kann PhoSca eine gemessene Aussenkontur
und vier zugehörige Ecken im Editor anzeigen. Ein überlappender
Teilvorschlag bleibt als eigener Prüfpunkt erhalten. Solange
beide Vorschläge dieselbe Fläche beanspruchen, bleiben beide
benannten Exporte gesperrt. Die Sperre gilt auch nach dem Speichern
der Ecken. Nach Klärung der Überlagerung kann PhoSca den ganzen Abzug
entlang seiner gemessenen Kontur mit Transparenz zur Prüfung
darstellen. Ein bewusst gewähltes Motiv im Innern kann separat
exportiert werden, sofern sein Rand innerhalb des gemessenen Papiers
und mit Abstand zu anderen Abzügen liegt.
[Überlappung und gezackte Papierkante](data/evaluation/README.md#show-scalloped-paper-in-ownership-review)

Für **Nur Bildmotiv** untersucht PhoSca zusätzlich die Übergänge
zwischen Papier und Bild. Der Renderer berechnet beide Ausgaben
direkt aus der Quellseite. Beim präzisen Sampling gehen in die
Interpolation nur Quellpixel ein, die zur jeweiligen Fläche gehören.

Liegt für einen klassischen Kontrastvorschlag ein grösserer Aussenumriss
vor, kann die App den bisherigen Zuschnitt und den neu entzerrten
Vergleich anzeigen. Das gilt auch, wenn kein Modell verfügbar ist.
PhoSca rendert den Vergleich nur dann, wenn beide Zuschnittmodi in
Prüfung stehen und die bisher enthaltenen geschützten Flächen
erhalten bleiben. Mehr als ein Quadratpixel Überschneidung im
Original mit einem aktiven oder offenen Vorschlag sperrt den
Vergleich. Gespeicherte Ecken, Anzahl und Exporte
ändern sich durch die Anzeige nicht. Legt die Person in der Prüfung
eine überlappende Nachbarkontur an, zieht PhoSca die Vergleichsvorschau
zurück. [Vergleich am Kontaktbogen](data/evaluation/README.md#render-the-measured-paper-comparison-in-classical-fallback)

Bei einem Kontaktbogen mit strittigen Ecken kann PhoSca im Original
vier getrennte Abschnitte eines grösseren Rands markieren. Es vergleicht
drei Farbschwellen, misst an jeder Kante den Übergang zwischen zwei
Farbbereichen und prüft den hellen Saum in der Originalauflösung. Ein
regelmässiges Raster mit dunkleren Lochreihen und helleren Bildfeldern
stützt den Hinweis. Die gemessenen Abschnitte erscheinen in Türkis neben
den vorhandenen violetten Kantenhinweisen. Das Original zeigt, ob der
Rand zu einem Papierabzug gehört oder beispielsweise Teile aufliegen.
Ein Vorschlag, der mehr als ein Quadratpixel des untersuchten Bereichs
beansprucht, blendet die türkisen Linien aus. Vor dem Speichern muss die
Person in der Prüfung die vier Ecken am sichtbaren Rand desselben Abzugs
bestätigen. PhoSca prüft diese Ecken erneut an der Originalaufnahme.
Die Bestätigungspflicht gilt auch nach erneuter Verarbeitung, wenn die
türkisen Linien durch einen Nachbarabzug ausgeblendet wurden.
[Gemessene Kanten am gedrehten Kontaktbogen](data/evaluation/README.md#locate-paper-edges-around-a-held-contact-grid)

Eine Negativhülle auf derselben Aufnahme kann mehrere Vorschläge
erzeugen. Bei zwei schmalen Modellmasken, die nur am Seitenverhältnis
scheitern, sucht PhoSca nach einem gemeinsamen Kontrastumriss. Es misst
dessen vier Seiten in den Originalpixeln nach, prüft den Streifen
zwischen den Masken auf ein regelmässiges Negativraster und sucht
ausserhalb des Umrisses nach einem lichtdurchlässigen Saum. Sichtbarer
Untergrund an der Trennstelle oder ein weiterer Vorschlag, der den
gemessenen Bereich beansprucht, sperrt den Gruppenhinweis. Die App
zeigt den Trennstreifen und den Saum als zwei offene Linien. Alle
Vorschläge behalten ihre Konturen und bleiben einzeln sichtbar, bis die
Person die Hülle als zusammengehörigen Gegenstand bestätigt. Ein
benachbarter Papierabzug behält seine eigene Prüfung in beiden
Zuschnittmodi.
[Negativhülle auf der gedrehten Aufnahme](data/evaluation/README.md#review-partial-film-groups-on-the-rotated-contact-page)

Scheitert bei einer solchen Hülle genau eine der vier Farbmessungen,
setzt PhoSca für diese Seite drei zusätzliche Messungen ein, jeweils
um ein, zwei und drei Originalpixel nach aussen versetzt. Die drei
Messungen müssen dieselbe Kante ergeben. An beiden Enden dürfen ihre
Ergebnisse höchstens 0,60 Pixel auseinanderliegen und maximal
2,50 Pixel weiter aussen liegen. Eine Messung auf der Innenseite
sperrt den Hinweis. Wenn eine weitere Kontur die ursprüngliche
Hüllenfläche beansprucht, bleibt diese Nachmessung gesperrt. PhoSca
prüft die rekonstruierte Kontur erneut auf die Lage der Filmstreifen
und auf konkurrierende Flächen, bevor es einen Gruppenhinweis anbietet.
[Prüfung der verlustfreien Aufnahme](data/evaluation/README.md#recheck-one-uncertain-sleeve-side-on-the-lossless-contact-page)

Beim Papierabzug kann auch ein Modellvorschlag diese vier
Kantenhinweise erhalten. PhoSca prüft dafür einen gemeinsamen
Modell- und Kontrastvorschlag mit unklarem Papierrand und
Seitenverhältnis mit derselben Messung am Original. Solange ein
anderer Vorschlag den gemessenen Umriss überlappt, bleiben die
Linien ausgeblendet. Nach der Stückzahlprüfung lässt sich der
Abzug auf Desktop und Mobil über **Papierkanten erneut prüfen**
neu messen. Die vier türkisen Abschnitte helfen dabei, die Ecken
des Papierabzugs zu setzen. Eine neue überlappende Kontur blendet
sie wieder aus. Zum Speichern braucht es die ausdrückliche
Bestätigung der vier Ecken am selben Abzug und die anschliessende
Prüfung des manuell gesetzten Umrisses in den Originalpixeln.
Bleibt eine Kante mehrdeutig, bleiben beide Zuschnittmodi in Prüfung.
[Kantenhinweise für den verlustfreien Papierabzug](data/evaluation/README.md#show-native-paper-edges-after-the-lossless-sleeve-review)

Erkennt die native Prüfung nach dem Speichern zwei konkurrierende
Linien, kann PhoSca den Übergang auf der Seite der zuvor geprüften
Filmhülle orange hervorheben. Dafür müssen alle drei Hüllenvorschläge
im selben Prüfschritt als Filmhülle bestätigt worden sein. PhoSca
vermisst die Hülle nochmals am Original. Es prüft, ob zwischen den
beiden Linien ein durchgehender Untergrundstreifen sichtbar ist und
ob weitere Vorschläge den Bereich beanspruchen. Violett bleibt die
Linie am Papier. Der Hinweis hilft beim Vergleich und beide
Zuschnittmodi bleiben in Prüfung.
[Vergleich der Papier- und Hüllenkante](data/evaluation/README.md#distinguish-the-previously-reviewed-sleeve-edge)

Liegt eine schwach gestützte gespeicherte Kante im Albumkarton,
kann PhoSca einen weiter innen sichtbaren Übergang als offene Linie
markieren. Es vergleicht dafür zwei Linien und prüft am Original,
welche zum hellen Untergrund überleitet. Eine freie Ecke muss an
denselben inneren Übergang anschliessen. Beansprucht ein anderer
Abzug oder ein offener Vorschlag die Messstelle, erscheint der
Hinweis nur, wenn sich der Übergang an einer anderen freien Ecke
abstützen lässt. Die Linie zeigt, wo die Person in der Prüfung den
Papierrand oder den Rand des Motivs vergleichen kann. Ein
geschützter Bildbereich im fraglichen Streifen sperrt den Hinweis.
Beide gespeicherten Zuschnittmodi bleiben in Prüfung.
[Dunkler Abzug und Albumkarton](data/evaluation/README.md#show-an-open-inner-edge-with-one-clear-corner)

Bleibt genau eine Papierkante strittig, kann PhoSca ihren Verlauf
anhand der anderen drei Seiten und der angrenzenden Ecken für
die Prüfung genauer messen. Vier im Original gestützte Motivkanten,
ein durchgehender Papiersaum und Abstand zu Nachbarabzügen erlauben
in diesem Fall eine separate Motivvorschau. Zeigt der abzutrennende
Papierstreifen eine mögliche Beschriftung oder eine andere kompakte
Abweichung zwischen ähnlich gefärbten Papierflanken, markiert der
Editor die Stelle auf der Originalseite und hält die enge Vorschau
zurück. Auf Desktop und Mobil lässt sich die vorhandene Vorschau
ansehen. Die Exporte beider Modi erfordern weiterhin eine Prüfung.
Der Aussenumriss wird zuerst am Original bestätigt.
Eine markierte Randstelle bleibt beim Speichern des Aussenumrisses
in der Motivprüfung, bis ein Motivzuschnitt ausdrücklich gewählt wird.
[Prüfvergleich für Motiv und Rand](data/evaluation/README.md#preview-a-motif-with-one-disputed-paper-edge)

Die Prüfung kann auch vor der Bestätigung des Papierrands auf mögliche
Beschriftung hinweisen. Das gilt für automatische Vorschläge mit unklarer
Papiergrenze, unbelegtem Seitenverhältnis und möglichem Inhaltsverlust.
PhoSca verlangt vier messbare Übergänge zwischen Papier und Motiv sowie
eine gekrümmte Motivkante. Die Übergänge grenzen den zu untersuchenden
Papierstreifen ein. Die Randprüfung vergleicht ihn in der bestehenden
entzerrten Auflösung und mit 800 Pixeln auf der langen Seite. Beide
Messungen müssen sich im Original bestätigen und dort mindestens 80 %
Flächenüberdeckung erreichen. Der Fund muss innerhalb der sichtbaren
Papierfläche liegen, mit Abstand zum Aussenrand und ausserhalb aller
anderen Abzüge. Eine neu gespeicherte Nachbarkontur, die den Fund
beansprucht, blendet die Markierung wieder aus. Der Hinweis erscheint
auf Desktop und Mobil schon bei **Ganzer Abzug**. Er beschreibt eine
Fundstelle und zeigt, welche Grenze zuerst geprüft werden muss. Für
beide Exporte gelten weiter die bestehenden Geometrieprüfungen.
[Beispiel am nativen Kontaktabzug](data/evaluation/README.md#locate-native-contact-print-lettering-while-the-paper-is-held)

Die Randprüfung betrachtet nur Papierstreifen zwischen Aussenkontur
und vorgeschlagener Motivgrenze. Für deutliche Markierungen verwendet
sie zusammenhängende Komponenten aus der Helligkeitsdifferenz zum
lokalen Papier. Der erste Test sucht Reihen mehrerer Komponenten.
Wenn er keine findet, sucht ein zweiter Test einzelne Zeichen oder
verbundene Schrift. Er verlangt einen vom Motiv getrennten
Papiersaum, annähernd gleichmässige Strichdicke und eine definierte
Form aus geschlossenen Innenräumen, einem Stamm oder einer Spitze
mit zwei Schenkeln. Für beide Tests wird der Strich in die
Aufnahme zurückprojiziert. Mindestens 70 % der abgetasteten
Strichpixel müssen dort den erforderlichen Kontrast zum Papier
erreichen.

Für schwache Schrift prüft PhoSca ausserdem schmale Striche mit
lokalem Helligkeits- oder Farbkontrast auf beiden Seiten. Bei
verbundenen blassen Zeichen zieht der Detektor den lokalen
Papiermedian ab und sucht ausgerichtete Strichblöcke mit gemeinsamer
Schriftlinie. Auch offene Zeichen können zu diesem Signal beitragen.
Die bestätigenden Originalpixel müssen auf demselben Papier und
ausserhalb des Motivs liegen. Nach einem deutlich bestätigten Fund
sucht PhoSca auf hellen Papierstreifen weiter. Es spart die bereits
gefundene Stelle samt vier Pixeln Abstand in der Papierebene aus. So
kann eine blasse Beschriftung neben der Kopfzeile oder auf einer anderen
Seite einen eigenen Hinweis erhalten. Bei einem bereits bestätigten
Schriftzug lässt sich der Hinweis über einen Bindestrich verlängern.
Der Strich und zwei weitere strukturierte Schriftteile müssen in
derselben Zeile liegen und sich einzeln in den Originalpixeln
bestätigen lassen. Ein bestätigter Fund setzt für
**Nur Bildmotiv** den
Prüfgrund `IMAGE_MARGIN_MARKS_REVIEW` und speichert die Fundstellen in
Koordinaten der Quelle. Der Hinweis fordert dazu auf,
den gesamten Rand zu prüfen. Die Prüfung verändert die Geometrie und
die Pixel von **Ganzer Abzug** nicht. Wenn
Fotoecken Papier verdecken, kann **Ganzer Abzug** in Prüfung liegen,
während das vollständig sichtbare Motiv geometrisch verwendbar ist.

Beim Sampling begrenzt das gewählte Projektionsviereck die zulässigen
Quellpixel für die JPEG-Ausgabe. An einer Kante fallen bilineare Taps
ausserhalb dieser Fläche oder des Rasters weg. Die übrigen Gewichte
werden renormalisiert. Wenn die Interpolationszelle am Eckpunkt keinen
zulässigen Tap enthält, kann der nächste zulässige Quellpixel in
höchstens zwei Originalpixeln Abstand verwendet werden. Fehlt auch
dieser, bleibt die Geometrie zur Prüfung. Eine Prüfvorschau kann eine
gemessene, nicht rechteckige Kontur mit Transparenz rendern. Die
Qualität des Samplers und die Richtigkeit der gewählten Papiergrenze
werden im Benchmark getrennt gemessen.
[Perspektive und Modi](backend/app/pipeline/perspective.py) ·
[Quellpixel-Sampling](backend/app/pipeline/crop_sampling.py) ·
[Motivrand und Beschriftung](backend/app/pipeline/motif_margin.py)
· [Formprüfung für einzelne Striche](backend/app/pipeline/margin_stroke_shapes.py)

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

Bei einem verfügbaren Ganzabzug kann der Motiv-Editor ein zum Papier
paralleles Rechteck direkt auf der Originalseite bearbeiten. Die
Position des Zeigers wird in die entzerrte Ebene des gespeicherten
Papiers umgerechnet. Beim Ziehen verschieben sich die beiden Kanten,
die an die aktive Ecke grenzen. Die gegenüberliegende Ecke bleibt
an ihrem Originalpunkt und die Motivkontur bleibt innerhalb des
Papiers. Der Editor rechnet die neuen Ecken zurück in die Aufnahme.
Bei gespeicherten oder bewusst geladenen schrägen Motiven lassen sich
die vier Ecken unabhängig verschieben.
[Motiv-Rechteck in der Papierebene](data/evaluation/README.md#resize-the-motif-in-the-saved-paper-plane)

Mobile zeigt offene Motivkanten und erkannte Randbeschriftungen auf der
Originalseite. Sobald ein manuell bestätigter Ganzabzug verfügbar ist,
lässt sich **Nur Bildmotiv** auch unter **Cropping prüfen** bearbeiten.
Die Motivecken bleiben ein eigener Entwurf. Ein Antippen, eine Bewegung
unter zwei Bildschirmpixeln oder eine zurückgezogene Ecke genügt beim
ersten Entwurf für die Speicherung nicht. Die Vorschau verwendet das
Seitenverhältnis aus der entzerrten Papierebene. Beim Speichern gehen
die unveränderten Papierecken, Masse und die gewählte Drehung mit den
Motivecken an die bestehende Editor-API. Die Freigabe des Motivs setzt
eine gültige Geometrie und die bestehende Prüfung voraus.
[Motivkorrektur am Kontaktabzug](data/evaluation/README.md#review-and-correct-the-motif-on-mobile)

Unter **Crop-Qualität** zeigt Mobile den Grund für eine gesperrte
Bestätigung. Für einen berechneten Ganzabzug richtet sich der Knopf
**Korrekt & vollständig** nach der vom Server bereitgestellten
Download-URL. Oben im Editor steht, ob du die vier Papierecken
prüfen, die Masse des Abzugs eintragen oder eine Aufnahme mit
sichtbarem Rand rundum machen kannst. Die App übernimmt nach jeder
Speicherung den aktuellen Prüfstand des Servers. Ist der
Ganzabzug verfügbar, zeigt sie den tatsächlich berechneten
Zuschnitt. Eine ausdrücklich geforderte Motivprüfung hat weiterhin
ihre eigene Bestätigung.
[Beispiel für eine nicht bestätigbare Geometrie](data/evaluation/README.md#explain-a-held-whole-print-in-mobile-geometry-review)

Nach einer Änderung rendert die API die Varianten erneut vom
Albumoriginal. Ein Modus kann eine Vorschau mit Korrekturhinweisen
zeigen, während sein Download gesperrt bleibt. Die Galerie lädt die
gewählte verfügbare Variante. Manuelle Korrekturen durchlaufen
weiterhin Geometrie-, Nachbarschafts- und Pixelprüfungen. Rechenläufe,
Bilddateien, Prüfentscheide und Korrekturrevisionen bleiben mit der
Quellseite verknüpft.

Bei einer automatisch vorgeschlagenen Kontrastkontur mit unbelegtem
Papierrand innerhalb der Aufnahme bleibt die Papierprüfung offen.
Prüfe im Original, ob alle vier Ecken zum selben Abzug gehören.
Bestätige die Papierkontur im Desktop- oder Mobile-Editor ausdrücklich,
bevor du «Ganzer Abzug» speicherst. Papiermasse und Ausrichtung reichen
dafür allein nicht aus.
Zeigt die markierte Fläche keinen eigenen Abzug, kannst du sie als
«Falsches Objekt» aus der aktiven Anzahl entfernen. Die Historie und
unabhängige Vorschläge bleiben erhalten.
[Beispiel einer unbelegten Kontrastkontur](data/evaluation/README.md#require-separate-confirmation-for-an-unsupported-contrast-outline)

Sind beide Zuschnitte in Prüfung, kann die Geometrieprüfung zwei
orange gestrichelte Abschnitte auf der vorgeschlagenen Kontur zeigen.
Dort ist beidseits der Linie zusammenhängende Bildstruktur messbar.
Prüfe an diesen Stellen den Papierrand im Original. Beim Verschieben
einer Ecke verschwinden die Markierungen für den ursprünglichen
Vorschlag. Die Markierungen allein ändern weder die Ecken noch die
Abzugsanzahl oder den Prüfstatus.
[Messung durchgehender Bildstruktur](data/evaluation/README.md#show-source-detail-across-an-unsupported-contrast-outline)

Nach der Bestätigung einer solchen Kontrastkontur misst PhoSca den
Papierrand erneut im Original. Die eingegebenen Ecken bestimmen die
gespeicherte Papierkontur. Gibt es mehrere plausible Kanten, bleibt
der Zuschnitt in Prüfung und PhoSca zeigt die messbaren Alternativen an. Bei einem
klaren, abweichenden Papierrand kann der Editor eine neue Kontur zum
Laden anbieten. Geprüfte Rundungen lassen sich in einer transparenten
Vorschau mit dem bisherigen Zuschnitt vergleichen. Ältere bestätigte
Zuschnitte dieses Typs werden vor dem Download erneut geprüft.
[Nachprüfung bestätigter Kontrastkonturen](data/evaluation/README.md#recheck-a-confirmed-contrast-outline-in-native-pixels)

Auf geripptem Albumkarton kann PhoSca einen möglichen Papierrand
blau gestrichelt ins Original einzeichnen und den ganzen Abzug entlang
dieser Kontur als Vergleichsvorschau entzerren. Vier stabile Kanten und
ein durchgehendes Rippenmuster auf dem Karton beidseits des Abzugs sind
Voraussetzung. Die gespeicherten Ecken bleiben massgebend. Die
Vergleichsvorschau enthält keine Aussage über die Form der Ecken,
und beide Ausschnitte bleiben in Prüfung. Sobald du eine Ecke
verschiebst oder ein neuer Abzug die Messfläche beansprucht, blendet
PhoSca die Vergleichsvorschau aus.
[Messung und Gegenproben auf geripptem Albumkarton](data/evaluation/README.md#compare-a-measured-outline-on-ribbed-backing)

Bei einem unbelegten Modellumriss auf einer breiten, hellen Fläche
kann PhoSca einen einzelnen Übergang zum sichtbaren Bild als
violette Prüfspur im Original markieren. Mehrere Messungen müssen
denselben Abschnitt stützen. Geschützte Inhalte, beanspruchte
Nachbarabzüge und ein belegter äusserer Rahmen können die Spur
verhindern. Die Spur hilft, den Bildübergang am Original zu finden.
Ob dort auch das Fotopapier beginnt, muss im Review geklärt werden.
Ecken, Zuschnitt und Freigabe werden weiterhin separat geprüft.
[Messung an Scan 66 und Gegenproben](data/evaluation/README.md#show-a-measured-image-transition-inside-a-held-detection)

Liegt ein gekrümmter sichtbarer Bildrand teilweise ausserhalb
des Zuschnitts, kann PhoSca den belegten Streifen in einer
transparenten Vergleichsvorschau ergänzen. Dafür müssen der
Übergang ins Bild und ein schmaler heller Nachbarstreifen in den
Originalpixeln über eine zusammenhängende Strecke messbar sein.
Die Vorschau erweitert den Ausschnitt in der bestehenden
Perspektive. Zu den Ecken bleibt ein Abstand, weil dort mehrere
Kanten zusammentreffen. Wenn ein Nachbarabzug die Messfläche
beansprucht, verschwindet der Vergleich. Ob der helle Streifen zum
Fotopapier gehört, bleibt eine eigene Prüfentscheidung.
[Messung und Vergleich auf Golden 12](data/evaluation/README.md#recover-a-measured-strip-along-a-bowed-image-edge)

Wenn sich hinter der Bildkante eine zweite Kante und danach
ein durchgehender Hintergrund messen lassen, kann dieselbe
Prüfvorschau einen belegten Abschnitt ausserhalb des hellen
Streifens ausblenden. Der helle Streifen selbst bleibt sichtbar.
Ein messbarer grösserer Papierrand, Schriftmerkmale oder
geschützte Inhalte lassen die Vorschau bei der Ergänzung des
Bildstreifens bleiben. Beide Zuschnittmodi bleiben in Prüfung.
[Messung des äusseren Streifens und Gegenproben](data/evaluation/README.md#compare-a-measured-exterior-band-beyond-a-pale-rim)

PhoSca vergleicht die fraglichen Pixel auch mit ihrer unmittelbaren
Umgebung. Zusammenhängende Helligkeits- oder Farbabweichungen
lassen den Rand sichtbar, damit feine Markierungen und Papierstruktur
im Original geprüft werden können.
[Messung feiner Randmarkierungen](data/evaluation/README.md#preserve-local-marks-in-the-exterior-comparison)

Wenn sich zwei gehaltene Modellvorschläge stark überdecken und
der grössere zugleich in einen weiteren offenen Vorschlag reicht,
kann die Prüfung die Originalmasken vergleichen. Violett liegt
in beiden Vorschlägen, Orange nur im grösseren und Türkis nur im
kleineren. Die Farben beschreiben die erkannten Flächen. Prüfe
am Original, welche Bildteile und Papierränder zusammengehören.
PhoSca berechnet den Vergleich aus den aktuellen Einträgen der Seite.
Wird einer der beteiligten Einträge korrigiert, verworfen oder
freigegeben, entfällt dieser Vergleich. Zuschnitte, Anzahl und
Freigabe werden weiterhin über die bestehenden Schritte geprüft.
[Überlappende Vorschläge auf Scan 66](data/evaluation/README.md#compare-overlapping-held-model-outlines)

Bei zwei stark überlappenden Modellmasken kann PhoSca zusätzlich
eine sichtbare Bildkante direkt in den Originalpixeln messen. Die
helle gestrichelte Linie erscheint in der Anzahl- und Geometrieprüfung,
wenn zwei Kantensuchen dieselbe Form finden und alle vier Seiten
einen deutlichen Übergang zum hellen Umfeld zeigen. Vergleiche die
orange und türkis markierten Bereiche mit dieser Linie. Ein Bereich
ausserhalb der Linie kann Papier, Albumuntergrund oder Bildinhalt
eines zweiten Abzugs zeigen. Prüfe die physische Papierkante separat.
Die Linie ist eine Orientierung für die Korrektur. Speichere die
Ecken des Abzugs und des Motivs über die jeweiligen Modi.
[Beispiel am Hochzeitsalbum](data/evaluation/README.md#compare-two-held-proposals-against-a-measured-motif-edge)

| Aktion | Verhalten |
| --- | --- |
| Anzahl bestätigen | Der `segmentation`-Entscheid bezieht sich auf die Seite. Die `photo_count_revision` und gegebenenfalls die IDs offener Vorschläge müssen zum aktuellen Stand passen. Eine veraltete Bestätigung liefert `409`. |
| Abzug korrigieren oder ergänzen | `PATCH /api/v1/photos/{id}/bounding-box` setzt die überprüfte Geometrie. `POST /api/v1/pages/{id}/manual-photos` ergänzt einen Abzug. Ungültige Ecken können mit `422` abgelehnt werden. Ein überlappender Vorschlag hält bei ungeklärter Papierzugehörigkeit beide Modi zurück. |
| Falsches Objekt markieren | `wrong_object` bezieht sich auf die aktuelle Box und Maske. Der Eintrag wird aus der aktiven Anzahl und Kundenansicht entfernt. Die Prüfgeometrie bleibt für die Historie erhalten. |
| Geometrie vollständig bestätigen | `POST /api/v1/review/events` archiviert den Entscheid zur aktuellen Geometrie. Ist der ganze Abzug gesperrt oder eine echte Motivprüfung offen, liefert `correct` ein `409`. Ein gleichlautender Wiederholungsaufruf mit demselben Idempotenzschlüssel wird als Wiederholung erkannt. |
| Erneut berechnen | `POST /api/v1/photos/{id}/reprocess` rendert mit den aktuellen Daten vom gespeicherten Albumoriginal. Eine Neuberechnung allein erledigt keinen historischen Entscheid, dessen Snapshot eine offene Motivprüfung enthält. |

Die Zustände liegen unter `enhancement_applied.correction.crop_modes`,
getrennt als `full_print` und `image_only`. Der Wechsel von
`ready` zu `review_needed` kann trotz vorhandener Geometrie erfolgen,
etwa bei fehlender Kalibrierung oder einem offenen Besitzkonflikt.
`reason_codes` erklären den Zustand. `geometry_status: ready` kann
dabei die bereits berechnete Geometrie kennzeichnen. Die Varianten
folgen diesem Vertrag.

| Abruf | Ergebnis |
| --- | --- |
| `GET /api/v1/pages/{id}/image` | Unveränderte gespeicherte Albumaufnahme. |
| `GET /api/v1/photos/{id}/image?variant=full_print` oder `image_only` | JPEG des angeforderten Modus, wenn er bereit und verfügbar ist. Ein gesperrter Modus liefert `404` und keine Download-URL. |
| `full_print_preview`, `image_only_preview` | Separat erzeugte PNG-Prüfvorschau, falls für den offenen Modus ein Vorschauartefakt existiert. Die Antwort verwendet `Cache-Control: private, no-store`. |
| `original`, `source_corrected` am Foto-Endpunkt | Quellengetreue, bereits perspektivisch korrigierte JPEG-Ansicht. Die unveränderte Albumdatei ist über den Seiten-Endpunkt verfügbar. |
| `source_corrected_untrimmed` | Entzerrte Quellenansicht für die Kontrolle. Ihre Verfügbarkeit erlaubt keine Freigabe der benannten Modi. |
| `source_corrected_trimmed` | Alter Name für einen getrimmten Zuschnitt. Beim präzisen Pfad gelten auch die Sperren von `image_only`. |

Eine manuelle Geometriekorrektur löscht die automatische
`segmentation_confidence`. Die Geometrie-, Zuordnungs- und
Samplingprüfungen gelten weiterhin. Bei einem Hinweis auf eine
unterbrochene Papierkante speichert PhoSca die Messabschnitte aus
dem Original im Prüfprotokoll. Beide Zuschnitte bleiben in Prüfung,
bis dieselbe manuell gewählte Seite beide Abschnitte mit höchstens
zwei nativen Pixeln Abweichung abdeckt. Eingetragene Masse ändern
das Seitenverhältnis. Ihre Eingabe löst die Kantenprüfung nicht auf.
Bei einer automatischen
Erkennung überprüft der Download-Endpunkt den Freigabenachweis auch
beim Abruf. Ein alter Zustand `ready` genügt deshalb nach einer
Änderung der Konfiguration oder des Freigabedokuments allein nicht.
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

OCR arbeitet auf der Seite und ordnet Text später anhand seiner Lage
zu. Bei fehlender OCR kann ein vorhandener Sidecar Text liefern.
Beim Wiederholen ersetzt der OCR-Schritt die ungeprüften Einträge.
Bestätigte Texte bleiben erhalten. Der Seitenlauf kann trotz gesperrter
Ganzabzüge OCR erzeugen und die nachgelagerte Verbesserung für diese
Abzüge auslassen.

Docker Compose verbindet Nginx und React mit FastAPI, PostgreSQL,
Redis, den Celery-Workern sowie dem Volume für Originale und
Derivate. Lokal kann SQLite als Entwicklungsdatenbank dienen.
Das Modellregister bindet die konfigurierten Gewichte an ihre Hashes.
Bei aktiviertem `AUTH_REQUIRED` prüft die API Rollen und Albumzugriff
über einen API-Schlüssel oder eine Browsersitzung.

Die Derivate liegen unter
`processed/<album>/runs/<run>/stages/<stage>/photos`. Ein Lauf bindet
Quellhash, Pipelineversion und Konfigurationsfingerabdruck.
Korrekturen wechseln die aktive Variantenreferenz. Speicherbereinigung
ist ein separater Vorgang. `Album.completed` beschreibt den Abschluss
der Verarbeitung. Seiten mit offenen Prüfungen zählen als
verarbeitet. Die Qualitätsfreigabe der Bildverbesserung und die
Freigabe der beiden benannten Zuschnitte verwenden getrennte Regeln.
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
Seitenverhältnis. Die Anforderungen für einzelne Ausgaben sind
strenger als ein hoher mittlerer IoU. Im Freigabereport wird jeder
vorhergesagte Abzug höchstens einer Referenz zugeordnet.

| Prüfbereich | Mindestanforderung |
| --- | --- |
| Instanzerkennung und finale Masken | Zuordnung bei IoU ≥ 0,85, mittlerer Polygon-IoU ≥ 0,85, Präzision und Trefferquote je ≥ 90 %, Kanten-F-Score ≥ 0,80. Auf dieser Stufe gelten zusätzlich höchstens 8 % Inhaltsverlust, höchstens 12 % fremde Fläche und keine als schwerwiegend klassifizierten Fehler. |
| Ausgabe pro Modus | Zugeordnete Instanz, richtige Orientierung, höchstens 2 Originalpixel Eckfehler, höchstens 1 % Seitenverhältnisfehler und, falls Referenzlandmarken vorhanden sind, höchstens 2 px Landmarkenfehler bei p95. |
| Sampling pro Modus | Null Inhaltsschnitt und null fremde Beiträge jenseits eines Bands von einem Originalpixel, null Abtastungen ausserhalb des Quellrasters, höchstens 1 px Samplingdefizit und höchstens 1 px Abweichung zwischen Homographie und Geometriemetadaten. Der synthetische Test verlangt zusätzlich null fremde Quellsamples über die gesamte Ausgabe. |
| Status pro Modus | Unwiederherstellbare Referenzen müssen in Prüfung liegen. Keine verifizierte falsche oder unwiederherstellbare Ausgabe darf als bereit gelten. |

Die automatische Freigabe braucht einen bestandenen Nachweis für die
eingesetzte Konfiguration. Die Herkunft der Modellgewichte wird mit
Quellen, Album-IDs, Bild-Hashes und Splits dokumentiert. Die
Detektorkonfidenz wird auf einem separaten Datensatz für die
eingesetzte Konfiguration kalibriert. Der Fit ist isoton und erfolgt
pro Detektorquelle, etwa für ONNX und klassische Konturen getrennt.
Ein Fit benötigt pro Quelle mindestens 20 Vorhersagen insgesamt,
darunter richtige und falsche Beispiele. Quellen ohne Fit bleiben
unkalibriert. Testquellen dürfen sich weder
nach Album-ID noch nach Bild-Hash mit Trainings-, Entwicklungs- oder
Kalibrierungsquellen überschneiden. Beide Modi brauchen überprüfte
Referenzen aus mindestens zwei gemeinsamen physischen Albumgruppen.
Pro Modus müssen mindestens **90 %** der wiederherstellbaren
Zuschnitte automatisch freigabefähig sein, insgesamt und in jeder
erforderlichen Gruppe. Der Server akzeptiert den Bericht nur mit
passenden Modell-, Profil-, Quellen- und Berichtshashes. Die
Release-Regel bezieht sich auf tatsächlich automatisch
freigabefähige Ausgaben.

Die automatische Freigabe bleibt gesperrt. Auf dem eingefrorenen
synthetischen Satz findet der Detektor-Kontrolllauf **18/18 Abzüge**.
Mit denselben automatisch gefundenen Vorschlägen bestehen
**6/11 Ganzabzüge** und **7/12 Motive** die jeweilige Geometrieprüfung.
Ein getrennter Lauf mit vorgegebenen, leicht versetzten
Startkonturen besteht **11/11 Ganzabzüge** und **12/12 Motive**.
Die Trennung zeigt, wie viel die Qualität der Erkennung zur
Ausgabequalität beiträgt. Unabhängige Originalraster und die passende
Kalibrierung für die Freigabe sind noch unvollständig.
[Prüfverfahren und Belege](data/evaluation/README.md) ·
[Auswertung im Code](scripts/evaluate_onnx_segmentation.py) ·
[Freigabeprüfung im Code](backend/app/pipeline/crop_validation.py)

## Technische Rückschau

### Ausgangsbefund und Fehlersuche

Der Ausgangsbefund aus neun echten Cloud-Ansichten waren 22 Treffer,
sechs fehlende Abzüge und 17 Fehlvorschläge. Das entspricht 78,6 %
Trefferquote und 56,4 % Präzision. Elf der Fehlvorschläge stammten aus
drei physischen Mehrbildabzügen. Der exakte Cloud-Lauf liess sich aus
den verfügbaren Originalen, Modellgewichten und Reports nicht erneut
ausführen. Der spätere Entwicklungsvergleich verwendet sieben andere
reproduzierbare Ansichten. Davon stammen mehrere aus demselben Scan.
[Ausgangsbefund und reproduzierbarer Vergleich](data/evaluation/README.md#replay-the-available-real-image-development-views)

Für die Fehlersuche vergleiche ich pro Abzug vier Zustände. Der rohe
Detektor zeigt, ob es überhaupt einen Vorschlag gab. Die finale Maske
zeigt, ob mehrere Abzüge zusammengelegt oder ein Abzug fragmentiert
wurde. Der auf die Quelle zurückprojizierte Zuschnitt zeigt, ob
Entzerrung und Randbehandlung Inhalt entfernen. Der API- und
Browserpfad zeigt, ob genau diese Variante als Vorschau, Prüfung
oder Download behandelt wird. So lässt sich derselbe Fehler gezielt
einer Stufe zuordnen. [Instanzdiagnostik und Renderprüfung](data/evaluation/README.md)

### Erkennung, Training und Fusion

| Versuch | Ergebnis und Entscheidung |
| --- | --- |
| Klassische Geometrie gezielt verbessern | Die Liniensuche holte auf Golden-Seite 11 einen Abzug mit 0,911 Masken-IoU zurück. Der zweite blieb unerkannt. Auf Seite 37 verdrängte ein Diagonalfragment die vollständige Kontur. Nach der Korrektur erreichte die Maske 0,994 IoU, während eine gemessene Papierkurve weiterhin Prüfung verlangte. Der klassische Golden-Lauf verbesserte sich insgesamt von 12/29 mit 16 Fehlvorschlägen auf 14/29 mit 15. [Liniensuche](data/evaluation/README.md#line-pair-search-after-rejected-contours) · [Konturauswahl](data/evaluation/README.md#nested-classical-contours-on-the-same-print) |
| Synthetisch auf physische Papierabzüge trainieren | Modell-only erkannte 12/12 Mehrbildabzüge und 32/34 getrennte Abzüge im gezielten Stock-Test bei IoU 0,85, ohne Fehlvorschläge. Der Hybrid blieb auf echten Golden-Seiten bei 14/29, mit 16 statt 15 Fehlvorschlägen gegenüber klassischer Geometrie. Der Kandidat wurde für den Einsatz verworfen. [Stock-Experiment](data/evaluation/README.md#synthetic-stock-detector-experiment) |
| Transfer mit wenigen echten Kameraoriginalen prüfen | Im Mischarm ersetzten 40 Derivate von fünf Originalen 40 der 96 Trainingsbilder. Checkpoint-Ausgangspunkt, Validierung, Input von 512 × 512 und Schwellen blieben im Vergleich gleich. Die Golden-Erkennung im Hybrid stieg von 14/29 mit 20 Fehlvorschlägen auf 29/29 mit zwei. Auf dem eingefrorenen Satz erreichte der Mischarm 17/18 mit vier Fehlvorschlägen, 9/11 bestandene Ganzabzüge und 10/12 Motive. Auf dem realen Kontaktbogen lieferte er drei Fehlvorschläge mehr als der synthetische Kontrollarm. Beide Gewichte bleiben Entwicklungsartefakte. Die Golden-Seiten teilen die physische Quellgruppe mit dem alten Kameramaterial. [Gepaarter Transfer](data/evaluation/README.md#compare-real-camera-pages-against-a-paired-synthetic-control) |
| Mehrbildabzüge mit dichterem Druckraster trainieren | Auf 24 neuen synthetischen Seiten stiegen die rohen Treffer von 8/12 auf 11/12 bei Mehrbildabzügen und von 18/24 auf 21/24 bei getrennten Papieren. Bei einem zuvor korrekten Paar schlug das dichte Modell jedoch eine gemeinsame Aussenkontur vor. Auf dem echten JPEG75-Kontaktbogen fand die Variante mit dem weniger dichten Raster einen Abzug ohne Fehlvorschlag. Bei der dichteren Variante verdrängte die klassische Fusion die bessere Modellkontur. Beide Varianten verschlechterten die Zuschnittprüfung in beiden Modi und bleiben Entwicklungsartefakte. [Gepaarter Mehrbildtest](data/evaluation/README.md#compare-sparse-and-dense-artwork-on-the-same-physical-stock) |
| Lokale Maskenverfeinerung nachtrainieren | Auf 166 dichten Validierungsausschnitten stieg der mittlere IoU von 0,562 auf 0,907. Auf dem Met-Entwicklungsscan sank die Trefferzahl von 44 auf 42. Zusätzlich fiel die eingefrorene Erkennungsprüfung aus. Der Verfeinerer wurde nicht übernommen. [Dense-Gutters-Auswertung](data/evaluation/README.md#precise-crop-validation) |
| Reale Entwicklungsansichten getrennt auswerten | Klassisch ergaben sich 22/24 Treffer und acht Fehlvorschläge, gemischt 23/24 und drei. Der Gesamtwert verdeckt einen Rückschritt beim angeschnittenen NerdScan-Abzug mit 35,3 % Verlust der Entwurfsfläche. Dieser Vergleich zeigt, warum ein Gewinn der Gesamterkennung und ein sicherer Zuschnitt getrennt bewertet werden. [Realer Entwicklungsvergleich](data/evaluation/README.md#replay-the-available-real-image-development-views) |
| Schwache Zusatzvorschläge am Original überprüfen | Zwei zusätzliche klassische Golden-Vorschläge wanderten nach nativer Grenzprüfung in die Zählprüfung. Die aktive Zählung erreichte damit 29/29 ohne zusätzlich gezählte Objekte. Beide Vorschläge blieben für den Reviewer sichtbar. [Prüfung von Fallback-Konturen](data/evaluation/README.md#review-unsupported-fallback-outlines-before-counting-a-print) |
| Szene innerhalb eines vorgeschlagenen Abzugs verfolgen | Auf einem 35-Megapixel-Albumscan lieferte die Fusion zu sechs sichtbaren Abzügen zwei falsche Zusatzkonturen. Beim mittleren rechten Foto laufen Bilddetails über zwei vermeintliche Kanten hinweg. Der Teilvorschlag liegt jetzt in der Anzahlprüfung. Ein zweiter Teilvorschlag im Original und fehlender Papierrand an den grösseren Konturen bleiben offen. In der Halbbreitenansicht bleiben sechs aktive Vorschläge und dieselbe Anzahlprüfung. [Original, Messwerte und Gegenproben](data/evaluation/README.md#trace-scene-detail-across-a-nested-flatbed-proposal) |
| Einen fehlenden oder angeschnittenen Abzug an Fotoecken und Papierkanten lokalisieren | Auf Willett 95 findet das Modell den oberen linken Abzug weder im Original noch in zwei kleineren Ansichten. In den kleineren Ansichten stützen vier rote Halter und sieben gemessene Papierabschnitte einen Vorschlag zur Anzahlprüfung. Im Original zeigt PhoSca beim bestehenden Abzug gemessene Kanten ausserhalb des angeschnittenen Zuschnitts. Die Fotoecken verdecken die Papierspitzen und beide Modi brauchen Prüfung. [Fund in den kleineren Ansichten](data/evaluation/README.md#recover-a-count-offer-from-red-photo-corners-and-open-paper-edges) · [Korrekturhilfe in Originalauflösung](data/evaluation/README.md#show-the-visible-paper-edges-of-an-already-counted-mounted-print) |
| Gemessene Teile der Brücke in der Prüfvorschau ergänzen | Auf Willett 95 kommen rund 280’000 sichtbare Originalpixel hinzu. Unsichere Randabschnitte bleiben transparent. Der vollständige Papierrand und die Motivgrenze brauchen weiterhin Prüfung. [Messvergleich und Gegenproben](data/evaluation/README.md#compare-only-measured-strips-beside-a-mounted-print) |

### Konturen, Perspektive und Ausgabesicherheit

- **Mehrbildabzug als physische Instanz.** Beim Kontaktbogen stützen
  übereinstimmende Rasterabstände, durchgehendes Papier an den Rändern
  und fehlende Hinweise auf einen eigenen Schatten die Zuordnung zu
  einem Abzug. Im 33-%-JPEG75-Fall steigt der Masken-IoU der angebotenen
  Aussenkontur gegen den Entwurf von 0,8317 auf 0,9885. In der
  verlustfreien Variante steigt er von 0,8286 auf 0,9898. Die
  konkurrierenden Fragmente bleiben sichtbar und zunächst gezählt.
  Bei weniger als zwei Originalpixeln eigenem Rand kann ein zweites
  Papier gleich aussehen. Ein lockerer Vorschlag mit drei gestützten
  Seiten überschritt bei Domoritz in 28 von 190 Gegenproben eine
  bekannte physische Trennung. Eine solche Geometrie bleibt ein
  Korrekturhinweis. [Kontaktbogen](data/evaluation/README.md#compare-the-whole-contact-print-with-its-counted-fragments) ·
  [Unschärfe-Gegenproben](data/evaluation/README.md#split-print-under-mild-blur)
- **Bessere Vorschau bei verbleibendem Randfehler.** Beim JPEG55
  verbesserte die neu entzerrte Aussenkontur den Zuschnitt-IoU
  gegen den Entwurf von 0,8363 auf 0,9884. Der fehlende Inhalt sank
  von 16,37 % auf 1,07 %. Der grösste Eckfehler blieb bei 4,84
  Originalpixeln. 220 Ausgabepixel erhielten fremde Beiträge jenseits
  des Ein-Pixel-Bands. Das Ergebnis blieb als Prüfvorschau verfügbar.
  Die nachträgliche Suche eines Rastermusters um acht Pixel in beide
  Richtungen lieferte auch bei 47 von 63 Kontrollen mit getrennten
  Papieren fälschlich dieselbe Rasterevidenz. Die automatische
  Entscheidung behält die engere Prüfung.
  [Vorschau und Messreihe](data/evaluation/README.md#straighten-the-proposed-contact-print-for-review)
- **Papierkurve und Motivkurve unterscheiden.** Die Eingabe von
  Papiermassen hatte auf Golden-Seite 37 früher eine Freigabe trotz
  offener Papierkurve erlaubt. Der neue Pfad hält die Kurve auch nach
  manueller Änderung fest. Er akzeptierte 36 belegte gerade
  Papiergrenzen mit gekrümmtem Druckmotiv und hielt 180 getestete
  Eckänderungen an tatsächlich gewölbten Abzügen zurück. Beim
  JPEG75-Kontaktabzug bleiben konkurrierende Motivübergänge und
  Beschriftung auf dem Papier entscheidend. Ein höherer
  Krümmungsschwellwert würde diese Fragen offenlassen.
  [Papierkurven](data/evaluation/README.md#keeping-a-measured-paper-curve-in-review-after-a-manual-edit) ·
  [Kontaktbogenmotiv](data/evaluation/README.md#require-an-intentional-motif-outline-in-review)
- **Manuelle Prüfung und Export vor Inhaltsverlust schützen.** Das
  alte, pauschal um 6 % eingerückte Motiv erhielt nur einen von
  1’938 vermessenen Beschriftungspunkten und schnitt fünf Felder
  der letzten Reihe an. Heute erfordert der Editor eine bewusste
  Motivkontur. Eine ältere allgemeine Geometriebestätigung kann
  eine offene Motivprüfung nicht ausblenden. Ausserdem hält ein
  überlappender aktiver Fragmentvorschlag beide Exporte zurück,
  bis seine Zuordnung geklärt ist. Am echten Kontaktabzug ergab
  der Standardweg nach ausdrücklicher Korrektur zwei Downloads
  mit allen 35 Bildfeldern und allen 1’938 geprüften
  Beschriftungspunkten im Motiv. [Manueller Ablauf](data/evaluation/README.md#require-an-intentional-motif-outline-in-review) ·
  [Alte Bestätigungen](data/evaluation/README.md#keep-genuine-motif-reviews-open-after-legacy-geometry-approvals) ·
  [Überlappende Papieransprüche](data/evaluation/README.md#keep-manual-exports-held-while-paper-proposals-overlap)

### Schrift am Motivrand und Datenlage

Die erste Randprüfung erkannte 25 von 54 zuvor geometrisch bereiten
Zuschnitten mit teilweisem oder vollständigem Schriftverlust. Der
zweite Schritt mit beidseitigem Originalkontrast fing drei weitere
Verluste ab. In derselben 162-Fälle-Reihe blieben alle Ganzabzüge
pixelgleich. Von den 54 Ausgangsfehlern bestehen noch 26. Sie
enthalten Bleistift, einzelne Zeichen, verbundene Schrift und
Ausschnitte einer real geformten Beschriftung. Der separate Test mit
216 Beispielen schwacher Beschriftung und 660 Gegenproben verbesserte
die Erkennung von 19 auf 52. Bei den Gegenproben kam keine
Fehlmarkierung hinzu. Eine pauschale niedrigere
Kontrastschwelle hatte in früheren Reihen auch Papierstruktur,
Spiegelungen und Kratzer markiert.
[Erster Randschutz](data/evaluation/README.md#review-visible-markings-outside-the-proposed-motif) ·
[Graphit und blaue Schrift](data/evaluation/README.md#check-faint-graphite-and-blue-ink-against-the-original-paper)

In einer weiteren Serie mit 216 markierten Streifen stieg die
Erkennung mit der Formprüfung für einzelne Komponenten von 11 auf 37.
Unter 198 Gegenproben kamen keine neuen Prüfhinweise hinzu.
Die neuen Treffer enthalten einzelne Zeichen wie **A** und **B**
sowie verbundene Schrift. Sehr blasses Graphit, manche
Einzelzeichen und kurze Schrift ohne geschlossene Innenräume bleiben
offen. [Formprüfung](data/evaluation/README.md#hold-motif-trims-for-isolated-strong-strokes)

Beim historischen Detektor gehören der archivierte 28/29-Bericht
(`e4168ae…`) und die spätere 29/29-Zusammenfassung (`4d823588…`)
zu verschiedenen Modell-Digests. Das Modellregister nennt für die
spätere Datei eine NCHW-Eingabe mit 1024 × 1024 Pixeln. In der
vorliegenden Arbeitskopie fehlen die registrierte ONNX-Datei und
Teile ihres Trainingsinventars. Bei den zusätzlichen
Archivkandidaten sind zum Teil getrennte Aufnahmen katalogisiert,
aber native Paare, überprüfte Grenzen, Nutzungsrechte oder der
Ausschluss von Trainingsüberlappung sind noch offen. An der Freigabe
fehlen damit konkrete Artefakte und eine dazu passende Kalibrierung.
[Modellregister](models/manifest.json) ·
[Historische Modellherkunft](data/evaluation/README.md#recover-detector-provenance-from-the-git-archive) ·
[Archivpaare](data/evaluation/README.md#verify-that-catalogued-pairs-represent-independent-acquisitions)

### Nächste technische Hypothesen

Am linken Rudge-Abzug verbessert eine aus den Originalpixeln gemessene
Kontur mit seitlicher Toleranz die Überlappung mit dem vorläufigen Entwurf
von 0,8104 auf 0,9414 und erhält alle darin markierten Pixelzentren.
Dieselbe Kantenmessung würde bei einem separat erfassten Nachbarabzug
rund 5 % der projizierten Papierfläche abschneiden. Sie eignet sich
deshalb in diesem Zustand für die Analyse und als Ausgangspunkt für
eine Prüfhilfe. Die App zeigt die beobachteten Kanten am Original.
Beide Zuschnittmodi dieses Rudge-Falls bleiben zur Prüfung zurückgehalten.
[Messung, Aufnahmepaar und Gegenprobe](data/evaluation/README.md#measure-the-curved-opening-on-the-left-of-rudge-page-12)

1. **Bei konkurrierenden Linien das Papier zuordnen.** Im eingefrorenen
   Kontrolllauf liegt bei allen sieben umstrittenen Seiten eine
   korrekte Papierkante bereits im Suchband. Die Bewertung kann eine
   etwa neun bis zwölf Pixel weiter innen liegende Motivkante
   bevorzugen. Zwei Prüfvorschauen gewinnen den fehlenden Rand mit
   einer materialgestützten Wiederholung. Für die übrigen Fälle
   stehen mehrere Seiten oder unterschiedlich heller Untergrund zur
   Prüfung. [Diagnose mit Startversatz und Linienvergleich](data/evaluation/README.md#measure-seed-displacement-and-competing-print-edges)
2. **Den Besitzkonflikt auf ScanCropper auflösen.** Ein zusätzlicher
   Modellvorschlag umfasst einen Teil des oberen Abzugs. In der
   Ablation ohne genau diesen Vorschlag stieg die Masken-IoU von
   0,85386 auf 0,99795. Der Fusionssieger blieb derselbe. Der Editor
   zeigt die gemessene Kontur inzwischen beim aktiven Konflikt.
   Sehr dünne aufgelegte Abzüge mit derselben Vorlage können
   dieselben Übergänge zeigen. Ihre Anzahl bleibt eine Prüfentscheidung.
   [Fusionsprüfung und ScanCropper-Ablation](data/evaluation/README.md#audit-cross-source-fusion-and-scalloped-stock-conflicts)
3. **Restfehler bei blasser und offener Schrift prüfen.** Der lokal
   geschätzte Papiermedian und die Gruppierung verbundener
   Striche bringen das blasse Beispieldatum in Prüfung. In der
   gepaarten Serie bleiben 22 von 54 früher verlustreichen
   Motivzuschnitten offen. Besonders schwacher Kontrast,
   Einzelzeichen und isolierte Striche brauchen weitere
   Gegenproben auf ganzen Originalen.
   [Vergleich der Schrifterkennung](data/evaluation/README.md#compare-local-paper-centering-and-noise-scaling-for-faint-dates)
4. **Zwei echte Albumgruppen als unabhängigen Endtest sichern.** Die
   Quellenaufarbeitung soll nutzbare Originale mit belegtem Papier-
   und Motivbezug liefern. Danach lassen sich getrennte
   Kalibrierungsquellen für die tatsächlich aktiven Detektorquellen
   beschaffen und das festgelegte Modell einmal auf dem
   zurückgehaltenen Freigabesatz prüfen. Für die Modellwahl und
   Grenzwertsuche sind separate Entwicklungsalben erforderlich.
   [Referenzformat und Splitregeln](data/evaluation/README.md)

## Beispiele und offene Fälle

![Drei generierte Fälle mit Eingabe und Ergebnissen. Eine Schrägaufnahme gelingt in beiden Modi. Bei einer verdeckten Papierecke bleibt der ganze Abzug in Prüfung. Bei einem blassen Datumsaufdruck liegt das Motiv in Prüfung.](docs/images/crop-examples.png)

- **Schrägaufnahme.** Beide Zuschnitte gelingen. PhoSca entzerrt den ganzen
  Abzug und entfernt im Motivmodus den hellen Papierrand. Die synthetische
  Probe hat bekannte Referenzgeometrie und einen leicht versetzten
  Startvorschlag.
- **Fotoecke auf Papier.** Für den ganzen Abzug erscheint eine
  Prüfvorschau. Das vollständig sichtbare Motiv lässt sich in dieser
  synthetischen Probe separat entzerren.
- **Blasser Datumsaufdruck.** Der frühere Motivzuschnitt verlor in
  dieser Probe alle 727 vermessenen Schriftpunkte. Jetzt markiert
  PhoSca den verdächtigen Randbereich im Original und hält den
  Motivexport für die gezielte Korrektur zurück. Nach der bewussten
  Korrektur enthielt der exportierte Zuschnitt alle 727 Messpunkte.
- **Echter Kontaktabzug mit 35 Bildfeldern.** Die bisherige Vorschau
  der verlustfreien Aufnahme erfasst 1’509 von 1’940 vermessenen
  Beschriftungspunkten. Der mögliche Aussenumriss zeigt alle 1’940.
  Der grösste Eckfehler gegenüber dem Referenzentwurf sinkt von 69,19
  auf 5,71 Originalpixel. Die drei Vorschläge auf der Negativhülle
  brauchen weiterhin eine eigene Anzahlprüfung. Die Motivgrenze bleibt
  separat zu bestimmen. Nach manueller Prüfung beider Grenzen enthielten
  beide Test-Downloads alle 35 Felder.
  [Prüfvorschau und Messwerte](data/evaluation/README.md#compare-the-disputed-whole-contact-print) ·
  [Manueller Downloadpfad](data/evaluation/README.md#review-visible-markings-outside-the-proposed-motif)
- **Derselbe Kontaktabzug in Originalauflösung.** PhoSca markiert einen
  Teil der Beschriftung «NYC, Brooklyn» bereits bei **Ganzer Abzug**.
  Ein Versuch entlang des Filmrasters hätte die ganze Kopfzeile
  abgeschnitten. Zwei Schutzmarkierungen im Holz lassen sich am
  Original einzeln prüfen. Die Grösse des Papiers und die Motivgrenze
  werden danach separat geprüft.
  [Beschriftung und Messwerte](data/evaluation/README.md#locate-native-contact-print-lettering-while-the-paper-is-held) ·
  [Prüfung der Schutzmarkierungen](data/evaluation/README.md#review-multiple-advisory-marks-at-a-measured-paper-edge)
- **Zwei Abzüge auf Golden-Seite 26.** Unten rechts stimmt die
  Fläche von **Ganzer Abzug** nahezu mit der Entwicklungsreferenz
  überein (IoU 0,996). Oben links folgt der Zuschnitt der hellen
  Albumkante und übernimmt rund 5,93 % Fremdpixelgewicht. Dort zeigt
  PhoSca einen weiter innen liegenden Übergang zur Kontrolle am
  Original. Beide Modi bleiben in Prüfung.
  [Vergleich und Messwerte](data/evaluation/README.md#show-an-inward-source-transition-below-an-album-edge)
- **Golden-Seite 26 gedreht und komprimiert.** Alle vier echten
  Abzüge bleiben erfasst. Zwei Teilkonturen innerhalb des oberen
  rechten Abzugs erscheinen zur Anzahlprüfung. Ein dritter
  Vorschlag reicht über Album und Untergrund bis zum Aufnahmerand.
  Er erscheint mit einer gemessenen Linie zur separaten Prüfung.
  [Innenkonturen](data/evaluation/README.md#review-two-fallback-contours-inside-a-detected-print) ·
  [Vorschlag am Aufnahmerand](data/evaluation/README.md#review-a-fallback-outline-crossing-the-album-edge)
- **Der rechte Rand des oberen linken Abzugs.** Auf derselben
  gedrehten Aufnahme zeigt eine transparente Prüfvorschau 791
  zusätzliche native Quellpixel. An der gebogenen Kante bleiben
  Abschnitte und Ecken offen. [Randvergleich und Prüfablauf](data/evaluation/README.md#recover-an-observed-bowed-edge-in-a-transparent-review-preview)
- **Echter Abzug am Aufnahmebildrand.** Beim NerdScan-Entwicklungsbild
  traf die getestete Konfiguration drei erwartete Abzüge und lieferte
  zwei zusätzliche Vorschläge. Beide Modi blieben in Prüfung.
  Beim unteren Abzug zeigt der Editor die gemessene obere und linke
  Papierkante. Rechts und unten endet die Aufnahme. Der
  überlappende Teilvorschlag kann separat beurteilt werden.
  Bei angeschnittenen Originalen hilft eine neue Aufnahme mit sichtbar
  belassenen Papierrändern. [Entwicklungsreferenz](data/evaluation/nerdscan-clipped-development.json) ·
  [Auswertung](data/evaluation/README.md#show-the-visible-paper-sides-in-a-clipped-overlap)
- **Derselbe Scan um 180° gedreht.** Die Erkennung trifft zwei von drei
  Abzügen und liefert vier zusätzliche Vorschläge. Drei Vorschläge
  überdecken Teile desselben randnahen Abzugs. Die Anzahlprüfung
  zeigt seine zwei sichtbaren Papierseiten und verknüpft damit die
  drei Vorschläge. Falsche Teilvorschläge lassen sich einzeln
  entfernen. Bei künstlich getrennten Abzügen auf demselben Scan
  bleiben die beiden Einzelmasken erhalten.
  [Original und Gegenproben](data/evaluation/README.md#review-a-shared-capture-corner-and-preserve-separate-prints)

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

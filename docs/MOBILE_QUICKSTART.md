# PhoSca Mobile – Kurzstart

## Start auf dem PC

In PowerShell im Projektordner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_mobile.ps1
```

Das Skript startet Backend und HTTPS-Frontend, aktiviert automatische Analyse
sowie konservative Enhancement-Kandidaten und zeigt die lokale Smartphone-URL.

Aktuell im lokalen Netz:

```text
https://192.168.254.129:5173/mobile
```

Die Adresse kann sich nach einem Router- oder PC-Neustart ändern. Massgeblich
ist immer die vom Startskript ausgegebene URL.

## Aufnahme

1. Smartphone und PC ins gleiche lokale Netz bringen.
2. Mobile-URL öffnen.
3. Lokales Zertifikat akzeptieren, falls das Gerät danach fragt.
4. `Albumseiten aufnehmen` wählen.
5. Album/Batch auswählen.
6. Ganze Seite mit etwas Rand fotografieren.

Nach der Hashbestätigung erscheint `Auf PC gespeichert, geprüft und zur
Analyse eingeplant`. Bei Netzunterbruch bleibt die Datei in der lokalen
Upload-Warteschlange.

## Fünf-Minuten-Reviews

- `Cropping prüfen` → zuerst Printanzahl, danach einzelne Crop-Grenzen.
- Bei `Crop-Qualität` können die zwei Rahmenecken direkt gezogen und gespeichert
  werden.
- `Enhancement vergleichen` → A/B ist verblindet; `Gleich gut / No-op` ist eine
  ausdrücklich richtige Antwort.
- `Unsicher` ist besser als eine geratene Bewertung.

Jede Antwort wird sofort gespeichert. Ein abgebrochener Task kann später erneut
erscheinen; bereits abgeschlossene Tasks werden aus der Queue entfernt.

## Aktuell vorbereiteter Batch

- Album: `Album 11.7.2026`
- 77 vorhandene Albumseiten verarbeitet;
- 384 Crop-Vorschläge;
- 324 Enhancement-A/B-Paare;
- 60 Crops mit möglichem Motivschnitt gezielt für Review zurückgehalten;
- eine alte Datenbankzeile verweist auf eine nicht mehr vorhandene Datei und
  bleibt nachvollziehbar als fehlgeschlagen markiert.

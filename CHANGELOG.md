# Änderungsprotokoll

## 2.2.2

- Google Translate ohne Schlüssel bündelt kurze Untertitel und prüft ihre Zuordnung vor der Übernahme. Bei HTTP 429 hält die App weitere Google-Anfragen an.
- Das Video Studio beendet hängende Anfragen nach einer Frist und fragt bei Verbindungsfehlern seltener nach dem Auftragsstatus.
- Der Worker wiederholt vorübergehend fehlgeschlagene Uploads kleiner Ergebnisdateien innerhalb der bestehenden Fristen.
- Colab legt Parakeet im gewählten Modellcache ab. Bei mehreren Videos zeigt der Abschlussbericht auch Warnungen einzelner Aufträge an.
- Quellcode, Website und Colab verweisen auf `riemenschneider2015-svg/linguoai-de`. Die Website lässt sich mit korrekten Sprach- und Asset-Pfaden über GitHub Pages veröffentlichen; Colab lädt das Release-ZIP direkt aus diesem Repository.
- Whisper erhält die bereits normalisierten PCM-Audiodaten direkt. Damit entfällt der fehlerhafte Aufruf von `av.open(..., metadata_errors=...)` mit neueren PyAV-Versionen.
- Fehlende Parakeet-Modellordner und ungeeignete Quellsprachen werden vor dem Kopieren der Videos geprüft. Die Meldung erklärt den Wechsel zu Whisper oder die nötige Einrichtung.
- Die Colab-Modellauswahl erklärt, welche Spracherkennung direkt startklar ist und welche einen eigenen Modellordner benötigt.
- Anleitungen erhalten eine gemeinsame Lesebreite, abgestufte Abstände für Abschnitte und Schritte sowie kompaktere Befehls- und Hilfeblöcke.

## 2.2.1

- Dynamische Gradio-Freigabe als Standard; benannte Cloudflare-Tunnel optional mit dokumentiertem Start und Deployment.
- Veröffentlichte Website, Quellcode und Download-Artefakte auf denselben Versionsstand gebracht.
- Lokale Sprachkataloge erhalten und die Kontext-, Timing- und Colab-Verbesserungen aus 2.2.0 zusammengeführt.
- Desktop- und CLI-Standard für API-Wiederholungen auf vier Versuche vereinheitlicht.
- Kontextlimits schließen den Kürzungsmarker ein; nicht zusammenführbare Cues blockieren keine weiteren Timing-Optimierungen.

## 2.2.0

- Colab-Setup und Start in einer einzigen selbstprüfenden Zelle; verhindert fehlende `linguoai`-Imports nach ausgelassenen Zellen.
- Mehrfach-Upload von Videos und sequenzielle Batch-Verarbeitung mit gemeinsamem Fortschritt.
- Optionales Fortsetzen des Batches, wenn ein einzelnes Video fehlschlägt.
- Gemini kann nach konfigurierbaren Pausen und exponentiellen Wartezeiten erneut angefragt werden.
- Bei Gemini-Ausfall kann automatisch und segmentweise auf Google Translate gewechselt werden.
- Standardmäßig vier API-Versuche, 2,5 Sekunden Mindestabstand zwischen Gemini-Anfragen und 2 Sekunden Basis-Retry-Pause.

## 2.1.0

- Gesamttranskript als Übersetzungskontext für Gemini und unterstützte DeepL-SDKs.
- Adaptive Nutzung freier Pausen vor und nach Sprachsegmenten.
- Zeitliche Neuverteilung kleiner benachbarter Segmentgruppen.
- Automatisches Zusammenfassen überlanger Nachbar-Cues vor starker Beschleunigung.
- Optionale inhaltstreue Kürzung problematischer Zieltexte mit Gemini und erneute TTS-Prüfung.
- Bevorzugtes Tempo 1,35x; Notfall-Maximum standardmäßig 2,20x.
- Neue Tests für Kontextübergabe und adaptive Zeitplanung.

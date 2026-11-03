# Allsky360 — Read-only Audit

> Erstellt: 2026-09-11 · Stand: Arbeitsbaum auf `main` (Commit `f610b82`, umfangreiche uncommittete Änderungen)
> Methodik: rein lesender Audit (keine Builds, keine Prozesse, keine Hardwarezugriffe).
> Statuslegende je Befund: `offen` / `in Arbeit` / `erledigt` — bitte beim Abarbeiten pflegen.

## Fortschritt

| ID | Titel | Severity | Status |
|---|---|---|---|
| SEC-01 | Sudoers-Regeln und `sudo`-Gruppe heben Privilegientrennung auf | Critical | **erledigt** |
| SEC-02 | Dienststeuerung per HTTP ohne Authentifizierung | High | **erledigt** |
| SEC-03 | Path Traversal im Static-File-Handler | Medium | **erledigt** |
| SEC-04 | Standort/Betreiber/Infrastruktur unauthentifiziert abrufbar | Medium | **erledigt** |
| BUG-01 | Out-of-Bounds-Lesezugriff auf `frames[3]`/`frames[4]` | High | **erledigt** |
| BUG-02 | Configparser verwirft Keys mit Leerzeichen vor `=` | High | **erledigt** |
| BUG-03 | `continue` in `TIMED_BLOCK` wirkt nicht auf die Schleife | Medium | **erledigt** |
| BUG-04 | `nighttime_sunangle` wirkungslos | Medium | **erledigt** |
| BUG-05 | `signal(SIGCHLD, SIG_IGN)` erzeugt Dauer-Falschfehler | Low | **erledigt** |
| REL-01 | Kein Retention/Cleanup; Hauptlog ungerotiert | High | **erledigt** |
| REL-02 | `make install` läuft nicht durch; Skripte nie installiert | High | **erledigt** |
| REL-03 | Postprocessing/End-of-Night scheitern systematisch | High | **erledigt** |
| REL-04 | Datei-I/O-Fehler per `assert()` → `abort()` | Medium | **erledigt** |
| REL-05 | Nicht-atomare Veröffentlichung von JSON-Dateien | Medium | **erledigt** |
| REL-06 | Nachtverarbeitung geht bei Neustart verloren | Medium | **erledigt** |
| REL-07 | SQLite ohne `busy_timeout` bei 4 Schreibern | Medium | **erledigt** |
| REL-08 | `postprocessing.sh` überschreibt Metadaten mit leerer Datei | Medium | **erledigt** |
| BUG-06 | `decodeURIComponent` beendet bei kaputtem Encoding den API-Prozess | Medium | **erledigt** |
| BUG-07 | 12 `config_t`-Felder ohne Default auf uninitialisiertem Struct | High | **erledigt** |
| BUG-08 | Belichtungsleiter unterschreitet `camera_min_exposure` | Medium | **erledigt** |
| BUG-09 | HDR-Fusion korrigiert die Belichtung doppelt (`exp_gain * c_gain`) | **High** | **erledigt** |
| BUG-10 | Hemisphären-Suffixe `S`/`W` in den Koordinaten werden ignoriert (+ unquotierte Werte am ersten Leerzeichen abgeschnitten) | Medium | **erledigt** |
| REL-09 | `systemctl reload` (SIGHUP ohne Handler) beendet den Kamera-Daemon | Medium | **erledigt** |
| REL-10 | Kamera-Unit wartet auf falschen INDIGO-Unit-Namen | Low | **erledigt** |
| REL-11 | Capture startet, bevor die INDIGO-Verbindung steht | Medium | **erledigt** |
| REL-12 | `make install` überschreibt eine vorhandene `config.ini` | High | **erledigt** |
| SEC-05 | Web-UI sendet Dienststeuerung als GET ohne Token (Knöpfe wirkungslos) | Medium | **erledigt** |
| SEC-06 | `config.ini` mit `api-token` ist für alle lokalen Benutzer lesbar (644) | Medium | **erledigt** |
| BUG-11 | Mount-Dienst in API/UI als `allsky360-mount.service` statt `-mount-position` | Low | **erledigt** |
| BUG-12 | Config-Werte > 99 Zeichen werden stillschweigend abgeschnitten (Kamera, Aurora, Meteo) | Medium | **erledigt** |
| BUG-13 | Makefile mischt CPU- und CUDA-Objekte beim Wechsel von `USE_CUDA` | Low | **erledigt** |
| REL-13 | Kein systemd-Watchdog für hängenden Capture-Loop (`Type=notify`, `WatchdogSec=600`) | Medium | **erledigt** |
| REL-14 | API liest Videos komplett in den RAM, kein HTTP-Range (→ Streaming + Range/ETag, Encoder faststart/CRF) | Medium | **erledigt** |
| SEC-07 | Node 20 ist End-of-Life (April 2026) → Node 24, sqlite3 6, `url.parse` ersetzt | Medium | **erledigt** (auf dem System: `sudo make node-deps`) |
| SEC-08 | API ohne TLS auf `0.0.0.0` als Default | Medium | offen |
| SEC-09 | systemd-Units ohne Sandbox-Härtung, Dienstkonto mit Login-Shell (Score 8.3–9.0 → 1.4–2.3, nologin) | Low | **erledigt** |
| SEC-10 | Dienststeuerung der API nie funktionsfähig: `LockPersonality` impliziert `NoNewPrivileges`, `sudo` verweigert → polkit statt sudoers | Medium | **erledigt** |
| BUG-14 | `allsky360-aurora`: Double-Free bricht ab, sobald ein NOAA-Abruf fehlschlägt | High | **erledigt** |
| INST-02 | Installation vereinfacht: `install.sh` (Preflight, Build als Benutzer, Assistent, Selbsttest), CUDA-Arch/Version automatisch, einheitliche INDIGO-Gerätenamen | Medium | **erledigt** |
| FEAT-01 | RS485/Modbus-Sensoren (Wetter, Wind, Schaltschrank) → `allsky360-sensors`, Tabelle `sensors`, `/api/sensors` | – | **erledigt** |
| REL-15 | `make uninstall` löschte mit `rm -rf $(CONFIG_DIR)/*` das ganze Home von allskyuser (Bilder, Videos, DBs); `UNIT_DIR` undefiniert, `{service,timer}` ohne Brace-Expansion in sh | High | **erledigt** |
| DATA-02 | Tagesdatenbank: Kamera/Sensoren benennen nach UTC-Tag, Meteo/Aurora nach lokalem Tag (fällt nur bei Zeitzone ≠ UTC auf) | Low | offen |
| HW-01 | Windmesser (Adresse 11) liefert in Register 0-3 stets 0, Ultraschall-Wettersensor gleichzeitig 3-5 m/s: blockiert, windgeschützt oder andere Register/Skalierung | – | offen (vor Ort prüfen) |
| DATA-01 | NOAA hat `solar-wind/mag-2-hour.json` und `plasma-2-hour.json` abgeschaltet (404), Kp-Index wird nicht mehr gelesen (-1) | Medium | offen |
| BUG-15 | `indigo_cameras -l` listet die Kamera nicht (zeitweise), `indigo_list_cameras` schon | Low | offen |
| INST-01 | YOLO-Setup nutzt JetPack-6.1/Python-3.10-Wheels → `make yolo` (YOLO26, Torch cu130, Python 3.12) | High | **erledigt** (auf dem System: `sudo make yolo`) |

---

# Audit-Zusammenfassung

**Repository-/Laufzeitkontext**

1. C-Hauptanwendung (`src/`, ~40 Dateien + CUDA-Varianten) baut `bin/allsky360-camera`; Kamerazugriff ausschließlich über INDIGO-TCP (kein direktes GPIO/I²C/UART im Code — Hardware-Abstraktion liegt vollständig bei INDIGO).
2. Zielplattform bestätigt NVIDIA Jetson Orin NX (`NVCCFLAGS = -arch=sm_87`, `Makefile:88`), Ubuntu/Debian, systemd; Build CPU-only oder `USE_CUDA=1`.
3. Nebenprozesse: Node-Webserver/API (`allsky-api/index.js`, Port 3000), oneshot-Tools (aurora/meteo/location/mount-position) via systemd-Timer, Shell-Skripte für End-of-Night-Timelapse und AI-Postprocessing via `sudo systemd-run`.
4. Persistenz: JPEG/JSON-Bildarchiv unter `/home/allskyuser/images/YYYYMMDD/`, Videos, tägliche SQLite-DBs `database_YYYYMMDD.db` (gemeinsam von Kamera, aurora, meteo, location beschrieben).
5. Externe Dienste: NOAA SWPC (aurora), Open-Meteo (meteo) per HTTPS/libcurl; optional dump1090-ADS-B aus `/run/dump1090-mutability/aircraft.json`.
6. Zugriffsmodell: **keine Authentifizierung, kein TLS, kein CORS/CSRF-Schutz** in der API; implizit „trusted LAN“ angenommen, im Repo nirgends abgesichert oder dokumentiert.
7. Konfiguration: ein flaches `key=value`-INI (`config.ini`), zur Laufzeit in **jedem** Capture-Zyklus neu geladen und als `config.json` in das Web-Docroot geschrieben.
8. Keine CI, keine Lint-/Format-Gates, `make tests` baut nur 2 von 9 Testdateien und führt nichts aus; Git-Historie sehr kurz (8 Commits).
9. Dokumentation ist ungewöhnlich gut (README, INSTALL, 8 Themen-MDs), **divergiert aber von Code/Installation** (z. B. HDR „bis 5 Belichtungen“ vs. `MAX_FRAMES 3`).

**Review-Abdeckung**

- Vollständig gelesen: `Makefile` (Install/Sudoers/Systemd/Cron), alle `etc/`-Units + Cron + Logrotate + Apache, `allsky-api/index.js`, alle `scripts/*.sh`, `include/allsky.h`, `src/main.c`, `src/config.c`, `src/daily_database.c`, `src/image_metadata.c`, `src/ai_training.c`, `src/sun_calc.c`, `src/dehaze_filter.c`, `config.ini.default`, `package.json`/`package-lock.json`, Web-UI-Sinks (gezielt).
- Gezielt/stichprobenartig: `src/hdr_merge.c`, `src/star_detector.c`, OpenMP-Pragmas aller `src/*.c`, `tools/aurora`, `tools/meteo`, `tools/timelapse`.
- **Bewusst nicht geprüft**: alle `*_cuda.cu`-Kernels (ohne Hardware nicht seriös beurteilbar), Bildqualitäts-Algorithmik jenseits offensichtlicher Speicher-/Threading-Fehler, Frontend-Layout/JS-Logik im Detail, `ai/*.py`, npm-Dependency-CVEs (kein Netzzugriff, keine erfundenen CVEs).

**Wichtigste drei Risiken**

1. **Lokale Root-Eskalation durch die eigene Installation**: `sudo`-Gruppenmitgliedschaft plus `NOPASSWD: /usr/bin/systemd-run` ohne Argumentbeschränkung (SEC-01).
2. **Unauthentifizierte Fernsteuerung privilegierter Dienste** über `/api/services/{start,stop,restart}` (SEC-02), kombiniert mit Path Traversal (SEC-03).
3. **Der unbeaufsichtigte 24/7-Betrieb ist nicht tragfähig**: kein Retention-/Cleanup-Mechanismus, Hauptlog nicht von logrotate erfasst (REL-01), `make install` defekt (REL-02/REL-03).

**Gesamturteil**

Der Bildverarbeitungskern ist ambitioniert und handwerklich überdurchschnittlich (konsequente `allsky_safe_*`-Wrapper, Prepared Statements, TLS-Verify bei NOAA, Service-Allowlist in der API, HTML-Escaping im ADS-B-View). Die **Betriebs- und Deploymentschicht ist dagegen nicht release-reif**: Das Installationsziel läuft nicht durch, mehrere installierte Features sind Dead Code, die Privilegientrennung ist durch die Sudoers-Regeln faktisch aufgehoben, und nichts verhindert, dass das Gerät sich innerhalb von Wochen die Platte füllt. Für ein isoliertes LAN mit manuell nachgezogener Installation betreibbar; als installierbares Produkt oder bei Internet-Exposition **nicht freigabefähig**. Korrekturaufwand für die Top-Risiken: überschaubar (überwiegend S/M).

---

# Architekturkarte

**Komponenten & Prozesse**

| Komponente | Prozess/Unit | Rechte | Verantwortung |
|---|---|---|---|
| `allsky360-camera` | `allsky360-camera.service`, `Restart=always`, `RestartSec=90` | `allskyuser`, **keine** Sandbox-Direktiven | Capture-Loop, HDR-Fusion, Filterpipeline, JPEG/JSON/Thumbnail/Panorama, SQLite-Insert, Anstoß von Postprocessing/End-of-Night |
| Web/API | `allsky360-api.service` | `allskyuser`, `ProtectSystem=full`, `PrivateTmp` | statisches Docroot + JSON-Endpunkte + **systemctl-Steuerung** |
| Fetcher | aurora/meteo/location/mount-position `.service`+`.timer` | `allskyuser`, `NoNewPrivileges`, `ProtectSystem=full` | externe Daten → gleiche Tages-DB + JSON ins Docroot |
| Shell-Jobs | via `sudo systemd-run` aus `main.c` | **root** → Unit als `allskyuser` | `endofnight.sh` (Timelapse), `postprocessing.sh` (YOLO-Wetterklasse → JSON-Patch) |
| INDIGO | `indigo-server.service` | `User=user1`, `PAMName=login` | Kamera-/Mount-Treiber, Port 7624 |
| Cron | `etc/cron.d/allsky360-tools` | `User=allsky` (existiert nicht) | dieselben Fetcher — **redundant zu den Timern** |

**Datenflüsse**

```
INDIGO (TCP 7624, BLOB/URL) ──► raw_buffer ──► indigo_raw_to_rgbf1 ──► frames[0..2]
      ──► HDR-Fusion ──► Filterkette ──► hdr_image ──► JPEG/Thumbnail/Panorama
            ├─► /home/allskyuser/images/YYYYMMDD/*.jpg|*.json   (Docroot via Symlink!)
            ├─► images/latest_image.json, images/config.json    (Docroot)
            ├─► database_YYYYMMDD.db  (gemeinsam mit aurora/meteo/location)
            └─► sudo systemd-run → postprocessing.sh → jq-Patch des Bild-JSON

HTTP-Client ──► Node-API ──► {Tages-DB (readonly), Docroot-Dateien, sudo systemctl}
NOAA/Open-Meteo (HTTPS) ──► Tages-DB + public/*.json
dump1090 (/run/.../aircraft.json) ──► /api/aircraft ──► aircraft.html
config.ini ──► Kamera-Daemon UND Node-API (jeweils eigener Parser!)
```

**Trust Boundaries & privilegierte Operationen**

- **TB1 – HTTP (untrusted, unauthentifiziert)**: `parsed.query.date/type/service`, `pathname`. Senken: SQLite-Dateipfad (reguliert), Dateisystem-Read (**nicht** reguliert, SEC-03), `sudo systemctl` (Allowlist, aber keine Authentifizierung, SEC-02).
- **TB2 – `config.ini`**: von `allskyuser` schreibbar (`-m 644 -o allskyuser`), speist Pfade, Geokoordinaten, Kamera- und Filterparameter in beide Prozesse.
- **TB3 – Kameradaten**: INDIGO-BLOB-Header bestimmt `raw_width/raw_height`; Plausibilisierung nur gegen Puffergröße (`main.c:1285-1298`).
- **TB4 – externe APIs**: NOAA/Open-Meteo-JSON → json-c-Parsing, unbegrenzte Response-Größe.
- **TB5 – Funkdaten**: ADS-B-Felder aus `aircraft.json` → Web-UI (korrekt escaped, `aircraft.html:541-560`).
- **Privilegiert**: `sudo systemctl …` (6 Units), `sudo /usr/bin/systemd-run` **ohne Argumentfilter**, `sudo /opt/allsky360/scripts/allsky360-ai-run.sh *`, Mitgliedschaft in `sudo`/`video`/`render`.

**Annahmen**

- A1: Das Gerät steht in einem vertrauenswürdigen LAN und ist nicht per Port-Forwarding exponiert. Im Repo gibt es dafür keine Absicherung — die Apache-Beispielconfigs enthalten einen auskommentierten Redirect auf eine öffentliche Domain (`etc/apache2/sites-available/indigo-allsky.conf:8`).
- A2: `allskyuser` hat kein Passwort (`useradd -r` ohne `-p`), d. h. die `sudo`-Gruppenmitgliedschaft allein ist nicht direkt nutzbar — die `NOPASSWD`-Regeln sind es sehr wohl.
- A3: Produktive Installationen wurden manuell nachkorrigiert (sonst wären Postprocessing und Timelapse nie gelaufen).
- A4: `npm ci` zieht `mime-types` transitiv über Express ein (deshalb läuft `index.js` trotz fehlender Direktabhängigkeit).

---

# Befunde

## [SEC-01] Sudoers-Regeln und `sudo`-Gruppe heben die Privilegientrennung vollständig auf
- Bereich: Security
- Severity: Critical
- Confidence: High
- Status: bestätigt
- Fundstelle: `Makefile:370` — Target `user`; `Makefile:482-483` — Target `sudoers`; `Makefile:435`
- Evidenz: `usermod -a -G sudo $(USERNAME)` nimmt den Dienstbenutzer in die `sudo`-Gruppe auf. `Makefile:483` schreibt `allskyuser ALL=(root) NOPASSWD: /usr/bin/systemd-run` — **ohne Argumentbeschränkung**. `systemd-run` startet beliebige Kommandos; `sudo systemd-run /bin/sh -c '…'` läuft als root. `Makefile:482` erlaubt zusätzlich `…/allsky360-ai-run.sh *`, und das Skript wird per `Makefile:435` mit `-o allskyuser -m 0644` installiert, d. h. der Dienstbenutzer darf die Datei, die er als root ausführen darf, selbst überschreiben.
- Auswirkung: Jede Kompromittierung von `allskyuser` eskaliert ohne weitere Hürde zu vollständiger Root-Kontrolle. Die Sandbox der Fetcher-Units (`NoNewPrivileges`, `ProtectSystem`) ist dadurch wertlos.
- Angriffsszenario: LAN-Angreifer erreicht Codeausführung als `allskyuser` → `sudo -n systemd-run --property=User=root /bin/sh -c 'cp /bin/bash /tmp/r; chmod u+s /tmp/r'` → persistente Root-Shell. Kein Passwort nötig.
- Empfehlung: `usermod -a -G sudo` entfernen. `systemd-run` aus den Sudoers entfernen, stattdessen dedizierte, vom Dienstbenutzer **nicht** schreibbare Units (`allsky360-endofnight.service`, `allsky360-postprocessing@.service`) per `systemctl start` aus der bestehenden Allowlist anstoßen. Alle root-ausführbaren Skripte `root:root 0755` installieren. Wildcard-Regel entfernen.
- Akzeptanzkriterien: (1) `sudo -l -U allskyuser` listet nur `systemctl {start,stop,restart,is-active,show}` auf die benötigten Units, kein `systemd-run`; (2) `id allskyuser` enthält nicht `sudo`; (3) alle Dateien unter `/opt/allsky360/scripts` sind `root:root` und nicht vom Dienstbenutzer schreibbar; (4) End-of-Night und Postprocessing laufen über die neuen Units.
- Aufwand: M
- Abhängigkeiten/Risiken: Greift in REL-03 ein (`main.c:562`/`main.c:658` muss mit). Bestehende Installationen brauchen Migration von `/etc/sudoers.d/allsky360`.

## [SEC-02] Dienststeuerung per HTTP ohne jede Authentifizierung
- Bereich: Security
- Severity: High
- Confidence: High
- Status: bestätigt
- Fundstelle: `allsky-api/index.js:71-103` — `executeSystemctl()`; `allsky-api/index.js:299-419`; Allowlist `index.js:61-68`; Sudoers `Makefile:476-481`
- Evidenz: `http.createServer` verarbeitet jeden Request ohne Auth-, Token-, Origin- oder Methodenprüfung (`GET` genügt). Der `service`-Parameter wird gegen `ALLOWED_SERVICES` geprüft — Command Injection ist verhindert —, anschließend aber über `sudo -n systemctl <command> <service>` als root ausgeführt. Kein Rate-Limit, kein CSRF-Schutz, kein Loopback-Bind (`index.js:460`).
- Auswirkung: Jeder im LAN kann den Kamera-Daemon dauerhaft stoppen (Totalausfall, Datenverlust für die Nacht). Per `<img src>`/`fetch` auch als CSRF durch den Browser eines Nutzers im gleichen Netz auslösbar.
- Angriffsszenario: `curl http://<host>:3000/api/services/stop?service=allsky360-camera.service` → Aufnahme endet, `Restart=always` greift nicht.
- Empfehlung: Schreiboperationen auf `POST` beschränken, ein in `config.ini` hinterlegtes Shared Secret als Header verlangen, `server.listen(PORT, BIND_ADDR)` konfigurierbar machen. Reverse-Proxy mit Auth/TLS in `INSTALL.md` dokumentieren.
- Akzeptanzkriterien: (1) `GET /api/services/stop?...` antwortet 405; (2) `POST` ohne gültiges Secret antwortet 401 und führt kein `sudo` aus; (3) ohne konfiguriertes Secret sind die Schreibendpunkte deaktiviert; (4) die Dashboard-UI funktioniert weiter (nutzt diese Endpunkte derzeit nicht).
- Aufwand: S

## [SEC-03] Path Traversal im Static-File-Handler
- Bereich: Security
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `allsky-api/index.js:106-107`, `allsky-api/index.js:435-455`
- Evidenz: `decodeURIComponent(parsed.pathname)` wird ohne Normalisierung an `path.join(PUBLIC_PATH, pathname)` übergeben. Node normalisiert `req.url` nicht, `url.parse` entfernt keine `..`-Segmente, `decodeURIComponent` erzeugt aus `%2e%2e%2f` wieder `../`. Es folgt **kein** Containment-Check, nur ein Endungs-Allowlist-Filter (Zeile 439-443).
- Auswirkung: Lesender Zugriff auf beliebige Host-Dateien mit den Endungen `.html .js .css .ico .png .jpg .jpeg .json .mp4`, soweit für `allskyuser` lesbar.
- Angriffsszenario: `curl --path-as-is 'http://<host>:3000/../../../../opt/allsky360/allsky-api/index.js'` bzw. `%2e%2e%2f`-kodiert.
- Empfehlung: `path.resolve(PUBLIC_PATH, '.' + pathname)` plus hartes `startsWith(PUBLIC_PATH + path.sep)` vor dem Endungsfilter.
- Akzeptanzkriterien: (1) `/../../<x>.json` liefert 403; (2) reguläre Pfade funktionieren unverändert; (3) `%2e%2e%2f` liefert ebenfalls 403; (4) Symlink-Auflösung in `public/images` bleibt funktionsfähig.
- Aufwand: XS

## [BUG-06] Kaputtes Prozent-Encoding beendet den gesamten API-Prozess
- Bereich: Bug / Security (DoS)
- Severity: Medium
- Confidence: High
- Status: bestätigt (beim Umbau für SEC-03 gefunden)
- Fundstelle: `allsky-api/index.js:106-107` (vor dem Fix) — Request-Handler
- Evidenz: `decodeURIComponent(parsed.pathname)` wirft bei ungültigem Prozent-Encoding synchron einen `URIError`. Der Aufruf steht direkt im `http.createServer`-Callback ohne `try`/`catch` und ohne `process.on('uncaughtException')` — die Ausnahme beendet den Node-Prozess.
- Auswirkung: Ein einzelnes `curl 'http://<host>:3000/%'` beendet den Webserver. `Restart=always`/`RestartSec=3` startet ihn zwar neu, in einer Schleife gesendet ist die API damit aber dauerhaft nicht verfügbar — unauthentifiziert auslösbar.
- Angriffsszenario: `while true; do curl -s 'http://<host>:3000/%' ; done`.
- Empfehlung: `decodeURIComponent` in `try`/`catch` mit 400-Antwort; zusätzlich NUL-Bytes im Pfad ablehnen.
- Akzeptanzkriterien: (1) `/%` liefert 400 und der Prozess lebt weiter; (2) `/foo%00.jpg` liefert 400; (3) reguläre Pfade unverändert.
- Aufwand: XS

## [SEC-04] Standort, Betreibername und interne Infrastruktur unauthentifiziert abrufbar
- Bereich: Security
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/main.c:944-948`; `src/config.c:1248-1280` (`config_to_json`, Felder `location_*`, `indigoserver*`); `allsky-api/index.js:249-260`
- Evidenz: In jedem Capture-Zyklus schreibt der Daemon die komplette Laufzeitkonfiguration als `config.json` nach `images_dir`, das per Symlink im Docroot liegt. Exportiert werden u. a. `location_latitude/longitude/altitude/city/owner`, `indigoservername`, `indigoserver`, `indigoport`, `indigocamera`. `/api/config` gibt die Datei ohne Auth aus.
- Auswirkung: Präzise Geokoordinaten, Klarname des Betreibers, interne IP/Port des INDIGO-Servers und Kameramodell für jeden erreichbaren Client abrufbar.
- Angriffsszenario: `curl http://<host>:3000/api/config`.
- Empfehlung: Allowlist der vom Frontend benötigten Felder (oder separate `config_public.json`); `indigoserver*`/`location_owner` ausschließen, Koordinaten gerundet veröffentlichen.
- Akzeptanzkriterien: (1) `/api/config` ohne `location_owner`/`indigoserver*`; (2) `config.html`/`dashboard.html` weiterhin funktionsfähig; (3) vollständige Konfiguration lokal außerhalb des Docroots verfügbar.
- Aufwand: S

## [BUG-01] Out-of-Bounds-Lesezugriff auf `frames[3]`/`frames[4]` — Root Cause / Multiplikator
- Bereich: Bug
- Severity: High
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/main.c:81-83` (`#define MAX_FRAMES 3`); `src/main.c:2132-2133`, `src/main.c:2137-2138`; `src/hdr_merge.h:24` (`#define MAX_IMAGES 5`)
- Evidenz: Das globale Array hat 3 Elemente; gelesen werden `frames[3]`/`frames[4]` (je 56 Byte jenseits des Endes). Ursache: zwei Konstanten für dieselbe Größe, `MAX_FRAMES 3` vs. `MAX_IMAGES 5`, wobei `hdr_multi_scale_fusion_laplacian_rgbf1(const struct HdrFrames frames[MAX_IMAGES], …)` ein 5-elementiges Array suggeriert. `CLAUDE.md` dokumentiert `MAX_FRAMES = 5`.
- Auswirkung: Undefiniertes Verhalten bei jedem Bild; benachbarte Globals werden als Float/Double reinterpretiert und als `exposure_t3/t4`, `sigma_noise_t3/t4` ins Archiv geschrieben. Bei geänderter Symbolanordnung jederzeit eskalierbar.
- Empfehlung: Eine einzige Konstante, `_Static_assert` für Array-Dimension vs. Metadatenfelder; nicht aufgenommene Frames explizit auf 0.
- Akzeptanzkriterien: (1) nur eine Definition im Baum; (2) ASan-Dryrun ohne OOB-Read; (3) `image-*.json` liefert reproduzierbar `0`; (4) Doku stimmt mit Code überein.
- Aufwand: S
- Abhängigkeiten/Risiken: Erhöhung auf 5 Frames kostet zwei weitere Vollbildpuffer (~80 MB je bei 2280×2180 RGBA-Float).

## [BUG-02] Configparser verwirft Schlüssel mit Leerzeichen vor dem `=` stillschweigend
- Bereich: Bug
- Severity: High
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/config.c:220-229`; `config.ini.default:276-280`
- Evidenz: `sscanf(trim_line, "%49[^=] = %99s", key, value)` — `%49[^=]` konsumiert auch das Leerzeichen vor `=`. Bei `clarity_filter_strength = 0.35` lautet `key` `"clarity_filter_strength "` und scheitert an jedem `strcmp` (`config.c:406-414`). Kein `else`-Zweig → keine Warnung. Genau dieses Muster nutzt die ausgelieferte `config.ini.default` in Zeile 276-280. Analog verpuffen `camara_cooling_temperature` (Zeile 46) und die zerschossene Zeile `image_zenith=y=1185` (Zeile 71) → `image_zenith_y` bleibt auf Default 500 (`config.c:164`).
- Auswirkung: Dokumentierte Parameter ohne Wirkung; stille Defaults (z. B. `clarity_filter_radius` 5 statt 24). `show_config()` zeigt die Defaults und suggeriert Korrektheit.
- Empfehlung: Key trimmen, `else`-Zweig mit Warnung für unbekannte Keys, `config.ini.default` normalisieren und Zeilen 46/71 korrigieren.
- Akzeptanzkriterien: (1) `clarity_filter_radius = 24` kommt als 24 an; (2) falscher Key erzeugt genau eine Warnung pro Ladevorgang; (3) `image_zenith_y` kommt korrekt an; (4) kein Regress bei `key=value`.
- Aufwand: XS
- Abhängigkeiten/Risiken: Bisher inaktive Filterparameter werden wirksam → Bildwirkung ändert sich. Vorher/Nachher-Vergleich einplanen.

## [BUG-07] Zwölf `config_t`-Felder ohne Default auf einem uninitialisierten Struct
- Bereich: Bug
- Severity: High
- Confidence: High
- Status: bestätigt (beim Umbau für BUG-02 gefunden)
- Fundstelle: `src/main.c:706` — `config_t config;`; `src/config.c` Default-Block (Zeilen 30-219); Nutzung `src/main.c:1485` und `src/main.c:1219-1221`; `src/config.c:1270-1271`
- Evidenz: `config_t config;` ist eine nicht initialisierte lokale Variable. `load_config()` setzt Defaults für 155 der 167 Strukturfelder — zwölf bleiben ohne Zuweisung: `crop_image`, `crop_width`, `crop_height`, `crop_x_offset`, `crop_y_offset`, `daytime_capture`, `daytime_meanbrightness`, `nighttime_meanbrightness`, `nighttime_fullmoongain`, `rotate_image`, `measure_brightness_area`, `indigoservername`. Fehlt der zugehörige Schlüssel in `config.ini`, enthält das Feld Stack-Müll. Zwei davon sind mit der ausgelieferten Konfiguration **garantiert** betroffen:
  - `measure_brightness_area`: Die Datei setzte `autoexposure_measure_radius` — einen Schlüssel, den der Parser nicht kennt (BUG-02). Der Wert geht bei `main.c:1485` als `(int)(measure_brightness_area * image_horizon_radius)` in den Messradius der Auto-Exposure ein.
  - `indigoservername`: wird nur durch den Schlüssel `indigo_servername` gesetzt (`config.c:595`), der in keiner mitgelieferten Konfiguration vorkommt. Er geht als erstes Argument in `indigo_connect_server()` (`main.c:1219-1221`) und zusätzlich in `json_object_new_string()` (`config.c:1270`) — dort wird bis zum ersten NUL gelesen, notfalls über die 100-Byte-Grenze hinaus.
- Auswirkung: Der Messradius der Belichtungsregelung ist undefiniert — er kann negativ, null oder größer als das Bild sein und schwankt zwischen Prozessstarts. Damit ist die Belichtungssteuerung jedes Bildes betroffen. Der INDIGO-Servername ist ein undefinierter String im Verbindungsaufbau und im veröffentlichten `config.json`.
- Empfehlung: `memset(config, 0, sizeof(*config))` zu Beginn von `load_config()` plus explizite Defaults für die zwölf Felder.
- Akzeptanzkriterien: (1) Eine Konfiguration, die nur `image_directory` und `database_directory` enthält, liefert für alle 167 Felder definierte Werte; (2) `measure_brightness_area` hat einen dokumentierten Default; (3) `indigoservername` ist ein gültiger, NUL-terminierter String; (4) keine Compiler-Warnungen.
- Aufwand: XS

## [BUG-08] Belichtungsleiter unterschreitet das konfigurierte Kameraminimum
- Bereich: Bug
- Severity: Medium
- Confidence: High
- Status: bestätigt (beim Umbau für BUG-01 gefunden)
- Fundstelle: `src/main.c:98` — `#define MIN_EXPOSURE 0.000032`; `src/main.c:1096-1105`; `src/autoexposure.c:25-35` — `calc_exposure_time()`
- Evidenz: Die HDR-Leiter berechnet `t_i = t0 / 2^(i * ev_steps)` und prüft das Ergebnis gegen die **fest einkompilierte** Konstante `MIN_EXPOSURE` (0,000032 s) statt gegen `config.camera_min_exposure`. Liegt ein Wert dazwischen, wird er unverändert an die Kamera geschickt. Die Absicht war offenbar eine andere: `main.c:1101` vergleicht `frames[i - 1].exposure == config.camera_min_exposure` — dieser Zweig kann nie zutreffen, weil der einzige zuweisende Pfad `frames[i].exposure = MIN_EXPOSURE` setzt. Zusätzlich nimmt `calc_exposure_time(float t0, int ev_steps, int index)` einen **int** entgegen, während `config.*_hdr_exposure_factor` ein `double` ist: 2.0 → 2, 5.0 → 5, ein Wert wie 1.5 würde stillschweigend zu 1.
- Auswirkung: Die Kamera wird um Belichtungszeiten gebeten, die die Konfiguration selbst als nicht unterstützt ausweist. Je nach Treiber wird still geklemmt (die Belichtungsreihe hat dann zwei praktisch identische Frames, die Fusion gewichtet Rauschen statt Dynamik), ein Fehler gemeldet oder gar nicht ausgelöst — im letzten Fall greift der `indigo_capture_timeout` und die gesamte Serie wird verworfen. Mit den produktiven Konfigurationen (`camera_min_exposure=0.0007`, `daytime_hdr_exposure_factor=5.0`) trifft das **heute jede Tages-HDR-Serie**: Frame 1 wird mit 0,000141 s angefordert, also dem Fünffachen unter dem Minimum.
- Empfehlung: Untergrenze auf `fmax(config.camera_min_exposure, MIN_EXPOSURE)` ziehen und die Leiter abbrechen, sobald sie darunter fällt; `ev_steps` in `calc_exposure_time()` auf `double`/`float` ändern oder die Ganzzahl-Abschneidung dokumentieren und beim Laden der Konfiguration prüfen.
- Akzeptanzkriterien: (1) Keine angeforderte Belichtung liegt unter `camera_min_exposure`; (2) die Serie endet sauber mit Meldung statt mit einem Kamera-Timeout; (3) ein `hdr_exposure_factor` von 1.5 wird entweder korrekt angewandt oder beim Laden mit Warnung abgelehnt; (4) Nacht- und Tagesleiter sind mit den produktiven Werten dokumentiert nachgerechnet.
- Aufwand: S
- Abhängigkeiten/Risiken: Ändert, wie viele Frames tatsächlich aufgenommen werden — zusammen mit der Wahl von `hdr_exposure_factor` zu entscheiden.

## [BUG-09] HDR-Fusion korrigiert die Belichtungsdifferenz doppelt
- Bereich: Bug
- Severity: High
- Confidence: High
- Status: bestätigt (Symptom vom Betreiber unabhängig bestätigt: „Rauschen statt Dynamik gewichtet")
- Fundstelle: `src/hdr_merge.c:430` und `:443` — Berechnung der Gains; `src/hdr_merge.c:518-520` — linearisierte Luma; `src/hdr_merge.c:298` — `build_gaussian_pyramid_channel()`, Level 0 der Ausgabepyramide; Aufrufe `src/hdr_merge.c:933-941`
- Evidenz: Es existieren zwei Gains, die **dieselbe physikalische Größe** ausdrücken:
  - `exp_gain[k] = frames[0].exposure / frames[k].exposure` — das Belichtungsverhältnis aus den Sollzeiten (`:430`).
  - `r_gain/g_gain/b_gain[k] = median_X0 / median_Xk` — das *gemessene* Helligkeitsverhältnis derselben zwei Frames (`:441-444`). Bei linearem Sensor gilt `median_Xk ≈ median_X0 · (t_k/t_0)`, also `X_gain[k] ≈ exp_gain[k]`.
  Beide werden multipliziert: `linG = px[1] * exp_gain[k] * g_gain[k]` (`:519`) und, entscheidend für das Ergebnis, `pyramid->data[0][i] = px[channel] * exp_g * c_gain` (`:298`) — das ist Level 0 der Pyramiden, aus denen das fusionierte Bild entsteht. Die Korrektur wird damit quadriert: `px · (t0/tk)²` statt `px · (t0/tk)`.
  Nachgerechnet (linearer Sensor, neutrale Szene, `factor 2.0`, korrekter Wert wäre überall 0,40):

  | Frame | exp_gain | g_gain | linG | Faktor zu hell |
  |---|---|---|---|---|
  | 0 | 1 | 1 | 0,40 | — |
  | 1 | 4 | 4 | 1,60 | 4× |
  | 2 | 16 | 16 | 6,40 | 16× |
  | 3 | 64 | 64 | 25,6 | 64× |
  | 4 | 256 | 256 | 102,4 | 256× |

  Dass hier zwei Bedeutungen vermischt wurden, zeigt auch die Inkonsistenz an `:719-721`: für die **Gewichtsberechnung** wird bewusst nur `exp_gain` verwendet („not color-gain-adjusted"), für die **Bilddaten** dagegen beides. Ergänzend ist die Kompensation `median_X / channel_scale_X` (`:434-440`) rechnerisch wirkungslos: Zähler und Nenner werden durch denselben Faktor geteilt, er kürzt sich im Quotienten heraus.
- Auswirkung: Das Referenzbild (k=0) wird korrekt skaliert, alle kürzeren Belichtungen um genau ihr eigenes Belichtungsverhältnis zu hell. Die Gewichtskarten sind dagegen korrekt (nur `exp_gain`) und ziehen die Lichter weiterhin aus den kurzen Frames — die kommen dadurch massiv überhöht in die Fusion. Das nachgelagerte Tone-Mapping bzw. `autostretch` komprimiert das Gesamtbild entsprechend, wodurch die Schattenpartien aus dem langen Frame in Richtung Rauschgrenze gedrückt werden. Genau das beschriebene Symptom: der Dynamikgewinn der Belichtungsreihe geht verloren, das Rauschen des langen Frames dominiert. Der Effekt wächst mit jedem zusätzlichen Frame und ist damit nach BUG-01 (5 Belichtungen) stärker als vorher.
- Empfehlung: Die beiden Größen sauber trennen — `exp_gain` trägt die Helligkeit, die Kanal-Gains nur noch die *Farbdrift* zwischen den Frames (auf Grün normiert, `g_gain ≡ 1`). Details und Alternativen siehe „Runde 4".
- Akzeptanzkriterien: (1) Bei linearem Sensor und konstanter Szene liefert jeder Frame nach Linearisierung denselben Luma-Wert (Abweichung < 2 %); (2) `g_gain[k] == 1` für alle k; (3) die wirkungslose `channel_scale`-Division ist entfernt oder durch eine wirksame ersetzt; (4) ein HDR-Testsatz (`--dryrun --img0..--img4` mit den vorhandenen `.raw`-Dateien) zeigt gegenüber vorher mehr Schattendetail bei gleicher Lichterzeichnung.
- Aufwand: S–M
- Abhängigkeiten/Risiken: Verändert das Bildergebnis deutlich. `hdr_y_max_expected`, `hdr_weight_sigma` und `autostretch` sind gegen das bisherige (falsche) Helligkeitsniveau eingestellt und müssen danach neu justiert werden.

## [REL-01] Kein Retention-/Cleanup-Mechanismus und ungerotiertes Hauptlog
- Bereich: Reliability
- Severity: High
- Confidence: High
- Status: bestätigt
- Fundstelle: Negativbefund über `scripts/`, `src/*.c`, `Makefile`, `etc/`; `etc/systemd/system/allsky360-camera.service:22-23`; `etc/logrotate.d/allsky360:1`
- Evidenz: Nirgends existiert Code oder ein Job, der alte Bilder, Thumbnails, Panoramen, JSONs, Debug-Bilder, Raw-Dumps (~16 MB je), Videos oder Tages-DBs löscht. Die Kamera-Unit schreibt alle Ausgaben nach `/var/log/allsky360.log`, logrotate erfasst aber nur `/var/log/allsky360/*.log` (anderer Pfad). Der Daemon läuft mit `-v` und loggt mehrere Zeilen pro Frame.
- Auswirkung: Bei `capture_interval=120` ~700 Serien/Tag, mit `hdr_save_exposures=1` mehrere GB/Tag. Volle Platte → JPEG-Writes, SQLite-Inserts schlagen fehl und der Daemon stirbt per `abort()` (REL-04).
- Empfehlung: (a) Retention-Timer mit Aufbewahrungszeiten je Datenklasse plus Mindest-Freiplatz-Wächter; (b) Logpfad auf `/var/log/allsky360/camera.log`; `hdr_save_exposures`/`debug_raw_exposures` im Default aus.
- Akzeptanzkriterien: (1) Löschjob trifft genau die Daten jenseits der Grenze; (2) `logrotate -d` listet das Kameralog; (3) bei zu wenig Platz klare Meldung statt Abbruch; (4) Grenzen in `config.ini` dokumentiert.
- Aufwand: M
- Abhängigkeiten/Risiken: Löschlogik muss Symlink-Struktur und laufende Timelapse-Jobs respektieren.

## [REL-02] `make install` läuft nicht durch; Shell-Skripte werden nie installiert; Cron defekt/doppelt
- Bereich: Reliability
- Severity: High
- Confidence: High
- Status: bestätigt
- Fundstelle: `Makefile:503`, `Makefile:521`, `Makefile:517`, `Makefile:627`, `Makefile:425-441`; `etc/cron.d/allsky360-tools:7-9`
- Evidenz: `etc/systemd/system/allsky360-mount-position.server` fehlt (Datei heißt `.service`) → `systemd`-Target bricht ab. `etc/cron.d/allsky360-tools.cron` fehlt (Datei ohne `.cron`) → `cron`-Target bricht ab. `scripts/allsky360-ai-run.sh` fehlt, wird aber installiert und in Sudoers referenziert. `Makefile:517` aktiviert `allsky360-mount-position` ohne Unit-Suffix. Target `shell-scripts` wird von `install` nie aufgerufen → `endofnight.sh`, `postprocessing.sh`, beide `-run.sh`-Wrapper und `ai/*.py` landen nie unter `/opt/allsky360/`. Die Cron-Datei nutzt User `allsky` (existiert nicht), `/usr/local/bin/…`, `/etc/allsky360/config.ini` und `/run/allsky360` (nie angelegt) — und dupliziert die Timer.
- Auswirkung: Installation nach `INSTALL.md` nicht reproduzierbar; Postprocessing und Timelapse strukturell unmöglich (REL-03); teilinstallierte Zustände schwer zu diagnostizieren.
- Empfehlung: Dateinamen angleichen, `shell-scripts` in die `install`-Kette, `allsky360-ai-run.sh` ergänzen oder alle Referenzen entfernen, `RUN_DIR`/`LOG_DIR` anlegen, Cron-Datei löschen, Skripte mit Executable-Bit installieren.
- Akzeptanzkriterien: (1) `sudo make install` läuft in sauberer VM durch; (2) alle referenzierten Pfade existieren danach; (3) `systemctl list-timers` zeigt alle Timer; (4) keine zweite Schedulingquelle.
- Aufwand: M

## [REL-03] Postprocessing und End-of-Night scheitern systematisch
- Bereich: Reliability
- Severity: High
- Confidence: High
- Status: bestätigt (Sudoers-Lücke), wahrscheinlich (Umgebungsverlust via `env_reset`, auf Debian/Ubuntu Standard)
- Fundstelle: `src/main.c:553-566`, `src/main.c:650-663`; `Makefile:476-483`; `scripts/allsky360-postprocessing-run.sh:5-22`; `scripts/postprocessing.sh:59-61`
- Evidenz: Beide Funktionen rufen `execl("/usr/bin/sudo", "sudo", "/opt/allsky360/scripts/allsky360-…-run.sh", …)`. Die Sudoers decken nur `systemctl`, `allsky360-ai-run.sh` und `systemd-run` ab — **nicht** diese Wrapper. `sudo` wird ohne `-n` aufgerufen, während stdio bereits auf `/dev/null` liegt → stummer Fehlschlag. Selbst mit Regel scheitert die Parameterübergabe: Werte kommen per `setenv("ALLSKY360_…")`, `sudo` setzt die Umgebung per `env_reset` zurück → `postprocessing.sh:59` greift mit leerem Pfad auf `/latest_image.json` zu. Zusätzlich fehlt `scripts/mask.jpg` (erwartet in `postprocessing.sh:73`).
- Auswirkung: „AI-Wetterklassifikation“ und „End-of-Night-Timelapse“ sind funktionslos; je Zyklus zwei Fork-Paare ohne Nutzen plus Fehlermeldung (verschärft durch BUG-05).
- Empfehlung: Zusammen mit SEC-01: Wrapper durch dedizierte Units ersetzen (`allsky360-endofnight.service`, `allsky360-postprocessing@.service` mit Instanzparameter statt Umgebungsvariablen), Anstoß per `sudo -n systemctl start`.
- Akzeptanzkriterien: (1) erfolgreicher Lauf im Postprocessing-Log mit korrektem Bildpfad; (2) Timelapse-MP4 beim Tagesübergang; (3) kein `systemd-run` in den Sudoers; (4) keine Fehlermeldungen mehr in diesem Pfad.
- Aufwand: M
- Abhängigkeiten/Risiken: Abhängig von SEC-01 und REL-02; Bildpfad als Unit-Instanzname escapen (`systemd-escape`).

## [BUG-03] `continue` innerhalb des `TIMED_BLOCK`-Makros wirkt nicht auf die Capture-Schleife
- Bereich: Bug
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `include/allsky.h:46-56`; `src/main.c:1314-1326`
- Evidenz: `TIMED_BLOCK` expandiert zu `do { … } while (0)`. Ein `continue` springt zur Bedingung dieser inneren Schleife und setzt die Ausführung **hinter dem Makro** fort, statt die `for`-Iteration zu überspringen. Der Fehlerpfad setzt `abort_captures = 1; failed_no++; continue;`, danach laufen Maskierung, Weißabgleich, Overexposure-Check und `use_max_images++` (`main.c:1333-1414`) auf dem fehlgeschlagen konvertierten, aus `realloc` stammenden (uninitialisierten) Puffer. Einzige Stelle dieser Art im Baum.
- Auswirkung: Filterkette auf uninitialisiertem Heap; NaN/Inf können Folgefilter stören oder zum Absturz führen. Serie wird danach ohnehin verworfen — der Aufwand ist verloren, die Ursache im Log überdeckt.
- Empfehlung: Rückgabewert in lokale Variable, Fehlerbehandlung **nach** dem `TIMED_BLOCK`; Makro so umbauen, dass `continue`/`break` darin nicht kompilieren.
- Akzeptanzkriterien: (1) bei erzwungenem Konvertierungsfehler wird keine Filterfunktion mehr aufgerufen; (2) `use_max_images` bleibt unverändert; (3) genau eine Fehlermeldung pro Frame; (4) kein weiteres `continue`/`break` in `TIMED_BLOCK`.
- Aufwand: XS

## [REL-04] Datei-I/O-Fehler werden per `assert()` behandelt
- Bereich: Reliability
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `include/allsky.h:176-203`; `Makefile:21` (`CFLAGS` ohne `-DNDEBUG`)
- Evidenz: `allsky_safe_fopen` endet bei `fopen == NULL` in `assert()`, `allsky_safe_fwrite` in `assert(result == count)`, `allsky_safe_fclose` in `assert(status == 0)`. Ohne `NDEBUG` sind die Asserts aktiv → `SIGABRT`. Die Speicher-Wrapper (`allsky.h:60-107`) wurden bereits auf explizites `abort()` mit Meldung umgestellt, die Datei-Wrapper nicht.
- Auswirkung: Volle Platte oder Rechtefehler beendet den Daemon hart; `Restart=always`/`RestartSec=90` erzeugt eine Crashloop, in der auch Tagesübergänge verpasst werden (REL-06). Mit `-DNDEBUG` kippt es ins andere Extrem: stillschweigend abgeschnittene JPEGs.
- Empfehlung: Die vier Datei-Wrapper auf Rückgabewerte + `errno`-Logging umstellen; kein `NDEBUG` vor diesem Umbau.
- Akzeptanzkriterien: (1) auf vollem tmpfs überspringt der Daemon den Zyklus und läuft weiter; (2) kein `SIGABRT` im Journal; (3) keine Asserts mehr in I/O-Pfaden; (4) Erfolgsfall unverändert.
- Aufwand: S

## [REL-05] Nicht-atomare Veröffentlichung von `latest_image.json`, `config.json` und Bild-Metadaten
- Bereich: Reliability
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/main.c:2186-2198`; `src/main.c:944-948`; `src/image_metadata.c:85-93`; Leser: `allsky-api/index.js:249-274`
- Evidenz: Alle drei Dateien werden mit `fopen(path, "w")` direkt am Zielpfad geschrieben — ohne Temporärdatei, ohne `rename()`, ohne `fsync()`. Der Bildpfad nutzt korrekt `allsky_safe_rename` (`main.c:2049`), das Muster war also bekannt. Leser: API, Dashboard im Sekundentakt, `scripts/postprocessing.sh:59-60`.
- Auswirkung: Leser können leere/abgeschnittene Dateien erhalten; `postprocessing.sh` berechnet daraus falsche Pfade. Bei Stromausfall bleibt eine korrupte `latest_image.json` zurück. Ergänzend: `latest_image.jpg` wird direkt nach dem Schreiben ins Archiv umbenannt (`main.c:2049`) und existiert praktisch nie.
- Empfehlung: `write_json_atomic(path, str)` — tmp schreiben, `fflush`+`fsync`, `fclose`, `rename()`.
- Akzeptanzkriterien: (1) Dauer-Leseloop über 50 Zyklen ohne Parse-Fehler; (2) nach hartem Reset entweder alte oder neue Version vollständig; (3) keine `*.tmp`-Reste; (4) Format unverändert.
- Aufwand: S
- Abhängigkeiten/Risiken: Temp-Datei im Zielverzeichnis (gleiches Dateisystem), sonst `EXDEV`.

## [REL-06] End-of-Night hängt an einem In-Process-Übergang und geht bei Neustart verloren
- Bereich: Design
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/main.c:876-879`, `src/main.c:950-962`; `etc/systemd/system/allsky360-camera.service:25-26`
- Evidenz: Der Tag/Nacht-Zustand lebt nur als lokale Variable; ausgelöst wird bei `was_night && !is_night`. Beim Start wird `is_night` frisch berechnet, es gibt keinen persistenten Zustand. Die Unit startet bei jedem Absturz neu.
- Auswirkung: Fällt der Prozess über den Dämmerungsübergang aus (auch durch `/api/services/restart`, SEC-02), entfällt die komplette Nachtverarbeitung ohne Nachholmechanismus.
- Empfehlung: Letzten verarbeiteten Nachtstempel persistieren und beim Start nachholen; alternativ idempotenter Timer.
- Akzeptanzkriterien: (1) nach `kill -9` während des Übergangs holt der nächste Start die Timelapse nach; (2) zweiter Start löst keinen Doppellauf aus; (3) Zustand überlebt Reboot; (4) ungestörter Lauf unverändert.
- Aufwand: M
- Abhängigkeiten/Risiken: Setzt REL-03 voraus.

## [REL-07] SQLite ohne `busy_timeout` bei vier konkurrierenden Schreibern
- Bereich: Reliability
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/daily_database.c:85-131`; `src/main.c:2216-2227`; `etc/systemd/system/allsky360-{aurora,meteo,location}.service`
- Evidenz: `insert_measurement` öffnet die DB, setzt `PRAGMA journal_mode=WAL`, schreibt und schließt — ohne `sqlite3_busy_timeout()` und ohne Retry. Aurora (5 min), Meteo und Location (10 min) schreiben in dieselbe Datei. `SQLITE_BUSY` → `return 3`, Aufrufer loggt nur „SQL insert failed“.
- Auswirkung: Sporadisch verlorene Messzeilen; Bild existiert, DB-Zeile fehlt → schwer erkennbare Inkonsistenz. Das Setzen von `journal_mode` ist selbst BUSY-anfällig und bricht dann vor dem Insert ab (`daily_database.c:96-100`).
- Empfehlung: `sqlite3_busy_timeout(db, 5000)` nach jedem Open, `journal_mode=WAL` nur in `create_daily_database()`, begrenzter Retry. Gleiches in den Fetcher-Tools.
- Akzeptanzkriterien: (1) Lasttest mit vier Prozessen verliert über 200 Iterationen keine Zeile; (2) `journal_mode` genau einmal pro Tag; (3) Lock-Konflikt erzeugt Warnung mit Retry-Zähler; (4) API-Lesezugriffe unbeeinträchtigt.
- Aufwand: S

## [BUG-04] `nighttime_sunangle` ist konfigurierbar, aber wirkungslos
- Bereich: Bug
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/sun_calc.c:58-88` (Schwelle hart `-6.0`, Zeile 81); Aufrufe `src/main.c:876`, `src/main.c:953`; `src/config.c:255-256`, `:642`, `:929-930`; `config.ini.default:120-121`
- Evidenz: Der Parameter wird geparst, angezeigt und nach `config.json` exportiert — aber nie gelesen. Das Umschalten nutzt `is_sun_below_6_degrees()` mit Literal `-6.0`.
- Auswirkung: Der dokumentierte Hauptschalter für den Umschaltzeitpunkt (Belichtungsregime, Nachtfilter, Anstoß der Nachtverarbeitung) hat keine Wirkung; das Frontend zeigt den Wert an und verstärkt die Fehlannahme.
- Empfehlung: `is_sun_below_angle(time_t, lon, lat, double angle_deg)` einführen, `config.nighttime_sunangle` übergeben, auf −18…0 klemmen, alte Funktionen als Wrapper erhalten (für `test/test_sun_calc.c`).
- Akzeptanzkriterien: (1) `-12` schaltet nachweislich später; (2) ohne Eintrag Verhalten wie heute; (3) unplausible Werte werden geklemmt + Warnung; (4) `bin/test_sun_calc` läuft.
- Aufwand: XS
- Abhängigkeiten/Risiken: Setzt BUG-02 voraus.

## [BUG-05] `signal(SIGCHLD, SIG_IGN)` macht jedes `waitpid` fehlschlagen
- Bereich: Bug
- Severity: Low
- Confidence: High
- Status: bestätigt
- Fundstelle: `src/main.c:829`; `src/main.c:511-516`, `src/main.c:602-609`; Aufrufer `src/main.c:958-961`, `src/main.c:2230-2233`
- Evidenz: Mit `SIGCHLD` auf `SIG_IGN` reapt der Kernel automatisch; `waitpid()` schlägt mit `ECHILD` fehl. Beide Spawner prüfen `if (waitpid(...) < 0) return 3;` → liefern **immer** 3. Aufrufer loggen „ERROR: Failed to …“, direkt gefolgt von der Erfolgsmeldung.
- Auswirkung: Pro Bild eine irreführende Fehlerzeile neben einer Erfolgsmeldung — verdeckt echte Ausfälle (REL-03) und belastet die Logrotation (REL-01).
- Empfehlung: `ECHILD` als Erfolg behandeln oder `signal(SIGCHLD, SIG_IGN)` entfernen (der Doppelfork verhindert Zombies bereits); „gestartet“ vs. „erfolgreich“ im Log trennen.
- Akzeptanzkriterien: (1) erfolgreicher Spawn ohne Fehlerzeile; (2) erzwungener Fork-Fehler erzeugt genau eine; (3) keine Zombies nach 20 Zyklen; (4) Rückgabecodes korrekt dokumentiert.
- Aufwand: XS

## [REL-08] `postprocessing.sh` überschreibt die Bild-Metadaten mit einer leeren Datei
- Bereich: Reliability
- Severity: Medium
- Confidence: High
- Status: bestätigt
- Fundstelle: `scripts/postprocessing.sh:127-140`; ergänzend `:72` und fehlendes `set -euo pipefail`
- Evidenz: `stat -c` ohne Existenzprüfung; `jq … "$JSON" > "$TMP_JSON"` **ohne** Exitcode-Prüfung, danach `mv -f "$TMP_JSON" "$JSON"`. Schlägt `jq` fehl (korruptes JSON, REL-05-Teilschreibung, `tonumber`-Fehler), ist `TMP_JSON` leer und ersetzt die vollständigen Metadaten durch 0 Byte. Ohne `set -e` läuft das Skript weiter und loggt Erfolg. Zeile 72 nutzt `mktemp --dry-run` in `/tmp` (Unit ohne `PrivateTmp`) → vorhersagbarer, nicht reservierter Pfad (TOCTOU, für sich Low).
- Auswirkung: Irreversibler Verlust der Aufnahmemetadaten (Belichtungen, Rauschwerte, Sternzahl, SQM, Mond/Sonne) — gehäuft genau dann, wenn ohnehin etwas schiefläuft.
- Empfehlung: `set -euo pipefail`, Existenz von `$JSON` prüfen, `jq`-Exitcode und Mindestgröße verifizieren, erst dann `mv`; `mktemp` ohne `--dry-run`, `PrivateTmp=true` in der Unit.
- Akzeptanzkriterien: (1) bei korruptem Input bleibt das Original unverändert, Exitcode ≠ 0; (2) im Erfolgsfall enthält die Datei `weather`+`weather_score` und alle Originalfelder; (3) Eigentümer/Modus unverändert; (4) kein `--dry-run`-mktemp mehr im Baum.
- Aufwand: S
- Abhängigkeiten/Risiken: Wird erst relevant, wenn REL-03 behoben ist.

---

# Priorisierter Maßnahmenplan

| Priorität | Ticket-Titel | Befund-IDs | Ziel | Aufwand | Risiko bei Nichtumsetzung | Abhängigkeiten |
|---|---|---|---|---|---|---|
| 1 | Sudoers härten, `sudo`-Gruppe entfernen, root-Skripte schreibschützen | SEC-01 | Privilegientrennung wiederherstellen | M | Triviale Root-Eskalation | Koordination mit REL-03 |
| 2 | API absichern: Auth + POST + konfigurierbarer Bind | SEC-02 | Keine unauthentifizierte Dienststeuerung | S | LAN-weiter Not-Aus der Kamera, CSRF | — |
| 3 | Path Traversal schließen + Docroot-Containment | SEC-03 | Lesezugriff strikt auf `public/` | XS | Auslesen beliebiger lesbarer Host-Dateien | — |
| 4 | Retention/Cleanup + korrekter Logpfad | REL-01 | 24/7-Betrieb ohne Volllaufen | M | Planbarer Geräteausfall | REL-04 hilfreich vorher |
| 5 | Install-Pipeline reparieren | REL-02 | `sudo make install` läuft durch | M | Nicht reproduzierbare Installation | nach SEC-01 |
| 6 | Postprocessing/End-of-Night über dedizierte Units | REL-03, SEC-01 | Features funktionieren; `systemd-run` entfällt | M | Zwei Kernfeatures funktionslos | SEC-01, REL-02 |
| 7 | Frame-Konstanten vereinheitlichen | BUG-01 | Kein UB; korrekte Metadaten | S | UB pro Bild, falsche Archivdaten | — |
| 8 | Configparser robust + Default-Config korrigieren | BUG-02, BUG-04 | Konfiguration wirkt wie dokumentiert | XS | Stille Fehlkonfiguration | BUG-04 baut darauf auf |
| 9 | Datei-I/O-Fehler statt `assert()` | REL-04 | Kein `abort()` bei ENOSPC | S | Crashloop bei voller Platte | — |
| 10 | Atomare JSON-Veröffentlichung | REL-05 | Leser sehen nie halbe Dateien | S | UI-Fehler, korrupte JSON | — |
| 11 | Nachtverarbeitung idempotent/neustartfest | REL-06 | Keine verlorene Nacht | M | Ausfall ganzer Nächte | REL-03 |
| 12 | SQLite `busy_timeout` + Retry | REL-07 | Keine verlorenen Messzeilen | S | Lücken in der Zeitreihe | — |
| 13 | Postprocessing-Skript gegen Datenverlust härten | REL-08 | Metadaten gehen nie verloren | S | Verlust von Aufnahmemetadaten | REL-03 |
| 14 | `TIMED_BLOCK` fehlersicher machen | BUG-03 | Fehlerpfade brechen wirklich ab | XS | Filterkette auf uninit. Speicher | — |
| 15 | `/api/config` auf Pflichtfelder reduzieren | SEC-04 | Kein Standort-/Infrastrukturleak | S | Deanonymisierung, Recon | — |
| 16 | Log-/Observability-Hygiene | BUG-05 | Logs diagnosetauglich; Tests laufen | S | Echte Fehler gehen unter | nach 4 |

---

# Empfohlene nächste Schritte

**Beste drei Maßnahmen nach Nutzen/Risiko/Aufwand:**

1. **SEC-03 + SEC-02** (`allsky-api/index.js`): Docroot-Containment, Schreibendpunkte auf `POST` + Token, konfigurierbarer Bind.
2. **BUG-02** (`src/config.c` + `config.ini.default`): Key-Trim, Warnung für unbekannte Keys, Default-Config normalisieren.
3. **SEC-01** (`Makefile`): `sudo`-Gruppe entfernen, `systemd-run` aus den Sudoers, Skript-Owner `root:root 0755`.

**Weitere atomare Aufgaben:**

4. **BUG-01**: Frame-Konstante vereinheitlichen, `_Static_assert`, `t3/t4` korrigieren; ASan-Dryrun mit den vorhandenen `.raw`-Dateien.
5. **REL-04**: Datei-Wrapper auf Rückgabewerte umstellen; Test gegen volles tmpfs.
6. **REL-03 + REL-06**: Zwei systemd-Units, Anstoß per `systemctl start`, Nachtzustand persistieren.
7. **REL-02**: Install-Targets reparieren und in sauberer VM end-to-end prüfen.
8. **REL-01**: Retention-Timer + Mindest-Freiplatz-Wächter; Logpfad korrigieren.
9. **REL-05 + REL-07**: `write_json_atomic()` und `sqlite3_busy_timeout` in allen Schreibern.
10. **BUG-03 + BUG-05 + BUG-04**: Drei lokale Korrektheitsfixes mit gezielten Negativtests.

---

# Umsetzungsnotizen

## Runde 1 — SEC-01, SEC-02, SEC-03 (+ BUG-06)

**`allsky-api/index.js`**
- SEC-03: Docroot-Containment über `path.resolve(PUBLIC_PATH, '.' + pathname)` + `startsWith(PUBLIC_PATH + path.sep)`, ausgeführt **vor** dem Endungsfilter. `PUBLIC_PATH` ist jetzt `path.resolve(...)`.
- BUG-06: `decodeURIComponent` in `try`/`catch` (400 statt Prozessende); NUL-Bytes im Pfad → 400.
- SEC-02: neue Optionen `api-bind` (Default `0.0.0.0`, Verhalten unverändert) und `api-token` (Default leer). `requireServiceAuth()` erzwingt für `/api/services/{start,stop,restart}`: Token konfiguriert (sonst 403), Methode `POST` (sonst 405), Header `X-API-Token` per `crypto.timingSafeEqual` (sonst 401). `/api/services/status` bleibt bewusst offener Lesezugriff. `server.listen(PORT, BIND_ADDR, …)`.
- Nebenbei: `parseConfig()` splittet jetzt am **ersten** `=` statt per `split("=")` — sonst würde ein Token mit `=` (Base64-Padding) stillschweigend abgeschnitten.

**`config.ini.default`**
- `api-bind` und `api-token` dokumentiert (inkl. `openssl rand -hex 32`).

**`Makefile`**
- SEC-01: `usermod -a -G sudo` entfernt; Sudoers-Regeln für `/usr/bin/systemd-run` und `allsky360-ai-run.sh *` entfernt; das `sudoers`-Target prüft jetzt aktiv, dass `systemd-run` **nicht** mehr gewährt ist und bricht sonst ab.
- Skripte (`scripts/`, `ai/*.py`) und alle Binaries nach `/opt/allsky360/bin` werden `root:root` installiert, Skripte mit `0755`. `$(DATA_DIR)` und `$(DATA_DIR)/bin` sind ebenfalls `root:root` — Schreibrecht auf dem **Verzeichnis** würde sonst weiterhin das Ersetzen der Dateien erlauben.
- Die pauschalen `chown -R $(USERNAME) $(DATA_DIR)` wurden auf `$(DATA_DIR)/allsky-api` eingegrenzt, damit sie die Root-Eigentümer nicht zurücksetzen.

**Verifikation**
- `node --check allsky-api/index.js` → OK.
- Containment-Logik isoliert getestet (11 Fälle): alle Traversal-Varianten (`../`, `%2e%2e%2f`, gemischt), `/%`, `%00` und `.db` werden abgewiesen; `dashboard.html`, `/js/*.js`, `/images/**`, `/videos/*.mp4` werden weiterhin ausgeliefert.
- `make -n` für `user`, `sudoers`, `binaries`, `shell-scripts`, `directories` → parst fehlerfrei. **Nicht** verifiziert: ein echter `sudo make install`-Durchlauf (bricht weiterhin an REL-02 ab) und ein Laufzeittest der API (`node_modules` sind nicht installiert).

**Bewusst nicht mitgemacht (bleibt offen)**
- `url.parse()` ist in Node ≥ 23 deprecated (DEP0169); Umstellung auf WHATWG-`URL` ist ein eigener, größerer Schnitt (Query-Handling) — nicht Teil dieser Runde.
- Der Entzug von `systemd-run` macht `run_end_of_night()`/`run_postprocessing()` in `main.c` endgültig wirkungslos. Diese Pfade waren bereits vorher defekt (REL-03); Ticket 6 ersetzt sie durch dedizierte Units.
- Bestehende Installationen brauchen eine Migration: `sudo make sudoers` neu ausführen und `gpasswd -d allskyuser sudo`.

## Runde 2 — BUG-02 (+ BUG-07)

**`src/config.c`**
- BUG-02: Zeilen werden vor dem Parsen links getrimmt (Leerzeichen/Tab/CR), leere Zeilen und Kommentare — auch eingerückte — werden übersprungen. Nach dem `sscanf` werden Leerzeichen/Tab/CR am Ende von `key` entfernt, damit `key = value` genauso funktioniert wie `key=value`.
- BUG-02: Neuer `else`-Zweig meldet unbekannte Schlüssel (`WARNING: <datei>: unknown configuration key '<key>' (ignored)`). Schlüssel mit Präfix `api-` werden bewusst stumm ignoriert (die liest die Node-API). Die Warnung erscheint **nur beim ersten Laden** pro Prozess — `load_config()` läuft in jedem Capture-Zyklus, sonst würde das Log geflutet (REL-01).
- BUG-07: `memset(config, 0, sizeof(*config))` am Funktionsanfang plus explizite Defaults für die zwölf bisher ungesetzten Felder.

**`config.ini.default`**
- `camara_cooling_temperature` → `camera_cooling_temperature` (Tippfehler).
- `image_zenith=y=1185` → `image_zenith_y=1185` (die Zeile hat den Zenit nie gesetzt).
- `autoexposure_measure_radius=600` → `measure_brightness_area=0.55` — der Parser kennt nur den zweiten Schlüssel, und er erwartet einen Bruchteil von `image_horizon_radius`; 0.55 × 1090 px ≈ die ursprünglich gemeinten 600 px.
- `camera_roi_*` auskommentiert: im Daemon gibt es dafür weder Felder noch Auswertung.
- Die fünf `clarity_filter_*`-Zeilen auf `key=value` normalisiert.

**Verifikation** (Testharness im Scratchpad, json-c als Stub — nichts installiert)
- `clang -Wall -Wextra -fsyntax-only` auf `src/config.c`: sauber.
- `load_config("config.ini.default")` real ausgeführt: `clarity_filter_strength=0.35`, `radius=24`, `midtone_width=0.25`, `mask_mode=1`, `image_zenith_y=1185`, `camera_cooling_temperature=5`, `measure_brightness_area=0.55`, `indigoservername='indigo_server'` — vorher wären das alles Defaults bzw. undefinierte Werte gewesen. Normale Keys (`capture_interval`, `jpeg_quality`, `gamma`, Pfade, quotierte Strings) unverändert.
- Negativtests: Tippfehler-Keys erzeugen genau eine Warnung, eingerückte Kommentare mit `=` erzeugen keine, `api-token=deadbeef==` erzeugt keine, `key   =   42` und `  key = 300` werden korrekt gelesen, CRLF-Dateien funktionieren, dreifaches Laden warnt nur einmal.

**Befund aus den produktiven Configs** (`config.ini.uranus-c`, `config.ini.saturn-c` — beide gitignored, von mir **nicht** geändert)
- Beide setzen `clarity_filter=1`, aber alle fünf Parameter standen als `key = value` da und wurden verworfen. Der Filter lief also mit `strength=0.5`, `radius=5`, `mask_mode=0` statt mit den gewollten `0.35 / 12 / 1`. **Nach dem Fix ändert sich das Bildergebnis sichtbar.**
- Unbekannte Schlüssel in beiden: `hdr_chroma_mode`, `ai_save_trainig_images`, `camara_cooling_temperature`, `camera_roi_*` bzw. `cmaera_roi_y`, `aurora`; in uranus-c zusätzlich `debayer` und `image_mask_center_x/y`, in saturn-c `mask_center_x/y` und `mask_radius`.
- Ohne Folgen: `debayer=vng` (der Default ist ohnehin `vng`) sowie die `mask_center_*`-Varianten (beide Dateien setzen zusätzlich die korrekten `image_center_x/y` und `image_mask_radius`).
- **Offen und relevant**: Keine der beiden Dateien setzt `image_zenith_x`/`image_zenith_y`. Damit stehen sie auf den Defaults (1000, 500), während die Bildmitte bei (1925, 1185) bzw. (1504, 1504) liegt. Rauschmessung (`nighttime_measure_noise_radius`) und Fokusmessung (`focus_radius=400`) messen also an der falschen Stelle im Bild.

## Runde 3 — BUG-01 (Entscheidung: 5 Belichtungen)

**`src/main.c`**
- `#define MAX_FRAMES 3` entfernt. Einzige Quelle ist jetzt `MAX_IMAGES` aus `hdr_merge.h` (= 5); alle 14 Verwendungen in `main.c` umbenannt, `frames[]` ist damit `[5]`.
- Neuer `_Static_assert(MAX_IMAGES == 5, …)` direkt an der Array-Definition: `image_metadata_t` hat genau `exposure_t0..t4` und `sigma_noise_t0..t4`, die drei Stellen dürfen nicht wieder auseinanderlaufen.
- Der harte Clamp `if (use_max_images > 3) use_max_images = 3;` war die eigentliche Begrenzung — jetzt gegen `MAX_IMAGES`.
- An der Fusion war **nichts** zu ändern: `hdr_multi_scale_fusion_laplacian_rgbf1()` ist vollständig über `use_max_images` parametrisiert und dimensioniert alle internen Arrays (`exp_gain`, `sigma_Y_lin`, `weight_maps`, `luma_maps`, `contrast_maps`) bereits mit `MAX_IMAGES`.

**Verifikation**
- Isolierter Testbau gegen `hdr_merge.h`/`image_metadata.h`: `frames[]` ist 5 × 72 Byte, gültige Indizes 0..4, ein Canary hinter dem Array bleibt nach Schreiben und Lesen von `frames[0..4]` unverändert. Die Zugriffe `frames[3]`/`frames[4]` aus `main.c:2138-2143` liegen damit im Array.
- Gegenprobe: `MAX_IMAGES` testweise auf 4 gesetzt → `_Static_assert` schlägt beim Kompilieren fehl.
- Ein vollständiger Build war hier nicht möglich (INDIGO, Cairo, libnova, json-c fehlen auf dem Mac) — der Fix ist auf dem Zielsystem mit `make USE_CUDA=1` zu kompilieren.

**Speicherbedarf**
- Pro zusätzlichem Frame ein RGBA-Float-Puffer: bei `crop 2280×2180` ≈ 75,8 MiB, ohne Crop (`3856×2180`, saturn-c) ≈ 128,3 MiB.
- Die Puffer werden **lazy** belegt: die Aufnahmeschleife bricht ab, sobald ein Frame nicht mehr überbelichtet ist. Frames 3 und 4 kosten also nur in Szenen mit extremem Dynamikumfang zusätzlichen Speicher — dann aber dauerhaft, weil die Puffer bis zum Prozessende gehalten werden.
- `hdr_save_exposures=1` schreibt pro Frame ein `latest_image_t<i>.jpg` — bei 5 Frames entsprechend mehr Schreiblast (siehe REL-01).

**Wichtig für die produktiven Stationen**
Mit den dort gesetzten Werten ändert der Schritt auf 5 Frames **vorerst nichts**. Nachgerechnet mit `calc_exposure_time()`:

| Konfiguration | Leiter | Ende |
|---|---|---|
| Nacht, Station (`factor 10.0`) | 10 s → 0,00977 s → 0,0000095 s | Abbruch **nach 2 Frames** (unter `MIN_EXPOSURE`) |
| Tag, Station (`factor 5.0`) | 0,0045 s → 0,000141 s → 0,0000044 s | Abbruch **nach 2 Frames**; Frame 1 liegt unter `camera_min_exposure` (BUG-08) |
| Nacht, `config.ini.default` (`factor 2.0`) | 10 / 2,5 / 0,625 / 0,156 / 0,039 s | alle **5 Frames** nutzbar |
| Tag, `config.ini.default` (`factor 2.0`) | 0,0045 / 0,001125 / 0,000281 / 0,00007 s | Abbruch nach 4 Frames; Frames 2 und 3 unter `camera_min_exposure` (BUG-08) |

Damit 5 Belichtungen tatsächlich entstehen, muss `nighttime_hdr_exposure_factor` auf 1.0–2.0 (statt 10.0) und `daytime_hdr_exposure_factor` auf 1.0–2.0 (statt 5.0) — genau das, was der Kommentar in `config.ini.default` empfiehlt. Davor sollte BUG-08 behoben sein, sonst werden am Tag Belichtungen unterhalb des Kameraminimums angefordert.

## Runde 4 — BUG-09 und BUG-08

**`src/hdr_merge.c` / `hdr_merge.h` / `main.c` (BUG-09, Variante „Farb-Gains auf Grün normiert")**
- Die Kanal-Gains tragen jetzt ausschließlich die **Farbdrift** zwischen den Frames: sie werden für jeden Frame gegen dessen eigenen Grünkanal normiert (`cb_r0/cb_rk`), wodurch `g_gain[k] ≡ 1.0` gilt. Die Helligkeitsdifferenz trägt allein `exp_gain[k]`. Damit ist `px * exp_gain * c_gain` wieder genau **eine** Korrektur statt `(t0/tk)²`.
- Die Parameter `channel_scale_r/g/b` sind entfallen (Signatur, Header und Aufrufstelle). Sie kürzten sich im Quotienten ohnehin heraus; die zugehörigen Locals in `main.c` sind mit entfernt.
- Als Nebeneffekt ist auch die Rauschgewichtung korrekt: `sigma_Y_lin[k] = sn * exp_gain[k] * g_gain[k]` war ebenfalls doppelt korrigiert und ist mit `g_gain ≡ 1` jetzt die saubere Linearisierung des Rauschens.

**`src/main.c` / `autoexposure.{c,h}` (BUG-08)**
- Die Leiter endet jetzt an `max(config.camera_min_exposure, MIN_EXPOSURE)` statt an der einkompilierten Konstante; die Meldung nennt Frame, berechnete Zeit, Grenze und den zu senkenden Parameter. Der tote Vergleich gegen `config.camera_min_exposure` ist weg.
- Ein nicht aufgenommener Frame bekommt `exposure = 0.0`, damit die Metadaten keine Phantom-Belichtung ausweisen.
- `calc_exposure_time()` nimmt `ev_steps` jetzt als `float` entgegen — vorher wurde der `double`-Configwert auf `int` abgeschnitten, ein `hdr_exposure_factor` von 1.5 wäre still zu 1 geworden.

**Verifikation — A/B auf dem echten Fusionscode**
`src/hdr_merge.c` ließ sich mit einem json-c-Stub und einem Stub für `save_jpeg_rgbf1()` eigenständig übersetzen. Damit wurde die **reale** `hdr_multi_scale_fusion_laplacian_rgbf1()` gegen eine Kopie mit der alten Gain-Rechnung getestet: synthetische Szene über 0,02–8,0 in Frame-0-Einheiten, drei Belichtungen 1,0 s / 0,25 s / 0,0625 s, Frame 0 sättigt ab Szenenwert 1,0, Frame 1 ab 4,0.

| Bereich | Szene (Soll) | vorher | vorher | nachher | nachher |
|---|---|---|---|---|---|
| Schatten | 0,0375 | 0,0375 | 1,00× | 0,0375 | 1,00× |
| Mitteltöne | 0,2460 | 0,2460 | 1,00× | 0,2460 | 1,00× |
| Lichter | 1,6155 | 7,8485 | **4,86×** | 1,5727 | 0,97× |
| Spitzlichter | 4,8430 | 66,0558 | **13,64×** | 4,5350 | 0,94× |

Schatten und Mitteltöne stammen aus Frame 0 (`exp_gain = 1`) und waren deshalb immer korrekt — der Fehler trat ausschließlich dort auf, wo die Fusion die Lichter aus den kurzen Belichtungen holt, also genau im Dynamikgewinn. Die Restabweichung von 3–6 % nach dem Fix ist die übliche Weichheit der Laplace-Rekonstruktion an einem steilen logarithmischen Verlauf, kein Skalierungsfehler.
Nebenbefund derselben Messung: die automatische `Y_max_expected`-Bestimmung meldete vorher `max Y: 128.000`, nachher `8.000` — sie war um denselben Faktor 16 verzerrt und hat die Gewichtung zusätzlich verfälscht.

**Nicht verifiziert**
Ein vollständiger Build von `main.c` war hier nicht möglich (INDIGO, Cairo, libnova fehlen). Aufrufstelle und Signatur wurden manuell gegengeprüft (15 Argumente auf beiden Seiten, keine `channel_scale`-Reste im Baum), der Build auf der Station steht aber noch aus.

**Nachjustierung erforderlich**
`hdr_y_max_expected`, `hdr_weight_sigma`, `autostretch_*`, `gamma` und `saturation` sind gegen das bisherige, in den Lichtern um das 4- bis 16-fache überhöhte Niveau eingestellt. Nach diesem Fix kommen die Lichter korrekt an, das Gesamtbild wird deshalb zunächst dunkler und flauer wirken. Empfohlene Reihenfolge: erst mit `--dryrun --img0/1/2` und den drei vorhandenen `.raw`-Dateien gegenprüfen, dann `hdr_y_max_expected` neu bestimmen, danach Autostretch und Gamma.

## Runde 5 — BUG-03, BUG-04, BUG-05

**`include/allsky.h` (BUG-03)**
Statt nur die eine Fundstelle umzubauen, ist die Ursache im Makro behoben: `TIMED_BLOCK` ist jetzt `if (1) { … } else ((void)0)` statt `do { … } while (0)`. Damit binden `continue` und `break` im eingebetteten Block an die Schleife der **Aufrufstelle** — die Fußangel ist für alle rund 30 Verwendungen weg, und der Fehlerpfad in `main.c:1314-1326` wirkt jetzt wie beabsichtigt.

*Verifikation:* eigener Testbau mit drei Fällen — `continue` überspringt die Iteration (4 verarbeitet / 1 übersprungen statt 5/1), `break` verlässt die Schleife (n=3), und ein nachfolgendes `else` wird nicht vom Makro eingefangen. Alle drei korrekt.

**`src/sun_calc.{c,h}`, `src/main.c`, `src/config.c` (BUG-04)**
- Neu: `is_sun_below_angle(timestamp, lon, lat, angle_deg)`, aufgesetzt auf das bereits vorhandene `sun_altitude()`. `is_sun_below_6_degrees()` bleibt als Wrapper erhalten (`test/test_sun_calc.c` benutzt ihn).
- Beide Aufrufstellen in `main.c` (Start und Capture-Schleife) übergeben jetzt `config.nighttime_sunangle`.
- `config.c` prüft den Wertebereich und klemmt auf −18,0 … 0,0 mit einmaliger Warnung.
- Nebenbei: `sun_calc.c` inkludierte seinen **eigenen Header nicht**. Mit der neuen Funktion wäre `sun_altitude()` implizit deklariert gewesen und der Build gebrochen — `#include "sun_calc.h"` ist jetzt drin, wodurch Deklaration und Definition dauerhaft gegeneinander geprüft werden.
- Die vier `printf`-Zeilen pro Zyklus aus der alten Funktion sind durch eine einzige ersetzt (Höhe, Schwelle, Ergebnis) — kleiner Beitrag zu REL-01.

*Verifikation:* `-6` und `-12` kommen unverändert an, `-25` → −18,0 mit Warnung, `+5` → 0,0 mit Warnung, `config.ini.default` ergibt −6,00 (also exakt das bisherige Verhalten).

**`src/main.c` (BUG-05)**
`signal(SIGCHLD, SIG_IGN)` entfernt; der Kommentar erklärt, warum es nicht zurückkommen darf. Die Meldungen unterscheiden jetzt zwischen „spawned" und Fehler, statt beides nacheinander auszugeben.

*Verifikation:* Nachbau des Doppel-Fork-Musters, je 20 Durchläufe. Mit `SIG_IGN`: **20 von 20** Aufrufen liefern Fehler (`waitpid: No child processes`). Ohne: **0 von 20**. Zombies entstehen in beiden Varianten keine — der Doppel-Fork erledigt das Reaping bereits, `SIG_IGN` war schlicht überflüssig und schädlich.

**Syntax-Checks dieser Runde** (mit json-c- und libnova-Stubs im Scratchpad, nichts installiert)
`clang -Wall -Wextra -std=gnu11 -fsyntax-only` auf `sun_calc.c`, `config.c` und `autoexposure.c`: keine Fehler, keine Warnungen. `main.c` bleibt mangels INDIGO/Cairo hier ungebaut.

## Runde 6 — REL-01 (Retention und Logrotation)

**Vorgabe des Betreibers:** Speichergrenze und Altersgrenze konfigurierbar, Defaults 500 GB und 365 Tage, es greift was zuerst eintritt; gleiche Behandlung für Bilder, Thumbnails, Videos und Tages-DB.

**Neu: `src/retention.c` / `src/retention.h`** (im Daemon, nicht als Skript — der Daemon läuft zuverlässig und unabhängig von der noch defekten Install-Pipeline, REL-02)
- Das Archiv ist nach UTC-Tagen organisiert, und ein Tag wird immer **als Ganzes** entfernt, über alle Klassen hinweg: `images/YYYYMMDD/` (Vollbilder, Thumbnails, Panoramen, JSON, Debugbilder), `raw_dump_directory/YYYYMMDD/`, `<mode>_YYYYMMDD.mp4/.png` und `database_YYYYMMDD.db` inkl. `-wal`/`-shm`.
- Zwei Limits, es gewinnt das zuerst erreichte: Alter über `retention_max_days`, danach Größe — solange das verwaltete Datenvolumen `retention_max_size_gb` überschreitet, fällt der jeweils älteste verbleibende Tag.
- `retention_keep_days` (Minimum und Default 2) schützt die neuesten Tage **unabhängig** von beiden Limits, damit eine zu klein geratene Grenze nicht die laufende, über Mitternacht reichende Nacht löscht.
- Aufruf einmal pro UTC-Tag aus der Capture-Schleife heraus, zusätzlich einmal direkt nach dem Start, damit eine bereits übervolle Station nicht bis Mitternacht warten muss.

**Sicherungen im Löschpfad** (es wird gelöscht, deshalb explizit):
- Gelöscht wird nur, was als Tagesschlüssel validiert ist: exakt acht Ziffern. `config.json`, `latest_image.json` und beliebige Fremdverzeichnisse im Bildordner sind dadurch strukturell außer Reichweite.
- `remove_tree()` benutzt ausschließlich `lstat()` und folgt keinem Symlink — ein Link kann keine fremden Daten in die Löschung ziehen; die Rekursionstiefe ist auf 8 begrenzt.
- Zielverzeichnisse werden vorher geprüft (nicht leer, nicht `/`, kein `..`, existiert, ist ein Verzeichnis).
- `retention_dry_run=1` meldet vollständig, was passieren würde, und löscht nichts.
- `config.c` prüft die Wertebereiche: `retention_max_days < 1` oder `retention_max_size_gb <= 0` schalten die Retention ab statt alles zu löschen; `retention_keep_days` wird auf mindestens 2 angehoben.

**Neue Schlüssel in `config.ini.default`:** `retention`, `retention_max_days` (365), `retention_max_size_gb` (500), `retention_keep_days` (2), `retention_dry_run` (0).

**Logrotation**
- `allsky360-camera.service` schreibt jetzt nach `/var/log/allsky360/camera.log` statt `/var/log/allsky360.log` — damit greift die vorhandene Regel `etc/logrotate.d/allsky360` (`/var/log/allsky360/*.log`, 30 Tage, komprimiert, `copytruncate`).
- `LOG_DIR` und `RUN_DIR` waren im Makefile nur als Variablen definiert und wurden nie angelegt; das `directories`-Target erzeugt sie jetzt.

**Verifikation** (synthetisches Archiv im Scratchpad, echte Löschvorgänge)
| Test | Erwartung | Ergebnis |
|---|---|---|
| Altersgrenze 3 Tage, 8 Tage vorhanden | 4 Tage weg (7–4 Tage alt), 20250105 (3 Tage) bleibt | korrekt; Videos und DBs im Gleichschritt entfernt |
| Größenlimit, Alter unbegrenzt | ältester Tag fällt, bis unter Limit | 5 Tage entfernt, 3 verbleiben |
| Dry-Run | meldet alles, löscht nichts | alle 8 Tage unverändert |
| Absurd kleines Limit | Schutz greift | genau 2 Tage bleiben stehen |
| Fremddateien im Bildordner | unangetastet | `config.json`, `latest_image.json`, `nicht_ein_tag/` alle noch da |
| Leeres Archiv / fehlendes Verzeichnis | kein Absturz | sauberer Abbruch mit Meldung |
| Raw-Dumps | mitgezählt und mitgelöscht | `raw/YYYYMMDD` folgt `images/YYYYMMDD` |

`clang -Wall -Wextra -std=gnu11 -fsyntax-only` auf `retention.c` und `config.c`: keine Fehler, keine Warnungen.

**Empfehlung vor dem Scharfschalten**
Einmal mit `retention_dry_run=1` laufen lassen und den Report im Log prüfen. Für die 500-GB-Platte ist `retention_max_size_gb=450` die sichere Wahl — das Limit zählt nur die verwalteten Daten, nicht Betriebssystem, Logs und `node_modules`.

## Runde 7 — REL-03, REL-02 und REL-08

**Entscheidende Erkenntnis:** `postprocessing.sh` lief über `sudo /usr/bin/systemd-run --property=User=allskyuser …` — also als **derselbe Benutzer, unter dem der Daemon ohnehin läuft**. Die ganze Kette aus `sudo`, `systemd-run` und zwei Wrapper-Skripten war funktionslos und war gleichzeitig die Ursache von SEC-01 (unbeschränktes `systemd-run` in den Sudoers) und der Hälfte von REL-03 (`env_reset` verschluckte die `ALLSKY360_*`-Variablen).

**`src/main.c` (REL-03)**
- Neu: `spawn_hook(script, arg)` — ein gemeinsamer, detachter Start per Doppel-Fork und direktem `execl()`. Kein `sudo`, kein `systemd-run`, kein Wrapper. Dadurch erreicht die `ALLSKY360_*`-Umgebung das Skript tatsächlich.
- `access(script, X_OK)` vor dem Fork: ein fehlendes oder nicht ausführbares Hook-Skript wird **einmal sauber gemeldet**, statt stumm in einem `_exit(127)` eines Enkelprozesses zu verschwinden.
- Ein leerer Pfad deaktiviert den Hook (Rückgabe 0, keine Fehlermeldung).
- `run_postprocessing()` übergibt den absoluten Pfad des archivierten Bildes als `argv[1]` — das Skript muss sich nicht mehr über `latest_image.json` zurückrechnen, was bei REL-05 (nicht-atomare JSON-Schreibvorgänge) fehlschlagen konnte.

**Neue Schlüssel:** `postprocessing_script` und `endofnight_script` (Default `/opt/allsky360/scripts/…`, leer = aus). Damit ist der Erweiterungspunkt verschiebbar — wer ein eigenes Skript im Homeverzeichnis pflegen will, trägt dort einfach den Pfad ein, statt die root-eigene Datei unter `/opt` zu editieren.

**`scripts/postprocessing.sh` — neu geschrieben (REL-03 + REL-08)**
- Bekommt das Bild als `$1` (Fallback `ALLSKY360_IMAGE_PATH`), `set -euo pipefail`.
- Eigenes Log nach `/var/log/allsky360/postprocessing.log` — nötig, weil `spawn_hook()` stdio auf `/dev/null` legt.
- `flock`-Guard: nur eine Instanz gleichzeitig. Bei `capture_interval=120` und einem langsamen Modell dürfen sich keine Jobs stapeln; ein übersprungenes Bild ist die richtige Abwägung.
- **REL-08 behoben:** der `jq`-Exitcode *und* eine Mindestgröße werden geprüft, bevor `mv` die Metadaten ersetzt. Schlägt etwas fehl, bleibt die Originaldatei unangetastet. `mktemp` ohne `--dry-run`, Aufräumen per `trap`.
- Maskierung ist optional: ohne `mask.jpg` oder ohne ImageMagick läuft die Inferenz auf dem Originalbild weiter, statt abzubrechen.
- Fehlt das AI-Backend (venv, Modell, `classes.txt`), endet der Hook als sauberer No-op mit Logzeile — kein Fehler alle zwei Minuten.
- Die `sed`-Muster nutzen jetzt POSIX-ERE (`sed -E`) statt GNU-BRE mit `\+`.

**`scripts/endofnight.sh`**
- Leitet die eigene Ausgabe nach `/var/log/allsky360/endofnight.log` (bisher ging sie ins Journal von `systemd-run`, künftig wäre sie in `/dev/null` verschwunden).
- `flock`-Guard, und eine Vorabprüfung auf `allsky360-timelapse` statt stiller Fehlschläge.
- Nutzt die `ALLSKY360_*`-Variablen mit `config.ini` als Fallback, damit der Handstart weiter funktioniert.

**Entfernt:** `scripts/allsky360-endofnight-run.sh` und `scripts/allsky360-postprocessing-run.sh`. Beide waren nur `systemd-run`-Wrapper mit `eval` auf Umgebungsinhalte und unquotierter `$SETENV_ARGS`-Expansion. Sie sind in git versioniert und damit wiederherstellbar.

**`Makefile` (REL-02)**
- `allsky360-mount-position.server` → `.service` (die Datei mit dem Tippfehler existierte nicht, das `systemd`-Target brach dort ab).
- `systemctl enable --now allsky360-mount-position` → `…mount-position.timer`.
- `shell-scripts` ist jetzt Teil der `install`-Kette — vorher wurden die Skripte **nie** installiert.
- `postprocessing.sh` wird installiert (fehlte komplett in der Liste); `run_ai_processing.sh` nur, wenn vorhanden.
- Referenzen auf das nicht existierende `allsky360-ai-run.sh` sind entfernt (auch in den Sudoers, siehe SEC-01).
- `cron` ist aus der `install`-Kette heraus: die systemd-Timer erledigen aurora/meteo/location bereits, die Cron-Datei hätte jeden Fetcher doppelt gestartet. Der Dateiname im Target ist korrigiert, das Target bleibt als Opt-in bestehen. **`etc/cron.d/allsky360-tools` selbst habe ich nicht angefasst — die Datei ist nicht in git und damit nicht wiederherstellbar.** Ihre Inhalte passen weiterhin nicht zur Installation (Benutzer `allsky` statt `allskyuser`, `/usr/local/bin` statt `/opt/allsky360/bin`, `/etc/allsky360/config.ini` statt `/home/allskyuser/config.ini`).

**Verifikation**
- `make -n` für alle Targets der Install-Kette löst auf; alle referenzierten Repo-Quellen unter `etc/`, `scripts/` und `ai/` existieren (geprüft per Skript).
- Hook-Tests mit einer nachgebauten Linux-Umgebung (GNU-`stat`-Shim, Fake-`yolo`):

| Test | Ergebnis |
|---|---|
| Argument fehlt | exit 1, klare Logzeile |
| gültiges JSON | `weather`/`weather_score` ergänzt, bestehende Felder erhalten |
| **kaputtes JSON** | exit 1, Datei **unverändert und nicht leer** |
| JSON fehlt | exit 0, Bild bleibt, Warnung im Log |
| AI nicht installiert | exit 0, No-op, JSON unverändert |
| `endofnight.sh` ohne Timelapse-Binary | exit 1 mit Klartextmeldung im eigenen Log |

`flock` ist auf dem Testrechner nicht vorhanden, der Lock-Pfad ist daher **nicht** praktisch getestet — das Skript überspringt die Sperre dann bewusst (`command -v flock`). Auf der Station ist `flock` vorhanden (die Cron-Datei benutzt es).

**Noch offen im Umfeld:** REL-06 — End-of-Night hängt weiterhin allein am In-Process-Übergang `was_night && !is_night`. Jetzt, da der Hook erstmals wirklich läuft, lohnt sich das: ein Neustart im Dämmerungsfenster kostet nach wie vor die komplette Nachtverarbeitung.

## Runde 8 — REL-06 und REL-07

### REL-06: Nachtverarbeitung neustartfest

**Neu: `src/run_state.c` / `run_state.h`** — ein kleiner persistenter Key/Value-Store in `<database_directory>/allsky360-state`. Geschrieben wird atomar (Temporärdatei, `fsync`, `rename`), beschädigte oder unbekannte Zeilen werden übersprungen statt den Daemon zu stören. Bewusst generisch gehalten, damit REL-05 denselben Mechanismus nutzen kann.

**Regeländerung in `main.c`:** Der Auslöser ist nicht mehr der im Prozess beobachtete Übergang `was_night && !is_night`, sondern: *einmal pro UTC-Tag, sobald Tagmodus herrscht, sofern für diesen Tag noch nicht gelaufen*. Der Zustand wird **vor** dem Start des Hooks geschrieben — der Hook läuft detacht und minutenlang; stirbt der Daemon dabei, darf der nächste Start keinen zweiten, konkurrierenden Lauf anstoßen.

**Ein Entwurfsfehler, der erst im Test auffiel:** Die erste Fassung hat bei einer Neuinstallation, die nachts startet, *keine* Statusdatei geschrieben („pending" nur im Speicher). Ein Absturz vor der Dämmerung hinterließ damit überhaupt keinen Zustand, und der Neustart — dann im Tagmodus — nahm den Zweig „frische Installation am Tag" und **unterdrückte genau den Lauf, für den dieser Mechanismus existiert**. Der Sentinel `pending` wird jetzt auf die Platte geschrieben.

*Verifikation* (echtes `run_state`, Entscheidungsregel aus `main.c` nachgebildet, da sie dort inline steht):

| Szenario | Ergebnis |
|---|---|
| Unit-Tests `run_state` (8 Fälle: schreiben, lesen, überschreiben, zweiter Schlüssel, unbekannter Schlüssel, Newline/`=` abgewiesen) | alle OK, keine `.tmp`-Reste |
| beschädigte Statusdatei | kein Absturz, gültige Zeile wird weiter gelesen |
| Neuinstallation nachts | Zustand `pending`, Lauf bei der folgenden Dämmerung |
| Neuinstallation tags | kein Lauf heute, Lauf am nächsten Morgen |
| **Absturz nachts, Neustart nach der Dämmerung** | **Lauf findet statt** (vorher verloren) |
| erneuter Neustart am selben Tag | kein Doppellauf |

### REL-07: SQLite unter Nebenläufigkeit

Betroffen waren drei Dateien mit identischem Muster: `src/daily_database.c`, `tools/aurora/src/aurora_database.c`, `tools/meteo/src/meteo_database.c`.
- `sqlite3_busy_timeout(db, 5000)` nach jedem `sqlite3_open()`.
- `PRAGMA journal_mode=WAL` nur noch in den `create_*`-Funktionen (einmal pro Datenbank) statt bei jedem Insert — es schreibt in den Datenbankkopf und war selbst sperrenanfällig, konnte also vor dem eigentlichen Insert abbrechen.
- Begrenzter Retry (3 Versuche mit `sqlite3_reset`) um `sqlite3_step()`, plus eine Fehlermeldung mit `sqlite3_errmsg()` und Code statt eines stummen `return 3`.
- Nebenbefunde mitbehoben: in aurora und meteo fehlte auf den `sqlite3_open`-Fehlerpfaden je zweimal das `sqlite3_close()` (`sqlite3_open()` legt auch im Fehlerfall ein Handle an), und `src/daily_database.h` benutzte `time_t` ohne `<time.h>` einzubinden — der Header war nicht eigenständig übersetzbar.

*Verifikation:* vier Prozesse schreiben gleichzeitig je 50 Zeilen in dieselbe Datenbank, drei Durchläufe:

| Variante | Zeilen in der DB | gemeldete Fehlschläge |
|---|---|---|
| vorher | 20 / 33 / 48 von 200 | 180 / 167 / 152 |
| nachher | **200 / 200 / 200** | 0 / 0 / 0 |

**Korrektur meiner ursprünglichen Einschätzung:** Ich hatte REL-07 als „sporadisch verlorene Messzeilen" beschrieben. Das untertreibt den Mechanismus — wenn zwei Schreiber kollidieren, verwarf der alte Code die Zeile *immer*. Im realen Betrieb kollidieren sie allerdings selten (Kamera alle 120 s, Fetcher alle 5–10 min), der Test mit vier Dauerschreibern ist deutlich aggressiver als die Praxis. Die Einstufung „Medium" bleibt damit richtig, die Begründung lautet aber: seltene Kollision, dann aber garantierter Verlust.

`clang -Wall -Wextra -std=gnu11 -fsyntax-only` auf allen geänderten Dateien: keine Fehler, keine Warnungen.

## [BUG-10] Hemisphären-Suffixe in den Koordinaten werden stillschweigend ignoriert
- Bereich: Bug
- Severity: Medium
- Confidence: High
- Status: bestätigt (beim Testen von SEC-04 gefunden)
- Fundstelle: `src/config.c:565-570` — `atof(value)` für `location_latitude`/`location_longitude`; `config.ini.default:10-11`
- Evidenz: Die mitgelieferte Konfiguration schreibt `location_latitude=51.2349N` und `location_longitude=7.4479E`, lehrt damit also die Suffix-Schreibweise. Gelesen wird mit `atof()`, das beim ersten Buchstaben abbricht — das Suffix wird verworfen, **auch `S` und `W`**. Gemessen:

  | Eingabe | ergibt | korrekt wäre |
  |---|---|---|
  | `51.2349N` / `7.4479E` | 51,2349 / 7,4479 | richtig (nur zufällig) |
  | `33.9249S` / `18.4241E` (Kapstadt) | **+33,9249** / 18,4241 | −33,9249 |
  | `40.7128N` / `74.0060W` (New York) | 40,7128 / **+74,0060** | −74,0060 |
  | `-33.9249` / `-18.4241` | −33,9249 / −18,4241 | richtig |

- Auswirkung: Auf der Süd- oder Westhalbkugel ist der Standort spiegelverkehrt, ohne jede Meldung. Betroffen ist alles Standortabhängige: die Tag/Nacht-Umschaltung, Sonnen- und Mondhöhe in den Metadaten, die Aurora-Wahrscheinlichkeit und die Wetterabfrage. Für die vorhandenen Stationen (N/E) folgenlos, aber die Beispielkonfiguration führt neue Nutzer direkt in die Falle.
- Empfehlung: Die Suffixe auswerten (`S`/`W` negieren, `N`/`E` ignorieren) und alles, was danach noch übrig bleibt, mit einer Warnung ablehnen — oder die Suffixe gar nicht erst anbieten und `config.ini.default` auf vorzeichenbehaftete Dezimalgrade umstellen. Die Entscheidung gehört dem Betreiber, deshalb hier nur dokumentiert.
- Akzeptanzkriterien: (1) `33.9249S` ergibt −33,9249; (2) `74.0060W` ergibt −74,0060; (3) vorzeichenbehaftete Werte funktionieren unverändert; (4) unverständliche Eingaben erzeugen eine Warnung statt eines stillen Fehlwerts.
- Aufwand: XS

---

## Runde 9 — REL-04, REL-05 und SEC-04

### REL-05: Atomare Veröffentlichung

**Neu `src/atomic_file.c/h`** mit zwei Formen: `write_text_file_atomic()` für fertige Inhalte und `atomic_file_open()`/`atomic_file_commit()`/`atomic_file_abort()` für inkrementell erzeugte. Beide schreiben in eine Temporärdatei **im Zielverzeichnis** (`rename()` ist nur innerhalb eines Dateisystems atomar), prüfen `ferror()` — das fängt Schreibfehler ab, die `fprintf()` stillschweigend verschluckt, etwa bei voller Platte — und machen `fsync()` vor dem `rename()`.

Umgestellt: `config.c` (`config.json`), `image_metadata.c` (`image-*.json`), `main.c` (`latest_image.json`), `star_detector.c` (`stars-*.json`) und `spektrum.c` (inkrementell, daher Open/Commit). `run_state.c` nutzt jetzt ebenfalls das gemeinsame Modul statt eigener `fsync`/`rename`-Logik.

Nicht umgestellt: `ai_training.c:103` — der Aufruf steht hinter `#if AI_TRAINING`, und das Makro ist auskommentiert (`main.c:100`), der Pfad ist also tot. Die Trainingsdaten werden zudem von keinem laufenden Konsumenten gelesen.

*Verifikation:* ein Schreiber rewritet die Datei 400× im Wechsel zwischen zwei unterschiedlich langen Versionen, ein Leser liest parallel ~40.000×:

| Variante | Lesevorgänge | davon unvollständig |
|---|---|---|
| vorher (direkt) | 39.851 / 39.843 | 2 / 1 |
| nachher (atomar) | 39.660 / 40.000 | **0 / 0** |

Keine `.tmp`-Reste. Die niedrige Fehlerrate bestätigt die Einstufung „sporadisch" — das Zeitfenster ist klein, aber bei einem Dashboard, das im Sekundentakt pollt, trifft es früher oder später zu.

### REL-04: Korrektur meiner Audit-Bewertung

**Der Befund war in seiner Wirkung falsch beschrieben.** Ich hatte geschrieben, eine volle Platte beende den Daemon per `SIGABRT`. Die Prüfung zeigt: `allsky_safe_fopen`, `_fread`, `_fwrite` und `_fclose` haben **repoweit keinen einzigen Aufrufer**. Der beschriebene Ausfall konnte so nie eintreten. Richtig bleibt der Kern — die Wrapper sind als Falle angelegt, wer sie benutzt, holt sich das Verhalten ein.

Umgesetzt:
- Die vier Datei-Wrapper liefern jetzt Fehler zurück und melden sie mit `strerror(errno)`, statt zu assertieren. `allsky_safe_fclose()` gibt wie `fclose()` einen `int` zurück (Signaturänderung gefahrlos, da ohne Aufrufer).
- `allsky_safe_strdup()` (3 Aufrufer) meldet OOM bzw. ein `NULL`-Argument mit Text und `abort()` — genau das Muster, das `allsky_safe_malloc/calloc/realloc` bereits verwenden.
- `allsky_safe_strcpy`/`_strcat` prüfen ihre Argumente und kehren zurück, statt zu assertieren.
- `include/allsky.h` enthält damit **kein `assert()` mehr**; `<assert.h>` ist entfernt. Der bisherige Zustand war doppelt riskant: ohne `NDEBUG` Absturz, mit `NDEBUG` stilles Ignorieren.

*Verifikation:* Laufzeittest mit `NULL`-Argumenten, zu langem `strcpy`, `fopen` auf einen nicht existierenden Pfad und `fclose(NULL)` — kein `abort()`, Fehler werden gemeldet, Puffer korrekt terminiert.

### SEC-04: Weniger in `config.json`

Zuerst geprüft, was das Frontend überhaupt liest: `location_latitude`, `location_longitude` und `location_altitude` werden von Dashboard, Panorama- und Flugzeugansicht für Sonnen-/Mondstand und Entfernungen gebraucht — sie können nicht einfach entfallen. `location_owner`, `indigoserver*`, `indigoport` und `indigomount` werden dagegen nirgends verwendet (nur in `config_help.json` als Hilfetext und in einer Gruppierungsliste in `config.html`, die fehlende Schlüssel einfach überspringt).

- Entfernt: `location_owner` (Klarname), `indigoservername`, `indigoserver`, `indigoport`, `indigomount` (interne Netztopologie).
- Behalten: `indigocamera` (Hardwareinfo, wird angezeigt), `location_city` (grob).
- Neu: `publish_exact_location` — Default 1 behält das bisherige Verhalten. Auf 0 werden die Koordinaten auf zwei Nachkommastellen (~1 km) gerundet veröffentlicht; für Sonnen-/Mondhöhe ist das folgenlos, die Entfernungsanzeige bei Flugzeugen wird um bis zu ~1,5 km ungenauer.

*Verifikation* (mit einem funktionsfähigen json-c-Ersatz im Scratchpad, damit echtes JSON entsteht): `config.json` aus `config.ini.default` enthält keines der fünf entfernten Felder, ist gültiges JSON mit 165 Schlüsseln, und der Schalter wirkt — `51.2349 / 7.4479` gegenüber `51.23 / 7.45`. Keine `.tmp`-Reste.

**Syntax-Checks dieser Runde:** `clang -Wall -Wextra -std=gnu11 -fsyntax-only` auf `atomic_file.c`, `config.c`, `image_metadata.c`, `star_detector.c`, `spektrum.c`, `run_state.c` und `include/allsky.h` — keine Fehler, keine Warnungen.

---

**Offene Rückfragen:**

1. **Zugriffsmodell**: Ist Port 3000 aus dem Internet erreichbar oder strikt LAN/VPN? Davon hängt die Einstufung von SEC-02/03/04 ab.
2. **Realer Installationszustand**: `sudo make install` oder manuell? `sudo -l -U allskyuser`, `ls -l /opt/allsky360/scripts/`, `systemctl list-timers` klären, welche REL-02/REL-03-Befunde im Feld real sind.
3. **Laufen Postprocessing/End-of-Night heute?** Auszug aus `/var/log/allsky360-postprocessing.log` und aktuelle MP4s im Videoverzeichnis.
4. **HDR-Zielkonfiguration**: 3 oder 5 Belichtungen? Entscheidet die Umsetzungsrichtung von BUG-01 (Speicherbedarf!).
5. **Aufbewahrungspolitik**: Wie lange Vollbilder, Raw-Dumps, Debug-Bilder, Videos, Tages-DBs — und wie groß ist der Datenträger? Ohne Zahlen kann REL-01 nur als Mechanismus umgesetzt werden.

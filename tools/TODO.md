# tools/ — Review-Befunde (offen)

Ergebnis eines vollständigen Source-Reviews von `tools/` (2026-09-28). Kein echter Build war
in der Review-Umgebung möglich (macOS, Xcode-Lizenz nicht akzeptiert, kein CUDA-Toolkit,
INDIGO nicht lokal verfügbar) — Build-bezogene Befunde (Abschnitt A, D) sind aus der
Makefile-/Include-Semantik abgeleitet, aber nicht durch echten Compile verifiziert. Vor dem
Abhaken bitte real bauen (`make USE_CUDA=1` auf Jetson Orin, `make tools`).

Format je Befund: `- [ ] Datei:Zeile — Beschreibung. Fix: ...`

## A. Build-brechende Regressionen (zuerst fixen)

- [x] (entfällt: Tool entfernt) `tools/location/Makefile` + `tools/location/src/config.c` (Symlink auf `src/config.c`) —
      Build kaputt: `config.c` inkludiert `"atomic_file.h"` und ruft `write_text_file_atomic()`
      auf (REL-05-Refactoring), aber das Makefile hat weder `-I../../src` noch
      `../../src/atomic_file.c` in `OBJECTS`. Fix: `INCLUDE_ROOT = ../../src`, `-I$(INCLUDE_ROOT)`
      ergänzen, `atomic_file.c` als zusätzliches Objekt aufnehmen (Muster aus
      `tools/panorama/Makefile` übernehmen).
- [x] (erledigt: panorama und mount-status sind in `make tools`; `indigo-list-cameras` wurde
      entfernt, ersetzt durch `indigo-cameras -l`)
      Root-`Makefile:285` (`tools:`) — `panorama`, `indigo-list-cameras`, `indigo-mount-status`
      fehlen komplett in `make tools` und in `binaries:` (Z. 452-465), obwohl CLAUDE.md sie als
      Build-Ziele dokumentiert (`make list-cameras` schlägt aktuell mit "No rule to make target"
      fehl). Fix: Targets ergänzen — oder CLAUDE.md korrigieren, falls Auslassung beabsichtigt
      ist (siehe offene Frage unten).
- [ ] `tools/aurora/Makefile:42,45` — fehlende Order-Only-Prerequisites `| $(OBJDIR)` /
      `| $(BINDIR)` (vgl. `tools/meteo/Makefile:43,46`, dort korrekt). Auf frischem Checkout
      (kein `obj/`/`bin/`, nicht versioniert) bricht der erste Compile mit "No such file or
      directory" ab. Fix: Order-only-Prerequisites analog meteo ergänzen.
- [x] (entfällt: Tool entfernt) `tools/indigo-mount-status/src/indigo_mount_status.c:197-213` — nutzt nicht-existente
      INDIGO-API (`indigo_client_attach()`, `indigo_service_attach()`, `indigo_client_detach()`).
      Alle anderen INDIGO-Clients im Repo (`indigo_cameras.c`, `indigo_mount_position.c`,
      `indigo_list_cameras.c`, `src/main.c`) nutzen einheitlich `indigo_attach_client()` /
      `indigo_connect_server()` / `indigo_disconnect_server()` / `indigo_detach_client()`.
      Fix: auf etabliertes Muster umstellen (analog `indigo_mount_position.c`); fehlendes
      `indigo_disconnect_server`-Äquivalent ergänzen.

## B. Speicherfehler / Undefined Behavior

- [ ] `tools/aurora/src/main.c:229,236-268` — Double-Free / Use-after-Free: `json_raw` wird nach
      `free()` (Z. 229) nicht auf `NULL` gesetzt. Schlägt `--kp`/`--mag`/`--plasma`-Fetch fehl,
      zeigt der Zeiger weiter auf freigegebenen Speicher (UAF bei `--debug`, danach erneuter
      Double-Free). Wiederholt sich für mag (Z. 243-254) und plasma (Z. 255-268).
      Fix: nach jedem `free(json_raw)` sofort `json_raw = NULL;`; Debug-Print an Fetch-Erfolg
      koppeln.
- [ ] `tools/meteo/src/main.c:176-185` — invertierte Freigabe-Bedingung: `if (!json_raw)
      free(json_raw);` sollte `if (json_raw)` heißen. Leak bei fehlgeschlagenem
      `parse_open_meteo_weather()`. Fix: Bedingung umkehren.
- [ ] `tools/meteo/src/open_meteo_fetch.c:36-37` — `mem->response = realloc(mem->response, ...)`
      ohne Zwischenvariable; bei `realloc`-Fehlschlag geht der alte Puffer verloren (Leak).
      Vergleiche korrekte Umsetzung in `tools/aurora/src/aurora_fetch.c:36-41`.
      Fix: Zwischenvariable verwenden wie in aurora_fetch.c.
- [ ] `tools/timelapse/src/image_loader_nvjpeg.cu:120-128` — GPU-Leak bei fehlgeschlagenem
      `nvjpegDecode()`: `d_dst` sowie `state`/`handle` werden nicht freigegeben (nur `d_src`/
      `jpeg_buffer`). Akkumuliert sich über lange Timelapse-Läufe.
      Fix: einheitlichen Cleanup-Pfad (`goto cleanup`) für alle Fehlerfälle verwenden.
- [ ] `tools/timelapse/src/image_loader_nvjpeg.cu:93-105` — GPU-Leak bei fehlgeschlagenem
      `cudaMalloc(d_src)`/`cudaMalloc(d_dst)`: `jpeg_buffer`/`state`/`handle` nicht freigegeben.
      Fix: siehe oben, gleicher Cleanup-Pfad.
- [ ] `tools/timelapse/src/stacker.c:33-59` — fehlende NULL-Prüfung für `ctx->background` bei
      `mode == STACK_SIGMA_CLIP` (nur für `STACK_DIFFERENCE` geprüft, Z. 54-56), obwohl
      `update_stack()` es auch im Sigma-Clip-Zweig nutzt. Bei OOM: NULL-Deref-Crash in
      OpenMP-Schleife. Fix: NULL-Check auf `mode == STACK_SIGMA_CLIP || mode ==
      STACK_DIFFERENCE` erweitern.

## C. Funktionale Regressionen / falsches Verhalten

- [ ] `tools/aurora/src/config.c:47-48`, `tools/meteo/src/config.c:47-48` — Leerzeichen um `=`
      werden nicht getrimmt → Config-Werte (`location_latitude`, `location_longitude`,
      `database_directory`) werden bei `key = value`-Schreibweise (Standard laut
      `config.ini.default`) stillschweigend verworfen. Der Haupt-Parser `src/config.c:275-287`
      wurde dafür bereits gefixt, die tools/-Kopien nie. Fix: Trim-Block aus
      `src/config.c:280-287` in beide `tools/*/src/config.c` übernehmen.
- [ ] `tools/aurora/src/config.c:50-53`, `tools/meteo/src/config.c:50-53` — Himmelsrichtungs-
      Suffixe S/W bei Config-Koordinaten ignoriert: `load_config()` nutzt `atof(value)` statt
      des im selben Verzeichnis vorhandenen, korrekten `coordinate_parser.c` (`parse_coordinate()`
      behandelt N/E/S/W korrekt inkl. Vorzeichenumkehr). Bei meteo ist `coordinate_parser`
      dadurch komplett toter Code. Gleiche Fehlerklasse wie BUG-10 in `src/`, hier gravierender.
      Fix: `atof(value)` durch `parse_coordinate(value, &config->location_latitude/longitude)`
      ersetzen.
- [ ] `tools/timelapse/src/stacker.c:40` — Motion-Stacking-Schwellwert-Regression:
      `ctx->motion_threshold = sigma_clip_threshold; //10.0f;` — auskommentierter Wert deutet auf
      früheren eigenständigen Default hin. Erbt jetzt unbedingt den `-t`-Sigma-Clip-Parameter
      (Default 2.0f) → rauschdominiertes statt ereignisdominiertes Motion-Ergebnis ohne
      explizites `-t 10`. Fix: eigenen Default für STACK_MOTION, unabhängig vom Sigma-Clip-
      Parameter.
- [ ] `tools/timelapse/src/video_encoder.c:58` — `init_encoder_nvenc()` nutzt kein NVENC: sucht
      `avcodec_find_encoder_by_name("libx264")`, kein Verweis auf `h264_nvenc` im Modul.
      Widerspricht Funktionsname, Header-Doku und CLAUDE.md ("nvJPEG + NVENC"). Videokodierung
      läuft aktuell immer CPU-gebunden. Fix: `h264_nvenc` mit Fallback auf `libx264` verwenden,
      oder Doku/Namensgebung korrigieren.
- [ ] `tools/timelapse/src/video_encoder.c:176` — fehlerhafte PTS-Vergabe: `frame->pts =
      frame_counter;`, aber `frame_counter` wird nur beim Paket-Empfang erhöht (Z. 212), nicht
      pro eingereichtem Frame. Mit `max_b_frames = 1` (Z. 93) → doppelte/nicht-monotone
      Timestamps im Ausgabevideo. Fix: separaten, pro `encode_frame()`-Aufruf inkrementierten
      Zähler für `pts` verwenden.
- [ ] `tools/timelapse/src/image_loader_nvjpeg.cu:113-136` — `cudaStreamSynchronize`
      synchronisiert falschen Stream: `nvjpegDecode()` läuft auf Stream `0`, `stream` wird erst
      danach angelegt und wirkungslos synchronisiert; `nppiResize` läuft auf `stream`.
      Funktioniert nur zufällig durch Legacy-Default-Stream-Semantik. Fix: `nvjpegDecode(...,
      stream)` mit dem tatsächlich genutzten Stream aufrufen, diesen synchronisieren.
- [ ] `tools/indigo-mount-position/indigo_mount_position.c:316-357` — `-m`/`--mount` wird von
      `-c`/`--config` überschrieben: `cli_mount` wird gesetzt aber mit `(void)cli_mount;`
      (Z. 357) nie ausgewertet, anders als bei `server`/`port`. Widerspricht dokumentierter
      Semantik "CLI arguments override values from -c/--config" (Z. 34). Fix: Mount-Namen
      analog Server/Port über Scratch-Buffer + `cli_mount`-Check behandeln.
- [x] (entfällt: Tool entfernt) `tools/location/src/magnetic_declination.c:244-247` — unbedingtes Debug-`printf` bei jedem
      Aufruf, unabhängig vom `--debug`-Flag in `main.c`. Verunreinigt reguläre Ausgabe.
      Fix: entfernen oder hinter globales `debug`-Flag stellen.

## D. Makefile-Probleme (Build-Robustheit)

- [x] (entfällt: Tool entfernt) `tools/indigo-mount-status/Makefile:40-58` (`tools/indigo-list-cameras` entfernt):
  - Z. 40: `-L$(INDIGO_LIBS)` — `INDIGO_LIBS` enthält bereits `-L.../lib -lindigo`, ergibt
    kaputtes `-L-L...` plus doppeltes `-lindigo` (Build funktioniert nur, weil Z. 46 den
    korrekten Flag separat liefert). Fix: Zeile 40 streichen.
  - Z. 42: `INCLUDE = $(INDIGOI_CFLAGS)` — Tippfehler, sollte `INDIGO_CFLAGS` heißen. Fix:
    korrigieren oder Zeile entfernen (redundant zu CFLAGS).
  - Z. 52-53: Binary wird nach `$(TARGET)` im Wurzelverzeichnis gelinkt statt
    `$(BINDIR)/$(TARGET)` — inkonsistent zu `indigo-cameras`/`indigo-mount-position`. Fix:
    Link-Ziel auf `$(BINDIR)/$(TARGET)` umstellen.
  - Z. 56: `rm -f $(OBJ) $(TARGET)` — `$(OBJ)` nirgends definiert, `clean` räumt `obj/` nie auf.
    Fix: `rm -rf $(OBJDIR) $(BINDIR)`.
- [ ] `tools/timelapse/Makefile:84` — `CU_SRCS =` statt `CU_SRC =` im CPU-Zweig (Tippfehler,
      aktuell folgenlos). Fix: `CU_SRC =` korrigieren.
- [ ] `tools/timelapse/Makefile:45` + Root-`Makefile:290` — `USE_NVJPEG` wird gesetzt/
      durchgereicht, aber nirgends ausgewertet (weder Make-Bedingung noch `-D`-Flag). Fix:
      entweder auswerten (nvJPEG optional abschaltbar machen) oder ersatzlos entfernen.
- [ ] `tools/timelapse/Makefile:76` — `-lcuda` (Driver-API) gelinkt, aber kein
      `cuInit`/`cuCtx*`/`cuDevice*`/`cuMemAlloc` im Quellcode gefunden — evtl. unnötige
      Abhängigkeit auf `libcuda.so` in treiberlosen Build-Umgebungen. Verifizieren, ob
      nvjpeg/npp es transitiv brauchen; sonst entfernen.
- [ ] `tools/timelapse/Makefile:67-76` — hart auf Jetson Orin zugeschnitten (`-arch=sm_87`,
      `gcc-11`, `aarch64-linux-gnu`-Pfade unconditional). Auf anderen Zielsystemen schlägt
      `make USE_CUDA=1` fehl oder baut falsche Architektur. Fix: `CUDA_ARCH ?= sm_87`
      parametrisierbar machen und dokumentieren.

## E. Plausible Risiken (nicht abschließend verifiziert)

- [x] (entfällt: Tool entfernt) `tools/location/src/main.c:138-151` — Fehlschlag von `fopen(outfile, "w")` wird nur auf
      stderr gemeldet, Exitcode bleibt `EXIT_SUCCESS`. Fix: bei Fehler `EXIT_FAILURE`
      zurückgeben.
- [ ] `tools/indigo-mount-position/indigo_mount_position.c:453-459` — bei Timeout mit aktivem
      `-o` bleibt eine leere `.tmp`-Ausgabedatei liegen (kein `remove()`); sammelt sich bei
      periodischem Cron-Aufruf an. Fix: `remove(output_tmp)` vor `return 2` ergänzen.
- [ ] `tools/aurora/src/aurora_fetch.c:110-116`, `tools/meteo/src/open_meteo_fetch.c:115-116` —
      `json_object_get_string()` kann `NULL` liefern (unerwartetes API-Schema); nachfolgende
      `strptime()`/`strcmp()` mit `NULL` sind UB. Fix: NULL-Rückgabe vor Verwendung abfangen.
- [ ] `tools/timelapse/src/main.c:256`, `stacker.c:34` — inkonsistente `size_t`-Casts bei
      Puffergrößenberechnung (`width * height * 3` teils ohne Cast); `-w`/`-h` aus `atoi()`
      ohne Obergrenze → theoretisches Int-Overflow-Risiko. Fix: konsequent `(size_t)width *
      (size_t)height * 3`; Obergrenze für -w/-h validieren.
- [ ] `tools/timelapse/src/image_loader_nvjpeg.cu:54-55` — Rückgabewerte von
      `nvjpegCreateSimple()`/`nvjpegJpegStateCreate()` ungeprüft. Fix: Rückgabewerte prüfen.
- [ ] `tools/timelapse/src/video_encoder.c:31,139-145` — `sws_ctx` (aus `sws_getContext()`) wird
      in `finalize_encoder()` nie via `sws_freeContext()` freigegeben. Für kurzlebigen Prozess
      folgenlos, aber inkonsistent. Fix: `sws_freeContext(sws_ctx)` ergänzen.

## F. Offene Fragen (Entscheidung/Verifikation durch Betreiber nötig)

- [x] (entfällt: Tool entfernt) WMM-Implementierung (`tools/location/src/magnetic_declination.c:124-249`) numerisch gegen
      offizielle NOAA-WMM-Testvektoren verifizieren.
- [x] (entfällt: Tool entfernt) Fehlende Root-Integration von panorama/list-cameras/mount-status (siehe A) — Regression
      oder nie fertig integriert? Entscheidet, ob Root-Makefile oder CLAUDE.md korrigiert wird.
- [x] (entfällt: Tool entfernt) `tools/indigo-mount-status/src/indigo_mount_status.c:127-136` — verbindet den Mount beim
      ersten Property-Update, trennt ihn aber nie wieder (anders als `indigo_cameras.c`).
      Beabsichtigt oder Versehen?
- [ ] Uneinheitliche curl-Absicherung zwischen `aurora_fetch.c` (Connect-Timeout,
      `FAILONERROR`, explizite SSL-Verifikation) und `open_meteo_fetch.c` (curl-Defaults) —
      Drift oder bewusst?
- [ ] `tools/timelapse/Makefile:99` installiert `libnpp-dev-12-6`, während `nppig` gelinkt wird
      — Paketaufteilung zwischen CUDA-Versionen nicht geprüft.

## G. Panorama

Keine Befunde — `tools/panorama/` ist in sich konsistent (main.c, Makefile, Funktionssignaturen
zwischen Header und Aufrufern stimmen, `apply_mask()` geht korrekt von `CAIRO_FORMAT_RGB24` aus).

---

**Hinweis:** `tools/location`, `tools/indigo-mount-status` und `tools/indigo-list-cameras` wurden entfernt.

**Empfohlene Reihenfolge:** A1 (location baut nicht) → A4 (indigo-mount-status vermutlich nicht
kompilierbar) → B1 (Double-Free aurora) → C4 (NVENC-Tool nutzt keine GPU-Kodierung) → Rest nach
Abschnitt.

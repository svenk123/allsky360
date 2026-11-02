# gpio-out

Schaltet die 4 Ausgänge `OUT1`-`OUT4` des I/O-Boards (PCA9534-I2C-GPIO-Expander,
`gpiochip2`) manuell **dauerhaft** ein oder aus und fragt ihren Zustand ab. Kein
RS485 -- nur GPIO. Gleiche Hardware und Zuordnung wie `magnetventil-steuern`,
aber ohne Zeitsteuerung.

| Ausgang | GPIO-Offset |
|---|---|
| OUT1 | 3 |
| OUT2 | 2 |
| OUT3 | 1 |
| OUT4 | 0 |

**Polarität (Annahme):** active-HIGH -- `ein` = Pegel HIGH.

## Bauen

```bash
make
sudo make install    # nach /usr/local/bin/gpio_out
```

Benötigt `libgpiod` (Entwicklungspaket) und Zugriff auf `/dev/gpiochip2`
(Gruppe `gpio`).

## Kommandozeile

```
gpio_out -o <1-4> -a <ein|aus> [-j]
gpio_out -s [-o <1-4>] [-j]
```

| Parameter | Kurzform | Beschreibung |
|---|---|---|
| `--output` | `-o` | Ausgang 1-4. Pflicht beim Schalten; bei `--status` optional (Default: alle vier). |
| `--action` | `-a` | `ein` oder `aus` (auch `an`, `on`, `off`, `1`, `0`). Der Ausgang bleibt dauerhaft in diesem Zustand, bis er wieder geschaltet wird. |
| `--status` | `-s` | Zustand abfragen, keine Änderung. Nicht mit `--action` kombinierbar. |
| `--json` | `-j` | Ausgabe (Erfolg und Fehler) als JSON statt Klartext. |
| `--help` | `-h` | Hilfe anzeigen. |

Exit-Code: `0` bei Erfolg, `1` bei Fehler.

## Wie "dauerhaft an" funktioniert

Auf diesem PCA9534 fällt ein auf HIGH gesetzter Ausgang sofort auf LOW zurück,
sobald der Prozess, der die GPIO-Line angefordert hat, sie freigibt oder
beendet wird (siehe `magnetventil-steuern`). Deshalb startet `--action ein` einen
kleinen **Halteprozess im Hintergrund** (`gpio_out_N`, sichtbar in `ps`), der die
Line auf HIGH hält. Das Tool selbst kehrt sofort zurück. `--action aus` beendet
diesen Prozess und setzt die Line auf LOW.

Folgen:
- Es gibt **keine Zeitbegrenzung und keinen Watchdog**: Ein Ausgang bleibt an,
  bis er ausgeschaltet wird. Für zeitbegrenztes Schalten von Ventilen
  `magnetventil_steuern` verwenden.
- Ein belegter Ausgang lässt sich von anderen Prozessen nicht anfordern
  (`gpioget` meldet `Device or resource busy`) -- das ist beabsichtigt.
- Wird der Halteprozess hart beendet (`kill -9`) oder das System neu gestartet,
  fällt der Ausgang auf LOW zurück (Fail-Safe). Nach einem Neustart sind alle
  Ausgänge aus.
- `ein` auf einen bereits eingeschalteten Ausgang ändert nichts und meldet
  Erfolg. Dasselbe gilt für `aus`.

## Beispiele

```
$ ./gpio_out -o 1 -a ein
OUT1: ein

$ ./gpio_out -s
OUT1: ein
OUT2: aus
OUT3: aus
OUT4: aus

$ ./gpio_out -o 1 -a aus
OUT1: aus
```

JSON:
```
$ ./gpio_out -o 4 -a ein -j
{"success": true, "tool": "gpio-out", "result": {"output": 4, "state": "ein"}, "timestamp": 1791996628}

$ ./gpio_out -s -j
{"success": true, "tool": "gpio-out", "result": {"outputs": [{"output": 1, "state": "aus"}, {"output": 2, "state": "aus"}, {"output": 3, "state": "aus"}, {"output": 4, "state": "aus"}]}, "timestamp": 1791996622}

$ ./gpio_out -o 4 -a aus -j
{"success": true, "tool": "gpio-out", "result": {"output": 4, "state": "aus", "changed": false}, "timestamp": 1791996630}
```

`changed: false` erscheint nur, wenn der Ausgang bereits im gewünschten Zustand
war. Ist eine Line von einem fremden Prozess belegt, meldet `--status`
`"state": "unbekannt"` mit `"busy_by": "<Verbraucher>"`.

## Fehlercodes

| error_code | wann | retry_safe |
|---|---|---|
| `INVALID_ARGS` | Ausgang außerhalb 1-4, fehlende Pflichtangabe, unbekannte Aktion/Option, `--status` mit `--action` | `false` |
| `HARDWARE_NOT_FOUND` | `gpiochip2` oder die Line existiert nicht | `false` |
| `PERMISSION_DENIED` | Kein Zugriff auf `/dev/gpiochip2` | `false` |
| `GPIO_WRITE_FAILED` | Line lässt sich nicht setzen/lesen, z.B. von einem fremden Prozess belegt | `true` |
| `HOLDER_INIT_FAILED` | `pipe()`/`fork()` für den Halteprozess schlägt fehl -- der Ausgang bleibt aus | `false` |

```
$ ./gpio_out -o 5 -a ein -j
{"success": false, "tool": "gpio-out", "error": "Ungueltiger Ausgang: 5 (erlaubt: 1-4)", "error_code": "INVALID_ARGS", "metadata": {"retry_safe": false}}
```

# modbus-scan

Sucht Modbus-RTU-Geräte am RS485-Bus. Das Tool fragt einen Adressbereich
nacheinander mit einem Read-Request ab (Default: Holding-Register 1). Jedes
Gerät, das mit einer CRC-gültigen Antwort reagiert, gilt als gefunden --
auch wenn es mit einer Modbus-Exception antwortet (z.B. "Register nicht
vorhanden"), denn auch das beweist, dass unter dieser Adresse ein Gerät sitzt.

Der Bus liegt wie bei den anderen Tools an `/dev/ttyTHS1` (Format 8N1); die
RTS/DE-Leitung wird über GPIO (libgpiod, `/dev/gpiochip0`, Line 112) gesteuert.
Die Baudrate ist einstellbar.

Für das Ändern einer Slave-Adresse siehe `modbus_tool`.

## Bauen

```bash
make
sudo make install    # nach /usr/local/bin/modbus_scan
```

Benötigt `libgpiod` (Entwicklungspaket).

## Kommandozeile

```
modbus_scan [-b BAUD] [-s START] [-e ENDE] [-r REGISTER] [-f FUNKTION] [-t TIMEOUT_MS] [-R VERSUCHE] [-j] [-h]
```

| Parameter | Kurzform | Pflicht | Wertebereich | Beschreibung |
|---|---|---|---|---|
| `--baud` | `-b` | optional | 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200 | Baudrate. Default: `4800`. |
| `--start` | `-s` | optional | 1-247 | Erste abzufragende Adresse. Default: `1`. |
| `--end` | `-e` | optional | 1-247 | Letzte abzufragende Adresse (muss ≥ Start sein). Default: `247`. |
| `--register` | `-r` | optional | 0-0xFFFF, dezimal oder `0x`-Hex | Abgefragtes Register. Default: `1`. |
| `--function` | `-f` | optional | `0x03`, `0x04` | Holding-Register (`0x03`) oder Input-Register (`0x04`). Default: `0x03`. |
| `--timeout` | `-t` | optional | 10-60000 | Timeout pro Anfrage in ms. Default: `1200`. |
| `--retries` | `-R` | optional | 1-10 | Versuche pro Adresse. Default: `2`. |
| `--json` | `-j` | optional | – | Ausgabe (Erfolg und Fehler) als JSON. |
| `--help` | `-h` | optional | – | Hilfe anzeigen. |

Exit-Code: `0` = mindestens ein Gerät gefunden, `2` = kein Gerät gefunden
(kein Fehler), `1` = Fehler.

**Dauer:** Für jede Adresse ohne Antwort wartet das Tool `TIMEOUT × VERSUCHE`.
Mit den Defaults (1200 ms, 2 Versuche) dauert ein voller Scan von 1-247 ohne
Treffer bis zu 10 Minuten. Die Defaults sind für Sensoren mit ~1 s Antwortzeit
gewählt -- mit weniger Timeout oder nur einem Versuch wurde der Wettersensor
nicht zuverlässig gefunden. Für schnelle Geräte geht es deutlich zügiger,
z.B. mit `-t 300 -R 1`. Zum Eingrenzen am besten `-s`/`-e` nutzen.

Findet der Scan bei der Standard-Baudrate nichts, andere Baudraten mit `-b`
probieren (Sensoren arbeiten oft mit 4800 oder 9600).

## Beispiele

Standard-Scan (Adressen 1-247, 4800 Baud):
```
$ ./modbus_scan
Scanne Adressen 1-247, 4800 Baud 8N1, Holding Register 1 ...
Adresse 1: Geraet gefunden, Register 1 = 1
Scan abgeschlossen: 1 Geraet(e) gefunden.
```

Eingeschränkter Bereich bei 9600 Baud, schneller Scan:
```
$ ./modbus_scan -b 9600 -s 1 -e 20 -t 300 -R 1
```

Anderes Register / Input-Register abfragen:
```
$ ./modbus_scan -r 0x01F4 -f 0x04
```

JSON:
```
$ ./modbus_scan -j -s 1 -e 2
{"success": true, "tool": "modbus-scan", "result": {"baud": 4800, "start": 1, "end": 2, "register": 1, "function": 3, "found": 1, "devices": [{"address": 1, "value": 1}]}, "timestamp": 1791994444}
```

Kein Gerät gefunden (Exit-Code 2):
```
$ ./modbus_scan -j -s 7 -e 7 -t 100 -R 1
{"success": true, "tool": "modbus-scan", "result": {"baud": 4800, "start": 7, "end": 7, "register": 1, "function": 3, "found": 0, "devices": []}, "timestamp": 1791994615}
```

| Feld in `result` | Bedeutung |
|---|---|
| `baud`, `start`, `end`, `register`, `function` | Verwendete Scan-Parameter |
| `found` | Anzahl gefundener Geräte |
| `devices[].address` | Modbus-Adresse des Geräts |
| `devices[].value` | Gelesener Registerwert (16 Bit, vorzeichenbehaftet) |
| `devices[].exception` | Modbus-Exception-Code, statt `value`, falls das Gerät mit einem Fehler antwortet |

## Fehlercodes

| error_code | wann | retry_safe |
|---|---|---|
| `INVALID_ARGS` | Baudrate nicht unterstützt, Adresse/Register/Timeout/Versuche außerhalb des Bereichs, Start > Ende, unbekannte Option | `false` |
| `HARDWARE_NOT_FOUND` | `/dev/ttyTHS1` oder `/dev/gpiochip0` nicht vorhanden bzw. GPIO-Line nicht verfügbar | `false` |
| `PERMISSION_DENIED` | Kein Zugriff auf serielles Gerät oder GPIO | `false` |
| `UNKNOWN_ERROR` | Anderer Fehler beim Öffnen des seriellen Geräts | `false` |

Ein leerer Bus ist **kein** Fehler (`success: true`, `found: 0`, Exit-Code 2).

```
$ ./modbus_scan -j -s 5 -e 3
{"success": false, "tool": "modbus-scan", "error": "Startadresse 5 liegt hinter der Endadresse 3", "error_code": "INVALID_ARGS", "metadata": {"retry_safe": false}}
```

# wind-direction

Liest einen RS485-Windrichtungsmesser per Modbus-RTU über `/dev/ttyTHS1`
(4800 Baud, 8N1) aus; die RTS/DE-Leitung wird wie bei `weather-sensor` über
GPIO (libgpiod, `/dev/gpiochip0`, Line 112) gesteuert. Die Windrichtung steht
im Holding-Register 1 (0-359 Grad). Ausgegeben werden Grad und
Himmelsrichtung (16-teilige Windrose, deutsche Kürzel: `O` = Ost).

## Bauen

```bash
make
sudo make install    # nach /usr/local/bin/wind_direction
```

Benötigt `libgpiod` (Entwicklungspaket).

## Kommandozeile

```
wind_direction [-a ADRESSE] [-o OFFSET] [-j] [-h]
```

| Parameter | Kurzform | Pflicht | Wertebereich | Beschreibung |
|---|---|---|---|---|
| `--address` | `-a` | optional | 1-247 | Modbus-Slave-Adresse. Default: `1`. |
| `--offset` | `-o` | optional | 0-359 | Korrekturwert für die Einbaulage. Wird auf die gemessene Richtung addiert (modulo 360), Ergebnis ist die absolute Richtung. Default: `0`. |
| `--json` | `-j` | optional | – | Ausgabe (Erfolg und Fehler) als JSON. |
| `--help` | `-h` | optional | – | Hilfe anzeigen. |

Exit-Code: `0` bei Erfolg, `1` bei jedem Fehler.

## Beispiele

Klartext:
```
$ ./wind_direction -o 90
Adresse: 1
Gemessen: 0 Grad
Offset: +90 Grad
Windrichtung: 90 Grad (O, Ost)
```

JSON:
```
$ ./wind_direction -j -o 90
{"success": true, "tool": "wind-direction", "result": {"address": 1, "direction_raw_deg": 0, "offset_deg": 90, "direction_deg": 90, "direction_abbr": "O", "direction_name": "Ost"}, "timestamp": 1791993122}
```

| Feld | Bedeutung |
|---|---|
| `direction_raw_deg` | Gemessener Wert des Sensors (ohne Offset) |
| `offset_deg` | Angewandter Korrekturwert |
| `direction_deg` | Korrigierte Windrichtung, 0-359 |
| `direction_abbr` / `direction_name` | Himmelsrichtung (z.B. `NNO` / `Nordnordost`) der korrigierten Richtung |

## Fehlercodes

| error_code | wann | retry_safe |
|---|---|---|
| `INVALID_ARGS` | Adresse oder Offset außerhalb des Bereichs, nicht-numerisch, unbekannte Option | `false` |
| `HARDWARE_NOT_FOUND` | `/dev/ttyTHS1` oder `/dev/gpiochip0` nicht vorhanden bzw. GPIO-Line nicht verfügbar | `false` |
| `PERMISSION_DENIED` | Kein Zugriff auf serielles Gerät oder GPIO | `false` |
| `SENSOR_TIMEOUT` | Keine gültige Antwort nach 3 Versuchen à 1,2 s | `true` |
| `SENSOR_OUT_OF_RANGE` | Sensorwert größer als 360 Grad | `false` |
| `UNKNOWN_ERROR` | Anderer Fehler beim Öffnen des seriellen Geräts | `false` |

```
$ ./wind_direction -j -a 5
{"success": false, "tool": "wind-direction", "error": "Keine gueltige Antwort erhalten (Timeout nach mehreren Versuchen), Adresse 5, Register 0x0001", "error_code": "SENSOR_TIMEOUT", "metadata": {"retry_safe": true}}
```

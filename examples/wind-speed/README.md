# wind-speed

Liest einen RS485-Windgeschwindigkeitssensor per Modbus-RTU über
`/dev/ttyTHS1` (4800 Baud, 8N1) aus; die RTS/DE-Leitung wird wie bei
`weather-sensor` über GPIO (libgpiod, `/dev/gpiochip0`, Line 112) gesteuert.
Die Windgeschwindigkeit steht im 16-Bit-Holding-Register 1. Ausgegeben werden
m/s, km/h, Knoten und die Beaufort-Stufe.

**Skalierung:** Default ist Teiler 1, d.h. der Rohwert wird als ganze m/s
interpretiert. Ein Vergleich mit dem Wetterstations-Sensor (gleich starkes
Anpusten: 3 m/s dort, Rohwert 3 hier) legt das nahe. **Noch nicht gegen eine
bekannte Windgeschwindigkeit verifiziert** -- liefert der Sensor in 0,1 m/s,
mit `-s 10` abfragen.

## Bauen

```bash
make
sudo make install    # nach /usr/local/bin/wind_speed
```

Benötigt `libgpiod` (Entwicklungspaket).

## Kommandozeile

```
wind_speed [-a ADRESSE] [-s TEILER] [-j] [-h]
```

| Parameter | Kurzform | Pflicht | Wertebereich | Beschreibung |
|---|---|---|---|---|
| `--address` | `-a` | optional | 1-247 | Modbus-Slave-Adresse. Default: `1`. |
| `--scale` | `-s` | optional | 0.001-100000 | Teiler: Rohwert / Teiler = m/s. Default: `1`. |
| `--json` | `-j` | optional | – | Ausgabe (Erfolg und Fehler) als JSON. |
| `--help` | `-h` | optional | – | Hilfe anzeigen. |

Exit-Code: `0` bei Erfolg, `1` bei jedem Fehler.

## Beispiele

Klartext:
```
$ ./wind_speed
Adresse: 1
Rohwert: 0 (Teiler 1)
Windgeschwindigkeit: 0.00 m/s = 0.00 km/h = 0.00 kn (Beaufort 0)
```

JSON:
```
$ ./wind_speed -j
{"success": true, "tool": "wind-speed", "result": {"address": 1, "raw": 0, "scale": 1, "wind_speed_ms": 0.00, "wind_speed_kmh": 0.00, "wind_speed_knots": 0.00, "beaufort": 0}, "timestamp": 1791993686}
```

Anderer Teiler (Rohwert = m/s × 10, Auflösung 0,1 m/s):
```
$ ./wind_speed -j -s 10
```

| Feld | Bedeutung |
|---|---|
| `raw` | Rohwert des Registers |
| `scale` | Verwendeter Teiler |
| `wind_speed_ms` / `_kmh` / `_knots` | Windgeschwindigkeit in m/s, km/h, Knoten |
| `beaufort` | Beaufort-Stufe 0-12 |

## Fehlercodes

| error_code | wann | retry_safe |
|---|---|---|
| `INVALID_ARGS` | Adresse oder Teiler außerhalb des Bereichs, nicht-numerisch, unbekannte Option | `false` |
| `HARDWARE_NOT_FOUND` | `/dev/ttyTHS1` oder `/dev/gpiochip0` nicht vorhanden bzw. GPIO-Line nicht verfügbar | `false` |
| `PERMISSION_DENIED` | Kein Zugriff auf serielles Gerät oder GPIO | `false` |
| `SENSOR_TIMEOUT` | Keine gültige Antwort nach 3 Versuchen à 1,2 s | `true` |
| `SENSOR_OUT_OF_RANGE` | Umgerechneter Wert über 60 m/s (Sensor defekt oder falscher Teiler) | `false` |
| `UNKNOWN_ERROR` | Anderer Fehler beim Öffnen des seriellen Geräts | `false` |

```
$ ./wind_speed -j -a 5
{"success": false, "tool": "wind-speed", "error": "Keine gueltige Antwort erhalten (Timeout nach mehreren Versuchen), Adresse 5, Register 0x0001", "error_code": "SENSOR_TIMEOUT", "metadata": {"retry_safe": true}}
```

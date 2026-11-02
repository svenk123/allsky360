# weather-sensor

Liest den RS485 9-in-1 Ultraschall-Wettersensor **Halisense HLS-UWS-SDTHA**
per Modbus-RTU über `/dev/ttyTHS1` (4800 Baud, 8N1) aus; die RTS/DE-Leitung
wird über GPIO (libgpiod, `/dev/gpiochip0`, Line 112) gesteuert. Alle
Messwerte werden in einem Block-Read (Register `0x01F4`-`0x0202`) geholt; schlägt
der fehl, liest das Tool jedes Register einzeln.

## Bauen

```bash
make
sudo make install    # nach /usr/local/bin/weather_sensor
```

## Kommandozeile

```
weather_sensor [-a ADRESSE] [-f FUNKTION] [-j] [-h]
```

| Parameter | Kurzform | Beschreibung |
|---|---|---|
| `--address` | `-a` | Modbus-Slave-Adresse 1-247, Default `1` (der Sensor im aktuellen Aufbau liegt auf Adresse `5`). |
| `--function` | `-f` | `0x03` Holding- (Default) oder `0x04` Input-Register. |
| `--json` | `-j` | Ausgabe als JSON statt Klartext. |
| `--help` | `-h` | Hilfe anzeigen. |

Exit-Code: `0` = alle Werte gelesen, `2` = mindestens ein Wert ohne Antwort,
`1` = Argumentfehler / Hardware nicht erreichbar.

## Registerbelegung

Die Belegung entspricht der DFRobot-SEN0658-Referenz. Die Register
`0x01F8`-`0x01FD` wurden am echten Sensor bestätigt (plausible Werte für
Feuchte, Temperatur, Noise und Druck). Die Reihenfolge weicht von der im
Internet gefundenen Liste ab (dort steht der Druck vor Noise).

| Register | JSON-Feld | Skalierung | Status |
|---|---|---|---|
| `0x01F4` | `wind_speed_ms` | Wert / 10 = m/s | nicht verifiziert (kein Wind beim Test) |
| `0x01F6` | `wind_direction_index` | 0-7, 0 = Nord, im Uhrzeigersinn | nicht verifiziert |
| `0x01F7` | `wind_direction_deg` | 0-360 Grad | nicht verifiziert |
| `0x01F8` | `humidity_percent` | Wert / 10 = %RH | verifiziert |
| `0x01F9` | `temperature_c` | signed, Wert / 10 = °C | verifiziert |
| `0x01FA` | `noise_db` | Wert / 10 = dB | verifiziert |
| `0x01FB` | `pm25_ug_m3` | µg/m³ | nicht verifiziert |
| `0x01FC` | `pm10_ug_m3` | µg/m³ | nicht verifiziert |
| `0x01FD` | `pressure_hpa` | Wert = kPa × 10 = hPa | verifiziert |
| `0x01FE`/`0x01FF` | `illuminance_lux` | 32 Bit, High-Word zuerst (angenommen) | nicht verifiziert |
| `0x0201` | `rainfall_mm` | Wert / 10 | nicht verifiziert |
| `0x0202` | `solar_irradiance_raw` | roh | **Adresse geraten**, unverifiziert |

Windrichtung und Windgeschwindigkeit haben eigene Tools (`wind-direction`,
`wind-speed`) für getrennte Sensoren.

## Beispiel

```
$ ./weather_sensor -a 5
Adresse: 5
Windgeschwindigkeit (m/s): 0.0
Windrichtung (Index 0-7): 0
Windrichtung (Grad): 0
Luftfeuchtigkeit (%RH): 57.7
Temperatur (C): 23.3
Noise (dB): 30.0
PM2.5 (ug/m3): 0
PM10 (ug/m3): 0
Luftdruck (hPa): 1010.0
Beleuchtung (Lux): 0
Regen (mm): 0.0
Solarstrahlung (roh, unverifiziert): 0
```

```
$ ./weather_sensor -a 5 -j
{"address":5,"wind_speed_ms":0.0,"wind_direction_index":0,"wind_direction_deg":0,"humidity_percent":57.7,"temperature_c":23.3,"noise_db":30.0,"pm25_ug_m3":0,"pm10_ug_m3":0,"pressure_hpa":1010.0,"illuminance_lux":0,"rainfall_mm":0.0,"solar_irradiance_raw":0}
```

Ein Wert, der nicht gelesen werden konnte, erscheint im JSON als `null` mit
zusätzlichem `<feld>_error`.

**Hinweis:** Die JSON-Feldnamen haben sich gegenüber der früheren Version
geändert (die alte Registerbelegung war falsch, z.B. `wind_speed_raw` →
`wind_speed_ms`). Das Ausgabeformat ist weiterhin ein flaches Objekt (kein
`success`/`result`-Wrapper wie bei den neueren Tools).

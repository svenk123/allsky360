# meteo

Ruft aktuelle Wetterdaten (Temperatur, Luftfeuchte, Taupunkt, Luftdruck,
Bewölkung, Sichtweite, Windgeschwindigkeit/-richtung) über die kostenlose
[Open-Meteo](https://open-meteo.com/) Forecast-API ab und gibt sie als JSON
aus. Anders als die übrigen Tools in diesem Verzeichnis liest `meteo` **keinen
lokalen Sensor**, sondern externe Wetterdaten für die konfigurierten
Koordinaten -- gedacht als Ergänzung zum lokalen `weather-sensor`-Tool
(dessen Register bis auf die Windgeschwindigkeit noch unverifiziert sind).

Übernommen aus [svenk123/allsky360](https://github.com/svenk123/allsky360)
(`tools/meteo`), unverändert bis auf:

- Installpfad `/usr/local/bin` statt `/opt/allsky360/bin` (Konvention dieses
  Repos).
- Neue Parameter `-a/--latitude` und `-o/--longitude`, mit denen die
  Koordinaten direkt übergeben werden können, ohne eine Config-Datei
  anzulegen (`-c/--config` bleibt möglich und wird bei gleichzeitiger Angabe
  von `--latitude`/`--longitude` von diesen überschrieben).

## Bauen

Benötigt `libcurl`, `json-c` und `sqlite3` (Entwicklungspakete):

```
sudo apt install libcurl4-openssl-dev libjson-c-dev libsqlite3-dev
make
sudo make install    # installiert nach /usr/local/bin/allsky360-meteo
```

## Kommandozeile

```
allsky360-meteo (-c KONFIGDATEI | -a BREITENGRAD -o LAENGENGRAD) [-t ZEITSTEMPEL] [-f AUSGABEDATEI] [-d DATENBANKVERZEICHNIS] [-v] [-h]
```

| Parameter | Kurzform | Pflicht | Beschreibung |
|---|---|---|---|
| `--latitude` | `-a` | ja, außer `-c` ist gesetzt | Breitengrad, Dezimalgrad oder mit N/S-Suffix (z.B. `50.3667` oder `50.3667N`). Überschreibt den Wert aus der Config-Datei. |
| `--longitude` | `-o` | ja, außer `-c` ist gesetzt | Längengrad, Dezimalgrad oder mit E/W-Suffix (z.B. `8.75` oder `8.75E`). **Achtung:** nur englische Suffixe `N/S/E/W` werden erkannt, kein deutsches `O` für Ost -- dafür einfach das Vorzeichen weglassen (Ost/Nord sind positiv). Überschreibt den Wert aus der Config-Datei. |
| `--config` | `-c` | ja, außer `-a`/`-o` sind beide gesetzt | Ini-Konfigurationsdatei mit `latitude`, `longitude`, `altitude`, `database_directory`. |
| `--timestamp` | `-t` | optional | UTC-Zeitstempel `YYYYMMDDhhmmss`, für den die Stunden-Vorhersage abgefragt wird. Default: jetzt. |
| `--outfile` | `-f` | optional | Schreibt das JSON zusätzlich in eine Datei. |
| `--databasedir` | `-d` | optional | Schreibt den Messwert zusätzlich in eine eigene SQLite-Tageszdatenbank (`database_YYYYMMDD.db`) in diesem Verzeichnis. Für die Integration in dieses Projekt **nicht genutzt** -- `poller/weather_poller.py` liest nur das JSON von stdout und schreibt selbst in die zentrale `state.db`. |
| `--debug` | `-v` | optional | Ausführliche Debug-Ausgaben (Rohantwort der API). |
| `--help` | `-h` | optional | Hilfetext anzeigen. |

Exit-Code: `0` bei Erfolg, `1` bei Fehler (Netzwerk, Parsing, fehlende
Koordinaten).

## Beispiel

```
$ ./bin/allsky360-meteo -a 50.3667 -o 8.75
-----------------------
{
  "source": "Open-Meteo",
  "temperature": 18.4,
  "humidity": 72,
  "dew_point": 13.2,
  "pressure_msl": 1017.3,
  "surface_pressure": 1005.1,
  "cloud_cover": 40,
  "cloud_low": 20,
  "cloud_mid": 10,
  "cloud_high": 15,
  "visibility": 24140,
  "wind_speed_10m": 11.2,
  "wind_dir_10m": 220,
  "wind_speed_300hPa": 55.8
}
```

Windgeschwindigkeiten (`wind_speed_10m`, `wind_speed_300hPa`) sind in **km/h**,
nicht m/s -- `poller/weather_poller.py` rechnet das beim Import in die
`state.db` um.

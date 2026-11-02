# modbus_tool

Hilfswerkzeug zum Scannen von Modbus-RTU-Slave-Adressen und zum Ändern
der Slave-Adresse eines RS485-Sensors (z.B. soil-sensor, solar_sensor).

## Adresse eines Sensors ändern

Voraussetzung: Das Konfigurationsregister für die Slave-Adresse muss
bekannt sein (siehe Datenblatt des Sensors). Für die hier eingesetzten
Sensoren ist das Register **0x7D0 (2000)**.

```bash
./modbus_tool --set-address -c AKTUELLE_ADRESSE -n NEUE_ADRESSE -r 0x7D0
```

Beispiel: Sensor antwortet aktuell unter Adresse 5, soll auf 10 geändert werden:

```bash
./modbus_tool --set-address -c 5 -n 10 -r 0x7D0
```

- `-c` Aktuelle Adresse (Default: 1, dem Werkszustand)
- `-n` Neue Adresse (1-247, Pflicht)
- `-r` Konfigurationsregister, dezimal oder als `0x`-Hex (Pflicht bei `--set-address` –
  bewusst kein Default, um nicht versehentlich das falsche Register zu beschreiben)
- `-j` Ausgabe als JSON statt Klartext

**Vorsicht:** Der Befehl schreibt sofort auf die echte Hardware. Danach
antwortet der Sensor nur noch unter der neuen Adresse.

## Adresse eines Sensors finden (Scan)

Falls die aktuelle Adresse unbekannt ist, alle Adressen 1-247 absuchen:

```bash
./modbus_tool --scan -r 0x7D0 -j
```

Liest bei jeder Adresse das angegebene Register; Treffer werden mit
Adresse und gelesenem Wert ausgegeben.

## Bauen

```bash
make
```

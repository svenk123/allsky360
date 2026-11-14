# RS485 Weather Sensors (optional)

Allsky360 can read its own weather station over an RS485 bus (Modbus RTU):
temperature, humidity, pressure, wind, rain, light and the climate in the
control cabinet. The service `allsky360-sensors` stores the values in the
daily database and the web interface shows them on the *Weather* page next to
the Open-Meteo data.

> **You do not need this.** Without sensors, all sensor addresses stay `0`
> (the default), the service does nothing and stops, and the *Weather* page
> hides the sensor tiles. Weather data from Open-Meteo is available without
> any hardware.

Technical reference (command-line options, database table, wind averaging):
[TOOLS.md – allsky360-sensors](TOOLS.md#8-allsky360-sensors-rs485--modbus-sensors).

---

## 1. Supported sensors

Any combination of these four sensors can be connected; each one is optional.

| Sensor | Values | Keys |
|---|---|---|
| **Weather sensor** Halisense HLS-UWS-SDTHA (also sold as DFRobot SEN0658) | temperature, humidity, dew point, pressure, wind speed and direction, noise, PM2.5, PM10, illuminance, rain | `modbus_weather_*` |
| **Anemometer** (Modbus RTU) | wind speed | `modbus_wind_speed_*` |
| **Wind vane** (Modbus RTU) | wind direction 0 – 359° | `modbus_wind_direction_*` |
| **Temperature/humidity sensor** in the control cabinet | temperature, humidity, dew point | `modbus_cabinet_*` |

The anemometer, wind vane and cabinet sensor are simple Modbus sensors that
deliver one value per register; the register numbers and the scale are set in
`config.ini`. Sensors with a different data layout need a change in
`tools/sensors/src/main.c`.

---

## 2. Wiring

RS485 is a two-wire bus: all sensors hang in parallel on the lines **A** and
**B**. Rules:

* Connect **A to A and B to B** on every device (some makers label them
  `D+`/`D-` or `485+`/`485-`).
* Use a twisted pair cable; for long cables put a **120 Ω terminating
  resistor** between A and B at both ends of the bus.
* Connect the **ground** of the sensors' power supply with the ground of the
  adapter.
* Power the sensors as their manual says (usually 12 or 24 V DC) – not from
  the RS485 adapter.

### Variant A: USB-RS485 adapter (simplest)

```
 Jetson / PC                USB-RS485 adapter            sensors
 ┌──────────┐   USB    ┌───────────────────┐   A ──┬──────┬──────┬───
 │          ├──────────┤  automatic        │   B ──┼──┬───┼──┬───┼──┬
 └──────────┘          │  direction        │  GND ─┼──┼─┬─┼──┼─┬─┼──┼─┬
                       └───────────────────┘       │  │ │ │  │ │ │  │ │
                                                 weather  anemo-   wind
                                                 sensor   meter    vane …
```

```ini
modbus_device=/dev/ttyUSB0
modbus_gpio_line=-1
```

`-1` means: the adapter switches between sending and receiving by itself.

### Variant B: Jetson UART with an RS485 transceiver

A 3.3 V RS485 transceiver module (e.g. with a MAX3485) on the 40-pin header
of the Jetson Orin. The transceiver needs a direction signal (DE/RE), which
Allsky360 switches with a GPIO:

| Jetson pin | Signal | Transceiver |
|---|---|---|
| 8 | UART1 TXD | DI |
| 10 | UART1 RXD | RO |
| 11 | UART1_RTS used as GPIO (PR.04) | DE and RE (bridged) |
| 1 | 3.3 V | VCC |
| 6 | GND | GND |

```ini
modbus_device=/dev/ttyTHS1
modbus_gpio_chip=/dev/gpiochip0
modbus_gpio_line=112
```

Check the pin assignment of your carrier board; the values above are for the
Jetson Orin Nano / Orin NX developer kit.

---

## 3. Sensor addresses

Every device on the bus needs its **own address** (1 – 247), and all must use
the **same baud rate**. Many sensors ship with address 1, so they cannot run
together out of the box:

1. Connect **one sensor at a time**.
2. Set its address (and the baud rate, if needed) as its manual describes –
   usually by writing a holding register with a Modbus tool or the maker's
   software.
3. Note the address, then connect the next sensor.

Example from the reference station: weather sensor `5`, anemometer `11`,
wind vane `12`, control cabinet `13`, all at 4800 baud.

---

## 4. Configuration

Enter the addresses with the configuration assistant:

```bash
sudo ./install.sh --configure
```

or directly in `/home/allskyuser/config.ini`:

```ini
modbus_baudrate=4800
modbus_function=3              # 3 = holding registers, 4 = input registers

modbus_weather_address=5
modbus_weather_register=0x01F4
modbus_weather_wind_direction_offset=0

modbus_wind_speed_address=11
modbus_wind_speed_register=1
modbus_wind_speed_scale=1.0    # m/s = raw value / scale

modbus_wind_direction_address=12
modbus_wind_direction_register=1
modbus_wind_direction_offset=0 # mounting correction in degrees

modbus_cabinet_address=13
modbus_cabinet_register=0      # humidity here, temperature at register + 1
```

`0` as address switches a sensor off. All keys:
[CONFIGURATION.md](CONFIGURATION.md#rs485--modbus-sensors-optional).

**Wind direction offset:** if the vane (or the weather sensor) is not mounted
with its north mark to geographic north, enter the angle by which it is
turned; it is added to every reading.

---

## 5. Test and start

Stop the service (it holds the serial port), read all sensors once without
writing anything, and look at the result:

```bash
sudo systemctl stop allsky360-sensors.service
sudo -u allskyuser /opt/allsky360/bin/allsky360-sensors -c /home/allskyuser/config.ini -n -v
```

`-v` shows the configuration and every failed attempt with the received
bytes. When all sensors answer with plausible values, start the service and
enable it at boot:

```bash
sudo systemctl enable --now allsky360-sensors.service
journalctl -u allsky360-sensors.service -f
```

After about two minutes of wind samples the first reading is stored, then one
every `modbus_interval` seconds (default 5 minutes). Every stored reading
writes a line to the journal, e.g.:

```
4 of 4 sensors; wind 10 min: 2.31 m/s (max 4.2), 245 deg, consistency 0.82, 41 of 60 samples windy
```

✅ **Check:** the *Weather* page shows the sensor tiles with current values.

---

## 6. What happens with the data

* **Database:** one row per reading in the table `sensors` of
  `database_YYYYMMDD.db` (UTC day).
* **Latest reading:** `/opt/allsky360/allsky-api/public/latest_sensor_data.json`,
  also served as `http://<hostname>:3000/latest_sensor_data.json`.
* **History of a day:** `GET /api/sensors?date=YYYY-MM-DD` ([API.md](API.md)).
* **Web interface:** tiles and charts on *Weather*, cabinet climate on
  *System Status*. A reading older than 15 minutes is marked *stale*.
* **Wind:** the anemometer and the vane are sampled every 10 seconds; a
  stored reading holds the **10-minute mean** (speed: average; direction:
  vector mean weighted by speed, as recommended by the WMO). In calm air the
  last direction is kept. Details in [TOOLS.md](TOOLS.md).
* **Plausibility:** impossible values (e.g. humidity above 100 %, pressure
  outside 300 – 1100 hPa) and values of sensors that did not answer are
  stored as empty (`NULL`) and shown as gaps.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Permission denied` on `/dev/ttyTHS1`, `/dev/ttyUSB0` or `/dev/gpiochip0` | `allskyuser` needs the groups `dialout` and `gpio`; `make install` adds them. Check with `id allskyuser`. |
| A sensor never answers | Wrong address or baud rate; A and B swapped; sensor not powered; missing common ground. Test this sensor alone on the bus. |
| Answers only sometimes | Missing termination on long cables; raise `modbus_timeout_ms` (sensors need up to ~1 s) or `modbus_retries`. |
| No sensor answers with the Jetson UART | `modbus_gpio_line` wrong or the transceiver's DE/RE not connected; with a USB adapter set `modbus_gpio_line=-1`. |
| Wrong wind speed | Adjust `modbus_wind_speed_scale` to the sensor (raw value / scale = m/s; e.g. `10` for a sensor that sends 0.1 m/s steps). |
| Wind direction turned | Set `modbus_wind_direction_offset` / `modbus_weather_wind_direction_offset`. |
| Service is "inactive" right after start | All addresses are `0` – no sensor configured. |
| `ERROR: … is busy (locked by another process)` | The service is still running; stop it before a manual test. |

The journal shows a warning when a sensor stays silent for 60 s and a line
when it answers again: `journalctl -u allsky360-sensors.service`.

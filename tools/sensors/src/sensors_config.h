/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#ifndef SENSORS_CONFIG_H
#define SENSORS_CONFIG_H

/* Settings for the RS485/Modbus sensors (keys modbus_* in config.ini).
 * A sensor address of 0 disables that sensor. */
typedef struct {
  char database_directory[256];

  /* Bus */
  char device[128];
  int baudrate;
  char gpio_chip[128];
  int gpio_line; /* -1: transceiver switches direction by itself */
  int timeout_ms;
  int retries;
  int function; /* 0x03 holding or 0x04 input registers */

  /* Weather sensor (Halisense HLS-UWS-SDTHA) */
  int weather_address;
  int weather_register; /* first register of the measurement block */
  int weather_wind_direction_offset;

  /* Anemometer */
  int wind_speed_address;
  int wind_speed_register;
  double wind_speed_scale; /* raw / scale = m/s */

  /* Wind vane */
  int wind_direction_address;
  int wind_direction_register;
  int wind_direction_offset; /* mounting correction, added modulo 360 */

  /* Control cabinet temperature/humidity (humidity at register,
   * temperature at register + 1) */
  int cabinet_address;
  int cabinet_register;
} sensors_config_t;

void sensors_config_defaults(sensors_config_t *config);

/**
 * Read the modbus_* keys and database_directory from config.ini.
 * Invalid values are reported on stderr and the default is kept.
 * @return 0 on success, -1 if the file cannot be read
 */
int sensors_config_load(const char *filename, sensors_config_t *config);

void sensors_config_print(const sensors_config_t *config);

#endif /* SENSORS_CONFIG_H */

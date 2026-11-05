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
#ifndef SENSORS_DATABASE_H
#define SENSORS_DATABASE_H

#include <time.h>

/* One reading of all RS485 sensors. NAN = not available (sensor disabled,
 * no answer or implausible value) and is stored as SQL NULL. */
typedef struct {
  time_t timestamp; /* UTC */

  /* Weather sensor */
  double temperature;       /* deg C */
  double humidity;          /* %RH */
  double dew_point;         /* deg C, calculated */
  double pressure;          /* hPa */
  double ws_wind_speed;     /* m/s */
  double ws_wind_direction; /* deg, 0 = north */
  double noise;             /* dB */
  double pm25;              /* ug/m3 */
  double pm10;              /* ug/m3 */
  double illuminance;       /* lux */
  double rainfall;          /* mm */
  double solar_irradiance_raw; /* unverified register, raw value */

  /* Anemometer and wind vane */
  double wind_speed;     /* m/s */
  double wind_direction; /* deg, 0 = north */

  /* Control cabinet */
  double cabinet_temperature; /* deg C */
  double cabinet_humidity;    /* %RH */
  double cabinet_dew_point;   /* deg C, calculated */
} sensors_reading_t;

/* Set all values to NAN */
void sensors_reading_init(sensors_reading_t *r, time_t timestamp);

/**
 * Create the table 'sensors' in the daily database if needed and insert the
 * reading (INSERT OR REPLACE on the timestamp).
 * @return 0 on success
 */
int sensors_database_store(const char *db_filename, const sensors_reading_t *r);

/**
 * Latest non-NULL value of a column of table 'sensors' with a timestamp
 * before 'before'. column must be a fixed column name, not user input.
 * @return 0 if found (value and its timestamp set), -1 if the database,
 *         the table or a value is missing
 */
int sensors_database_last_value(const char *db_filename, const char *column,
                                time_t before, double *value, time_t *timestamp);

#endif /* SENSORS_DATABASE_H */

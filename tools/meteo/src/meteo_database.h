/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license 
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#ifndef METEO_DATABASE_H
#define METEO_DATABASE_H

#include <time.h>

/**
 * Create a new SQLite database file with the required table schema,
 * if the file does not already exist.
 *
 * @param db_filename Path to the SQLite database file.
 * @return 0 on success, >0 on error.
 */
int create_meteo_database(const char *db_filename);

/**
 * Insert a new measurement into the SQLite database.
 *
 * @param db_filename Path to the SQLite database file.
 * @param timestamp   Time of the measurement (Unix timestamp).
 * @return 0 on success, >0 on failure.
 */
int insert_measurement(
    const char *db_filename,
    time_t timestamp,
    double temperature,
    double humidity,
    double dew_point,
    double pressure_msl,
    double surface_pressure,
    double cloud_cover,
    double cloud_low,
    double cloud_mid,
    double cloud_high,
    double visibility,
    double wind_speed_10m,
    double wind_dir_10m,
    double wind_speed_300hPa);

#endif // METEO_DATABASE_H


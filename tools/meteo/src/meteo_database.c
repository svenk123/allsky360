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
#include <sqlite3.h>
#include <sys/stat.h>
#include <stdio.h>
#include "meteo_database.h"

int create_meteo_database(const char *db_filename) {
//    struct stat st;
//    if (stat(db_filename, &st) == 0) {
        // File already exists
//        return 0;
//    }

    sqlite3 *db;
    int rc = sqlite3_open(db_filename, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    const char *sql = "CREATE TABLE IF NOT EXISTS meteo ("
                      "timestamp INTEGER PRIMARY KEY, "
                      "temperature REAL, "
                      "humidity REAL, "
                      "dew_point REAL, "
		      "pressure_msl REAL, "
		      "surface_pressure REAL, "
		      "cloud_cover REAL, "
                      "cloud_low REAL, "
		      "cloud_mid REAL, "
                      "cloud_high REAL, "
		      "visibility REAL, "
		      "wind_speed_10m REAL, "
		      "wind_dir_10m REAL, "
		      "wind_speed_300hPa REAL);";

    char *errmsg = NULL;
    rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", errmsg);
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return 2;
    }

    sqlite3_close(db);
    return 0;
}

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
    double wind_speed_300hPa) 
{
    sqlite3 *db;
    sqlite3_stmt *stmt;
    int rc = sqlite3_open(db_filename, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    // Enable WAL mode for crash resilience
    rc = sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
	fprintf(stderr, "Failed to set WAL mode: %s\n", sqlite3_errmsg(db));
	sqlite3_close(db);
	return 3;
    }

    const char *sql = "INSERT INTO meteo (timestamp, temperature, humidity, dew_point, pressure_msl, surface_pressure, cloud_cover, cloud_low, cloud_mid, cloud_high, visibility, wind_speed_10m, wind_dir_10m, wind_speed_300hPa) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare insert: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 2;
    }

    sqlite3_bind_int64(stmt, 1, timestamp);
    sqlite3_bind_double(stmt, 2, temperature);
    sqlite3_bind_double(stmt, 3, humidity);
    sqlite3_bind_double(stmt, 4, dew_point);
    sqlite3_bind_double(stmt, 5, pressure_msl);
    sqlite3_bind_double(stmt, 6, surface_pressure);
    sqlite3_bind_double(stmt, 7, cloud_cover);
    sqlite3_bind_double(stmt, 8, cloud_low);
    sqlite3_bind_double(stmt, 9, cloud_mid);
    sqlite3_bind_double(stmt, 10, cloud_high);
    sqlite3_bind_double(stmt, 11, visibility);
    sqlite3_bind_double(stmt, 12, wind_speed_10m);
    sqlite3_bind_double(stmt, 13, wind_dir_10m);
    sqlite3_bind_double(stmt, 14, wind_speed_300hPa);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return (rc == SQLITE_DONE) ? 0 : 3;
}

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
#include "aurora_database.h"

int create_aurora_database(const char *db_filename) {
//    struct stat st;
//    if (stat(db_filename, &st) == 0) {
        // File already exists
//        return 0;
//    }

    sqlite3 *db;
    int rc = sqlite3_open(db_filename, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db); /* sqlite3_open() allocates a handle even on failure */
        return 1;
    }


    /* WAL once per database instead of on every insert: it writes to the
     * database header and is itself subject to locking. */
    sqlite3_busy_timeout(db, 5000);
    rc = sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to set WAL mode: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 3;
    }

    const char *sql = "CREATE TABLE IF NOT EXISTS aurora ("
                      "timestamp INTEGER PRIMARY KEY, "
                      "probability_percent REAL, "
                      "probability_max REAL, "
                      "probability_avg REAL, "
                      "kp_index REAL, "
                      "bt REAL, "
                      "bz REAL, "
                      "density REAL, "
                      "speed REAL, "
                      "temperature REAL);";

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
    double probability_percent,
    double probability_max,
    double probability_avg,
    double kp_index,
    double bt,
    double bz,
    double density,
    double speed,
    double temperature) 
{
    sqlite3 *db;
    sqlite3_stmt *stmt;
    int rc = sqlite3_open(db_filename, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db); /* sqlite3_open() allocates a handle even on failure */
        return 1;
    }

    /* Four processes write to the same daily database (camera daemon, aurora,
     * meteo, location). Without a busy timeout a concurrent writer turns into
     * SQLITE_BUSY immediately and the measurement is lost (review.md, REL-07). */
    sqlite3_busy_timeout(db, 5000);

    const char *sql = "INSERT INTO aurora (timestamp, probability_percent, probability_max, probability_avg, kp_index, bt, bz, density, speed, temperature) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare insert: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 2;
    }

    sqlite3_bind_int64(stmt, 1, timestamp);
    sqlite3_bind_double(stmt, 2, probability_percent);
    sqlite3_bind_double(stmt, 3, probability_max);
    sqlite3_bind_double(stmt, 4, probability_avg);
    sqlite3_bind_double(stmt, 5, kp_index);
    sqlite3_bind_double(stmt, 6, bt);
    sqlite3_bind_double(stmt, 7, bz);
    sqlite3_bind_double(stmt, 8, density);
    sqlite3_bind_double(stmt, 9, speed);
    sqlite3_bind_double(stmt, 10, temperature);

    /* Retry a few times: busy_timeout already waits, but a writer that gives up
     * the lock late can still surface as SQLITE_BUSY here. */
    int attempt = 0;
    do {
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_BUSY && rc != SQLITE_LOCKED)
            break;
        attempt++;
        fprintf(stderr, "Database busy, retrying insert (%d/3)\n", attempt);
        sqlite3_reset(stmt);
    } while (attempt < 3);

    if (rc != SQLITE_DONE)
        fprintf(stderr, "Insert failed: %s (code %d)\n", sqlite3_errmsg(db), rc);

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return (rc == SQLITE_DONE) ? 0 : 3;
}

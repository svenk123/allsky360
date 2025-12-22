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
#ifndef AURORA_DATABASE_H
#define AURORA_DATABASE_H

#include <time.h>

/**
 * Create a new SQLite database file with the required table schema,
 * if the file does not already exist.
 *
 * @param db_filename Path to the SQLite database file.
 * @return 0 on success, >0 on error.
 */
int create_aurora_database(const char *db_filename);

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
    double probability_percent,
    double probability_max,
    double probability_avg,
    double kp_index,
    double bt,
    double bz,
    double density,
    double speed,
    double temperature);

#endif // AURORA_DATABASE_H


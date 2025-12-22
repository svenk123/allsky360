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
#ifndef DAILY_DATABASE_H
#define DAILY_DATABASE_H

/**
 * Create a new SQLite database file with the required table schema,
 * if the file does not already exist.
 *
 * @param db_filename: path to the SQLite database file.
 * @return: 0 on success, >0 on error.
 */
int create_daily_database(const char *db_filename);

/**
 * Insert a new measurement into the SQLite database.
 *
 * @param db_filename: path to the SQLite database file.
 * @param timestamp: time of the measurement (Unix timestamp).
 * @param timezone_offset: timezone offset (seconds).
 * @param exposure: exposure time (seconds).
 * @param gain: camera gain.
 * @param brightness: mean image brightness.
 * @param mean_r: mean red channel value.
 * @param mean_g: mean green channel value.
 * @param mean_b: mean blue channel value.
 * @param noise: image noise level.
 * @param hdr: whether HDR was used.
 * @param night_mode: 1 if in night mode, 0 otherwise.
 * @param stars: number of stars detected.
 * @param sqm: SQM value.
 * @param focus: focus value.
 * @return: 0 on success, >0 on failure.
 */
int insert_measurement(
    const char *db_filename,
    time_t timestamp,
    int timezone_offset,
    double exposure,
    double gain,
    double brightness,
    double mean_r,
    double mean_g,
    double mean_b,
    double noise,
    int hdr,
    int night_mode,
    int stars,
    double sqm,
    double focus);

#endif // DAILY_DATABASE_H


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
#ifndef AURORA_FETCH_H
#define AURORA_FETCH_H

#include <stddef.h>

/* Return codes */
#define FETCH_SUCCESS                0
#define FETCH_ERROR_CURL_INIT        1
#define FETCH_ERROR_CURL_TRANSFER    2
#define PARSE_ERROR_JSON             3
#define PARSE_ERROR_DATA             4

/* Solar wind values farther than this (seconds) from the requested time are
 * stale and not used (RTSW data arrive with a few minutes delay). */
#define SOLAR_WIND_MAX_AGE           (30 * 60)

/* NOAA SWPC endpoints */
#define URL_OVATION   "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
/* Estimated planetary Kp, one value per minute (last ~6 hours) */
#define URL_KP_1M     "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
/* Planetary Kp, 3-hourly (last 7 days), fallback for older timestamps */
#define URL_KP_3H     "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
/* Real-time solar wind, 1-minute values of all spacecraft (last 24 hours);
 * replaced products/solar-wind/{mag,plasma}-2-hour.json (gone, HTTP 404) */
#define URL_RTSW_MAG  "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
#define URL_RTSW_WIND "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"

/**
 * @brief Download raw JSON data from a given HTTPS endpoint.
 *
 * @param url The full URL to the JSON file (e.g., NOAA API).
 * @param out_json Pointer to store the allocated JSON string (must be free()-ed by caller).
 * @return int 0 on success, >0 on error (see FETCH_* macros).
 */
int fetch_aurora_json(const char *url, char **out_json);

/**
 * Parses the NOAA Ovation Aurora JSON data and extracts relevant forecast values
 * for a given geographic location and timestamp.
 *
 * @param raw_json          Raw JSON string fetched from the NOAA endpoint.
 * @param latitude          Latitude of the observation site (in degrees, positive = North).
 * @param longitude         Longitude of the observation site (in degrees, positive = East).
 * @param timestamp         UTC timestamp to compare with the observation time (time_t).
 * @param out_best_val      Pointer to int receiving the probability (%) at the closest point to (lat, lon).
 * @param out_max_val       Pointer to int receiving the maximum probability (%) within a 5°x5° region.
 * @param out_avg_val       Pointer to double receiving the average probability (%) in the 5°x5° region.
 * @param out_obs_time      Pointer receiving the OVATION observation time (UTC).
 * @param out_fore_time     Pointer receiving the OVATION forecast time (UTC).
 *
 * @return 0 on success, or a non-zero error code (e.g., PARSE_ERROR_JSON or PARSE_ERROR_DATA).
 */
int parse_noaa_json(const char *raw_json, double latitude, double longitude, time_t timestamp,
                    int *out_best_val, int *out_max_val, double *out_avg_val,
                    time_t *out_obs_time, time_t *out_fore_time);

/**
 * Parse NOAA planetary K-index JSON (1-minute estimate or 3-hourly product).
 * @param raw_json Raw JSON content
 * @param timestamp Time to match entry
 * @param out_kp Pointer to double to store Kp index
 * @return 0 on success, >0 on error
 */
int parse_kp_json(const char *raw_json, time_t timestamp, double *out_kp);

/**
 * Parse NOAA RTSW magnetometer JSON (active spacecraft only, Bz in GSM).
 * @param raw_json Raw JSON content
 * @param timestamp Time to match entry
 * @param out_bt Pointer to store total B field
 * @param out_bz Pointer to store Bz value
 * @return 0 on success, >0 on error
 */
int parse_mag_json(const char *raw_json, time_t timestamp, double *out_bt, double *out_bz);

/**
 * Parse NOAA RTSW solar wind plasma JSON (active spacecraft only).
 * @param raw_json Raw JSON content
 * @param timestamp Time to match entry
 * @param out_density Pointer to plasma density
 * @param out_speed Pointer to solar wind speed
 * @param out_temp Pointer to plasma temperature
 * @return 0 on success, >0 on error
 */
int parse_plasma_json(const char *raw_json, time_t timestamp, double *out_density, double *out_speed, double *out_temp);


#endif // AURORA_FETCH_H

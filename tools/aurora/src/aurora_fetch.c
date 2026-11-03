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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <math.h>
#include <time.h>
#include <json-c/json.h>
#include "aurora_fetch.h"

/* data buffer for curl */
struct mem_buffer {
    char *data;
    size_t size;
};

/* Callback for curl */
static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t total = size * nmemb;
    struct mem_buffer *buf = userdata;
    char *tmp = realloc(buf->data, buf->size + total + 1);

    if (!tmp) 
	return 0;

    buf->data = tmp;
    memcpy(buf->data + buf->size, ptr, total);
    buf->size += total;
    buf->data[buf->size] = '\0';

    return total;
}

/**
 * Fetch JSON from NOAA or other service.
 * @param url HTTPS endpoint
 * @param out_json [out] allocated JSON string, must be free()-d
 * @return int error code (0 = success)
 */
int fetch_aurora_json(const char *url, char **out_json) {
    CURL *curl = curl_easy_init();
    struct mem_buffer buf = { .data = NULL, .size = 0 };
    long code;

    if (!curl) 
	return FETCH_ERROR_CURL_INIT;

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    /* Accept gzip/deflate: the RTSW solar wind files are ~1.5-2.5 MB of JSON,
     * compressed ~100-150 kB (the timer runs every 5 minutes). */
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");

    if(curl_easy_perform(curl) != CURLE_OK) {
        curl_easy_cleanup(curl);
        free(buf.data);

        return FETCH_ERROR_CURL_TRANSFER;
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(curl);

    if (code < 200 || code >= 300){
        free(buf.data);
        return FETCH_ERROR_CURL_TRANSFER;
    }

    *out_json = buf.data;
    return FETCH_SUCCESS;
}




/**
 * Parse an SWPC time tag: "2026-11-03T11:29:00", "2026-11-03T11:22:00Z",
 * "2026-11-03 11:29:00.000" - always UTC.
 */
static int parse_time_tag(const char *str, time_t *out) {
    struct tm tm = {0};
    const char *end;

    if (!str)
        return -1;
    end = strptime(str, "%Y-%m-%dT%H:%M:%S", &tm);
    if (!end) {
        memset(&tm, 0, sizeof(tm));
        end = strptime(str, "%Y-%m-%d %H:%M:%S", &tm);
    }
    if (!end)
        return -1;
    *out = timegm(&tm);
    return 0;
}

/* Number member of a JSON object; json null, strings and missing keys fail */
static int get_number(json_object *obj, const char *key, double *out) {
    json_object *val;

    if (!json_object_object_get_ex(obj, key, &val))
        return -1;
    if (!json_object_is_type(val, json_type_double) &&
        !json_object_is_type(val, json_type_int))
        return -1;
    *out = json_object_get_double(val);
    return 0;
}

/**
 * Find the entry of an SWPC array of objects ([{"time_tag": ..., ...}, ...])
 * closest to timestamp that has all keys as numbers.
 *
 * @param only_active  RTSW files hold one row per minute and spacecraft
 *                     (DSCOVR/SOLAR1, ACE, IMAP); "active": true marks the one
 *                     SWPC uses operationally, the others are skipped.
 * @param max_diff     Maximum distance in seconds, farther entries are stale.
 * @return Borrowed entry or NULL.
 */
static json_object *find_nearest_entry(json_object *root, time_t timestamp,
                                       double max_diff, int only_active,
                                       const char *const *keys, int nkeys) {
    json_object *best = NULL;
    double best_diff = max_diff;
    size_t n = json_object_array_length(root);

    for (size_t i = 0; i < n; i++) {
        json_object *entry = json_object_array_get_idx(root, i);
        json_object *val;
        time_t t;
        double dummy;
        int ok = 1;

        if (!json_object_is_type(entry, json_type_object))
            continue;
        if (!json_object_object_get_ex(entry, "time_tag", &val) ||
            parse_time_tag(json_object_get_string(val), &t) != 0)
            continue;
        if (only_active &&
            (!json_object_object_get_ex(entry, "active", &val) ||
             !json_object_get_boolean(val)))
            continue;
        for (int k = 0; k < nkeys && ok; k++)
            ok = get_number(entry, keys[k], &dummy) == 0;
        if (!ok)
            continue;

        double diff = fabs(difftime(timestamp, t));
        if (diff <= best_diff) {
            best_diff = diff;
            best = entry;
        }
    }
    return best;
}

/* Parse raw_json and make sure it is a JSON array */
static json_object *parse_array(const char *raw_json) {
    json_object *root = json_tokener_parse(raw_json);

    if (root && !json_object_is_type(root, json_type_array)) {
        json_object_put(root);
        root = NULL;
    }
    return root;
}

/* Smallest distance between two longitudes in degrees (0..180) */
static double lon_distance(double a, double b) {
    double d = fmod(fabs(a - b), 360.0);
    return d > 180.0 ? 360.0 - d : d;
}

int parse_noaa_json(const char *raw_json, double latitude, double longitude, time_t timestamp,
                    int *out_best_val, int *out_max_val, double *out_avg_val,
                    time_t *out_obs_time, time_t *out_fore_time)
{
    json_object *root = json_tokener_parse(raw_json);
    if (!root)
        return PARSE_ERROR_JSON;

    json_object *j_obs, *j_fore, *j_coords;
    if (!json_object_object_get_ex(root, "Observation Time", &j_obs) ||
        !json_object_object_get_ex(root, "Forecast Time", &j_fore) ||
        !json_object_object_get_ex(root, "coordinates", &j_coords) ||
        !json_object_is_type(j_coords, json_type_array)) {
        json_object_put(root);
        return PARSE_ERROR_DATA;
    }

    time_t obs_time = 0, forecast_time = 0;
    parse_time_tag(json_object_get_string(j_obs), &obs_time);
    parse_time_tag(json_object_get_string(j_fore), &forecast_time);

    if (fabs(difftime(obs_time, timestamp)) > 3600)
        fprintf(stderr, "Warning: OVATION data is more than 1 hour away from the requested time.\n");

    double best_diff = 1e9;
    int best_val = -1, max_val = -1;
    double avg_sum = 0.0;
    int avg_count = 0;

    /* The OVATION grid uses longitudes 0..359 (east), the site -180..180:
     * compare across the 0/360 seam, otherwise western sites never match. */
    size_t n = json_object_array_length(j_coords);
    for (size_t i = 0; i < n; i++) {
        json_object *entry = json_object_array_get_idx(j_coords, i);
        if (!json_object_is_type(entry, json_type_array) || json_object_array_length(entry) < 3)
            continue;

        double lon = json_object_get_double(json_object_array_get_idx(entry, 0));
        double lat = json_object_get_double(json_object_array_get_idx(entry, 1));
        int val = json_object_get_int(json_object_array_get_idx(entry, 2));
        double dlon = lon_distance(lon, longitude);

        // Find nearest point
        double dist = hypot(lat - latitude, dlon);
        if (dist < best_diff) {
            best_diff = dist;
            best_val = val;
        }

        // Maximum and average within a 5x5 degree box
        if (fabs(lat - latitude) <= 2.5 && dlon <= 2.5) {
            if (val > max_val)
                max_val = val;
            avg_sum += val;
            avg_count++;
        }
    }

    json_object_put(root);

    if (best_val < 0 || avg_count == 0)
        return PARSE_ERROR_DATA;

    if (out_best_val)
        *out_best_val = best_val;
    if (out_max_val)
        *out_max_val = max_val;
    if (out_avg_val)
        *out_avg_val = avg_sum / avg_count;
    if (out_obs_time)
        *out_obs_time = obs_time;
    if (out_fore_time)
        *out_fore_time = forecast_time;

    return 0;
}

int parse_kp_json(const char *raw_json, time_t timestamp, double *out_kp) {
    /* planetary_k_index_1m.json: estimated Kp per minute ("estimated_kp");
     * noaa-planetary-k-index.json: 3-hourly Kp ("Kp"). The 3-hour value is
     * valid for its interval, so allow up to 3 hours distance. */
    static const char *const key_1m[] = { "estimated_kp" };
    static const char *const key_3h[] = { "Kp" };
    json_object *root = parse_array(raw_json);
    json_object *entry;
    double kp;

    if (!root)
        return PARSE_ERROR_JSON;

    entry = find_nearest_entry(root, timestamp, 3 * 3600, 0, key_1m, 1);
    if (entry)
        get_number(entry, key_1m[0], &kp);
    else if ((entry = find_nearest_entry(root, timestamp, 3 * 3600, 0, key_3h, 1)))
        get_number(entry, key_3h[0], &kp);

    json_object_put(root);
    if (!entry)
        return PARSE_ERROR_DATA;
    *out_kp = kp;
    return 0;
}

int parse_mag_json(const char *raw_json, time_t timestamp, double *out_bt, double *out_bz) {
    /* rtsw_mag_1m.json; Bz in GSM coordinates is the one relevant for aurora */
    static const char *const keys[] = { "bt", "bz_gsm" };
    json_object *root = parse_array(raw_json);
    json_object *entry;

    if (!root)
        return PARSE_ERROR_JSON;

    entry = find_nearest_entry(root, timestamp, SOLAR_WIND_MAX_AGE, 1, keys, 2);
    if (entry) {
        get_number(entry, "bt", out_bt);
        get_number(entry, "bz_gsm", out_bz);
    }
    json_object_put(root);
    return entry ? 0 : PARSE_ERROR_DATA;
}

int parse_plasma_json(const char *raw_json, time_t timestamp, double *out_density, double *out_speed, double *out_temp) {
    /* rtsw_wind_1m.json */
    static const char *const keys[] = { "proton_density", "proton_speed", "proton_temperature" };
    json_object *root = parse_array(raw_json);
    json_object *entry;

    if (!root)
        return PARSE_ERROR_JSON;

    entry = find_nearest_entry(root, timestamp, SOLAR_WIND_MAX_AGE, 1, keys, 3);
    if (entry) {
        get_number(entry, "proton_density", out_density);
        get_number(entry, "proton_speed", out_speed);
        get_number(entry, "proton_temperature", out_temp);
    }
    json_object_put(root);
    return entry ? 0 : PARSE_ERROR_DATA;
}

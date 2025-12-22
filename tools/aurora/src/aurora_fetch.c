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




int parse_noaa_json(const char *raw_json, double latitude, double longitude, time_t timestamp,
                    int *out_best_val, int *out_max_val, double *out_avg_val,
                    time_t *out_obs_time, time_t *out_fore_time)
{
    int debug = 0;

    json_object *root = json_tokener_parse(raw_json);
    if (!root)
        return PARSE_ERROR_JSON;

    json_object *j_obs, *j_fore, *j_coords;
    if (!json_object_object_get_ex(root, "Observation Time", &j_obs) ||
        !json_object_object_get_ex(root, "Forecast Time", &j_fore) ||
        !json_object_object_get_ex(root, "coordinates", &j_coords)) {
        json_object_put(root);
        return PARSE_ERROR_DATA;
    }

    const char *obs = json_object_get_string(j_obs);
    const char *fore = json_object_get_string(j_fore);

    // Parse observation time
    struct tm tm_obs = {0};
    time_t obs_time = 0;
    if (strptime(obs, "%Y-%m-%dT%H:%M:%SZ", &tm_obs)) {
        obs_time = timegm(&tm_obs);
    }

    // Parse forecast time
    struct tm tm_fore = {0};
    time_t forecast_time = 0;
    if (strptime(fore, "%Y-%m-%dT%H:%M:%SZ", &tm_fore)) {
        forecast_time = timegm(&tm_fore);
    }

    if (debug) {
        printf("NOAA observation time: %s", ctime(&obs_time));
        printf("Requested timestamp:   %s", ctime(&timestamp));
    }

    if (fabs(difftime(obs_time, timestamp)) > 3600) {
        if (debug)
            fprintf(stderr, "Warning: NOAA data is older/newer than 1 hour.\n");
    }

    double best_diff = 1e9;
    int best_val = -1, max_val = -1;
    double avg_sum = 0.0;
    int avg_count = 0;

    size_t n = json_object_array_length(j_coords);
    for (size_t i = 0; i < n; i++) {
        json_object *entry = json_object_array_get_idx(j_coords, i);
        if (!json_object_is_type(entry, json_type_array) || json_object_array_length(entry) < 3)
            continue;

        double lon = json_object_get_double(json_object_array_get_idx(entry, 0));
        double lat = json_object_get_double(json_object_array_get_idx(entry, 1));
        int val = json_object_get_int(json_object_array_get_idx(entry, 2));

        // Find nearest point
        double dist = hypot(lat - latitude, lon - longitude);
        if (dist < best_diff) {
            best_diff = dist;
            best_val = val;
        }

        // Average within 5x5 degree box
        if (fabs(lat - latitude) <= 2.5 && fabs(lon - longitude) <= 2.5) {
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
    json_object *root = json_tokener_parse(raw_json);
    if (!root || !json_object_is_type(root, json_type_array)) 
	return PARSE_ERROR_JSON;

    size_t n = json_object_array_length(root);
    double best_kp = -1.0;
    double best_diff = 1e9;

    for (size_t i = 1; i < n; i++) {  // skip header
        json_object *entry = json_object_array_get_idx(root, i);
        if (!json_object_is_type(entry, json_type_array) || json_object_array_length(entry) < 2) 
	    continue;

        const char *time_str = json_object_get_string(json_object_array_get_idx(entry, 0));
        double kp_val = json_object_get_double(json_object_array_get_idx(entry, 1));

        struct tm tm = {0};
        if (strptime(time_str, "%Y-%m-%d %H:%M:%S", &tm)) {
            time_t t = timegm(&tm);
            double diff = fabs(difftime(timestamp, t));
            if (diff < best_diff) {
                best_diff = diff;
                best_kp = kp_val;
            }
        }
    }

    json_object_put(root);
    if (best_kp < 0) 
	return PARSE_ERROR_DATA;
    *out_kp = best_kp;
    return 0;
}

int parse_mag_json(const char *raw_json, time_t timestamp, double *out_bt, double *out_bz) {
    json_object *root = json_tokener_parse(raw_json);
    if (!root || !json_object_is_type(root, json_type_array)) 
	return PARSE_ERROR_JSON;

    size_t n = json_object_array_length(root);
    double best_bt = -1, best_bz = -1, best_diff = 1e9;

    for (size_t i = 1; i < n; i++) {
        json_object *entry = json_object_array_get_idx(root, i);
        if (!json_object_is_type(entry, json_type_array) || json_object_array_length(entry) < 6) 
	    continue;

        const char *time_str = json_object_get_string(json_object_array_get_idx(entry, 0));
        double bt = json_object_get_double(json_object_array_get_idx(entry, 2));
        double bz = json_object_get_double(json_object_array_get_idx(entry, 4));

        struct tm tm = {0};
        if (strptime(time_str, "%Y-%m-%d %H:%M:%S", &tm)) {
            time_t t = timegm(&tm);
            double diff = fabs(difftime(timestamp, t));
            if (diff < best_diff) {
                best_diff = diff;
                best_bt = bt;
                best_bz = bz;
            }
        }
    }

    json_object_put(root);
    if (best_bt < 0 || best_bz < -1e5) 
	return PARSE_ERROR_DATA;
    *out_bt = best_bt;
    *out_bz = best_bz;
    return 0;
}

int parse_plasma_json(const char *raw_json, time_t timestamp, double *out_density, double *out_speed, double *out_temp) {
    json_object *root = json_tokener_parse(raw_json);
    if (!root || !json_object_is_type(root, json_type_array)) 
	return PARSE_ERROR_JSON;

    size_t n = json_object_array_length(root);
    double best_diff = 1e9;
    double best_density = -1, best_speed = -1, best_temp = -1;

    for (size_t i = 1; i < n; i++) {
        json_object *entry = json_object_array_get_idx(root, i);
        if (!json_object_is_type(entry, json_type_array) || json_object_array_length(entry) < 5) 
	    continue;

        const char *time_str = json_object_get_string(json_object_array_get_idx(entry, 0));
        double density = json_object_get_double(json_object_array_get_idx(entry, 1));
        double speed = json_object_get_double(json_object_array_get_idx(entry, 2));
        double temp = json_object_get_double(json_object_array_get_idx(entry, 3));

        struct tm tm = {0};
        if (strptime(time_str, "%Y-%m-%d %H:%M:%S", &tm)) {
            time_t t = timegm(&tm);
            double diff = fabs(difftime(timestamp, t));
            if (diff < best_diff) {
                best_diff = diff;
                best_density = density;
                best_speed = speed;
                best_temp = temp;
            }
        }
    }

    json_object_put(root);
    if (best_density < 0 || best_speed < 0 || best_temp < 0) 
	return PARSE_ERROR_DATA;
    *out_density = best_density;
    *out_speed = best_speed;
    *out_temp = best_temp;
    return 0;
}

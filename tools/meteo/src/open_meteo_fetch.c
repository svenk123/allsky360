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
/* timegm() and strptime() are not declared with only _XOPEN_SOURCE */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <math.h>
#include <time.h>
#include <json-c/json.h>
#include "open_meteo_fetch.h"

#define MAX_URL_LEN 1024

struct memory {
    char *response;
    size_t size;
};

static size_t write_callback(void *data, size_t size, size_t nmemb, void *userp) {
    size_t total = size * nmemb;
    struct memory *mem = (struct memory *)userp;
    /* Keep the old buffer on failure, the caller frees it */
    char *tmp = realloc(mem->response, mem->size + total + 1);
    if (!tmp) return 0;
    mem->response = tmp;
    memcpy(&(mem->response[mem->size]), data, total);
    mem->size += total;
    mem->response[mem->size] = 0;
    return total;
}

int fetch_open_meteo_weather(double latitude, double longitude, const int *elevation, time_t timestamp, char **out_json) {
    char url[MAX_URL_LEN];
    char elevation_param[32] = "";
    if (elevation)
        snprintf(elevation_param, sizeof(elevation_param), "&elevation=%d", *elevation);

    /* One request for the current hour (database) and the forecast: the
     * hourly values from the full hour of timestamp up to FORECAST_HOURS
     * later (start_hour/end_hour are inclusive, FORECAST_HOURS + 1 values). */
    time_t start = timestamp - timestamp % 3600;
    time_t end = start + FORECAST_HOURS * 3600;
    struct tm tm_start, tm_end;
    char start_hour[32], end_hour[32];
    gmtime_r(&start, &tm_start);
    gmtime_r(&end, &tm_end);
    strftime(start_hour, sizeof(start_hour), "%Y-%m-%dT%H:00", &tm_start);
    strftime(end_hour, sizeof(end_hour), "%Y-%m-%dT%H:00", &tm_end);

    snprintf(url, MAX_URL_LEN,
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=%.4f&longitude=%.4f%s"
        "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
        "pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,"
        "cloud_cover_mid,cloud_cover_high,visibility,"
        "wind_speed_10m,wind_direction_10m,wind_speed_300hPa,"
        "wind_gusts_10m,precipitation,precipitation_probability,is_day"
        "&timezone=UTC"
        "&start_hour=%s&end_hour=%s",
        latitude, longitude, elevation_param, start_hour, end_hour
    );

    CURL *curl = curl_easy_init();
    if (!curl) return 1;

    struct memory chunk = { .response = NULL, .size = 0 };

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &chunk);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || chunk.size == 0) {
        free(chunk.response);
        return 2;
    }

    *out_json = chunk.response;
    return 0;
}

int parse_open_meteo_weather(const char *json_raw, time_t target, open_meteo_weather_t *out)
{
    if (!json_raw || !out) return 1;

    json_object *root = json_tokener_parse(json_raw);
    if (!root) return 2;

    json_object *j_hourly = NULL;
    if (!json_object_object_get_ex(root, "hourly", &j_hourly)) {
        json_object_put(root);
        return 3;
    }

    // Format target time to match Open-Meteo hourly timestamp: "YYYY-MM-DDTHH:00"
    struct tm tm_target;
    gmtime_r(&target, &tm_target);
    char target_hour[32];
    strftime(target_hour, sizeof(target_hour), "%Y-%m-%dT%H:00", &tm_target);


    json_object *j_time = NULL;
    if (!json_object_object_get_ex(j_hourly, "time", &j_time)) {
        json_object_put(root);
        return 4;
    }

    int matched_index = -1;
    int n = json_object_array_length(j_time);
    for (int i = 0; i < n; i++) {
        const char *t = json_object_get_string(json_object_array_get_idx(j_time, i));
        if (t && strcmp(t, target_hour) == 0) {
            matched_index = i;
            break;
        }
    }

    if (matched_index == -1) {
        json_object_put(root);
        return 5; // Time not found
    }

    // Macro for safely extracting a float value by field name and writing it to the struct
#define EXTRACT_FIELD(NAME, FIELD) \
    do { \
        json_object *arr = NULL; \
        if (json_object_object_get_ex(j_hourly, NAME, &arr)) { \
            int arr_len = (int)json_object_array_length(arr); \
            if (matched_index >= 0 && matched_index < arr_len) { \
                out->FIELD = json_object_get_double(json_object_array_get_idx(arr, matched_index)); \
            } else { \
                out->FIELD = NAN; \
            } \
        } else { \
            out->FIELD = NAN; \
        } \
    } while (0)

    EXTRACT_FIELD("temperature_2m", temperature);
    EXTRACT_FIELD("relative_humidity_2m", humidity);
    EXTRACT_FIELD("dew_point_2m", dew_point);
    EXTRACT_FIELD("pressure_msl", pressure_msl);
    EXTRACT_FIELD("surface_pressure", surface_pressure);
    EXTRACT_FIELD("cloud_cover", cloud_cover);
    EXTRACT_FIELD("cloud_cover_low", cloud_low);
    EXTRACT_FIELD("cloud_cover_mid", cloud_mid);
    EXTRACT_FIELD("cloud_cover_high", cloud_high);
    EXTRACT_FIELD("visibility", visibility);
    EXTRACT_FIELD("wind_speed_10m", wind_speed_10m);
    EXTRACT_FIELD("wind_direction_10m", wind_dir_10m);
    EXTRACT_FIELD("wind_speed_300hPa", wind_speed_300hPa);

    json_object_put(root);
    return 0;
}

/* Value i of the hourly array name as a JSON number with the given decimals,
 * NULL (JSON null) when the array, the entry or the value is missing. */
static json_object *hourly_value(json_object *j_hourly, const char *name, int i, int decimals)
{
    json_object *arr = NULL, *val;
    char buf[32];

    if (!json_object_object_get_ex(j_hourly, name, &arr) ||
        !json_object_is_type(arr, json_type_array) ||
        i >= (int)json_object_array_length(arr))
        return NULL;
    val = json_object_array_get_idx(arr, i);
    if (!json_object_is_type(val, json_type_double) &&
        !json_object_is_type(val, json_type_int))
        return NULL;
    if (decimals == 0)
        return json_object_new_int((int)lround(json_object_get_double(val)));
    /* the string keeps the JSON short (no 4.2999999999999998) */
    snprintf(buf, sizeof(buf), "%.*f", decimals, json_object_get_double(val));
    return json_object_new_double_s(strtod(buf, NULL), buf);
}

int write_open_meteo_forecast(const char *json_raw, time_t from, const char *path)
{
    /* Open-Meteo name -> name in forecast JSON (as in the database), decimals */
    static const struct { const char *src, *dst; int decimals; } fields[] = {
        { "temperature_2m",            "temperature",               1 },
        { "relative_humidity_2m",      "humidity",                  0 },
        { "dew_point_2m",              "dew_point",                 1 },
        { "pressure_msl",              "pressure_msl",              1 },
        { "cloud_cover",               "cloud_cover",               0 },
        { "cloud_cover_low",           "cloud_low",                 0 },
        { "cloud_cover_mid",           "cloud_mid",                 0 },
        { "cloud_cover_high",          "cloud_high",                0 },
        { "precipitation_probability", "precipitation_probability", 0 },
        { "precipitation",             "precipitation",             1 },
        { "visibility",                "visibility",                0 },
        { "wind_speed_10m",            "wind_speed_10m",            1 },
        { "wind_gusts_10m",            "wind_gusts_10m",            1 },
        { "wind_direction_10m",        "wind_dir_10m",              0 },
        { "wind_speed_300hPa",         "wind_speed_300hPa",         1 },
        { "is_day",                    "is_day",                    0 },
    };
    json_object *root, *j_hourly = NULL, *j_time = NULL, *j_elev = NULL;
    time_t from_hour = from - from % 3600;

    if (!json_raw || !path)
        return 1;
    root = json_tokener_parse(json_raw);
    if (!root)
        return 2;
    if (!json_object_object_get_ex(root, "hourly", &j_hourly) ||
        !json_object_object_get_ex(j_hourly, "time", &j_time) ||
        !json_object_is_type(j_time, json_type_array)) {
        json_object_put(root);
        return 3;
    }

    json_object *jout = json_object_new_object();
    json_object *hours = json_object_new_array();
    json_object_object_add(jout, "source", json_object_new_string("Open-Meteo"));
    json_object_object_add(jout, "generated", json_object_new_int64((int64_t)time(NULL)));
    json_object_object_add(jout, "from", json_object_new_int64((int64_t)from_hour));
    if (json_object_object_get_ex(root, "elevation", &j_elev))
        json_object_object_add(jout, "elevation", json_object_get(j_elev));

    int n = (int)json_object_array_length(j_time);
    for (int i = 0; i < n && (int)json_object_array_length(hours) <= FORECAST_HOURS; i++) {
        const char *t = json_object_get_string(json_object_array_get_idx(j_time, i));
        struct tm tm = {0};
        if (!t || !strptime(t, "%Y-%m-%dT%H:%M", &tm))
            continue;
        time_t ts = timegm(&tm);
        if (ts < from_hour)
            continue;

        json_object *h = json_object_new_object();
        json_object_object_add(h, "timestamp", json_object_new_int64((int64_t)ts));
        for (size_t f = 0; f < sizeof(fields) / sizeof(fields[0]); f++)
            json_object_object_add(h, fields[f].dst,
                                   hourly_value(j_hourly, fields[f].src, i, fields[f].decimals));
        json_object_array_add(hours, h);
    }
    json_object_object_add(jout, "hours", hours);
    json_object_put(root);

    /* Write to a temporary file and rename: the web page never reads a
     * half-written file */
    char tmp_path[1024];
    int rc = 0;
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    FILE *fp = fopen(tmp_path, "w");
    if (!fp ||
        fprintf(fp, "%s\n", json_object_to_json_string_ext(jout, JSON_C_TO_STRING_PLAIN)) < 0) {
        rc = 4;
    }
    if (fp && fclose(fp) != 0)
        rc = 4;
    if (rc == 0 && rename(tmp_path, path) != 0)
        rc = 4;
    if (rc != 0)
        remove(tmp_path);

    json_object_put(jout);
    return rc;
}

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
#include "open_meteo_fetch.h"

#define MAX_URL_LEN 1024

struct memory {
    char *response;
    size_t size;
};

static size_t write_callback(void *data, size_t size, size_t nmemb, void *userp) {
    size_t total = size * nmemb;
    struct memory *mem = (struct memory *)userp;
    mem->response = realloc(mem->response, mem->size + total + 1);
    if (!mem->response) return 0;
    memcpy(&(mem->response[mem->size]), data, total);
    mem->size += total;
    mem->response[mem->size] = 0;
    return total;
}

int fetch_open_meteo_weather(double latitude, double longitude, time_t timestamp, char **out_json) {
    char url[MAX_URL_LEN];
    struct tm *utc_time = gmtime(&timestamp);
    char iso_time[32];
    strftime(iso_time, sizeof(iso_time), "%Y-%m-%dT%H:00", utc_time);

    snprintf(url, MAX_URL_LEN,
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=%.4f&longitude=%.4f"
        "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
        "pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,"
        "cloud_cover_mid,cloud_cover_high,visibility,"
        "wind_speed_10m,wind_direction_10m,wind_speed_300hPa"
        "&timezone=UTC"
        "&start_date=%.4d-%.2d-%.2d&end_date=%.4d-%.2d-%.2d",
        latitude, longitude,
        utc_time->tm_year + 1900, utc_time->tm_mon + 1, utc_time->tm_mday,
        utc_time->tm_year + 1900, utc_time->tm_mon + 1, utc_time->tm_mday
    );

    CURL *curl = curl_easy_init();
    if (!curl) return 1;

    struct memory chunk = { .response = NULL, .size = 0 };

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &chunk);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

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
        if (strcmp(t, target_hour) == 0) {
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

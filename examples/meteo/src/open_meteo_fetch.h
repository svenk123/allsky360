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
#ifndef OPEN_METEO_FETCH_H
#define OPEN_METEO_FETCH_H

#include <stddef.h>

typedef struct {
    double temperature;           // Temperature at 2 meters (°C)
    double humidity;              // Relative humidity at 2 meters (%)
    double dew_point;             // Dewpoint at 2 meters (°C)
    double pressure_msl;         // Mean sea level pressure (hPa)
    double surface_pressure;     // Surface pressure (hPa)
    double cloud_cover;          // Total cloud cover (%)
    double cloud_low;            // Low cloud cover (%)
    double cloud_mid;            // Mid-level cloud cover (%)
    double cloud_high;           // High cloud cover (%)
    double visibility;           // Visibility (meters)
    double wind_speed_10m;       // Wind speed at 10 meters (km/h)
    double wind_dir_10m;         // Wind direction at 10 meters (degrees)
    double wind_speed_300hPa;    // Wind speed at 300 hPa / 9.2 km altitude (km/h)
} open_meteo_weather_t;

int fetch_open_meteo_weather(double latitude, double longitude, time_t timestamp, char **out_json);

/**
 * @brief Parse weather data from Open-Meteo JSON response for a specific hour.
 * 
 * @param json_raw   Raw JSON string from Open-Meteo hourly endpoint.
 * @param target     The target timestamp (UTC) for which to extract weather data (will be matched to the full hour).
 * @param out        Pointer to open_meteo_weather_t struct where parsed values will be stored.
 * @return int       0 on success, >0 on failure.
 */
int parse_open_meteo_weather(const char *json_raw, time_t target, open_meteo_weather_t *out);

#endif // OPEN_METEO_FETCH_H

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
#include <time.h>
#include <libnova/ln_types.h>
#include <libnova/lunar.h>
#include <libnova/julian_day.h>
#include <libnova/transform.h>
#include "moon_calc.h"

double moon_altitude(time_t timestamp, double longitude, double latitude) {
    struct ln_date date;
    struct ln_equ_posn equatorial;
    struct ln_hrz_posn horizontal;
    struct ln_lnlat_posn observer;

    // Convert time_t to julian date
    struct tm *utc_time = gmtime(&timestamp);
    ln_get_date_from_tm(utc_time, &date);
    double jd = ln_get_julian_day(&date);

    // Calculate moon position
    ln_get_lunar_equ_coords(jd, &equatorial);

    observer.lat = latitude;
    observer.lng = longitude;

    // Convert to horizontal coordinates
    ln_get_hrz_from_equ(&equatorial, &observer, jd, &horizontal);

    return horizontal.alt;
}

const char* moon_phase(time_t timestamp) {
    struct ln_date date;

    struct tm *utc_time = gmtime(&timestamp);
    ln_get_date_from_tm(utc_time, &date);
    double jd = ln_get_julian_day(&date);

    // Calculate moon phase
    double phase = ln_get_lunar_phase(jd);

    if (phase < 1.84566)
        return "New moon";
    else if (phase < 5.53699)
        return "Waxing crescent moon";
    else if (phase < 9.22831)
        return "First quarter moon";
    else if (phase < 12.91963)
        return "Waxing gibbous moon";
    else if (phase < 16.61096)
        return "Full moon";
    else if (phase < 20.30228)
        return "Waning gibbous moon";
    else if (phase < 23.99361)
        return "Last quarter moon";
    else
        return "Waning crescent moon";
}

double moon_phase_percentage(time_t timestamp) {
    struct ln_date date;
    struct tm *utc_time = gmtime(&timestamp);
    ln_get_date_from_tm(utc_time, &date);
    double jd = ln_get_julian_day(&date);

    double illumination = ln_get_lunar_disk(jd);

    return illumination * 100.0;
}

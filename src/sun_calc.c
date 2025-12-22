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
#include <libnova/solar.h>
#include <libnova/julian_day.h>
#include <libnova/transform.h>

int is_sun_below_18_degrees(time_t timestamp, double longitude, double latitude) {
    struct ln_date date;
    struct ln_equ_posn equatorial;
    struct ln_hrz_posn horizontal;
    struct ln_lnlat_posn observer; // Correct structure for the observer position

    /* Optimization: if latitude > 54° or < -54°, sun may never reach -18° */
    if (latitude > 54.0 || latitude < -54.0) {
        printf("Latitude |%.2f| > 54° → Sun may never go below -18°.\n", latitude);
        return 0;
    }

    /* Convert time_t to julian date */
    struct tm *utc_time = gmtime(&timestamp);
    ln_get_date_from_tm(utc_time, &date);
    double jd = ln_get_julian_day(&date);
    printf("Julian date: %.5f\n", jd);

    /* Sun position */
    ln_get_solar_equ_coords(jd, &equatorial);
    printf("Solar equatorial (RA, Dec): %.5f, %.5f\n", equatorial.ra, equatorial.dec);

    observer.lat = latitude;
    observer.lng = longitude;

    /* Convert to horizontal coordinates */
    ln_get_hrz_from_equ(&equatorial, &observer, jd, &horizontal);
    printf("Sun altitude: %.5f, azimuth: %.5f\n", horizontal.alt, horizontal.az);

    if (horizontal.alt < -18.0) {
        printf("The sun is more than 18 degrees below the horizon (astronomical night).\n");
        return 1;
    } else {
        printf("The sun is not yet 18 degrees below the horizon.\n");
        return 0;
    }
}


int is_sun_below_6_degrees(time_t timestamp, double longitude, double latitude) {
    struct ln_date date;
    struct ln_equ_posn equatorial;
    struct ln_hrz_posn horizontal;
    struct ln_lnlat_posn observer; // Correct structure for the observer position

    /* Latitude optimization: If latitude > 60° or < -60° => sun can never go below -6° */
    if (latitude > 60.0 || latitude < -60.0) {
        printf("Latitude is outside the critical range (|%.2f| > 60°). Calculation not required.\n", latitude);
        return 0; // Sonne geht nie mehr als 6° unter
    }

    /* Convert time_t to julian date */
    struct tm *utc_time = gmtime(&timestamp);
    ln_get_date_from_tm(utc_time, &date);
    double jd = ln_get_julian_day(&date);
    printf("Julian date: %.5f\n", jd);

    /* Sun position */
    ln_get_solar_equ_coords(jd, &equatorial);
    printf("Solar equatorial (RA, Dec): %.5f, %.5f\n", equatorial.ra, equatorial.dec);

    observer.lat = latitude;
    observer.lng = longitude;

    /* Convert to horizontal coordinates */
    ln_get_hrz_from_equ(&equatorial, &observer, jd, &horizontal);
    printf("Sun altitude: %.5f, azimuth: %.5f\n", horizontal.alt, horizontal.az);

    if (horizontal.alt < -6.0) {
        printf("The sun is more than 6 degrees below the horizon.\n");
        return 1;
    } else {
        printf("The sun is not yet 6 degrees below the horizon.\n");
        return 0;
    }
}

double sun_altitude(time_t timestamp, double longitude, double latitude) {
    struct ln_date date;
    struct ln_equ_posn equatorial;
    struct ln_hrz_posn horizontal;
    struct ln_lnlat_posn observer;

    /* Convert time_t to julian date */
    struct tm *utc_time = gmtime(&timestamp);
    ln_get_date_from_tm(utc_time, &date);
    double jd = ln_get_julian_day(&date);

    /* Sun position */
    ln_get_solar_equ_coords(jd, &equatorial);

    observer.lat = latitude;
    observer.lng = longitude;

    /* Convert to horizontal coordinates */
    ln_get_hrz_from_equ(&equatorial, &observer, jd, &horizontal);

    return horizontal.alt;
}

int calculate_civil_twilight(int year, int month, int day, double longitude, double latitude, time_t *dawn, time_t *dusk) {
    struct ln_lnlat_posn observer;
    struct ln_date date;
    struct ln_rst_time rst;

    observer.lat = latitude;
    observer.lng = longitude;

    date.years = year;
    date.months = month;
    date.days = day;
    date.hours = 12;  // Lunch time as reference
    date.minutes = 0;
    date.seconds = 0;

    double jd = ln_get_julian_day(&date);

    /* Calculate civil twilight */
    int result = ln_get_solar_rst_horizon(jd, &observer, -6.0, &rst);

    /* No times found (i.e. polar day or polar night) */
    if (result != 0) {
        return 1;
    }

    /* Convert julian date to time_t */
    struct tm tm_dawn, tm_dusk;

    /* Begin of civil twilight (dawn) */
    ln_get_date(jd + rst.rise, &date);
    tm_dawn.tm_year = date.years - 1900;
    tm_dawn.tm_mon = date.months - 1;
    tm_dawn.tm_mday = date.days;
    tm_dawn.tm_hour = date.hours;
    tm_dawn.tm_min = date.minutes;
    tm_dawn.tm_sec = (int)date.seconds;
    tm_dawn.tm_isdst = -1;
    *dawn = timegm(&tm_dawn);

    /* End of civil twilight (dusk) */
    ln_get_date(jd + rst.set, &date);
    tm_dusk.tm_year = date.years - 1900;
    tm_dusk.tm_mon = date.months - 1;
    tm_dusk.tm_mday = date.days;
    tm_dusk.tm_hour = date.hours;
    tm_dusk.tm_min = date.minutes;
    tm_dusk.tm_sec = (int)date.seconds;
    tm_dusk.tm_isdst = -1;
    *dusk = timegm(&tm_dusk);

    return 0;
}


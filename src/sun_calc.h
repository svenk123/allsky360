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
#ifndef SUN_CALC_H
#define SUN_CALC_H

#include <time.h>

/**
 * Checks if the sun is more than 18 degrees below the horizon (astronomical night).
 * @param timestamp: Current time as time_t.
 * @param longitude: Longitude of the location (in degrees).
 * @param latitude: Latitude of the location (in degrees).
 * @return: 1 if the sun is more than 18 degrees below the horizon, otherwise 0.
 */
int is_sun_below_18_degrees(time_t timestamp, double longitude, double latitude);

/**
 * Checks if the sun is more than 6 degrees below the horizon (dusk/dawn).
 * @param timestamp: Current time as time_t.
 * @param longitude: Longitude of the location (in degrees).
 * @param latitude: Latitude of the location (in degrees).
 * @return: 1 if the sun is more than 6 degrees below the horizon, otherwise 0.
 */
int is_sun_below_6_degrees(time_t timestamp, double longitude, double latitude);

/**
 * Calculates the current solar altitude (elevation) in degrees.
 * @param timestamp: Current time as time_t.
 * @param longitude: Longitude of the location (in degrees).
 * @param latitude: Latitude of the location (in degrees).
 * @return: Solar altitude in degrees (negative means below the horizon).
 */
double sun_altitude(time_t timestamp, double longitude, double latitude);

/**
 * Calculates the start and end of civil twilight for a given day.
 * @param year: Year (e.g., 2024)
 * @param month Month (1–12)
 * @param day Day of the month (1–31)
 * @param longitude Longitude of the location in degrees.
 * @param latitude Latitude of the location in degrees.
 * @param dawn Pointer to a time_t variable that will receive the dawn time.
 * @param dusk Pointer to a time_t variable that will receive the dusk time.
 * @return 0 on success, -1 if no times were found (e.g., polar night or polar day).
 */
int calculate_civil_twilight(int year, int month, int day, double longitude, double latitude, time_t *dawn, time_t *dusk);


#endif /* SUN_CALC_H */

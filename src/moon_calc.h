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
#ifndef MOON_CALC_H
#define MOON_CALC_H

#include <time.h>

/**
 * Calculates the current moon altitude (elevation) in degrees.
 *
 * @param timestamp: current time as time_t.
 * @param longitude: longitude of the location (in degrees).
 * @param latitude: latitude of the location (in degrees).
 * @return: Moon altitude in degrees (negative means below the horizon).
 */
double moon_altitude(time_t timestamp, double longitude, double latitude);

/**
 * Calculates the current moon phase based on the given date.
 *
 * @param timestamp: current time as time_t.
 * @return: string representing the moon phase ("New Moon", "Waxing Crescent", etc.).
 */
const char* moon_phase(time_t timestamp);

/**
 * Calculates the current moon phase as a percentage (0% = New Moon, 100% = Full Moon).
 *
 * @param timestamp: current time as time_t.
 * @return: moon phase as a percentage (0% to 100%).
 */
double moon_phase_percentage(time_t timestamp);

#endif // MOON_CALC_H

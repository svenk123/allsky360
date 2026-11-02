/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#ifndef COORDINATES_H
#define COORDINATES_H

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>

/*
 * Parse a latitude or longitude from config.ini.
 *
 * Accepted forms (decimal degrees):
 *   51.2349   -33.9249   +7.4479      signed
 *   51.2349N  33.9249S   74.0060W     hemisphere suffix (case-insensitive,
 *   33.9249 S                         blanks before the suffix allowed)
 *
 * S and W make the value negative. A sign combined with a suffix
 * ("-33.9S"), unknown suffixes, trailing garbage and values outside
 * +-90 (latitude) / +-180 (longitude) are rejected, so a typo can never turn
 * into a silently mirrored location.
 *
 * @param value       text from config.ini
 * @param is_latitude 1 for latitude (N/S, +-90), 0 for longitude (E/W, +-180)
 * @param result      parsed value in signed decimal degrees
 * @return 0 on success, -1 if the value is invalid (result unchanged)
 */
static inline int parse_lat_lon(const char *value, int is_latitude,
                                   double *result) {
  if (!value || !result)
    return -1;

  while (isspace((unsigned char)*value))
    value++;

  int has_sign = (*value == '+' || *value == '-');

  char *end = NULL;
  errno = 0;
  double v = strtod(value, &end);
  if (end == value || errno != 0 || !isfinite(v))
    return -1;

  while (isspace((unsigned char)*end))
    end++;

  if (*end != '\0') {
    char suffix = (char)toupper((unsigned char)*end);
    char positive = is_latitude ? 'N' : 'E';
    char negative = is_latitude ? 'S' : 'W';

    if (has_sign || (suffix != positive && suffix != negative))
      return -1;

    end++;
    while (isspace((unsigned char)*end))
      end++;
    if (*end != '\0')
      return -1;

    if (suffix == negative)
      v = -v;
  }

  if (fabs(v) > (is_latitude ? 90.0 : 180.0))
    return -1;

  *result = v;
  return 0;
}

#endif /* COORDINATES_H */

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
#include <math.h>
#include <string.h>

#include "wind_average.h"

#define DEG2RAD (M_PI / 180.0)

void wind_buffer_init(wind_buffer_t *b) { memset(b, 0, sizeof(*b)); }

void wind_buffer_add(wind_buffer_t *b, time_t t, double speed, double direction) {
  if (isnan(speed) && isnan(direction))
    return;
  b->samples[b->head] = (wind_sample_t){t, (float)speed, (float)direction};
  b->head = (b->head + 1) % WIND_BUFFER_SIZE;
  if (b->count < WIND_BUFFER_SIZE)
    b->count++;
}

void wind_buffer_average(const wind_buffer_t *b, time_t now, int window_s,
                         double calm_speed, int min_direction_samples,
                         wind_average_t *out) {
  memset(out, 0, sizeof(*out));
  double speed_sum = 0.0, speed_max = 0.0, weight_sum = 0.0, east = 0.0, north = 0.0;

  for (int i = 0; i < b->count; i++) {
    const wind_sample_t *s = &b->samples[i];
    if (s->t > now || s->t <= now - window_s)
      continue;
    out->samples++;

    if (!isnan(s->speed)) {
      out->speed_samples++;
      speed_sum += s->speed;
      if (s->speed > speed_max)
        speed_max = s->speed;
    }
    if (isnan(s->direction))
      continue;
    out->direction_samples++;

    double weight = 1.0; /* no anemometer: plain unit-vector mean */
    if (!isnan(s->speed)) {
      if (calm_speed > 0.0 && s->speed < calm_speed)
        continue; /* calm: the vane points anywhere */
      weight = s->speed;
    }
    if (weight <= 0.0)
      continue; /* calm rule off and 0 m/s: no direction either */
    out->used_for_direction++;
    weight_sum += weight;
    east += weight * sin(s->direction * DEG2RAD);
    north += weight * cos(s->direction * DEG2RAD);
  }

  out->speed = out->speed_samples > 0 ? speed_sum / out->speed_samples : NAN;
  out->speed_max = out->speed_samples > 0 ? speed_max : NAN;
  out->direction = NAN;
  out->consistency = NAN;
  if (weight_sum <= 0.0)
    return;

  double resultant = hypot(east, north);
  out->consistency = resultant / weight_sum;
  /* Too few windy samples, or directions that cancel out exactly */
  if (out->used_for_direction < min_direction_samples || out->consistency < 1e-6)
    return;
  double deg = atan2(east, north) / DEG2RAD;
  if (deg < 0.0)
    deg += 360.0;
  out->direction = deg >= 359.5 ? 0.0 : deg; /* stored rounded to whole degrees */
}

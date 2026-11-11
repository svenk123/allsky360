/* Unit test of the wind ring buffer and vector mean (make test) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "wind_average.h"

static int failures = 0;

static void check(const char *what, double got, double want, double tol) {
  int ok = isnan(want) ? isnan(got) : fabs(got - want) <= tol;
  printf("%s %-55s got %8.3f want %8.3f\n", ok ? "ok  " : "FAIL", what, got, want);
  if (!ok)
    failures++;
}

int main(void) {
  wind_buffer_t *b = malloc(sizeof(*b));
  wind_average_t a;
  const time_t t0 = 1762800000;

  /* Around north: 350 and 10 deg average to 0, not 180 */
  wind_buffer_init(b);
  for (int i = 0; i < 60; i++)
    wind_buffer_add(b, t0 + i, 3.0, i % 2 ? 350.0 : 10.0);
  wind_buffer_average(b, t0 + 59, 600, 0.3, 30, &a);
  check("350/10 deg -> 0 deg", a.direction, 0.0, 0.01);
  check("mean speed", a.speed, 3.0, 1e-6);
  check("consistency cos(10 deg)", a.consistency, cos(10 * M_PI / 180), 1e-4);

  /* Weighted by speed: 90 deg at 3 m/s, 180 deg at 1 m/s -> atan2(3, -1) */
  wind_buffer_init(b);
  for (int i = 0; i < 60; i++)
    wind_buffer_add(b, t0 + i, i % 2 ? 1.0 : 3.0, i % 2 ? 180.0 : 90.0);
  wind_buffer_average(b, t0 + 59, 600, 0.3, 30, &a);
  check("speed-weighted mean", a.direction, atan2(3.0, -1.0) * 180 / M_PI, 0.01);

  /* Calm: vane random at 0.1 m/s, a few windy samples are not enough */
  wind_buffer_init(b);
  for (int i = 0; i < 600; i++)
    wind_buffer_add(b, t0 + i, i < 20 ? 1.0 : 0.1, (i * 37) % 360);
  wind_buffer_average(b, t0 + 599, 600, 0.3, 30, &a);
  check("calm: direction NAN (caller keeps the last)", a.direction, NAN, 0);
  check("calm: windy samples used", a.used_for_direction, 20, 0);
  check("calm: vane samples", a.direction_samples, 600, 0);

  /* Calm samples are left out, the windy ones give the direction */
  wind_buffer_init(b);
  for (int i = 0; i < 600; i++)
    wind_buffer_add(b, t0 + i, i % 2 ? 0.0 : 2.0, i % 2 ? (i * 37) % 360 : 225.0);
  wind_buffer_average(b, t0 + 599, 600, 0.3, 30, &a);
  check("calm samples ignored -> 225 deg", a.direction, 225.0, 0.01);
  check("mean speed includes calm samples", a.speed, 1.0, 1e-6);

  /* Window: only the last 10 minutes count */
  wind_buffer_init(b);
  for (int i = 0; i < 1200; i++)
    wind_buffer_add(b, t0 + i, 4.0, i < 600 ? 90.0 : 270.0);
  wind_buffer_average(b, t0 + 1199, 600, 0.3, 30, &a);
  check("window drops older samples", a.direction, 270.0, 0.01);
  check("samples in window", a.samples, 600, 0);

  /* Ring buffer wraps after WIND_BUFFER_SIZE samples */
  wind_buffer_init(b);
  for (int i = 0; i < WIND_BUFFER_SIZE + 500; i++)
    wind_buffer_add(b, t0 + i, 5.0, 45.0);
  check("buffer count capped", b->count, WIND_BUFFER_SIZE, 0);
  wind_buffer_average(b, t0 + WIND_BUFFER_SIZE + 499, 3600, 0.3, 30, &a);
  check("full hour after wrap", a.samples, WIND_BUFFER_SIZE, 0);

  /* Without anemometer: unweighted mean, no calm rule */
  wind_buffer_init(b);
  for (int i = 0; i < 60; i++)
    wind_buffer_add(b, t0 + i, NAN, 120.0);
  wind_buffer_average(b, t0 + 59, 600, 0.3, 30, &a);
  check("no anemometer: direction", a.direction, 120.0, 0.01);
  check("no anemometer: speed NAN", a.speed, NAN, 0);

  /* Opposite directions cancel out */
  wind_buffer_init(b);
  for (int i = 0; i < 60; i++)
    wind_buffer_add(b, t0 + i, 2.0, i % 2 ? 0.0 : 180.0);
  wind_buffer_average(b, t0 + 59, 600, 0.3, 30, &a);
  check("cancelling directions -> NAN", a.direction, NAN, 0);

  free(b);
  printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}

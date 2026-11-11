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
#ifndef WIND_AVERAGE_H
#define WIND_AVERAGE_H

#include <time.h>

/*
 * Ring buffer of wind samples (anemometer + wind vane, sampled together about
 * once per second) held in RAM only, and their mean over a time window as
 * recommended by the WMO (CIMO guide, WMO-No. 8): 10 minutes.
 *
 * Direction is a vector mean weighted by the wind speed of the same sample
 * (the "resultant wind"), so 350 deg and 10 deg average to 0 deg, not 180 deg.
 * Samples below the calm speed are left out: in calm air the vane turns at
 * random. Without an anemometer (speed NAN) every sample has weight 1.
 */

/* One hour at one sample per second, the largest averaging window */
#define WIND_BUFFER_SIZE 3600

typedef struct {
  time_t t;
  float speed;     /* m/s, NAN = no anemometer reading */
  float direction; /* deg 0..360, NAN = no vane reading */
} wind_sample_t;

typedef struct {
  wind_sample_t samples[WIND_BUFFER_SIZE];
  int head;  /* next slot to write */
  int count; /* valid samples, at most WIND_BUFFER_SIZE */
} wind_buffer_t;

typedef struct {
  int samples;            /* samples in the window */
  int speed_samples;      /* ... with a wind speed */
  int direction_samples;  /* ... with a vane reading */
  int used_for_direction; /* ... that went into the direction (not calm) */
  double speed;           /* scalar mean m/s, NAN without speed samples */
  double speed_max;       /* highest single sample m/s, NAN without */
  double direction;       /* vector mean deg 0..360, NAN if not enough wind */
  double consistency;     /* |resultant| / sum of weights, 0..1: 1 = steady
                             direction, near 0 = variable; NAN if no direction */
} wind_average_t;

void wind_buffer_init(wind_buffer_t *b);

/* Add a sample; the oldest is overwritten when the buffer is full. A sample
 * with neither speed nor direction is not stored. */
void wind_buffer_add(wind_buffer_t *b, time_t t, double speed, double direction);

/**
 * Mean over the samples with now - window_s < t <= now.
 * calm_speed: samples with a speed below it do not count for the direction
 * (0 = no calm rule). min_direction_samples: fewer usable samples leave
 * direction NAN (calm or too little data, the caller decides what to keep).
 */
void wind_buffer_average(const wind_buffer_t *b, time_t now, int window_s,
                         double calm_speed, int min_direction_samples,
                         wind_average_t *out);

#endif /* WIND_AVERAGE_H */

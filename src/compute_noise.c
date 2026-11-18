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
#include "compute_noise.h"
#include "allsky.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Compare function for qsort (float values)
 */
static int float_cmp(const void *a, const void *b) {
  float fa = *(const float *)a;
  float fb = *(const float *)b;
  return (fa > fb) - (fa < fb);
}

/**
 * Compute the median of an array of float values
 */
static float compute_median_float(float *data, size_t n) {
  qsort(data, n, sizeof(float), float_cmp);
  if (n == 0)
    return 0.0f; // Defensive 
    
  if (n % 2 == 0) {
    return 0.5f * (data[n / 2 - 1] + data[n / 2]);
  } else {
    return data[n / 2];
  }
}

int compute_background_noise_mad_rgbf1(
    const float *image, int width, int height, float median, float *sigma_out,
    int cx,    // center x position of the circle
    int cy,    // center y position of the circle
    int radius // radius of the circle
) {
  if (!image || !sigma_out || width < 1 || height < 1 || radius <= 0.0f)
    return 1;

  printf("Compute background noise: median: %f, radius: %d\n", median, radius);

  int num_pixels = width * height;
  int radius2 = radius * radius;

  /* First pass: count valid (non-masked + within circle) pixels */
  int count = 0;
  float min_val = 1.0f, max_val = 0.0f;
#pragma omp parallel for reduction(+ : count) reduction(min : min_val)         \
    reduction(max : max_val)
  for (int i = 0; i < num_pixels; i++) {
    int y = i / width;
    int x = i % width;
    int dx = x - cx;
    int dy = y - cy;
    if ((dx * dx + dy * dy) <= radius2) {
      const float *pixel = image + CHANNELS * i; // RGBA
      if (pixel[1] > 0.0f) {
        count++;
        if (pixel[1] < min_val)
          min_val = pixel[1];
        if (pixel[1] > max_val)
          max_val = pixel[1];
      }
    }
  }
  printf("Valid pixels: %d min: %f max: %f\n", count, min_val, max_val);

  if (count < 1)
    return 1; // No valid pixels

  /* Allocate array for absolute deviations */
  float *absdev = (float *)allsky_safe_malloc(count * sizeof(float));
  if (!absdev)
    return 1;

  /* Second pass: compute absolute deviations. Serial, directly into absdev
   * (per-thread buffers of count floats each multiplied the peak memory by
   * the thread count). Only the bounding box of the circle can contain
   * valid pixels; the condition is the same as in the first pass. */
  int idx = 0;
  int y0 = cy - radius > 0 ? cy - radius : 0;
  int y1 = cy + radius < height - 1 ? cy + radius : height - 1;
  int x0 = cx - radius > 0 ? cx - radius : 0;
  int x1 = cx + radius < width - 1 ? cx + radius : width - 1;
  for (int y = y0; y <= y1 && idx < count; y++) {
    int dy = y - cy;
    for (int x = x0; x <= x1 && idx < count; x++) {
      int dx = x - cx;
      if ((dx * dx + dy * dy) <= radius2) {
        float val = image[CHANNELS * ((size_t)y * width + x) + 1]; // green
        if (val > 0.0f)
          absdev[idx++] = fabsf(val - median);
      }
    }
  }

  /* Compute MAD and sigma */
  float mad = compute_median_float(absdev, idx);
  *sigma_out = 1.4826f * mad;

  allsky_safe_free(absdev);

  printf("Background noise: MAD: %f, sigma: %f\n", mad, *sigma_out);
  return 0;
}
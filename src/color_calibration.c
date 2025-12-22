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
#include <limits.h> // For UINT16_MAX (65535)
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allsky.h"
#include "color_calibration.h"

#ifndef USE_CUDA

/* RGB to HSV (0.0..1.0) */
static void rgb_to_hsv_f(float r, float g, float b, float *h, float *s,
                         float *v) {
  float max = fmaxf(r, fmaxf(g, b));
  float min = fminf(r, fminf(g, b));
  *v = max;

  float d = max - min;
  *s = (max == 0.0f) ? 0.0f : d / max;

  if (max == min) {
    *h = 0.0f;
  } else {
    if (max == r) {
      *h = (g - b) / d + (g < b ? 6.0f : 0.0f);
    } else if (max == g) {
      *h = (b - r) / d + 2.0f;
    } else {
      *h = (r - g) / d + 4.0f;
    }

    *h /= 6.0f;
  }
}

/* HSV to RGB (0.0..1.0) */
static void hsv_to_rgb_f(float h, float s, float v, float *r, float *g,
                         float *b) {
  int i = (int)(h * 6.0f);
  float f = h * 6.0f - i;
  float p = v * (1.0f - s);
  float q = v * (1.0f - f * s);
  float t = v * (1.0f - (1.0f - f) * s);

  switch (i % 6) {
  case 0:
    *r = v;
    *g = t;
    *b = p;
    break;
  case 1:
    *r = q;
    *g = v;
    *b = p;
    break;
  case 2:
    *r = p;
    *g = v;
    *b = t;
    break;
  case 3:
    *r = p;
    *g = q;
    *b = v;
    break;
  case 4:
    *r = t;
    *g = p;
    *b = v;
    break;
  case 5:
    *r = v;
    *g = p;
    *b = q;
    break;
  }
}

void adjust_saturation_rgbf1(float *rgba, int width, int height,
                             float saturation_factor) {
  if (!rgba || width <= 0 || height <= 0) {
    return;
  }

  if (saturation_factor == 1.0f) {
    printf("Saturation factor: 1.0 (no change)\n");
    return;
  }

  int pixel_count = width * height;
  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;
    float r = rgba[idx + 0]; // already in 0…1
    float g = rgba[idx + 1];
    float b = rgba[idx + 2];

    float h, s, v;
    rgb_to_hsv_f(r, g, b, &h, &s, &v);

    s *= saturation_factor;
    if (s > 1.0f)
      s = 1.0f;

    hsv_to_rgb_f(h, s, v, &r, &g, &b);

    /* Clamp to 0.0 ... 1.0 */
    rgba[idx + 0] = clampf1(r);
    rgba[idx + 1] = clampf1(g);
    rgba[idx + 2] = clampf1(b);
    rgba[idx + 3] = 1.0f; // Alpha
  }

  printf("Saturation: ok. Factor: %.2f\n", saturation_factor);
}



int apply_gamma_correction_rgbf1(float *rgba, int width, int height,
                                 float gamma) {
  if (!rgba || width <= 0 || height <= 0 || gamma <= 0.0f) {
    return 1;
  }

  int pixel_count = width * height;

#pragma omp parallel for
  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;

    float r = rgba[idx + 0];
    float g = rgba[idx + 1];
    float b = rgba[idx + 2];

    /* Apply gamma */
    r = powf(clampf1(r), gamma);
    g = powf(clampf1(g), gamma);
    b = powf(clampf1(b), gamma);

    rgba[idx + 0] = r;
    rgba[idx + 1] = g;
    rgba[idx + 2] = b;
    rgba[idx + 3] = 1.0f;
  }

  printf("Gamma correction: ok. Factor: %.3f\n", gamma);

  return 0;
}

#endif

void apply_white_balance_rgb16(uint16_t *rgb, int width, int height,
                               double scale_r, double scale_b) {
  if (!rgb || width <= 0 || height <= 0)
    return;

  int pixel_count = width * height;

  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;

    uint16_t r = rgb[idx];
    uint16_t g = rgb[idx + 1];
    uint16_t b = rgb[idx + 2];

    r = (uint16_t)fmin(r * scale_r, 65535.0);
    b = (uint16_t)fmin(b * scale_b, 65535.0);

    rgb[idx] = r;
    rgb[idx + 1] = g; // Green remains unchanged
    rgb[idx + 2] = b;
    // Alpha remains unchanged
  }
}
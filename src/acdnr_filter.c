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
#include "acdnr_filter.h"
#include "allsky.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA

/* Laplace-Operator 3x3 (isotropic) */
static const int laplace_kernel[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};

static float compute_laplace(const float *luminance, int x, int y, int width,
                             int height) {
  float result = 0.0f;
  for (int ky = -1; ky <= 1; ky++) {
    for (int kx = -1; kx <= 1; kx++) {
      int ix = x + kx;
      int iy = y + ky;
      if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
        result += luminance[iy * width + ix] * laplace_kernel[ky + 1][kx + 1];
      }
    }
  }
  return fabsf(result);
}

static void box_blur(float *channel, int width, int height, int radius,
                     float amount) {
  float *temp = (float *)allsky_safe_malloc(width * height * sizeof(float));
  if (!temp)
    return;
  memcpy(temp, channel, width * height * sizeof(float));

#pragma omp parallel for collapse(2)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = 0.0f;
      int count = 0;

      for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
          int ix = x + dx;
          int iy = y + dy;

          /* Skip out-of-bounds pixels */
          if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
            sum += temp[iy * width + ix];
            count++;
          }
        }
      }
      float blurred = sum / count;
      channel[y * width + x] =
          (1.0f - amount) * temp[y * width + x] + amount * blurred;
    }
  }

  allsky_safe_free(temp);
}

int acdnr_filter_rgbf1(float *rgba, int width, int height, float stddev_l,
                       float amount_l, int iterations_l, int structure_size_l,
                       float stddev_c, float amount_c, int iterations_c,
                       int structure_size_c) {
  (void)stddev_c;

  if (!rgba || width <= 0 || height <= 0)
    return 1;

  int pixel_count = width * height;
  float *Y = allsky_safe_calloc(pixel_count, sizeof(float));
  float *U = allsky_safe_calloc(pixel_count, sizeof(float));
  float *V = allsky_safe_calloc(pixel_count, sizeof(float));

  if (!Y || !U || !V) {
    allsky_safe_free(Y);
    allsky_safe_free(U);
    allsky_safe_free(V);
    return 2;
  }

printf("ACDNR Filter: stddev_l=%f, amount_l=%f, iterations_l=%d, structure_size_l=%d, stddev_c=%f, amount_c=%f, iterations_c=%d, structure_size_c=%d\n", stddev_l, amount_l, iterations_l, structure_size_l, stddev_c, amount_c, iterations_c, structure_size_c);

/* RGB -> YUV (Rec.709, linear, normalized 0..1) */
#pragma omp parallel for
  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;
    float r = rgba[idx + 0];
    float g = rgba[idx + 1];
    float b = rgba[idx + 2];

    Y[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    U[i] = -0.114572f * r - 0.385428f * g + 0.5f * b;
    V[i] = 0.5f * r - 0.454153f * g - 0.045847f * b;
  }

  /* Luminance filter (ACDNR logic) */
  for (int it = 0; it < iterations_l; it++) {
    float *mask = calloc(pixel_count, sizeof(float));
    if (!mask)
      break;

#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        int i = y * width + x;
        float contrast = compute_laplace(Y, x, y, width, height);
        mask[i] = (contrast < stddev_l) ? 1.0f : 0.0f;
      }
    }

    box_blur(Y, width, height, structure_size_l, amount_l);

    allsky_safe_free(mask);
  }

  /* Chrominance filter */
  for (int it = 0; it < iterations_c; it++) {
    box_blur(U, width, height, structure_size_c, amount_c);
    box_blur(V, width, height, structure_size_c, amount_c);
  }

/* YUV -> RGB */
#pragma omp parallel for
  for (int i = 0; i < pixel_count; i++) {
    float yv = Y[i];
    float u = U[i];
    float v = V[i];

    float r = yv + 1.13983f * v;
    float g = yv - 0.39465f * u - 0.58060f * v;
    float b = yv + 2.03211f * u;

    int idx = i * CHANNELS;
    rgba[idx + 0] = clampf1(r);
    rgba[idx + 1] = clampf1(g);
    rgba[idx + 2] = clampf1(b);
    rgba[idx + 3] = 1.0f; // Alpha = 1.0 im 0..1 Raum
  }

  allsky_safe_free(Y);
  allsky_safe_free(U);
  allsky_safe_free(V);

  printf("ACDNR Filter: ok\n");

  return 0;
}

#endif

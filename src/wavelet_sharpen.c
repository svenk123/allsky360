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
#include "wavelet_sharpen.h"

#include "allsky.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#ifndef USE_CUDA

static void gaussian_blur_rgbf1(const float *src, float *dst, int width,
                                int height, float sigma) {
  int radius = (int)ceilf(3.0f * sigma);
  int size = 2 * radius + 1;
  float *kernel = (float *)allsky_safe_malloc(size * sizeof(float));

  float sum = 0.0f;
  for (int i = -radius; i <= radius; i++) {
    float v = expf(-(i * i) / (2.0f * sigma * sigma));
    kernel[i + radius] = v;
    sum += v;
  }

  for (int i = 0; i < size; i++)
    kernel[i] /= sum;

  // Temporary buffer for horizontal pass
  float *temp =
      (float *)allsky_safe_malloc((size_t)width * height * CHANNELS * sizeof(float));

// Horizontal pass
#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float r = 0, g = 0, b = 0, a = 0;

      for (int k = -radius; k <= radius; k++) {
        int xx = x + k;

        if (xx < 0)
          xx = 0;

        if (xx >= width)
          xx = width - 1;

        int idx = (y * width + xx) * CHANNELS;
        float w = kernel[k + radius];
        r += src[idx + 0] * w;
        g += src[idx + 1] * w;
        b += src[idx + 2] * w;
        a += src[idx + 3] * w;
      }

      int dst_idx = (y * width + x) * CHANNELS;
      temp[dst_idx + 0] = r;
      temp[dst_idx + 1] = g;
      temp[dst_idx + 2] = b;
      temp[dst_idx + 3] = a;
    }
  }

// Vertical pass
#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float r = 0, g = 0, b = 0, a = 0;

      for (int k = -radius; k <= radius; k++) {
        int yy = y + k;

        if (yy < 0)
          yy = 0;

        if (yy >= height)
          yy = height - 1;

        int idx = (yy * width + x) * CHANNELS;
        float w = kernel[k + radius];
        r += temp[idx + 0] * w;
        g += temp[idx + 1] * w;
        b += temp[idx + 2] * w;
        a += temp[idx + 3] * w;
      }

      int dst_idx = (y * width + x) * CHANNELS;
      dst[dst_idx + 0] = r;
      dst[dst_idx + 1] = g;
      dst[dst_idx + 2] = b;
      dst[dst_idx + 3] = a;
    }
  }

  allsky_safe_free(kernel);
  allsky_safe_free(temp);
}

int wavelet_sharpen_rgbf1(float *rgba, int width, int height, float gain_small,
                          float gain_medium, float gain_large) {
  if (!rgba || width <= 0 || height <= 0)
    return 1;

  printf("wavelet_sharpen_rgbf1: gain_small: %.1f gain_medium: %.1f gain_large: %.1f\n", gain_small, gain_medium, gain_large);

  size_t sz = (size_t)width * height * CHANNELS * sizeof(float);
  float *blur1 = (float *)allsky_safe_malloc(sz);
  float *blur2 = (float *)allsky_safe_malloc(sz);
  float *blur3 = (float *)allsky_safe_malloc(sz);

  // Multi-scale Gaussian blurs
  printf("Multi-scale Gaussian blurs:\n");
  gaussian_blur_rgbf1(rgba, blur1, width, height, 1.0f);
  printf("Scale 1: sigma: 1.0\n");
  gaussian_blur_rgbf1(rgba, blur2, width, height, 3.0f);
  printf("Scale 2: sigma: 3.0\n");
  gaussian_blur_rgbf1(rgba, blur3, width, height, 8.0f);
  printf("Scale 3: sigma: 8.0\n");

  // In-place wavelet sharpening
  const int pixels = width * height;
  
  // Debug: Calculate statistics of wavelet coefficients
  float max_small = 0.0f, max_medium = 0.0f, max_large = 0.0f;
  double sum_small = 0.0, sum_medium = 0.0, sum_large = 0.0;
  float max_change = 0.0f;
  
#pragma omp parallel
  {
    float local_max_small = 0.0f, local_max_medium = 0.0f, local_max_large = 0.0f;
    double local_sum_small = 0.0, local_sum_medium = 0.0, local_sum_large = 0.0;
    float local_max_change = 0.0f;
    
#pragma omp for schedule(static)
    for (int i = 0; i < pixels; i++) {
      int idx = i * CHANNELS;

      for (int c = 0; c < 3; c++) { // R, G, B only
        float orig = rgba[idx + c];
        float small = orig - blur1[idx + c];
        float medium = blur1[idx + c] - blur2[idx + c];
        float large = blur2[idx + c] - blur3[idx + c];

        // Track statistics
        float abs_small = fabsf(small);
        float abs_medium = fabsf(medium);
        float abs_large = fabsf(large);
        if (abs_small > local_max_small) local_max_small = abs_small;
        if (abs_medium > local_max_medium) local_max_medium = abs_medium;
        if (abs_large > local_max_large) local_max_large = abs_large;
        local_sum_small += abs_small;
        local_sum_medium += abs_medium;
        local_sum_large += abs_large;

        float val =
            orig + gain_small * small + gain_medium * medium + gain_large * large;

        float change = fabsf(val - orig);
        if (change > local_max_change) local_max_change = change;

        rgba[idx + c] = clampf1(val);
      }
    }
    
#pragma omp critical
    {
      if (local_max_small > max_small) max_small = local_max_small;
      if (local_max_medium > max_medium) max_medium = local_max_medium;
      if (local_max_large > max_large) max_large = local_max_large;
      if (local_max_change > max_change) max_change = local_max_change;
      sum_small += local_sum_small;
      sum_medium += local_sum_medium;
      sum_large += local_sum_large;
    }
  }
  
  float avg_small = (float)(sum_small / (pixels * 3));
  float avg_medium = (float)(sum_medium / (pixels * 3));
  float avg_large = (float)(sum_large / (pixels * 3));
  
  printf("Sharpened wavelet coefficients:\n");
  printf("Scale 1: max: %.6f avg: %.6f\nScale 2: max: %.6f avg: %.6f\nScale 3: max: %.6f avg: %.6f\nMax change: %.6f\n",
         max_small, avg_small, max_medium, avg_medium, max_large, avg_large, max_change);

  allsky_safe_free(blur1);
  allsky_safe_free(blur2);
  allsky_safe_free(blur3);

  printf("Wavelet sharpen: ok\n");
  return 0;
}

#endif

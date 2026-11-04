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
#include "white_balance.h"
#include "allsky.h"
#include <math.h>
#include <stdio.h>

#define DEBUG_SCALING 1

// Smoothstep 0..1 between edge0..edge1
static inline float smoothstep(float edge0, float edge1, float x) {
  if (x <= edge0)
    return 0.f;
  if (x >= edge1)
    return 1.f;
  float t = (x - edge0) / (edge1 - edge0);
  return t * t * (3.f - 2.f * t);
}

int white_balance_rgbf1(
    float *rgba, int width, int height, float scale_r, float scale_g,
    float scale_b,
    float lp_strength,    // 0..1: how softly we approach the hard limit, if 0.0 then light protection is disabled
    int hdr_frame         // 1: no attenuation/light protection (see header)
    ) {
  if (!rgba || width <= 0 || height <= 0)
    return 1;
  if (scale_r <= 0.f || scale_g <= 0.f || scale_b <= 0.f)
    return 2;
    
  if (lp_strength < 0.f)
    lp_strength = 0.f;
  if (lp_strength > 1.f)
    lp_strength = 1.f;

  /* HDR frames must keep identical gains across the series and must keep
   * clipped pixels at 1.0: a content-dependent attenuation (or the per-pixel
   * limiter) would turn a sensor-clipped (1,1,1) into e.g. (0.55,0.53,1.0),
   * i.e. a blue cast the fusion no longer recognises as clipped, and would
   * break the exposure-time ratio between the frames. */
  if (hdr_frame)
    lp_strength = 0.f;


  printf("White Balance scaling: red: %.6f, green: %.6f, blue: %.6f, lp_strength: %.6f\n", scale_r,scale_g,scale_b, lp_strength);
  const int N = width * height;

  // 1) find per-channel maxima (pre-scale), only needed for the attenuation
  float r_max = 0.f, g_max = 0.f, b_max = 0.f;
  for (int i = 0; !hdr_frame && i < N; ++i) {
    const int idx = i * CHANNELS;
    float r = rgba[idx + 0], g = rgba[idx + 1], b = rgba[idx + 2];
    if (r > r_max)
      r_max = r;
    if (g > g_max)
      g_max = g;
    if (b > b_max)
      b_max = b;
  }

  // 2) global attenuation (only if strongly needed)
  //    Use a threshold so minor overshoots are handled by the per-pixel
  //    limiter.
  float post_r = r_max * scale_r;
  float post_g = g_max * scale_g;
  float post_b = b_max * scale_b;
  float post_max = fmaxf(post_r, fmaxf(post_g, post_b));

  const float ATTEN_TRIG = 1.10f; // start global attenuation only if > +10%
  float atten =
      (!hdr_frame && post_max > ATTEN_TRIG) ? (1.f / post_max) : 1.f;

  float sR = scale_r * atten;
  float sG = scale_g * atten;
  float sB = scale_b * atten;


// 3) apply WB with per-pixel hard-safe limiter (hue-preserving)
//    If a pixel's max channel would exceed 1.0 after global scaling,
//    we damp all three channels by the minimal factor, softened by lp_strength.
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    const int idx = i * CHANNELS;

    float R1 = rgba[idx + 0] * sR;
    float G1 = rgba[idx + 1] * sG;
    float B1 = rgba[idx + 2] * sB;

    if (lp_strength > 0.f) {
      float M = fmaxf(R1, fmaxf(G1, B1)); // max channel
      if (M > 1.f) {
        // minimal factor to avoid clipping exactly:
        float k_hard = 1.f / M; // in (0..1]
        // soften towards 1 by lp_strength (0=no soften, 1=just enough):
        float k = 1.f - (1.f - k_hard) * lp_strength; // in [k_hard..1]
        R1 *= k;
        G1 *= k;
        B1 *= k;
      }
    }

    rgba[idx + 0] = clampf1(R1);
    rgba[idx + 1] = clampf1(G1);
    rgba[idx + 2] = clampf1(B1);
    rgba[idx + 3] = rgb_to_luminance_rec709(R1, G1, B1);
  }

  printf("White Balance applied: red: %.6f, green: %.6f, blue: %.6f\n", sR,
         sG, sB);
  return 0;
}

int estimate_gray_scaling_rgbf1(const float *rgba, int width, int height,
                                int x0, int y0, int x1, int y1, float *scale_r,
                                float *scale_b) {
  if (!rgba || width <= 0 || height <= 0 || !scale_r || !scale_b)
    return 1;

  double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
  int total = 0;

#pragma omp parallel for reduction(+ : sum_r, sum_g, sum_b, total)
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = (y * width + x) * CHANNELS;
        float r = rgba[idx + 0];
        float g = rgba[idx + 1];
        float b = rgba[idx + 2];

        // Optional: ignore clipped/underexposed values
        if (r < 0.99f && g < 0.99f && b < 0.99f && r > 0.01f && g > 0.01f &&
            b > 0.01f) {
          sum_r += r;
          sum_g += g;
          sum_b += b;
          total++;
        }
      }
    }
  }

  if (total > 0) {
    *scale_r = (float)(sum_g / sum_r);
    *scale_b = (float)(sum_g / sum_b);
  } else {
    *scale_r = 1.0f;
    *scale_b = 1.0f;
  }

  return 0;
}

#ifndef USE_CUDA

int auto_color_calibration_rgbf1(float *rgba, int width, int height,
                                 float subframe_percent) {
  if (!rgba || width <= 0 || height <= 0 || subframe_percent <= 0.0f ||
      subframe_percent > 100.0f)
    return 1;

  printf("Auto Color Calibration: subframe %.1f%%\n", subframe_percent);

  int x0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * width);
  int y0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * height);
  int x1 = width - x0;
  int y1 = height - y0;

  double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
  double sum_r2 = 0.0, sum_g2 = 0.0, sum_b2 = 0.0;
  int count = 0;

  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      int idx = (y * width + x) * 4;
      float r = rgba[idx + 0];
      float g = rgba[idx + 1];
      float b = rgba[idx + 2];

      if (r < 1.0f && g < 1.0f && b < 1.0f) {
        sum_r += r;
        sum_g += g;
        sum_b += b;
        sum_r2 += r * r;
        sum_g2 += g * g;
        sum_b2 += b * b;
        count++;
      }
    }
  }

  if (count == 0)
    return 1;

  double mean_r = sum_r / count;
  double mean_g = sum_g / count;
  double mean_b = sum_b / count;

  double stddev_r = sqrt(sum_r2 / count - mean_r * mean_r);
  double stddev_g = sqrt(sum_g2 / count - mean_g * mean_g);
  double stddev_b = sqrt(sum_b2 / count - mean_b * mean_b);

  double target = fmax(fmax(mean_r, mean_g), mean_b);

  float scale_r = (float)(target / mean_r);
  float scale_g = (float)(target / mean_g);
  float scale_b = (float)(target / mean_b);

  float nSigma = (float)(3.0 * fmax(fmax(stddev_r, stddev_g), stddev_b));

  float max_val = (float)(target + nSigma);

#pragma omp parallel for
  for (int i = 0; i < width * height; ++i) {
    int idx = i * 4;
    float r = rgba[idx + 0];
    float g = rgba[idx + 1];
    float b = rgba[idx + 2];

    // Background neutralization
    r = fminf(r, max_val);
    g = fminf(g, max_val);
    b = fminf(b, max_val);

    // White balance
    r *= scale_r;
    g *= scale_g;
    b *= scale_b;

    // Clamping
    rgba[idx + 0] = clampf1(r);
    rgba[idx + 1] = clampf1(g);
    rgba[idx + 2] = clampf1(b);
    rgba[idx + 3] = rgb_to_luminance_rec709(r, g, b);
  }

  printf("Auto Color Calibration: ok\n");

  return 0;
}

static inline float saturation_rgb(float r, float g, float b) {
  float maxc = fmaxf(r, fmaxf(g, b));
  float minc = fminf(r, fminf(g, b));

  if (maxc <= 1e-6f)
    return 0.0f;

  return (maxc - minc) / maxc;
}


#endif

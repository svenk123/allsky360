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
#include "auto_gamma.h"
#include "allsky.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>

float compute_auto_gamma_rgbf1(const float *rgbf, int width, int height,
                               float target_mean, float gamma_min,
                               float gamma_max) {
  if (!rgbf || width <= 0 || height <= 0)
    return 1.0f;

  const int pixel_count = width * height;
  const float eps = 1e-6f;

  double sum_Y = 0.0;
  long count = 0;

#pragma omp parallel for reduction(+ : sum_Y, count)
  for (int i = 0; i < pixel_count; i++) {
    const float *px = &rgbf[i * CHANNELS];

    /* Skip masked pixels */
    if (px[0] <= eps && px[1] <= eps && px[2] <= eps)
      continue;

    float Y = rgb_to_luminance_rec709(px[0], px[1], px[2]);

    /* Only count pixels in valid 0..1 range; values above 1.0 (HDR) would
     * skew the mean and are handled by Reinhard afterwards */
    if (Y > eps && Y <= 1.0f) {
      sum_Y += Y;
      count++;
    }
  }

  if (count == 0) {
    printf("Auto-Gamma: no valid pixels, using gamma=1.0\n");
    return 1.0f;
  }

  float mean_Y = (float)(sum_Y / (double)count);

  if (mean_Y <= eps) {
    printf("Auto-Gamma: scene too dark (mean Y=%.6f), using gamma=1.0\n",
           mean_Y);
    return 1.0f;
  }

  float gamma = logf(target_mean) / logf(mean_Y);
  gamma = fmaxf(gamma_min, fminf(gamma_max, gamma));

  printf("Auto-Gamma: scene mean Y=%.4f, target=%.4f -> gamma=%.3f\n",
         mean_Y, target_mean, gamma);

  return gamma;
}

#ifndef USE_CUDA

int reinhard_tonemap_rgbf1(float *rgbf, int width, int height,
                           float white_point) {
  if (!rgbf || width <= 0 || height <= 0)
    return 1;

  const int pixel_count = width * height;
  const float eps = 1e-6f;

  /* In simple mode (white_point == 0) the naive operator Y/(1+Y) maps
   * Y_max=1.0 → 0.5, cutting the histogram at 128/255.  To avoid this,
   * auto-detect the scene maximum luminance and use the extended Reinhard
   * formula with that value as the white point.  The extended formula
   * satisfies Y_out(Y_max) == 1.0 by construction:
   *
   *   Y_out = Y_in * (1 + Y_in/Lw²) / (1 + Y_in)
   *   Y_out(Lw) = Lw * (1 + 1/Lw) / (1 + Lw) = (Lw+1)/(1+Lw) = 1.0  ✓
   *
   * If the caller supplies an explicit white_point > 0 it is used directly.
   */
  float lw = white_point; /* user-supplied or 0 for auto */

  if (lw <= eps) {
    /* Auto-detect maximum luminance */
    float max_Y = 0.0f;
#pragma omp parallel for reduction(max : max_Y)
    for (int i = 0; i < pixel_count; i++) {
      const float *px = &rgbf[i * CHANNELS];
      if (px[0] <= eps && px[1] <= eps && px[2] <= eps)
        continue;
      float Y = rgb_to_luminance_rec709(px[0], px[1], px[2]);
      if (Y > max_Y)
        max_Y = Y;
    }
    lw = (max_Y > eps) ? max_Y : 1.0f;
    printf("Reinhard TMO: white_point=%.4f (auto-detected)\n", lw);
  } else {
    printf("Reinhard TMO: white_point=%.4f (user-defined)\n", lw);
  }

  const float lw2 = lw * lw;

#pragma omp parallel for
  for (int i = 0; i < pixel_count; i++) {
    float *px = &rgbf[i * CHANNELS];

    float Y_in = rgb_to_luminance_rec709(px[0], px[1], px[2]);
    if (Y_in <= eps)
      continue;

    /* Extended Reinhard: maps Y_in=lw exactly to 1.0 */
    float Y_out = Y_in * (1.0f + Y_in / lw2) / (1.0f + Y_in);

    /* Scale RGB proportionally to preserve hue */
    float scale = Y_out / Y_in;
    px[0] = clampf1(px[0] * scale);
    px[1] = clampf1(px[1] * scale);
    px[2] = clampf1(px[2] * scale);
  }

  printf("Reinhard TMO: ok\n");
  return 0;
}

int apply_tone_curve_rgbf1(float *rgbf, int width, int height,
                           float shadow_lift, float midtone_contrast) {
  if (!rgbf || width <= 0 || height <= 0)
    return 1;

  if (shadow_lift < 0.0f)
    shadow_lift = 0.0f;
  if (midtone_contrast < 0.0f)
    midtone_contrast = 0.0f;

  const int pixel_count = width * height;
  const float eps = 1e-6f;

  printf("Tone Curve: shadow_lift=%.3f  midtone_contrast=%.3f\n",
         shadow_lift, midtone_contrast);

#pragma omp parallel for
  for (int i = 0; i < pixel_count; i++) {
    float *px = &rgbf[i * CHANNELS];

    /* Skip masked / black pixels */
    if (px[0] <= eps && px[1] <= eps && px[2] <= eps)
      continue;

    float Y_in = rgb_to_luminance_rec709(px[0], px[1], px[2]);
    if (Y_in <= eps)
      continue;

    float x = Y_in;

    /* Shadow lift: 4·x·(1−x)²
     * Bell curve peaking at x≈0.33 — only affects shadows/lower midtones.
     * Always positive → raises the lower tonal range. */
    float sl = shadow_lift * 4.0f * x * (1.0f - x) * (1.0f - x);

    /* S-curve: 4·x·(1−x)·(1−2x)
     * Antisymmetric around 0.5: positive for x < 0.5 (shadows up),
     * negative for x > 0.5 (highlights down). Pivot at x = 0.5. */
    float sc = midtone_contrast * 4.0f * x * (1.0f - x) * (1.0f - 2.0f * x);

    float Y_out = clampf1(x + sl + sc);

    /* Scale RGB proportionally to preserve hue */
    float scale = Y_out / Y_in;
    px[0] = clampf1(px[0] * scale);
    px[1] = clampf1(px[1] * scale);
    px[2] = clampf1(px[2] * scale);
  }

  printf("Tone Curve: ok\n");
  return 0;
}

#endif

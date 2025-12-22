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

#include "allsky.h"
#include "hdrmt.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline int iclamp(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

/* Simple midtones transfer (PixInsight-like midtones balance).
 * m in [0,1], x in [0,1]. m=0.5 -> identity.
 */
static inline float midtones_xfer(float x, float m) {
  if (m <= 0.0f)
    return (x <= 0.0f) ? 0.0f : 1.0f;

  if (m >= 1.0f)
    return x;

  float num = (x - m);
  float den = (1.0f - 2.0f * m);

  if (fabsf(den) < 1e-6f)
    return x;

  float y = (num / den + 0.5f);

  return clampf1(y);
}

/* À trous separable B3-spline blur (dilated by 'step'). */
static void atrous_pass(const float *src, float *dst, int w, int h, int step) {
  static const float h1[5] = {1.f / 16.f, 4.f / 16.f, 6.f / 16.f, 4.f / 16.f,
                              1.f / 16.f};
  float *tmp = (float *)allsky_safe_malloc((size_t)w * h * sizeof(float));

  if (!tmp) {
    memcpy(dst, src, (size_t)w * h * sizeof(float));
    return;
  }

/* Horizontal */
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const float *row = src + (size_t)y * w;
    float *out = tmp + (size_t)y * w;

    for (int x = 0; x < w; ++x) {
      double acc = 0.0;

      for (int k = -2; k <= 2; ++k) {
        int xx = iclamp(x + k * step, 0, w - 1);
        acc += row[xx] * (double)h1[k + 2];
      }

      out[x] = (float)acc;
    }
  }

/* Vertical */
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    float *out = dst + (size_t)y * w;

    for (int x = 0; x < w; ++x) {
      double acc = 0.0;

      for (int k = -2; k <= 2; ++k) {
        int yy = iclamp(y + k * step, 0, h - 1);
        acc += tmp[(size_t)yy * w + x] * (double)h1[k + 2];
      }

      out[x] = (float)acc;
    }
  }

  allsky_safe_free(tmp);
}

/* Local contrast compression curve: v' = v / (1 + k*|v|) */
static inline float compress_curve(float v, float k) {
  float a = fabsf(v);
  return (a > 0.0f) ? v / (1.0f + k * a) : v;
}

/* Shadow/Highlight protection blend factor in [0,1] */
static inline float protection_factor(float y, float sh, float hi) {
  float pf_sh = 1.0f - sh * (1.0f - y);
  float pf_hi = 1.0f - hi * (y);
  float pf = pf_sh * pf_hi;

  if (pf < 0.0f)
    pf = 0.0f;

  if (pf > 1.0f)
    pf = 1.0f;

  return pf;
}

int hdrmt_rgbf1(float *rgb, int width, int height, int levels, int start_level,
                float strength, float strength_boost, float midtones,
                float shadow_protect, float highlight_protect, float epsilon,
                float gain_cap) {
  if (!rgb || width <= 1 || height <= 1)
    return 1;

  if (levels < 1 || levels > 10)
    return 2;

  if (start_level < 0)
    return 3;

  if (start_level >= levels)
    return 4;

  if (epsilon <= 0.0f)
    return 5;

  const size_t pixel_count = (size_t)width * height;
  
  printf("HDR multiscale transform: levels: %d, start_level: %d, strength: %f, "
         "strength_boost: %f, midtones: %f, shadow_protect: %f, "
         "highlight_protect: %f, epsilon: %f, gain_cap: %f\n",
         levels, start_level, strength, strength_boost, midtones,
         shadow_protect, highlight_protect, epsilon, gain_cap);
  
  printf("Processing image: %dx%d pixels (%zu total)\n", width, height, pixel_count);

  float *Y = (float *)allsky_safe_malloc(pixel_count * sizeof(float));
  float *A = (float *)allsky_safe_malloc(pixel_count * sizeof(float));

  if (!Y || !A) {
    allsky_safe_free(Y);
    allsky_safe_free(A);
    return 6;
  }

/* Build luminance + init approximation */
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < pixel_count; ++i) {
    int idx = i * CHANNELS;
    float r = rgb[idx + 0];
    float g = rgb[idx + 1];
    float b = rgb[idx + 2];
    float y = rgb_to_luma(r, g, b);
    Y[i] = y;
    A[i] = y;
  }

  /* Multiscale decomposition (à trous) */
  printf("Starting multiscale decomposition with %d levels...\n", levels);
  float **D = (float **)allsky_safe_calloc((size_t)levels, sizeof(float *));
  if (!D) {
    allsky_safe_free(Y);
    allsky_safe_free(A);
    return 7;
  }

  for (int l = 0; l < levels; ++l) {
    printf("Processing level %d/%d...\n", l + 1, levels);
    D[l] = (float *)allsky_safe_malloc(pixel_count * sizeof(float));

    if (!D[l]) {
      for (int j = 0; j < l; ++j)
        allsky_safe_free(D[j]);
      allsky_safe_free(D);
      allsky_safe_free(Y);
      allsky_safe_free(A);
      return 8;
    }

    float *B = (float *)allsky_safe_malloc(pixel_count * sizeof(float));
    if (!B) {
      for (int j = 0; j <= l; ++j)
        allsky_safe_free(D[j]);
      allsky_safe_free(D);
      allsky_safe_free(Y);
      allsky_safe_free(A);
      return 9;
    }

    int step = 1 << l;
    atrous_pass(A, B, width, height, step);

/* D_l = A - B */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < pixel_count; ++i) {
      D[l][i] = A[i] - B[i];
    }

    memcpy(A, B, pixel_count * sizeof(float));
    allsky_safe_free(B);
    printf("Completed level %d/%d\n", l + 1, levels);
  }
  /* A is coarse residual */

  const float base_k = clampf1(strength);
  const float boost = clampf1(strength_boost);
  const float mt = clampf1(midtones);
  const float sh_prot = clampf1(shadow_protect);
  const float hi_prot = clampf1(highlight_protect);

  float *Yp = (float *)allsky_safe_malloc(pixel_count * sizeof(float));
  if (!Yp) {
    for (int l = 0; l < levels; ++l)
      allsky_safe_free(D[l]);
    allsky_safe_free(D);
    allsky_safe_free(Y);
    allsky_safe_free(A);
    return 10;
  }

/* Start from compressed residual */
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < pixel_count; ++i) {
    float a = A[i];
    float t = midtones_xfer(clampf1(a), mt);
    float delta_mid = t - clampf1(a);
    float k_res = base_k * (1.0f + 0.5f * boost);
    float pf = protection_factor(clampf1(Y[i]), sh_prot, hi_prot);
    float a_comp = compress_curve(a, k_res * pf);
    Yp[i] = a_comp + 0.3f * delta_mid;
  }

  /* Add detail bands back (compressed from start_level upwards) */
  printf("Adding detail bands back...\n");
  for (int l = 0; l < levels; ++l) {
    const int apply = (l >= start_level);
    float scale = (levels > 1) ? ((float)l / (float)(levels - 1)) : 1.0f;
    float k = base_k * (apply ? (1.0f + boost * scale) : 0.0f);

    if (!apply || k <= 0.0f) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < pixel_count; ++i) {
        Yp[i] += D[l][i];
      }
    } else {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < pixel_count; ++i) {
        float pf = protection_factor(clampf1(Y[i]), sh_prot, hi_prot);
        float d = D[l][i];
        float dc = compress_curve(d, k * pf);
        Yp[i] += dc;
      }
    }
  }

  /* Apply luminance gain to RGB */
  printf("Applying luminance gain to RGB...\n");
  const float eps = epsilon;
  const float cap = (gain_cap >= 1.0f) ? gain_cap : 0.0f;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < pixel_count; ++i) {
    int idx = i * CHANNELS;
    float y = Y[i];
    float yp = Yp[i];
    float denom = y + eps;
    if (denom < eps)
      denom = eps;
    float gain = yp / denom;

    if (cap >= 1.0f) {
      if (gain > cap)
        gain = cap;
      if (gain < 1.0f / cap)
        gain = 1.0f / cap;
    }

    float r = rgb[idx + 0] * gain;
    float g = rgb[idx + 1] * gain;
    float b = rgb[idx + 2] * gain;

    rgb[idx + 0] = clampf1(r);
    rgb[idx + 1] = clampf1(g);
    rgb[idx + 2] = clampf1(b);
  }
  
  printf("HDR multiscale transform: ok\n");

  /* Cleanup */
  for (int l = 0; l < levels; ++l)
    allsky_safe_free(D[l]);
  allsky_safe_free(D);
  allsky_safe_free(Yp);
  allsky_safe_free(A);
  allsky_safe_free(Y);
  return 0;
}

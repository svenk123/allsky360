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
#include "clarity_filter.h"
#include "allsky.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA

/* --- Helpers ------------------------------------------------------------- */

/* Safe ratio (avoid division by zero). */
static inline float safe_ratio(float num, float den) {
  const float eps = 1e-8f;
  float d = (fabsf(den) < eps) ? (den >= 0.f ? eps : -eps) : den;

  return num / d;
}

/* Decide whether a pixel is masked according to mask_mode. */
static inline int is_masked(const float *rgba, int mask_mode) {
  if (mask_mode == CLARITY_MASK_NONE)
    return 0;
  const float r = rgba[0], g = rgba[1], b = rgba[2], a = rgba[3];
  const float eps = 1e-12f;
  int alpha_zero = (fabsf(a) <= eps);
  int rgb_zero = (fabsf(r) <= eps) && (fabsf(g) <= eps) && (fabsf(b) <= eps);

  switch (mask_mode) {
  case CLARITY_MASK_ALPHA_ZERO:
    return alpha_zero;
  case CLARITY_MASK_RGB_ALL_ZERO:
    return rgb_zero;
  case CLARITY_MASK_RGB_OR_ALPHA:
    return alpha_zero || rgb_zero;
  default:
    return 0;
  }
}

/* Build 1D Gaussian kernel (sigma ~ radius/3). */
static int make_gauss_kernel(int radius, float **out_k, int *out_len) {
  if (radius < 1) {
    *out_k = NULL;
    *out_len = 0;

    return 1;
  }

  int len = 2 * radius + 1;
  float *k = (float *)malloc(sizeof(float) * (size_t)len);
  if (!k)
    return 2;

  float sigma = (float)radius / 3.0f;
  if (sigma < 0.5f)
    sigma = 0.5f;
  float inv2s2 = 1.0f / (2.0f * sigma * sigma);

  float sum = 0.0f;
  for (int i = 0; i < len; ++i) {
    int x = i - radius;
    float v = expf(-(x * x) * inv2s2);
    k[i] = v;
    sum += v;
  }

  for (int i = 0; i < len; ++i)
    k[i] /= sum;

  *out_k = k;
  *out_len = len;

  printf("gauss_kernel: len=%d\n", len);
  return 0;
}

/* Separable Gaussian blur, masked: horizontal pass.
 * Computes two convolutions in one:
 *   num = sum(k * Y_valid),  den = sum(k * valid)
 * Then out = num / max(den, tiny). Border handling: mirror. */
static void gauss_blur_h_masked(const float *Y, const unsigned char *valid,
                                float *dst, int width, int height,
                                const float *k, int radius) {
 // const int klen = 2 * radius + 1;
  const float tiny = 1e-8f;

#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    float *drow = dst + (size_t)y * width;
    const float *sY = Y + (size_t)y * width;
    const unsigned char *sV = valid + (size_t)y * width;

    for (int x = 0; x < width; ++x) {
      float num = 0.0f, den = 0.0f;

      for (int i = -radius; i <= radius; ++i) {
        int xi = x + i;
        if (xi < 0)
          xi = -xi;
        if (xi >= width)
          xi = 2 * width - 2 - xi;

        float w = k[i + radius];
        if (sV[xi]) {
          num += sY[xi] * w;
          den += w;
        }
      }

      //printf("gauss_blur_h_masked: num=%f, den=%f\n", num, den);
      drow[x] = (den > tiny) ? (num / den) : sY[x];
    }
  }
}

/* Vertical masked blur (same scheme as horizontal). */
static void gauss_blur_v_masked(const float *Y, const unsigned char *valid,
                                float *dst, int width, int height,
                                const float *k, int radius) {
//  const int klen = 2 * radius + 1;
//  const float tiny = 1e-8f;

#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    float *drow = dst + (size_t)y * width;

    for (int x = 0; x < width; ++x) {
      float num = 0.0f, den = 0.0f;

      for (int i = -radius; i <= radius; ++i) {
        int yi = y + i;
        if (yi < 0)
          yi = -yi;
        if (yi >= height)
          yi = 2 * height - 2 - yi;

        const float *rowY = Y + (size_t)yi * width;
        const unsigned char *rowV = valid + (size_t)yi * width;

        float w = k[i + radius];
        if (rowV[x]) {
          num += rowY[x] * w;
          den += w;
        }
      }

      //printf("gauss_blur_v_masked: num=%f, den=%f\n", num, den);
      drow[x] = (den > 1e-8f) ? (num / den) : Y[(size_t)y * width + x];
    }
  }
}

/* Soft limiter to reduce halos. */
static inline float soft_limiter(float x) { 
    return tanhf(x); 
}

/* Midtone weighting (bell around 0.5). */
static inline float midtone_bell(float y, float width) {
  float w = (width <= 0.05f) ? 0.05f : (width > 1.0f ? 1.0f : width);
  float sigma = 0.25f * w;
  float d = y - 0.5f;
  float inv2s2 = 1.0f / (2.0f * sigma * sigma);
  //printf("midtone_bell: y=%f, width=%f, sigma=%f, d=%f, inv2s2=%f\n", y, width, sigma, d, inv2s2);
  return expf(-(d * d) * inv2s2);
}

/* Optional highlight rolloff. */
static inline float highlight_rolloff(float y) {
  if (y <= 0.8f)
    return 1.0f;
  float t = (y - 0.8f) / 0.2f;
  if (t < 0.f)
    t = 0.f;
  if (t > 1.f)
    t = 1.f;

  //printf("highlight_rolloff: y=%f, t=%f\n", y, t);
  return 0.5f * (1.0f + cosf((float)M_PI * t));
}

/* --- Public API ---------------------------------------------------------- */

int clarity_filter_rgbf_masked(float *rgba, int width, int height,
                               float strength, int radius, float midtone_width,
                               int preserve_highlights, int mask_mode) {
  if (!rgba || width <= 0 || height <= 0)
    return 1;

  if (fabsf(strength) < 1e-6f)
    return 0;

  if (radius < 1)
    radius = 1;

  if (midtone_width <= 0.f)
    midtone_width = 0.35f;

  if (midtone_width > 1.5f)
    midtone_width = 1.5f;

  printf("Clarity Filter: strength=%f, radius=%d, midtone_width=%f, preserve_highlights=%d, mask_mode=%d\n", strength, radius, midtone_width, preserve_highlights, mask_mode);

  const size_t N = (size_t)width * (size_t)height;

  float *Y = (float *)allsky_safe_malloc(sizeof(float) * N);
  float *Ytmp = (float *)allsky_safe_malloc(sizeof(float) * N);
  float *Yblur = (float *)allsky_safe_malloc(sizeof(float) * N);
  unsigned char *V =
      (unsigned char *)allsky_safe_malloc(sizeof(unsigned char) * N); /* validity mask */
  if (!Y || !Ytmp || !Yblur || !V) {
    allsky_safe_free(Y);
    allsky_safe_free(Ytmp);
    allsky_safe_free(Yblur);
    allsky_safe_free(V);
    fprintf(stderr, "clarity_filter_rgbf_masked: error allocating memory\n");
    return 2;
  }

/* Build validity map + luminance (masked pixels get V=0, Y arbitrary). */
#pragma omp parallel for
  for (int i = 0; i < (int)N; ++i) {
    const float *p = rgba + 4 * (size_t)i;
    int masked = is_masked(p, mask_mode);
    V[i] = masked ? 0 : 1;

    float y = rgb_to_luma(p[0], p[1], p[2]);
    /* For masked pixels Y is irrelevant, but keep plausible value. */
    Y[i] = clampf1(y);
  }

  /* Gaussian kernel */
  float *k = NULL;
  int klen = 0;
  int mk = make_gauss_kernel(radius, &k, &klen);
  if (mk != 0) {
    allsky_safe_free(Y);
    allsky_safe_free(Ytmp);
    allsky_safe_free(Yblur);
    allsky_safe_free(V);
    fprintf(stderr, "clarity_filter_rgbf_masked: error building gaussian kernel\n");
    return 3;
  }

  /* Masked separable blur: horizontal then vertical. */
  gauss_blur_h_masked(Y, V, Ytmp, width, height, k, radius);
  gauss_blur_v_masked(Ytmp, V, Yblur, width, height, k, radius);
  allsky_safe_free(k);

  const float pre_gain = 1.0f;

/* Apply clarity on VALID pixels only; masked pixels remain unchanged. */
#pragma omp parallel for
  for (int i = 0; i < (int)N; ++i) {
    if (!V[i])
      continue; /* skip masked */

    float y = Y[i];
    float yb = Yblur[i];
    float detail = soft_limiter((y - yb) * pre_gain);

    float w = midtone_bell(y, midtone_width);
    if (preserve_highlights)
      w *= highlight_rolloff(y);

    float y_new = clampf1(y + strength * w * detail);
    Y[i] = y_new;
  }

/* Remap RGB by luma ratio (only for VALID pixels). Masked: pass-through. */
#pragma omp parallel for
  for (int i = 0; i < (int)N; ++i) {
    float *p = rgba + 4 * (size_t)i;
    if (!V[i])
      continue; /* keep original masked pixel (including alpha) */

    float y_old = rgb_to_luma(p[0], p[1], p[2]);
    float scale = safe_ratio(Y[i], y_old);

    if (y_old < 1e-4f) {
      float t = y_old / 1e-4f; /* fade-in to avoid noise pop */
      scale = t * scale + (1.0f - t);
    }

    p[0] = clampf1(p[0] * scale);
    p[1] = clampf1(p[1] * scale);
    p[2] = clampf1(p[2] * scale);
    /* alpha unchanged */
  }

  allsky_safe_free(Y);
  allsky_safe_free(Ytmp);
  allsky_safe_free(Yblur);
  allsky_safe_free(V);

  printf("Clarity Filter: ok\n");

  return 0;
}

#endif

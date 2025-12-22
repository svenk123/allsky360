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
    float lp_strength     // 0..1: how softly we approach the hard limit, if 0.0 then light protection is disabled
    ) {
  if (!rgba || width <= 0 || height <= 0)
    return 1;
  if (scale_r <= 0.f || scale_g <= 0.f || scale_b <= 0.f)
    return 2;
    
  if (lp_strength < 0.f)
    lp_strength = 0.f;
  if (lp_strength > 1.f)
    lp_strength = 1.f;


  printf("White Balance scaling: red: %.6f, green: %.6f, blue: %.6f, lp_strength: %.6f\n", scale_r,scale_g,scale_b, lp_strength);
  const int N = width * height;

  // 1) find per-channel maxima (pre-scale)
  float r_max = 0.f, g_max = 0.f, b_max = 0.f;
  for (int i = 0; i < N; ++i) {
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
  float atten = (post_max > ATTEN_TRIG) ? (1.f / post_max) : 1.f;

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
    // alpha unchanged
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

SceneType detect_scene_rgbf1(float *rgba, int width, int height)
{
    if (!rgba || width <= 0 || height <= 0) {
        fprintf(stderr, "SceneDetect: invalid image or dimensions\n");
        return SCENE_NIGHT_CLEAR;
    }

    const int total = width * height;

    double sumY  = 0.0;
    double sumY2 = 0.0;

    long bright_cnt      = 0;  // very bright pixels (moon/sun)
    long blue_sky_cnt    = 0;  // strong blue sky candidates
    long day_cloud_cnt   = 0;  // bright, low-saturation clouds (daytime)
    long night_aurora_cnt= 0;  // high-sat, mid/dark luminance (aurora)
    long night_cloud_cnt = 0;  // mid luminance, low-sat (night clouds)

#pragma omp parallel for reduction(+:sumY, sumY2, bright_cnt, blue_sky_cnt, day_cloud_cnt, night_aurora_cnt, night_cloud_cnt)
    for (int i = 0; i < total; ++i) {
        int idx = i * 4;
        float r = rgba[idx + 0];
        float g = rgba[idx + 1];
        float b = rgba[idx + 2];

        /* Clamp sanity */
        if (r < 0.0f || r > 1.0f ||
            g < 0.0f || g > 1.0f ||
            b < 0.0f || b > 1.0f) {
            continue;
        }

        float Y   = rgb_to_luma(r, g, b);
        float sat = saturation_rgb(r, g, b);

        double dY = (double)Y;
        sumY  += dY;
        sumY2 += dY * dY;

        /* Very bright → moon/sun candidate */
        if (Y > 0.85f) {
            bright_cnt++;
        }

        /* Blue sky candidate (daytime): moderately dark but highly saturated */
        if (Y > 0.10f && Y < 0.40f && sat > 0.50f) {
            blue_sky_cnt++;
        }

        /* Daytime clouds: brighter and desaturated */
        if (Y > 0.45f && sat < 0.35f) {
            day_cloud_cnt++;
        }

        /* Aurora candidates (night): high saturation, not too bright */
        if (Y < 0.40f && sat > 0.55f) {
            night_aurora_cnt++;
        }

        /* Night clouds: mid-luminance, low saturation */
        if (Y > 0.15f && Y < 0.65f && sat < 0.25f) {
            night_cloud_cnt++;
        }
    }

    /* Global statistics */
    double meanY = sumY / (double)total;
    double varY  = (sumY2 / (double)total) - meanY * meanY;
    if (varY < 0.0)
        varY = 0.0;
    double stdY = sqrt(varY);

    double bright_pct       = (double)bright_cnt       / (double)total;
    double blue_sky_pct     = (double)blue_sky_cnt     / (double)total;
    double day_cloud_pct_f  = (double)day_cloud_cnt    / (double)total;
    double night_aurora_pct = (double)night_aurora_cnt / (double)total;
    double night_cloud_pct_f= (double)night_cloud_cnt  / (double)total;

    printf("SceneDetect(CPU): meanY=%.4f stdY=%.4f bright=%.5f blue_sky=%.5f day_cloud=%.5f night_aurora=%.5f night_cloud=%.5f\n",
           meanY, stdY, bright_pct, blue_sky_pct, day_cloud_pct_f, night_aurora_pct, night_cloud_pct_f);

    /* Basic day/night/twilight classification */
    int isDay      = 0;
    int isNight    = 0;
    int isTwilight = 0;

    if (meanY > 0.18) {
        isDay = 1;
    } else if (meanY < 0.05) {
        isNight = 1;
    } else {
        isTwilight = 1;
    }

    /* --- Daytime classification --- */
    if (isDay) {
        /* If large fraction is bright and desaturated → cloudy */
        if (day_cloud_pct_f > 0.25) {
            return SCENE_DAY_CLOUDY;
        }

        /* If large fraction is blue, saturated sky → clear */
        if (blue_sky_pct > 0.30) {
            return SCENE_DAY_CLEAR;
        }

        /* Fallback: if not clearly cloudy, treat as clear */
        return SCENE_DAY_CLEAR;
    }

    /* --- Night classification --- */
    if (isNight) {
        /* Strong bright fraction → moon / strong light source */
        if (bright_pct > 0.0005) { // 0.05% of pixels
            return SCENE_NIGHT_MOON;
        }

        /* Significant aurora-like content */
        if (night_aurora_pct > 0.01) { // >1% of pixels
            return SCENE_NIGHT_AURORA;
        }

        /* Night clouds: many mid-luminance, low-sat pixels */
        if (night_cloud_pct_f > 0.20) {
            return SCENE_NIGHT_CLOUDY;
        }

        return SCENE_NIGHT_CLEAR;
    }

    /* --- Twilight --- */
    /* Twilight is naturally ambiguous; we do not distinguish clear/cloudy here. */
    return SCENE_TWILIGHT;
}

#if 0
SceneType detect_scene_rgbf1(float *rgba, int width, int height)
{
    if (!rgba || width <= 0 || height <= 0) {
        fprintf(stderr, "Scene Detection: invalid image\n");
        return SCENE_NIGHT_CLEAR;
    }

    const int total = width * height;
    long bright_pixels = 0;
    long aurora_pixels = 0;
    long cloudy_pixels = 0;

    double sumY = 0.0;
    double sumY2 = 0.0;

    /* Using luminance histogram-like analysis */
#pragma omp parallel for reduction(+:bright_pixels, aurora_pixels, cloudy_pixels, sumY, sumY2)
    for (int i = 0; i < total; i++) {
        int idx = i * 4;
        float r = rgba[idx + 0];
        float g = rgba[idx + 1];
        float b = rgba[idx + 2];

        float Y   = rgb_to_luma(r, g, b);
        float sat = saturation_rgb(r, g, b);

        sumY  += Y;
        sumY2 += (double)Y * (double)Y;

        /* 1) very bright pixels → possible moon or sun */
        if (Y > 0.85f)
            bright_pixels++;

        /* 2) highly saturated = typical for aurora */
        if (sat > 0.40f && Y < 0.85f)
            aurora_pixels++;

        /* 3) cloudiness uses midrange luminance */
        if (Y > 0.15f && Y < 0.65f)
            cloudy_pixels++;
    }

    double meanY = sumY / total;
    double varY  = (sumY2 / total) - meanY * meanY;
    if (varY < 0) varY = 0;
    double stdY  = sqrt(varY);

    double bright_pct = (double)bright_pixels / (double)total;
    double aurora_pct = (double)aurora_pixels / (double)total;
    double cloudy_pct = (double)cloudy_pixels / (double)total;

    printf("SceneDetect: meanY=%.4f stdY=%.4f bright=%.4f aurora=%.4f cloudy=%.4f\n",
           meanY, stdY, bright_pct, aurora_pct, cloudy_pct);

    /* ------------ DAY / NIGHT separation by mean Y ------------ */

    if (meanY > 0.25) {
        /* DAYTIME */
        if (cloudy_pct > 0.25)
            return SCENE_DAY_CLOUDY;
        else
            return SCENE_DAY_CLEAR;
    }

    if (meanY < 0.02) {
        /* NIGHT */
        if (bright_pct > 0.0002)   // ~0.02% bright pixels = moon
            return SCENE_NIGHT_MOON;

        if (aurora_pct > 0.01)     // >1% strong saturated pixels
            return SCENE_NIGHT_AURORA;

        if (cloudy_pct > 0.20)
            return SCENE_NIGHT_CLOUDY;

        return SCENE_NIGHT_CLEAR;
    }

    /* ------------ TWILIGHT (between night and day) ------------ */

    /* strong horizontal gradient? approximate by stdY */
    if (stdY > 0.08 || fabs(meanY - 0.10) < 0.05)
        return SCENE_TWILIGHT;

    return SCENE_TWILIGHT;
}

#endif

int ambience_color_calibration_rgbf1(float *rgba,
                                     int width,
                                     int height,
                                     float subframe_percent,
                                     float luma_clip_high,
                                     float sat_clip_high,
                                     float sigma_factor,
                                     float mix_factor)
{
  printf("Ambience-based color calibration: subframe %.1f%%\n", subframe_percent);

  if (!rgba || width <= 0 || height <= 0) {
    fprintf(stderr, "Ambience Color Calibration: invalid image pointer or dimensions\n");

    return 1;
  }

  if (subframe_percent <= 0.0f || subframe_percent > 100.0f) {
    fprintf(stderr, "Ambience Color Calibration: invalid subframe_percent %.2f\n", subframe_percent);

    return 2;
  }

  if (mix_factor <= 0.0f) {
    /* Nothing to do, but this is not an error */
    printf(" mix_factor <= 0.0, skipping calibration\n");

    return 0;
  }

  /* Compute central subframe region */
  int x0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * (float)width);
  int y0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * (float)height);
  int x1 = width  - x0;
  int y1 = height - y0;

  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > width)
    x1 = width;
  if (y1 > height)
    y1 = height;

  printf(" subframe %.1f%% -> region x=[%d,%d), y=[%d,%d)\n",
         subframe_percent, x0, x1, y0, y1);

  /* ---------- First pass: basic filtering and Y statistics ---------- */

  double sumR = 0.0, sumG = 0.0, sumB = 0.0;
  double sumY = 0.0, sumY2 = 0.0;
  long   count = 0;

  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      int idx = (y * width + x) * 4;
      float r = rgba[idx + 0];
      float g = rgba[idx + 1];
      float b = rgba[idx + 2];

      /* Basic sanity clamp (just in case) */
      if (r < 0.0f || r > 1.0f ||
          g < 0.0f || g > 1.0f ||
          b < 0.0f || b > 1.0f) {
        continue;
      }

      /* Luminance (Rec.709 style) */
      float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;

      /* Hard reject very bright pixels (moon, sun, bright clouds, lamps) */
      if (Y > luma_clip_high)
        continue;

      /* Approximate saturation: (max - min) / max */
      float maxc = fmaxf(fmaxf(r, g), b);
      float minc = fminf(fminf(r, g), b);
      float sat = 0.0f;
      if (maxc > 0.0f)
        sat = (maxc - minc) / maxc;

      /* Reject highly saturated pixels (aurora, city lights, strong colored objects) */
      if (sat > sat_clip_high)
        continue;

      sumR += r;
      sumG += g;
      sumB += b;
      sumY += Y;
      sumY2 += (double)Y * (double)Y;
      count++;
    }
  }

  if (count < 100) {
    /* Too few samples, bail out */
    fprintf(stderr, " not enough valid samples in subframe (count=%ld)\n", count);
    
    return 3;
  }

  double meanR = sumR / (double)count;
  double meanG = sumG / (double)count;
  double meanB = sumB / (double)count;
  double meanY = sumY / (double)count;
  double varY  = (sumY2 / (double)count) - meanY * meanY;
  if (varY < 0.0)
    varY = 0.0;
  double stdY  = sqrt(varY);

  printf(" first-pass samples=%ld  meanRGB=(%.4f, %.4f, %.4f)  meanY=%.4f  stdY=%.4f\n",
         count, meanR, meanG, meanB, meanY, stdY);

  /* ---------- Second pass: restrict to ±sigma_factor * stdY band ---------- */

  double sumR2band = 0.0, sumG2band = 0.0, sumB2band = 0.0;
  long   countBand = 0;

  double bandLow  = meanY - sigma_factor * stdY;
  double bandHigh = meanY + sigma_factor * stdY;

  if (stdY <= 0.0) {
    /* No contrast in Y, fall back to first-pass means */
    bandLow  = -1e9;
    bandHigh =  1e9;
  }

  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      int idx = (y * width + x) * 4;
      float r = rgba[idx + 0];
      float g = rgba[idx + 1];
      float b = rgba[idx + 2];

      if (r < 0.0f || r > 1.0f ||
          g < 0.0f || g > 1.0f ||
          b < 0.0f || b > 1.0f) {
        continue;
      }

      float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;

      if (Y > luma_clip_high)
        continue;

      float maxc = fmaxf(fmaxf(r, g), b);
      float minc = fminf(fminf(r, g), b);
      float sat = 0.0f;
      if (maxc > 0.0f)
        sat = (maxc - minc) / maxc;

      if (sat > sat_clip_high)
        continue;

      /* Only accept pixels within the Y-band */
      if (Y < bandLow || Y > bandHigh)
        continue;

      sumR2band += r;
      sumG2band += g;
      sumB2band += b;
      countBand++;
    }
  }

  if (countBand < 50) {
    /* If band-filtering removed too much, fallback to first-pass means */
    printf(" band filtering left few samples (countBand=%ld), falling back to first-pass means\n",
           countBand);
    countBand = count;
    sumR2band = sumR;
    sumG2band = sumG;
    sumB2band = sumB;
  }

  double ambienceR = sumR2band / (double)countBand;
  double ambienceG = sumG2band / (double)countBand;
  double ambienceB = sumB2band / (double)countBand;

  /* Compute ambience luminance (this defines the "light color" intensity) */
  double ambienceY = 0.2126 * ambienceR + 0.7152 * ambienceG + 0.0722 * ambienceB;

  printf(" ambienceRGB=(%.4f, %.4f, %.4f)  ambienceY=%.4f  bandSamples=%ld\n",
         ambienceR, ambienceG, ambienceB, ambienceY, countBand);

  if (ambienceR <= 0.0 || ambienceG <= 0.0 || ambienceB <= 0.0 || ambienceY <= 0.0) {
    fprintf(stderr, " invalid ambience values, skipping\n");
    return 4;
  }

  /* Compute full WB gain (what Gray-World would use) */
  double fullScaleR = ambienceY / ambienceR;
  double fullScaleG = ambienceY / ambienceG;
  double fullScaleB = ambienceY / ambienceB;

  /* Clamp extreme scales to avoid crazy corrections */
  const double SCALE_MIN = 0.25;
  const double SCALE_MAX = 4.0;
  if (fullScaleR < SCALE_MIN) fullScaleR = SCALE_MIN;
  if (fullScaleR > SCALE_MAX) fullScaleR = SCALE_MAX;
  if (fullScaleG < SCALE_MIN) fullScaleG = SCALE_MIN;
  if (fullScaleG > SCALE_MAX) fullScaleG = SCALE_MAX;
  if (fullScaleB < SCALE_MIN) fullScaleB = SCALE_MIN;
  if (fullScaleB > SCALE_MAX) fullScaleB = SCALE_MAX;

  /* Blend towards 1.0 (no correction) using mix_factor */
  float scaleR = (float)(1.0 + (fullScaleR - 1.0) * (double)mix_factor);
  float scaleG = (float)(1.0 + (fullScaleG - 1.0) * (double)mix_factor);
  float scaleB = (float)(1.0 + (fullScaleB - 1.0) * (double)mix_factor);

  printf(" fullScale=(%.4f, %.4f, %.4f)  mix_factor=%.3f  finalScale=(%.4f, %.4f, %.4f)\n",
         fullScaleR, fullScaleG, fullScaleB, mix_factor, scaleR, scaleG, scaleB);

  /* If final scales are almost 1.0, we can short-circuit */
  if (fabsf(scaleR - 1.0f) < 1e-3f &&
      fabsf(scaleG - 1.0f) < 1e-3f &&
      fabsf(scaleB - 1.0f) < 1e-3f) {
    printf(" scales ~1.0, no visible correction needed\n");
    return 0;
  }

  /* ---------- Apply scales to the whole image (OpenMP accelerated) ---------- */

  const int totalPixels = width * height;

#pragma omp parallel for
  for (int i = 0; i < totalPixels; ++i) {
    int idx = i * 4;
    float r = rgba[idx + 0];
    float g = rgba[idx + 1];
    float b = rgba[idx + 2];

    r *= scaleR;
    g *= scaleG;
    b *= scaleB;

    rgba[idx + 0] = clampf1(r);
    rgba[idx + 1] = clampf1(g);
    rgba[idx + 2] = clampf1(b);
    /* alpha channel is left untouched */
  }

  printf("Ambience-based color calibration: ok\n");
  return 0;
}

#endif

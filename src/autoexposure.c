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
#include "autoexposure.h"
#include "allsky.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float calc_exposure_time(float t0, float ev_steps, int index) {
  if (index <= 0) {
    // index 0 = longest exposure → no reduction
    return t0;
  }

  /* ev_steps is a double in config.ini (*_hdr_exposure_factor). It used to be
   * truncated to int here, so 1.5 silently became 1. */
  float exponent = (float)index * ev_steps;
  float factor = powf(2.0f, exponent);

  return t0 / factor;
}

int adjust_exposure_gain(double brightness, double target_brightness,
                         double max_exposure, double min_exposure,
                         double max_gain, double min_gain, double *exposure,
                         double *gain) {
  if (brightness == 0.0 || target_brightness == 0.0) {
    printf("No brightness or target brightness\n");
    return 1;
  }

  double prev_exposure = *exposure;
  double prev_gain = *gain;

  double error = target_brightness - brightness;
  double abs_error = fabs(error);

  printf("Brightness (t0): %.3f ====> target: %.3f\n", brightness,
         target_brightness);

  /* No andjustment, when brightness is within 2% tolerance */
  double hysteresis = 0.002;
  if (abs_error < hysteresis) {
    printf("Brightness within tolerances: no adjustment necessary.\n");
    return 1;
  }

  /* Calculate adjustment factor */
  double factor = 1.0 + error * 3.0; // Less aggressive

  /* Limitation */
  if (factor > 2.0)
    factor = 2.0; // Not more than 2x
  if (factor < 0.5)
    factor = 0.5; // Not less than 1/2x

  printf("Exposure control: error=%.3f, factor=%.3f\n", error, factor);

  /*  Image is bright enough */
  if (error == 0) {
    printf("Perfect! No adjustment necessary.\n");
    return 0;
  }

  double new_exposure = *exposure;
  double new_gain = *gain;
  double gain_step = max_gain * 0.1; // 10% step size

  /* Image is too bright */
  if (error < 0) {
    /* Exposure limit reached */
    if (prev_exposure == max_exposure) {
      if (prev_gain == min_gain) {
        // Decrease exposure
        new_exposure = *exposure * factor;
      } else {
        // Decrease gain 10%
        new_gain -= gain_step;
      }
    } else {
      // Decrease exposure
      new_exposure = *exposure * factor;
    }
  }

  /* Image is too dark */
  if (error > 0) {
    /* Exposure limit reached */
    if (prev_exposure == max_exposure) {
      // Increase gain 10%
      new_gain += gain_step;
    } else {
      // Increase exposure
      new_exposure = *exposure * factor;
    }
  }

  /* Min/Max Exposure limits */
  if (new_exposure > max_exposure) {
    new_exposure = max_exposure;
  } else if (new_exposure < min_exposure) {
    new_exposure = min_exposure;
  }

  /* Min/Max gain limits */
  if (new_gain > max_gain) {
    new_gain = max_gain;
  } else if (new_gain < min_gain) {
    new_gain = min_gain;
  }

  /* Exposure limits */
  if (new_exposure > max_exposure) {
    new_exposure = max_exposure;
  } else if (new_exposure < min_exposure) {
    new_exposure = min_exposure;
  }

  *exposure = new_exposure;
  *gain = new_gain;

  printf("Adjustment: before: exposure=%.6f s, gain=%.1f => after: "
         "exposure=%.6f s, gain=%.1f, brightness=%.2f, target=%.2f\n",
         prev_exposure, prev_gain, new_exposure, new_gain, brightness,
         target_brightness);

  return 0;
}

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

int update_exposure_controller(ExposureController *ctrl, float current_median,
                               float target_median) {
  if (!ctrl || current_median <= 0.0f || current_median > 1.0f)
    return 1;

  // --- Update history buffer ---
  ctrl->median_history[ctrl->history_index] = current_median;
  ctrl->history_index = (ctrl->history_index + 1) % HISTORY_SIZE;
  if (ctrl->history_count < HISTORY_SIZE)
    ctrl->history_count++;

  // --- Compute smoothed median ---
  float sum = 0.0f;
  printf(" median_history: ");
  for (int i = 0; i < ctrl->history_count; ++i) {
    sum += ctrl->median_history[i];
    printf("%.2f ", ctrl->median_history[i]);
  }
  printf("\n");
  float median_avg = sum / ctrl->history_count;

  float target = 0.4f;
  if (target_median > 0.0f) {
    // Manual target brightness
    target = target_median;
  } else {
    // --- Dynamic target brightness: brighter at night ---
    // Example: day = 0.3, night = 0.8
    target =
        0.3f + 0.3f * expf(-4.0f * median_avg); // adjust exponent as needed
  }
  printf(" current_median: %.2f, target_median: %.2f (%s)\n", current_median,
         target, target_median > 0.0f ? "fixed" : "dynamic");

  // --- Scale factor ---
  float scale = target / current_median;
  float damped_scale = 1.0f + ctrl->response * (scale - 1.0f);

  // --- Adjust shutter and gain ---
  if (scale > 1.0f) {
    // too dark → increase light
    if (ctrl->shutter < ctrl->shutter_max) {
      ctrl->shutter *= damped_scale;
      if (ctrl->shutter > ctrl->shutter_max)
        ctrl->shutter = ctrl->shutter_max;
    } else {
      ctrl->gain *= damped_scale;
      if (ctrl->gain > ctrl->gain_max)
        ctrl->gain = ctrl->gain_max;
    }
  } else {
    // too bright → reduce light
    if (ctrl->gain > ctrl->gain_min) {
      ctrl->gain *= damped_scale;
      if (ctrl->gain < ctrl->gain_min)
        ctrl->gain = ctrl->gain_min;
    } else {
      ctrl->shutter *= damped_scale;
      if (ctrl->shutter < ctrl->shutter_min)
        ctrl->shutter = ctrl->shutter_min;
    }
  }

  /* Light control with hysteresis */
  float light_on_threshold = 0.25f;
  float light_off_threshold = light_on_threshold + ctrl->hysteresis_threshold;

  if (!ctrl->lights_on && current_median < light_on_threshold)
    ctrl->lights_on = 1;
  else if (ctrl->lights_on && current_median > light_off_threshold)
    ctrl->lights_on = 0;

  return 0;
}

#define MEDIAN_BINS 1024 /* Number of histogram bins in the range 0.0..1.0f */

static double compute_channel_median(const unsigned int *histogram,
                                     int total_pixels) {
  if (total_pixels == 0)
    return 0.0;

  int lower_clip = (int)(total_pixels * 0.02);
  int upper_clip = (int)(total_pixels * 0.98);
  int clip_mid = (lower_clip + upper_clip) / 2;

  int count = 0;
  int median_bin = 0;

  for (int i = 0; i < MEDIAN_BINS; i++) {
    count += histogram[i];
    if (count >= lower_clip) {
      if (count >= clip_mid) {
        median_bin = i;
        break;
      }
    }
  }

  return (double)median_bin / (MEDIAN_BINS - 1);
}

int compute_filtered_median_brightness_rgbf1(const float *rgba, int width,
                                             int height, double *red_brightness,
                                             double *green_brightness,
                                             double *blue_brightness,
                                             double *luminance_brightness,
                                             int cx, int cy, int radius) {
  if (!rgba || width <= 0 || height <= 0 || !red_brightness ||
      !green_brightness || !blue_brightness || !luminance_brightness ||
      cx < 0 || cy < 0 || radius <= 0)
    return 1;

  printf("Compute filtered median brightness: cx: %d, cy: %d, radius: %d\n", cx,
         cy, radius);

  int radius2 = radius * radius;

  /* Calculate the boundaries of the enclosing rectangle for optimization */
  int x_start = (cx - radius > 0) ? cx - radius : 0;
  int x_end = (cx + radius < width) ? cx + radius + 1 : width;
  int y_start = (cy - radius > 0) ? cy - radius : 0;
  int y_end = (cy + radius < height) ? cy + radius + 1 : height;

  unsigned int histogram_r[MEDIAN_BINS] = {0};
  unsigned int histogram_g[MEDIAN_BINS] = {0};
  unsigned int histogram_b[MEDIAN_BINS] = {0};
  unsigned int histogram_l[MEDIAN_BINS] = {0};
  int total_pixels = 0;

  for (int y = y_start; y < y_end; y++) {
    for (int x = x_start; x < x_end; x++) {
      /* Calculate the distance from the pixel to the center */
      int dx = x - cx;
      int dy = y - cy;
      int dist2 = dx * dx + dy * dy;

      /* Only process pixels within the circle */
      if (dist2 > radius2)
        continue;

      int offset = (y * width + x) * CHANNELS;
      float r = rgba[offset + 0];
      float g = rgba[offset + 1];
      float b = rgba[offset + 2];
      float l = rgba[offset + 3];

      /* Ignore pixels with zero RGB or zero luminance values */
      if (r == 0.0f && g == 0.0f && b == 0.0f && l == 0.0f)
        continue;

      int r_bin = (int)(clampf1(r) * (MEDIAN_BINS - 1) + 0.5f);
      int g_bin = (int)(clampf1(g) * (MEDIAN_BINS - 1) + 0.5f);
      int b_bin = (int)(clampf1(b) * (MEDIAN_BINS - 1) + 0.5f);
      int l_bin = (int)(clampf1(l) * (MEDIAN_BINS - 1) + 0.5f);

      histogram_r[r_bin]++;
      histogram_g[g_bin]++;
      histogram_b[b_bin]++;
      histogram_l[l_bin]++;
      total_pixels++;
    }
  }

  if (total_pixels == 0) {
    *red_brightness = 0.0;
    *green_brightness = 0.0;
    *blue_brightness = 0.0;
    *luminance_brightness = 0.0;
    return 0;
  }

  *red_brightness = compute_channel_median(histogram_r, total_pixels);
  *green_brightness = compute_channel_median(histogram_g, total_pixels);
  *blue_brightness = compute_channel_median(histogram_b, total_pixels);
  *luminance_brightness = compute_channel_median(histogram_l, total_pixels);

  printf("Filtered median brightness calculation (circular region, radius=%d): "
         "R=%.3f G=%.3f B=%.3f L=%.3f\n",
         radius, *red_brightness, *green_brightness, *blue_brightness,
         *luminance_brightness);

  return 0;
}

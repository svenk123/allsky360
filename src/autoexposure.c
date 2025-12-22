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

int adjust_exposure_gain(double brightness, double target_brightness,
                         double max_exposure, double min_exposure,
                         double max_gain, double min_gain, double *exposure,
                         double *gain) {
  if (brightness == 0.0 || target_brightness == 0.0)
    return 1;

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

    // float ev = log2f(ctrl->shutter * ctrl->gain);  // pseudo Belichtungswert
    //  target = 0.04f + 0.36f * sigmoid(ev);

    // day = 0.4, night = 0.04
    //	target = 0.04f + 0.36f * (1.0f - expf(-5.0f * median_avg));
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

int compute_filtered_median_brightness_rgbf1(
    const float *rgba, int width, int height, double center_pct,
    double *red_brightness, double *green_brightness, double *blue_brightness) {
  if (!rgba || width <= 0 || height <= 0 || !red_brightness ||
      !green_brightness || !blue_brightness)
    return 1;

  if (center_pct < 0.1)
    center_pct = 0.1;
  if (center_pct > 1.0)
    center_pct = 1.0;

  double margin = (1.0 - center_pct) / 2.0;
  int x_start = (int)(width * margin);
  int x_end = (int)(width * (1.0 - margin));
  int y_start = (int)(height * margin);
  int y_end = (int)(height * (1.0 - margin));

  unsigned int histogram_r[MEDIAN_BINS] = {0};
  unsigned int histogram_g[MEDIAN_BINS] = {0};
  unsigned int histogram_b[MEDIAN_BINS] = {0};
  int total_pixels = 0;

  for (int y = y_start; y < y_end; y++) {
    for (int x = x_start; x < x_end; x++) {
      int offset = (y * width + x) * CHANNELS;
      float r = rgba[offset + 0];
      float g = rgba[offset + 1];
      float b = rgba[offset + 2];

      // Pixel mit RGB=0 ignorieren
      if (r == 0.0f && g == 0.0f && b == 0.0f)
        continue;

      int r_bin = (int)(clampf1(r) * (MEDIAN_BINS - 1) + 0.5f);
      int g_bin = (int)(clampf1(g) * (MEDIAN_BINS - 1) + 0.5f);
      int b_bin = (int)(clampf1(b) * (MEDIAN_BINS - 1) + 0.5f);

      histogram_r[r_bin]++;
      histogram_g[g_bin]++;
      histogram_b[b_bin]++;
      total_pixels++;
    }
  }

  if (total_pixels == 0) {
    *red_brightness = 0.0;
    *green_brightness = 0.0;
    *blue_brightness = 0.0;
    return 0;
  }

  *red_brightness = compute_channel_median(histogram_r, total_pixels);
  *green_brightness = compute_channel_median(histogram_g, total_pixels);
  *blue_brightness = compute_channel_median(histogram_b, total_pixels);

  printf("Filtered median brightness calculation (RGB, %.0f%% center region): "
         "R=%.3f G=%.3f B=%.3f\n",
         center_pct * 100, *red_brightness, *green_brightness,
         *blue_brightness);

  return 0;
}

int compute_filtered_median_brightness_green_rgbf1(const float *rgba, int width,
                                                   int height,
                                                   double center_pct,
                                                   double *brightness) {
  if (!brightness)
    return 1;

  double r = 0.0;
  double g = 0.0;
  double b = 0.0;

  int status = compute_filtered_median_brightness_rgbf1(
      rgba, width, height, center_pct, &r, &g, &b);
  if (status != 0)
    return status;

  *brightness = g;

  return 0;
}

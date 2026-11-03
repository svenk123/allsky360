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
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <json-c/json.h>

#include "atomic_file.h"
#include "config.h"
#include "coordinates.h"

/* Unknown keys are reported only on the first load (see load_config()). */
static int warn_unknown_keys = 1;

/* Maximum length of one line and of one value in config.ini */
#define CONFIG_LINE_MAX 1024
#define CONFIG_VALUE_MAX 512

/**
 * Copy a string value into a fixed-size config field.
 * Reports values that do not fit instead of silently truncating them
 * (a cut-off directory path otherwise only shows up as a confusing
 * "directory does not exist" error later on).
 */
static void config_copy_string(char *dst, size_t dst_size, const char *key,
                               const char *value) {
  size_t len = strlen(value);

  if (len >= dst_size && warn_unknown_keys)
    fprintf(stderr,
            "WARNING: config.ini: value of '%s' is %zu characters long, "
            "only %zu are supported - it is truncated\n",
            key, len, dst_size - 1);

  snprintf(dst, dst_size, "%.*s", (int)dst_size - 1, value);
}

int load_config(const char *filename, config_t *config) {
  if (!filename || !config)
    return 1;

  /* config_t is an uninitialised local in the caller. Zero it first so that a
   * field whose key is missing from config.ini can never hold stack garbage,
   * even if someone adds a struct member without a default below. */
  memset(config, 0, sizeof(*config));

  FILE *file = fopen(filename, "r");
  if (!file) {
    return 1;
  }

  // Set default values
  config->daytime_capture = 1;
  config->daytime_meanbrightness = 0.20;
  config->nighttime_meanbrightness = 0.15;
  config->nighttime_fullmoongain = 0;
  /* Radius of the auto-exposure brightness measurement, as a fraction
   * (0.0 - 1.0) of image_horizon_radius. */
  config->measure_brightness_area = 0.55;
  config->crop_image = 0;
  config->crop_width = 0;
  config->crop_height = 0;
  config->crop_x_offset = 0;
  config->crop_y_offset = 0;
  config->rotate_image = 0;
  snprintf(config->indigoservername, sizeof(config->indigoservername),
           "indigo_server");
  config->daytime_startexposure = 0.05;
  config->daytime_startgain = 0.0;
  config->daytime_move_blackpoint = 0.05;
  config->daytime_move_blackpoint_min_shift_pct = 0.005;
  config->daytime_move_blackpoint_max_shift_pct = 0.02;
  config->daytime_move_blackpoint_dark_threshold = 0.2;
  config->daytime_hdr_exposure_factor = 12.0;
  config->daytime_white_balance_red = 1.0;
  config->daytime_white_balance_blue = 1.0;
  config->daytime_white_balance_light_protect = 0.0;
  config->nighttime_sunangle = -6.0;
  config->nighttime_startexposure = 1.0;
  config->nighttime_startgain = 100.0;
  config->nighttime_move_blackpoint = 0.05;
  config->nighttime_move_blackpoint_min_shift_pct = 0.005;
  config->nighttime_move_blackpoint_max_shift_pct = 0.02;
  config->nighttime_move_blackpoint_dark_threshold = 0.2;
  config->nighttime_hdr_exposure_factor = 6.0;
  config->nighttime_white_balance_red = 1.0;
  config->nighttime_white_balance_blue = 1.0;
  config->nighttime_white_balance_light_protect = 0.0;
  config->nighttime_measure_noise = 1;
  config->nighttime_measure_noise_radius = 250;
  config->nighttime_multiscale_median_filter = 0;
  config->nighttime_multiscale_median_filter_max_scale = 2;
  config->nighttime_multiscale_median_filter_amount = 0.5f;

  config->target_mean = 0.5;
  config->black_clipping_point = 0.02;
  config->white_clipping_point = 0.98;

  config->hdr = 1;
  config->hdr_clipping_threshold = 0.98f;
  config->hdr_min_clipped_pixels = 0.00001f;
  config->hdr_y_max_expected = 1.0;
  config->hdr_weight_sigma = 0.25;
  config->hdr_weight_clip_factor = 0.90;
  config->hdr_contrast_weight_strength = 0.5;
  config->hdr_noise_fusion_strength = 1.0;
  config->hdr_pyramid_levels_override = 6;
  config->hdr_weight_maps = 0;
  config->hdr_weight_stats = 0;
  config->hdr_save_exposures = 1;
  config->hdrmt = 0;

  config->spektrum_enabled = 0;
  config->spektrum_cx = -1;
  config->spektrum_cy = -1;
  config->spektrum_radius = 50;
  config->spektrum_regularization = 0.01;
  snprintf(config->spektrum_output_path, sizeof(config->spektrum_output_path),
           "images/spektrum.json");

  config->autostretch = 1;
  config->autostretch_min_val = 0.01;
  config->autostretch_max_val = 0.99;

  config->acdnr_filter = 1;
  config->acdnr_filter_lum_stddev = 0.002f;
  config->acdnr_filter_lum_amount = 0.015f;
  config->acdnr_filter_lum_iterations = 3;
  config->acdnr_filter_lum_kernel_size = 5;
  config->acdnr_filter_chrom_stddev = 0.002f;
  config->acdnr_filter_chrom_amount = 0.015f;
  config->acdnr_filter_chrom_iterations = 3;
  config->acdnr_filter_chrom_kernel_size = 5;

  config->gamma = 0.8;
  config->auto_gamma = 0;
  config->auto_gamma_target = 0.18;
  config->auto_gamma_min = 0.15;
  config->auto_gamma_max = 1.2;
  config->reinhard_tonemap = 0;
  config->reinhard_white_point = 0.0;

  config->tone_curve = 0;
  config->tone_curve_shadow_lift = 0.2;
  config->tone_curve_midtone_contrast = 0.3;

  config->saturation = 2.8;

  config->dehaze_amount = 0.8;
  config->dehaze_estimate = 0.01;
  config->dehaze_gamma = 1.0;
  config->dehaze_saturation = 1.2;

  config->clarity_filter = 1;
  config->clarity_filter_strength = 0.5;
  config->clarity_filter_radius = 5;
  config->clarity_filter_midtone_width = 0.35;
  config->clarity_filter_preserve_highlights = 1;
  config->clarity_filter_mask_mode = 0;

  config->auto_color_calibration_area = 80.0;

  config->local_histogram_normalization = 1;
  config->local_histogram_normalization_kernel_radius = 64;
  config->local_histogram_normalization_contrast_limit = 1.5;

  config->scnr_filter = 1;
  config->scnr_filter_amount = 1.0f;
  snprintf(config->scnr_filter_protection,
           sizeof(config->scnr_filter_protection), "average_neutral");

  config->wavelet_sharpen = 0;
  config->wavelet_sharpen_gain_small = 0.5;
  config->wavelet_sharpen_gain_medium = 0.0;
  config->wavelet_sharpen_gain_large = 0.0;

  config->sqm = 0;
  config->sqm_max_stars = 1000;
  config->sqm_radius = 0;
  config->sqm_center_x = 0;
  config->sqm_center_y = 0;
  config->sqm_star_sigma = 1.0;
  config->sqm_detect_sigma = 5.0;
  config->sqm_max_fwhm = 8.0;
  config->sqm_max_elongation = 0.5;
  config->sqm_intercept = 18.0f;
  config->sqm_slope = 1.5f;

  config->debug_raw_exposures = 0;

  config->image_center_x = 1000;
  config->image_center_y = 500;
  config->image_mask_radius = 0;
  config->image_horizon_radius = 1000;
  config->image_north_angle = 10.0f;
  config->image_zenith_x = 1000;
  config->image_zenith_y = 500;

  config->focus_radius = 400;

  snprintf(config->font_path, sizeof(config->font_path),
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
  config->font_size = 20;

  config->camera_monochrome = 0;
  config->camera_max_exposure = 60.0;
  config->camera_min_exposure = 0.000032;
  config->camera_max_gain = 350;
  config->camera_min_gain = 0;
  config->camera_cooling = 0;
  config->camera_cooling_temperature = -5;
  config->camera_binning2x2 = 0;
  config->camera_r_factor = 0.2126f;
  config->camera_g_factor = 0.7152f;
  config->camera_b_factor = 0.0722f;
  snprintf(config->camera_debayer, sizeof(config->camera_debayer), "vng");
  config->thumbnail = 1;
  config->thumbnail_width = 100;

  config->panorama = 0;
  config->panorama_center_x = 1155;
  config->panorama_center_y = 1155;
  config->panorama_width = 2048;
  config->panorama_height = 1090;
  config->panorama_horizontal_start = 90;


  config->location_latitude = 51.2349;
  config->location_longitude = 7.4479;
  config->location_altitude = 114;
  snprintf(config->location_city, sizeof(config->location_city), "Musterstadt/Germany");
  snprintf(config->location_owner, sizeof(config->location_owner), "Max Mustermann");

  config->capture_interval = 120;

  config->publish_exact_location = 1;

  snprintf(config->postprocessing_script, sizeof(config->postprocessing_script),
           "/opt/allsky360/scripts/postprocessing.sh");
  snprintf(config->endofnight_script, sizeof(config->endofnight_script),
           "/opt/allsky360/scripts/endofnight.sh");

  config->retention = 1;
  config->retention_max_days = 365;
  config->retention_max_size_gb = 500.0;
  config->retention_keep_days = 2;
  config->retention_dry_run = 0;
  snprintf(config->working_directory, sizeof(config->working_directory), "/opt/allsky360");
  snprintf(config->image_directory, sizeof(config->image_directory), "/home/allskyuser/images");
  snprintf(config->raw_dump_directory, sizeof(config->raw_dump_directory), "/home/allskyuser/raw");
  snprintf(config->video_directory, sizeof(config->video_directory), "/home/allskyuser/videos");
  snprintf(config->database_directory, sizeof(config->database_directory), "/home/allskyuser/database");

  config->jpeg_quality = 100;
  snprintf(config->lens_name, sizeof(config->lens_name), "ZWO 2.1mm f/2.0");
  config->lens_focallength = 2.1;
  config->lens_focalratio = 1.6;

  snprintf(config->indigoserver, sizeof(config->indigoserver), "127.0.0.1");
  config->indigoport = 7624;
  snprintf(config->indigocamera, sizeof(config->indigocamera),
           "Uranus-C @ localhost");
  config->indigomount[0] = '\0';
  config->indigo_capture_timeout = 20;

  char line[CONFIG_LINE_MAX], key[64], value[CONFIG_VALUE_MAX];
  while (fgets(line, sizeof(line), file)) {
    /* A line without '\n' that is not the last one did not fit into the
     * buffer. Skip the rest of it instead of parsing the remainder as a new
     * line, and do not use the truncated part either. */
    if (!strchr(line, '\n') && !feof(file)) {
      int c;
      while ((c = fgetc(file)) != EOF && c != '\n')
        ;
      if (warn_unknown_keys)
        fprintf(stderr,
                "WARNING: config.ini: line longer than %d characters "
                "ignored: %.40s...\n",
                CONFIG_LINE_MAX - 1, line);
      continue;
    }

    char *trim_line = strtok(line, "\n");
    if (!trim_line)
      continue;

    /* Skip leading blanks so that indented keys and comments behave the same
     * as unindented ones. */
    while (*trim_line == ' ' || *trim_line == '\t' || *trim_line == '\r')
      trim_line++;

    if (*trim_line == '\0' || *trim_line == '#')
      continue;

    /* The widths must match key[64] and value[CONFIG_VALUE_MAX] above */
    if (sscanf(trim_line, "%63[^=] = \"%511[^\"]\"", key, value) == 2 ||
        sscanf(trim_line, "%63[^=] = %511[^\n]", key, value) == 2) {
      /* Unquoted values run to the end of the line. "%s" used to stop at
       * the first blank and silently drop the rest, e.g. the 'S' of
       * "33.9249 S" or everything after "Uranus-C" in a camera name.
       * Only trailing blanks are removed. */
      size_t value_len = strlen(value);
      while (value_len > 0 && (value[value_len - 1] == ' ' ||
                               value[value_len - 1] == '\t' ||
                               value[value_len - 1] == '\r'))
        value[--value_len] = '\0';

      /* "%49[^=]" matches everything up to '=', including the blanks in
       * "key = value". Strip them, otherwise every strcmp() below fails and
       * the setting is silently ignored. */
      size_t key_len = strlen(key);
      while (key_len > 0 && (key[key_len - 1] == ' ' ||
                             key[key_len - 1] == '\t' ||
                             key[key_len - 1] == '\r'))
        key[--key_len] = '\0';

      if (key_len == 0)
        continue;

      if (strcmp(key, "daytime_capture") == 0)
        config->daytime_capture = atoi(value);
      else if (strcmp(key, "daytime_startexposure") == 0)
        config->daytime_startexposure = atof(value);
      else if (strcmp(key, "daytime_startgain") == 0)
        config->daytime_startgain = atoi(value);
      else if (strcmp(key, "daytime_meanbrightness") == 0)
        config->daytime_meanbrightness = atof(value);
      else if (strcmp(key, "daytime_move_blackpoint") == 0)
        config->daytime_move_blackpoint = atof(value);
      else if (strcmp(key, "daytime_move_blackpoint_min_shift_pct") == 0)
        config->daytime_move_blackpoint_min_shift_pct = atof(value);
      else if (strcmp(key, "daytime_move_blackpoint_max_shift_pct") == 0)
        config->daytime_move_blackpoint_max_shift_pct = atof(value);
      else if (strcmp(key, "daytime_move_blackpoint_dark_threshold") == 0)
        config->daytime_move_blackpoint_dark_threshold = atof(value);
      else if (strcmp(key, "daytime_hdr_exposure_factor") == 0)
        config->daytime_hdr_exposure_factor = atof(value);
      else if (strcmp(key, "daytime_white_balance_red") == 0)
        config->daytime_white_balance_red = atof(value);
      else if (strcmp(key, "daytime_white_balance_blue") == 0)
        config->daytime_white_balance_blue = atof(value);
      else if (strcmp(key, "daytime_white_balance_light_protect") == 0)
        config->daytime_white_balance_light_protect = atof(value);

      else if (strcmp(key, "nighttime_sunangle") == 0)
        config->nighttime_sunangle = atof(value);
      else if (strcmp(key, "nighttime_startexposure") == 0)
        config->nighttime_startexposure = atof(value);
      else if (strcmp(key, "nighttime_startgain") == 0)
        config->nighttime_startgain = atoi(value);
      else if (strcmp(key, "nighttime_meanbrightness") == 0)
        config->nighttime_meanbrightness = atof(value);
      else if (strcmp(key, "nighttime_fullmoongain") == 0)
        config->nighttime_fullmoongain = atoi(value);
      else if (strcmp(key, "nighttime_move_blackpoint") == 0)
        config->nighttime_move_blackpoint = atof(value);
      else if (strcmp(key, "nighttime_move_blackpoint_min_shift_pct") == 0)
        config->nighttime_move_blackpoint_min_shift_pct = atof(value);
      else if (strcmp(key, "nighttime_move_blackpoint_max_shift_pct") == 0)
        config->nighttime_move_blackpoint_max_shift_pct = atof(value);
      else if (strcmp(key, "nighttime_move_blackpoint_dark_threshold") == 0)
        config->nighttime_move_blackpoint_dark_threshold = atof(value);
      else if (strcmp(key, "nighttime_hdr_exposure_factor") == 0)
        config->nighttime_hdr_exposure_factor = atof(value);
      else if (strcmp(key, "nighttime_white_balance_red") == 0)
        config->nighttime_white_balance_red = atof(value);
      else if (strcmp(key, "nighttime_white_balance_blue") == 0)
        config->nighttime_white_balance_blue = atof(value);
      else if (strcmp(key, "nighttime_white_balance_light_protect") == 0)
        config->nighttime_white_balance_light_protect = atof(value);
      else if (strcmp(key, "nighttime_measure_noise") == 0)
        config->nighttime_measure_noise = atoi(value);
      else if (strcmp(key, "nighttime_measure_noise_radius") == 0)
        config->nighttime_measure_noise_radius = atoi(value);
      else if (strcmp(key, "nighttime_multiscale_median_filter") == 0)
        config->nighttime_multiscale_median_filter = atoi(value);
      else if (strcmp(key, "nighttime_multiscale_median_filter_max_scale") == 0)
        config->nighttime_multiscale_median_filter_max_scale = atoi(value);
      else if (strcmp(key, "nighttime_multiscale_median_filter_amount") == 0)
        config->nighttime_multiscale_median_filter_amount = atof(value);

      else if (strcmp(key, "measure_brightness_area") == 0)
        config->measure_brightness_area = atof(value);

      else if (strcmp(key, "target_mean") == 0)
        config->target_mean = atof(value);
      else if (strcmp(key, "black_clipping_point") == 0)
        config->black_clipping_point = atof(value);
      else if (strcmp(key, "white_clipping_point") == 0)
        config->white_clipping_point = atof(value);

      else if (strcmp(key, "hdr") == 0)
        config->hdr = atoi(value);
      else if (strcmp(key, "hdr_clipping_threshold") == 0)
        config->hdr_clipping_threshold = atof(value);
      else if (strcmp(key, "hdr_min_clipped_pixels") == 0)
        config->hdr_min_clipped_pixels = atof(value);
      else if (strcmp(key, "hdr_y_max_expected") == 0)
        config->hdr_y_max_expected = atof(value);
      else if (strcmp(key, "hdr_weight_sigma") == 0)
        config->hdr_weight_sigma = atof(value);
      else if (strcmp(key, "hdr_weight_clip_factor") == 0)
        config->hdr_weight_clip_factor = atof(value);
      else if (strcmp(key, "hdr_contrast_weight_strength") == 0)
        config->hdr_contrast_weight_strength = atof(value);
      else if (strcmp(key, "hdr_noise_fusion_strength") == 0)
        config->hdr_noise_fusion_strength = atof(value);
      else if (strcmp(key, "hdr_pyramid_levels_override") == 0)
        config->hdr_pyramid_levels_override = atoi(value);
      else if (strcmp(key, "hdr_weight_maps") == 0)
        config->hdr_weight_maps = atoi(value);
      else if (strcmp(key, "hdr_weight_stats") == 0)
        config->hdr_weight_stats = atoi(value);
      else if (strcmp(key, "hdr_save_exposures") == 0)
        config->hdr_save_exposures = atoi(value);
      else if (strcmp(key, "hdrmt") == 0)
        config->hdrmt = atoi(value);

      else if (strcmp(key, "spektrum_enabled") == 0)
        config->spektrum_enabled = atoi(value);
      else if (strcmp(key, "spektrum_cx") == 0)
        config->spektrum_cx = atoi(value);
      else if (strcmp(key, "spektrum_cy") == 0)
        config->spektrum_cy = atoi(value);
      else if (strcmp(key, "spektrum_radius") == 0)
        config->spektrum_radius = atoi(value);
      else if (strcmp(key, "spektrum_regularization") == 0)
        config->spektrum_regularization = atof(value);
      else if (strcmp(key, "spektrum_output_path") == 0)
        config_copy_string(config->spektrum_output_path, sizeof(config->spektrum_output_path), key,
                           value);

      else if (strcmp(key, "autostretch") == 0)
        config->autostretch = atoi(value);
      else if (strcmp(key, "autostretch_min_val") == 0)
        config->autostretch_min_val = atof(value);
      else if (strcmp(key, "autostretch_max_val") == 0)
        config->autostretch_max_val = atof(value);

      else if (strcmp(key, "acdnr_filter") == 0)
        config->acdnr_filter = atoi(value);
      else if (strcmp(key, "acdnr_filter_lum_stddev") == 0)
        config->acdnr_filter_lum_stddev = atof(value);
      else if (strcmp(key, "acdnr_filter_lum_amount") == 0)
        config->acdnr_filter_lum_amount = atof(value);
      else if (strcmp(key, "acdnr_filter_lum_iterations") == 0)
        config->acdnr_filter_lum_iterations = atoi(value);
      else if (strcmp(key, "acdnr_filter_lum_kernel_size") == 0)
        config->acdnr_filter_lum_kernel_size = atoi(value);

      else if (strcmp(key, "acdnr_filter_chrom_stddev") == 0)
        config->acdnr_filter_chrom_stddev = atof(value);
      else if (strcmp(key, "acdnr_filter_chrom_amount") == 0)
        config->acdnr_filter_chrom_amount = atof(value);
      else if (strcmp(key, "acdnr_filter_chrom_iterations") == 0)
        config->acdnr_filter_chrom_iterations = atoi(value);
      else if (strcmp(key, "acdnr_filter_chrom_kernel_size") == 0)
        config->acdnr_filter_chrom_kernel_size = atoi(value);

      else if (strcmp(key, "gamma") == 0)
        config->gamma = atof(value);
      else if (strcmp(key, "auto_gamma") == 0)
        config->auto_gamma = atoi(value);
      else if (strcmp(key, "auto_gamma_target") == 0)
        config->auto_gamma_target = atof(value);
      else if (strcmp(key, "auto_gamma_min") == 0)
        config->auto_gamma_min = atof(value);
      else if (strcmp(key, "auto_gamma_max") == 0)
        config->auto_gamma_max = atof(value);
      else if (strcmp(key, "reinhard_tonemap") == 0)
        config->reinhard_tonemap = atoi(value);
      else if (strcmp(key, "reinhard_white_point") == 0)
        config->reinhard_white_point = atof(value);
      else if (strcmp(key, "tone_curve") == 0)
        config->tone_curve = atoi(value);
      else if (strcmp(key, "tone_curve_shadow_lift") == 0)
        config->tone_curve_shadow_lift = atof(value);
      else if (strcmp(key, "tone_curve_midtone_contrast") == 0)
        config->tone_curve_midtone_contrast = atof(value);

      else if (strcmp(key, "saturation") == 0)
        config->saturation = atof(value);

      else if (strcmp(key, "dehaze_amount") == 0)
        config->dehaze_amount = atof(value);
      else if (strcmp(key, "dehaze_estimate") == 0)
        config->dehaze_estimate = atof(value);
      else if (strcmp(key, "dehaze_gamma") == 0)
        config->dehaze_gamma = atof(value);
      else if (strcmp(key, "dehaze_saturation") == 0)
        config->dehaze_saturation = atof(value);

      else if (strcmp(key, "clarity_filter") == 0)
        config->clarity_filter = atoi(value);
      else if (strcmp(key, "clarity_filter_strength") == 0)
        config->clarity_filter_strength = atof(value);
      else if (strcmp(key, "clarity_filter_radius") == 0)
        config->clarity_filter_radius = atoi(value);
      else if (strcmp(key, "clarity_filter_midtone_width") == 0)
        config->clarity_filter_midtone_width = atof(value);
      else if (strcmp(key, "clarity_filter_preserve_highlights") == 0)
        config->clarity_filter_preserve_highlights = atoi(value);
      else if (strcmp(key, "clarity_filter_mask_mode") == 0)
        config->clarity_filter_mask_mode = atoi(value);

      else if (strcmp(key, "auto_color_calibration_area") == 0)
        config->auto_color_calibration_area = atof(value);

      else if (strcmp(key, "local_histogram_normalization") == 0)
        config->local_histogram_normalization = atoi(value);
      else if (strcmp(key, "local_histogram_normalization_kernel_radius") == 0)
        config->local_histogram_normalization_kernel_radius = atof(value);
      else if (strcmp(key, "local_histogram_normalization_contrast_limit") == 0)
        config->local_histogram_normalization_contrast_limit = atof(value);

      else if (strcmp(key, "scnr_filter") == 0)
        config->scnr_filter = atoi(value);
      else if (strcmp(key, "scnr_filter_amount") == 0)
        config->scnr_filter_amount = atof(value);
      else if (strcmp(key, "scnr_filter_protection") == 0)
        config_copy_string(config->scnr_filter_protection, sizeof(config->scnr_filter_protection), key,
                           value);

      else if (strcmp(key, "wavelet_sharpen") == 0)
        config->wavelet_sharpen = atoi(value);
      else if (strcmp(key, "wavelet_sharpen_gain_small") == 0)
        config->wavelet_sharpen_gain_small = atof(value);
      else if (strcmp(key, "wavelet_sharpen_gain_medium") == 0)
        config->wavelet_sharpen_gain_medium = atof(value);
      else if (strcmp(key, "wavelet_sharpen_gain_large") == 0)
        config->wavelet_sharpen_gain_large = atof(value);

      else if (strcmp(key, "sqm") == 0)
        config->sqm = atoi(value);
      else if (strcmp(key, "sqm_max_stars") == 0)
        config->sqm_max_stars = atoi(value);
      else if (strcmp(key, "sqm_radius") == 0)
        config->sqm_radius = atoi(value);
      else if (strcmp(key, "sqm_center_x") == 0)
        config->sqm_center_x = atoi(value);
      else if (strcmp(key, "sqm_center_y") == 0)
        config->sqm_center_y = atoi(value);
      /* Settings of the former template-matching detector */
      else if (strcmp(key, "sqm_threshold") == 0 ||
               strcmp(key, "sqm_min_brightness") == 0 ||
               strcmp(key, "sqm_star_size") == 0 ||
               strcmp(key, "sqm_template") == 0) {
        if (warn_unknown_keys)
          fprintf(stderr,
                  "NOTE: %s: '%s' is no longer used (the star detector works "
                  "on the raw frame now, see sqm_detect_sigma in "
                  "config.ini.default)\n",
                  filename, key);
      }
      else if (strcmp(key, "sqm_star_sigma") == 0)
        config->sqm_star_sigma = atof(value);
      else if (strcmp(key, "sqm_detect_sigma") == 0)
        config->sqm_detect_sigma = atof(value);
      else if (strcmp(key, "sqm_max_fwhm") == 0)
        config->sqm_max_fwhm = atof(value);
      else if (strcmp(key, "sqm_max_elongation") == 0)
        config->sqm_max_elongation = atof(value);
      else if (strcmp(key, "sqm_intercept") == 0)
        config->sqm_intercept = atof(value);
      else if (strcmp(key, "sqm_slope") == 0)
        config->sqm_slope = atof(value);

      else if (strcmp(key, "debug_raw_exposures") == 0)
        config->debug_raw_exposures = atoi(value);

      else if (strcmp(key, "camera_monochrome") == 0)
        config->camera_monochrome = atoi(value);
      else if (strcmp(key, "camera_max_exposure") == 0)
        config->camera_max_exposure = atof(value);
      else if (strcmp(key, "camera_min_exposure") == 0)
        config->camera_min_exposure = atof(value);
      else if (strcmp(key, "camera_max_gain") == 0)
        config->camera_max_gain = atoi(value);
      else if (strcmp(key, "camera_min_gain") == 0)
        config->camera_min_gain = atoi(value);
      else if (strcmp(key, "camera_binning2x2") == 0)
        config->camera_binning2x2 = atoi(value);
      else if (strcmp(key, "camera_cooling") == 0)
        config->camera_cooling = atoi(value);
      else if (strcmp(key, "camera_cooling_temperature") == 0)
        config->camera_cooling_temperature = atoi(value);
      else if (strcmp(key, "camera_r_factor") == 0)
        config->camera_r_factor = atof(value);
      else if (strcmp(key, "camera_g_factor") == 0)
        config->camera_g_factor = atof(value);
      else if (strcmp(key, "camera_b_factor") == 0)
        config->camera_b_factor = atof(value);
      else if (strcmp(key, "camera_debayer") == 0)
        config_copy_string(config->camera_debayer, sizeof(config->camera_debayer), key,
                           value);

      else if (strcmp(key, "jpeg_quality") == 0)
        config->jpeg_quality = atoi(value);
      else if (strcmp(key, "lens_name") == 0)
        config_copy_string(config->lens_name, sizeof(config->lens_name), key,
                           value);
      else if (strcmp(key, "lens_focallength") == 0)
        config->lens_focallength = atof(value);
      else if (strcmp(key, "lens_focalratio") == 0)
        config->lens_focalratio = atof(value);

      else if (strcmp(key, "image_center_x") == 0)
        config->image_center_x = atoi(value);
      else if (strcmp(key, "image_center_y") == 0)
        config->image_center_y = atoi(value);
      else if (strcmp(key, "image_mask_radius") == 0)
        config->image_mask_radius = atoi(value);
      else if (strcmp(key, "image_horizon_radius") == 0)
        config->image_horizon_radius = atoi(value);
      else if (strcmp(key, "image_north_angle") == 0)
        config->image_north_angle = atof(value);
      else if (strcmp(key, "image_zenith_x") == 0)
        config->image_zenith_x = atoi(value);
      else if (strcmp(key, "image_zenith_y") == 0)
        config->image_zenith_y = atoi(value);

      else if (strcmp(key, "focus_radius") == 0)
        config->focus_radius = atoi(value);

      else if (strcmp(key, "font_path") == 0)
        config_copy_string(config->font_path, sizeof(config->font_path), key,
                           value);
      else if (strcmp(key, "font_size") == 0)
        config->font_size = atoi(value);

      else if (strcmp(key, "crop_image") == 0)
        config->crop_image = atoi(value);
      else if (strcmp(key, "crop_width") == 0)
        config->crop_width = atoi(value);
      else if (strcmp(key, "crop_height") == 0)
        config->crop_height = atoi(value);
      else if (strcmp(key, "crop_x_offset") == 0)
        config->crop_x_offset = atoi(value);
      else if (strcmp(key, "crop_y_offset") == 0)
        config->crop_y_offset = atoi(value);
      else if (strcmp(key, "rotate_image") == 0)
        config->rotate_image = atoi(value);

      else if (strcmp(key, "thumbnail") == 0)
        config->thumbnail = atoi(value);
      else if (strcmp(key, "thumbnail_width") == 0)
        config->thumbnail_width = atoi(value);

      else if (strcmp(key, "panorama") == 0)
        config->panorama = atoi(value);
      else if (strcmp(key, "panorama_center_x") == 0)
        config->panorama_center_x = atoi(value);
      else if (strcmp(key, "panorama_center_y") == 0)
        config->panorama_center_y = atoi(value);
      else if (strcmp(key, "panorama_width") == 0)
        config->panorama_width = atoi(value);
      else if (strcmp(key, "panorama_height") == 0)
        config->panorama_height = atoi(value);
      else if (strcmp(key, "panorama_horizontal_start") == 0)
        config->panorama_horizontal_start = atoi(value);

      else if (strcmp(key, "location_latitude") == 0 ||
               strcmp(key, "location_longitude") == 0) {
        int is_latitude = (strcmp(key, "location_latitude") == 0);
        double *target = is_latitude ? &config->location_latitude
                                     : &config->location_longitude;
        /* Not atof(): it drops the S/W suffix and mirrors the location
         * (review.md, BUG-10). On error the default is kept. */
        if (parse_lat_lon(value, is_latitude, target) != 0)
          fprintf(stderr,
                  "WARNING: config.ini: invalid %s '%s' (expected e.g. %s), "
                  "using %.4f\n",
                  key, value, is_latitude ? "51.2349N, 33.9249S or -33.9249"
                                          : "7.4479E, 74.0060W or -74.0060",
                  *target);
      }
      else if (strcmp(key, "location_altitude") == 0)
        config->location_altitude = atoi(value);
      else if (strcmp(key, "location_city") == 0)
        config_copy_string(config->location_city, sizeof(config->location_city), key,
                           value);
      else if (strcmp(key, "location_owner") == 0)
        config_copy_string(config->location_owner, sizeof(config->location_owner), key,
                           value);
      else if (strcmp(key, "working_directory") == 0)
        config_copy_string(config->working_directory, sizeof(config->working_directory), key,
                           value);
      else if (strcmp(key, "image_directory") == 0)
        config_copy_string(config->image_directory, sizeof(config->image_directory), key,
                           value);
      else if (strcmp(key, "raw_dump_directory") == 0)
        config_copy_string(config->raw_dump_directory, sizeof(config->raw_dump_directory), key,
                           value);
      else if (strcmp(key, "video_directory") == 0)
        config_copy_string(config->video_directory, sizeof(config->video_directory), key,
                           value);
      else if (strcmp(key, "database_directory") == 0)
        config_copy_string(config->database_directory, sizeof(config->database_directory), key,
                           value);

      else if (strcmp(key, "capture_interval") == 0)
        config->capture_interval = atoi(value);
      else if (strcmp(key, "indigo_servername") == 0)
        config_copy_string(config->indigoservername, sizeof(config->indigoservername), key,
                           value);
      else if (strcmp(key, "indigo_server") == 0)
        config_copy_string(config->indigoserver, sizeof(config->indigoserver), key,
                           value);
      else if (strcmp(key, "indigo_port") == 0)
        config->indigoport = atoi(value);
      else if (strcmp(key, "indigo_camera") == 0)
        config_copy_string(config->indigocamera, sizeof(config->indigocamera), key,
                           value);
      else if (strcmp(key, "indigo_mount") == 0)
        config_copy_string(config->indigomount, sizeof(config->indigomount), key,
                           value);
      else if (strcmp(key, "indigo_capture_timeout") == 0)
        config->indigo_capture_timeout = atoi(value);

      else if (strcmp(key, "publish_exact_location") == 0)
        config->publish_exact_location = atoi(value);

      else if (strcmp(key, "postprocessing_script") == 0)
        config_copy_string(config->postprocessing_script, sizeof(config->postprocessing_script), key,
                           value);
      else if (strcmp(key, "endofnight_script") == 0)
        config_copy_string(config->endofnight_script, sizeof(config->endofnight_script), key,
                           value);

      else if (strcmp(key, "retention") == 0)
        config->retention = atoi(value);
      else if (strcmp(key, "retention_max_days") == 0)
        config->retention_max_days = atoi(value);
      else if (strcmp(key, "retention_max_size_gb") == 0)
        config->retention_max_size_gb = atof(value);
      else if (strcmp(key, "retention_keep_days") == 0)
        config->retention_keep_days = atoi(value);
      else if (strcmp(key, "retention_dry_run") == 0)
        config->retention_dry_run = atoi(value);

      /* Consumed by the Node.js API (allsky-api/index.js), not by this daemon. */
      else if (strncmp(key, "api-", 4) == 0) {
        /* intentionally ignored */
      }

      /* Consumed by the RS485 sensor tool (tools/sensors), not by this daemon. */
      else if (strncmp(key, "modbus_", 7) == 0) {
        /* intentionally ignored */
      }

      /* Anything else is a typo or a stale setting. Report it once per process:
       * load_config() runs on every capture cycle, so warning every time would
       * flood the log. */
      else if (warn_unknown_keys)
        fprintf(stderr,
                "WARNING: %s: unknown configuration key '%s' (ignored)\n",
                filename, key);
    }
  }

  fclose(file);

  /* Range checks for values that would otherwise misbehave silently. */
  if (config->retention_max_days < 1) {
    if (warn_unknown_keys)
      fprintf(stderr,
              "WARNING: %s: retention_max_days %d is below 1, retention "
              "disabled\n",
              filename, config->retention_max_days);
    config->retention = 0;
  }
  if (config->retention_max_size_gb <= 0.0) {
    if (warn_unknown_keys)
      fprintf(stderr,
              "WARNING: %s: retention_max_size_gb %.2f is not positive, "
              "retention disabled\n",
              filename, config->retention_max_size_gb);
    config->retention = 0;
  }
  if (config->retention_keep_days < 2) {
    if (warn_unknown_keys && config->retention_keep_days != 0)
      fprintf(stderr,
              "WARNING: %s: retention_keep_days %d is below the minimum of 2, "
              "using 2\n",
              filename, config->retention_keep_days);
    config->retention_keep_days = 2;
  }

  if (config->nighttime_sunangle > 0.0 || config->nighttime_sunangle < -18.0) {
    double clamped = config->nighttime_sunangle > 0.0 ? 0.0 : -18.0;
    if (warn_unknown_keys)
      fprintf(stderr,
              "WARNING: %s: nighttime_sunangle %.2f is outside -18.0..0.0, "
              "clamped to %.2f\n",
              filename, config->nighttime_sunangle, clamped);
    config->nighttime_sunangle = clamped;
  }

  warn_unknown_keys = 0;

  return 0;
}

int show_config(config_t *config) {
  printf("\nLoaded configuration:\n");
  printf("daytime_capture: %d (%s)\n", config->daytime_capture,
         config->daytime_capture == 1 ? "yes" : "no");
  printf("daytime_startexposure: %.6f\n", config->daytime_startexposure);
  printf("daytime_startgain: %d\n", config->daytime_startgain);
  printf("daytime_meanbrightness: %.2f\n", config->daytime_meanbrightness);
  printf("daytime_move_blackpoint: %.2f\n", config->daytime_move_blackpoint);
  printf("daytime_move_blackpoint_min_shift_pct: %.1f\n",
         config->daytime_move_blackpoint_min_shift_pct);
  printf("daytime_move_blackpoint_max_shift_pct: %.1f\n",
         config->daytime_move_blackpoint_max_shift_pct);
  printf("daytime_move_blackpoint_dark_threshold: %.1f\n",
         config->daytime_move_blackpoint_dark_threshold);
  printf("daytime_hdr_exposure_factor: %.1f\n",
         config->daytime_hdr_exposure_factor);
  printf("daytime_white_balance_red: %.1f\n",
         config->daytime_white_balance_red);
  printf("daytime_white_balance_blue: %.1f\n",
         config->daytime_white_balance_blue);
  printf("daytime_white_balance_light_protect: %.1f\n",
         config->daytime_white_balance_light_protect);

  printf("postprocessing_script: %s\n",
         config->postprocessing_script[0] ? config->postprocessing_script
                                          : "(disabled)");
  printf("endofnight_script: %s\n",
         config->endofnight_script[0] ? config->endofnight_script
                                      : "(disabled)");
  printf("retention: %d (%s)\n", config->retention,
         config->retention == 1 ? "on" : "off");
  printf("retention_max_days: %d\n", config->retention_max_days);
  printf("retention_max_size_gb: %.2f\n", config->retention_max_size_gb);
  printf("retention_keep_days: %d\n", config->retention_keep_days);
  printf("retention_dry_run: %d\n", config->retention_dry_run);
  printf("nighttime_sunangle: %.2f\n", config->nighttime_sunangle);
  printf("nighttime_startexposure: %.6f\n", config->nighttime_startexposure);
  printf("nighttime_startgain: %d\n", config->nighttime_startgain);
  printf("nighttime_meanbrightness: %.2f\n", config->nighttime_meanbrightness);
  printf("nighttime_fullmoongain: %d\n", config->nighttime_fullmoongain);
  printf("nighttime_move_blackpoint: %.2f\n",
         config->nighttime_move_blackpoint);
  printf("nighttime_move_blackpoint_min_shift_pct: %.1f\n",
         config->nighttime_move_blackpoint_min_shift_pct);
  printf("nighttime_move_blackpoint_max_shift_pct: %.1f\n",
         config->nighttime_move_blackpoint_max_shift_pct);
  printf("nighttime_move_blackpoint_dark_threshold: %.1f\n",
         config->nighttime_move_blackpoint_dark_threshold);
  printf("nighttime_hdr_exposure_factor: %.1f\n",
         config->nighttime_hdr_exposure_factor);
  printf("nighttime_white_balance_red: %.1f\n",
         config->nighttime_white_balance_red);
  printf("nighttime_white_balance_blue: %.1f\n",
         config->nighttime_white_balance_blue);
  printf("nighttime_white_balance_light_protect: %.1f\n",
         config->nighttime_white_balance_light_protect);
  printf("nighttime_measure_noise: %d (%s)\n", config->nighttime_measure_noise,
         config->nighttime_measure_noise == 1 ? "yes" : "no");
  printf("nighttime_measure_noise_radius: %dpx\n",
         config->nighttime_measure_noise_radius);
  printf("nighttime_multiscale_median_filter: %d (%s)\n",
         config->nighttime_multiscale_median_filter,
         config->nighttime_multiscale_median_filter == 1 ? "yes" : "no");
  printf("nighttime_multiscale_median_filter_max_scale: %d\n",
         config->nighttime_multiscale_median_filter_max_scale);
  printf("nighttime_multiscale_median_filter_amount: %.1f\n",
         config->nighttime_multiscale_median_filter_amount);

  printf("measure_brightness_area: %.1f\n", config->measure_brightness_area);
  printf("target_mean: %.2f\n", config->target_mean);
  printf("black_clipping_point: %.2f\n", config->black_clipping_point);
  printf("white_clipping_point: %.2f\n", config->white_clipping_point);

  printf("hdr: %d (%s)\n", config->hdr, config->hdr == 1 ? "yes" : "no");
  printf("hdr_clipping_threshold: %.2f\n", config->hdr_clipping_threshold);
  printf("hdr_min_clipped_pixels: %.2f\n", config->hdr_min_clipped_pixels);
  printf("hdr_y_max_expected: %.1f\n", config->hdr_y_max_expected);
  printf("hdr_weight_sigma: %.2f\n", config->hdr_weight_sigma);
  printf("hdr_weight_clip_factor: %.2f\n", config->hdr_weight_clip_factor);
  printf("hdr_contrast_weight_strength: %.1f\n",
         config->hdr_contrast_weight_strength);
  printf("hdr_noise_fusion_strength: %.2f\n",
         config->hdr_noise_fusion_strength);
  printf("hdr_pyramid_levels_override: %d\n",
         config->hdr_pyramid_levels_override);
  printf("hdr_weight_maps: %d (%s)\n", config->hdr_weight_maps,
         config->hdr_weight_maps == 1 ? "yes" : "no");
  printf("hdr_weight_stats: %d (%s)\n", config->hdr_weight_stats,
         config->hdr_weight_stats == 1 ? "yes" : "no");
  printf("hdr_save_exposures: %d (%s)\n", config->hdr_save_exposures,
         config->hdr_save_exposures == 1 ? "yes" : "no");
  printf("hdrmt: %d (%s)\n", config->hdrmt, config->hdrmt == 1 ? "yes" : "no");

  printf("spektrum_enabled: %d (%s)\n", config->spektrum_enabled,
         config->spektrum_enabled == 1 ? "yes" : "no");
  printf("spektrum_cx: %d\n", config->spektrum_cx);
  printf("spektrum_cy: %d\n", config->spektrum_cy);
  printf("spektrum_radius: %d px\n", config->spektrum_radius);
  printf("spektrum_regularization: %.4f\n", config->spektrum_regularization);
  printf("spektrum_output_path: %s\n", config->spektrum_output_path);

  printf("autostretch: %d (%s)\n", config->autostretch,
         config->autostretch == 1 ? "yes" : "no");
  printf("autostretch_min_val: %.2f\n", config->autostretch_min_val);
  printf("autostretch_max_val: %.2f\n", config->autostretch_max_val);

  printf("acdnr_filter: %d (%s)\n", config->acdnr_filter,
         config->acdnr_filter == 1 ? "yes" : "no");
  printf("acdnr_filter_lum_stddev: %.4f\n", config->acdnr_filter_lum_stddev);
  printf("acdnr_filter_lum_amount: %.4f\n", config->acdnr_filter_lum_amount);
  printf("acdnr_filter_lum_iterations: %d\n",
         config->acdnr_filter_lum_iterations);
  printf("acdnr_filter_lum_kernel_size: (%dx%d)\n",
         config->acdnr_filter_lum_kernel_size,
         config->acdnr_filter_lum_kernel_size);

  printf("acdnr_filter_chrom_stddev: %.4f\n",
         config->acdnr_filter_chrom_stddev);
  printf("acdnr_filter_chrom_amount: %.4f\n",
         config->acdnr_filter_chrom_amount);
  printf("acdnr_filter_chrom_iterations: %d\n",
         config->acdnr_filter_chrom_iterations);
  printf("acdnr_filter_chrom_kernel_size: (%dx%d)\n",
         config->acdnr_filter_chrom_kernel_size,
         config->acdnr_filter_chrom_kernel_size);

  printf("gamma: %.1f\n", config->gamma);
  printf("auto_gamma: %d (%s)\n", config->auto_gamma,
         config->auto_gamma == 1 ? "yes" : "no");
  printf("auto_gamma_target: %.3f\n", config->auto_gamma_target);
  printf("auto_gamma_min: %.2f\n", config->auto_gamma_min);
  printf("auto_gamma_max: %.2f\n", config->auto_gamma_max);
  printf("reinhard_tonemap: %d (%s)\n", config->reinhard_tonemap,
         config->reinhard_tonemap == 1 ? "yes" : "no");
  printf("reinhard_white_point: %.3f%s\n", config->reinhard_white_point,
         config->reinhard_white_point == 0.0 ? " (simple)" : " (extended)");
  printf("tone_curve: %d (%s)\n", config->tone_curve,
         config->tone_curve == 1 ? "yes" : "no");
  printf("tone_curve_shadow_lift: %.3f\n", config->tone_curve_shadow_lift);
  printf("tone_curve_midtone_contrast: %.3f\n",
         config->tone_curve_midtone_contrast);
  printf("saturation: %.1f\n", config->saturation);

  printf("dehaze_amount: %.1f\n", config->dehaze_amount);
  printf("dehaze_estimate: %.2f\n", config->dehaze_estimate);
  printf("dehaze_gamma: %.1f\n", config->dehaze_gamma);
  printf("dehaze_saturation: %.1f\n", config->dehaze_saturation);

  printf("clarity_filter: %d (%s)\n", config->clarity_filter,
         config->clarity_filter == 1 ? "yes" : "no");
  printf("clarity_filter_strength: %.1f\n", config->clarity_filter_strength);
  printf("clarity_filter_radius: %d\n", config->clarity_filter_radius);
  printf("clarity_filter_midtone_width: %.1f\n",
         config->clarity_filter_midtone_width);
  printf("clarity_filter_preserve_highlights: %d (%s)\n",
         config->clarity_filter_preserve_highlights,
         config->clarity_filter_preserve_highlights == 1 ? "yes" : "no");
  printf("clarity_filter_mask_mode: %d\n", config->clarity_filter_mask_mode);

  printf("auto_color_calibration_area: %.1f%s\n",
         config->auto_color_calibration_area,
         config->auto_color_calibration_area == 0.0 ? " (off)" : " ");

  printf("local_histogram_normalization: %d (%s)\n",
         config->local_histogram_normalization,
         config->local_histogram_normalization == 1 ? "yes" : "no");
  printf("local_histogram_normalization_kernel_radius: %.1f\n",
         config->local_histogram_normalization_kernel_radius);
  printf("local_histogram_normalization_contrast_limit: %.1f\n",
         config->local_histogram_normalization_contrast_limit);

  printf("scnr_filter: %d (%s)\n", config->scnr_filter,
         config->scnr_filter == 1 ? "yes" : "no");
  printf("scnr_filter_amount: %.1f\n", config->scnr_filter_amount);
  printf("scnr_filter_protection: %s\n", config->scnr_filter_protection);

  printf("wavelet_sharpen: %d (%s)\n", config->wavelet_sharpen,
         config->wavelet_sharpen == 1 ? "yes" : "no");
  printf("wavelet_sharpen_gain_small: %.1f\n",
         config->wavelet_sharpen_gain_small);
  printf("wavelet_sharpen_gain_medium: %.1f\n",
         config->wavelet_sharpen_gain_medium);
  printf("wavelet_sharpen_gain_large: %.1f\n",
         config->wavelet_sharpen_gain_large);

  printf("sqm: %d (%s)\n", config->sqm, config->sqm == 1 ? "yes" : "no");
  printf("sqm_max_stars: %d\n", config->sqm_max_stars);
  printf("sqm_radius: %dpx\n", config->sqm_radius);
  printf("sqm_center_x: %d\n", config->sqm_center_x);
  printf("sqm_center_y: %d\n", config->sqm_center_y);
  printf("sqm_star_sigma: %.2f px\n", config->sqm_star_sigma);
  printf("sqm_detect_sigma: %.1f\n", config->sqm_detect_sigma);
  printf("sqm_max_fwhm: %.1f px\n", config->sqm_max_fwhm);
  printf("sqm_max_elongation: %.2f\n", config->sqm_max_elongation);
  printf("sqm_intercept: %.1f\n", config->sqm_intercept);
  printf("sqm_slope: %.1f\n", config->sqm_slope);

  printf("debug_raw_exposures: %d (%s)\n", config->debug_raw_exposures,
         config->debug_raw_exposures == 1 ? "yes" : "no");

  printf("camera_monochrome: %d (%s)\n", config->camera_monochrome,
         config->camera_monochrome == 1 ? "yes" : "no");
  printf("camera_max_exposure: %.6fs\n", config->camera_max_exposure);
  printf("camera_min_exposure: %.6fs\n", config->camera_min_exposure);
  printf("camera_max_gain: %d\n", config->camera_max_gain);
  printf("camera_min_gain: %d\n", config->camera_min_gain);
  printf("camera_binning2x2: %d (%s)\n", config->camera_binning2x2,
         config->camera_binning2x2 == 1 ? "yes" : "no");
  printf("camera_cooling: %d (%s)\n", config->camera_cooling,
         config->camera_cooling == 1 ? "yes" : "no");
  printf("camera_cooling_temperature: %d °C\n",
         config->camera_cooling_temperature);
  printf("camera_r_factor: %.4f\n", config->camera_r_factor);
  printf("camera_g_factor: %.4f\n", config->camera_g_factor);
  printf("camera_b_factor: %.4f\n", config->camera_b_factor);
  printf("camera_debayer: %s\n", config->camera_debayer);
  printf("jpeg_quality: %d%%\n", config->jpeg_quality);
  printf("lensname: %s\n", config->lens_name);
  printf("lensfocallength: %.2f\n", config->lens_focallength);
  printf("lensfocalratio: %.2f\n", config->lens_focalratio);

  printf("image_center_x: %d\n", config->image_center_x);
  printf("image_center_y: %d\n", config->image_center_y);
  printf("image_mask_radius: %d px (%s)\n", config->image_mask_radius,
         config->image_mask_radius > 0 ? "enabled" : "disabled");
  printf("image_horizon_radius: %d px\n", config->image_horizon_radius);
  printf("image_north_angle: %.1f degrees\n", config->image_north_angle);
  printf("image_zenith_x: %d\n", config->image_zenith_x);
  printf("image_zenith_y: %d\n", config->image_zenith_y);
  printf("focus_radius: %d\n", config->focus_radius);
  printf("crop_image: %d\n", config->crop_image);
  printf("crop_width: %d px\n", config->crop_width);
  printf("crop_height: %d px\n", config->crop_height);
  printf("crop_x_offset: %d px\n", config->crop_x_offset);
  printf("crop_y_offset: %d px\n", config->crop_y_offset);

  printf("font_path: %s\n", config->font_path);
  printf("font_size: %d\n", config->font_size);

  printf("rotate_image: %d\n", config->rotate_image);

  printf("thumbnail: %d (%s)\n", config->thumbnail,
         config->thumbnail == 1 ? "yes" : "no");
  printf("thumbnail_width: %dpx\n", config->thumbnail_width);

  printf("panorama: %d (%s)\n", config->panorama,
         config->panorama == 1 ? "yes" : "no");
  printf("panorama_center_x: %d\n", config->panorama_center_x);
  printf("panorama_center_y: %d\n", config->panorama_center_y);
  printf("panorama_width: %d\n", config->panorama_width);
  printf("panorama_height: %d\n", config->panorama_height);
  printf("panorama_horizontal_start: %d deg.\n",
         config->panorama_horizontal_start);

  printf("location_latitude: %.6f\n", config->location_latitude);
  printf("location_longitude: %.6f\n", config->location_longitude);
  printf("location_altitude: %dm\n", config->location_altitude);
  printf("location_city: %s\n", config->location_city);
  printf("location_owner: %s\n", config->location_owner);
  printf("working_directory: %s\n", config->working_directory);
  printf("image_directory: %s\n", config->image_directory);
  printf("raw_dump_directory: %s\n", config->raw_dump_directory);
  printf("video_directory: %s\n", config->video_directory);
  printf("database_directory: %s\n", config->database_directory);

  printf("capture_interval: %d\n", config->capture_interval);
  printf("indigo_servername: %s\n", config->indigoservername);
  printf("indigo_server: %s\n", config->indigoserver);
  printf("indigo_port: %d\n", config->indigoport);
  printf("indigo_camera: %s\n", config->indigocamera);
  printf("indigo_mount: %s\n", config->indigomount);
  printf("indigo_capture_timeout: %ds\n", config->indigo_capture_timeout);

  return 0;
}

int config_to_json(const char *filename, const config_t *config) {
  if (!filename || !config)
    return 1;

  struct json_object *jroot = json_object_new_object();

  // Generated input from generate_config_to_json.py
  json_object_object_add(jroot, "daytime_capture",
                         json_object_new_int(config->daytime_capture));
  json_object_object_add(jroot, "daytime_startexposure",
                         json_object_new_double(config->daytime_startexposure));
  json_object_object_add(jroot, "daytime_startgain",
                         json_object_new_int(config->daytime_startgain));
  json_object_object_add(
      jroot, "daytime_meanbrightness",
      json_object_new_double(config->daytime_meanbrightness));
  json_object_object_add(
      jroot, "daytime_move_blackpoint",
      json_object_new_double(config->daytime_move_blackpoint));
  json_object_object_add(
      jroot, "daytime_move_blackpoint_min_shift_pct",
      json_object_new_double(config->daytime_move_blackpoint_min_shift_pct));
  json_object_object_add(
      jroot, "daytime_move_blackpoint_max_shift_pct",
      json_object_new_double(config->daytime_move_blackpoint_max_shift_pct));
  json_object_object_add(
      jroot, "daytime_move_blackpoint_dark_threshold",
      json_object_new_double(config->daytime_move_blackpoint_dark_threshold));
  json_object_object_add(
      jroot, "daytime_hdr_exposure_factor",
      json_object_new_double(config->daytime_hdr_exposure_factor));
  json_object_object_add(
      jroot, "daytime_white_balance_red",
      json_object_new_double(config->daytime_white_balance_red));
  json_object_object_add(
      jroot, "daytime_white_balance_blue",
      json_object_new_double(config->daytime_white_balance_blue));
  json_object_object_add(
      jroot, "daytime_white_balance_light_protect",
      json_object_new_double(config->daytime_white_balance_light_protect));

  json_object_object_add(jroot, "retention",
                         json_object_new_int(config->retention));
  json_object_object_add(jroot, "retention_max_days",
                         json_object_new_int(config->retention_max_days));
  json_object_object_add(jroot, "retention_max_size_gb",
                         json_object_new_double(config->retention_max_size_gb));
  json_object_object_add(jroot, "nighttime_sunangle",
                         json_object_new_double(config->nighttime_sunangle));
  json_object_object_add(
      jroot, "nighttime_startexposure",
      json_object_new_double(config->nighttime_startexposure));
  json_object_object_add(jroot, "nighttime_startgain",
                         json_object_new_int(config->nighttime_startgain));
  json_object_object_add(
      jroot, "nighttime_meanbrightness",
      json_object_new_double(config->nighttime_meanbrightness));
  json_object_object_add(jroot, "nighttime_fullmoongain",
                         json_object_new_int(config->nighttime_fullmoongain));
  json_object_object_add(
      jroot, "nighttime_move_blackpoint",
      json_object_new_double(config->nighttime_move_blackpoint));
  json_object_object_add(
      jroot, "nighttime_move_blackpoint_min_shift_pct",
      json_object_new_double(config->nighttime_move_blackpoint_min_shift_pct));
  json_object_object_add(
      jroot, "nighttime_move_blackpoint_max_shift_pct",
      json_object_new_double(config->nighttime_move_blackpoint_max_shift_pct));
  json_object_object_add(
      jroot, "nighttime_move_blackpoint_dark_threshold",
      json_object_new_double(config->nighttime_move_blackpoint_dark_threshold));
  json_object_object_add(
      jroot, "nighttime_hdr_exposure_factor",
      json_object_new_double(config->nighttime_hdr_exposure_factor));
  json_object_object_add(
      jroot, "nighttime_white_balance_red",
      json_object_new_double(config->nighttime_white_balance_red));
  json_object_object_add(
      jroot, "nighttime_white_balance_blue",
      json_object_new_double(config->nighttime_white_balance_blue));
  json_object_object_add(
      jroot, "nighttime_white_balance_light_protect",
      json_object_new_double(config->nighttime_white_balance_light_protect));
  json_object_object_add(jroot, "nighttime_measure_noise",
                         json_object_new_int(config->nighttime_measure_noise));
  json_object_object_add(
      jroot, "nighttime_measure_noise_radius",
      json_object_new_int(config->nighttime_measure_noise_radius));
  json_object_object_add(
      jroot, "nighttime_multiscale_median_filter",
      json_object_new_int(config->nighttime_multiscale_median_filter));
  json_object_object_add(
      jroot, "nighttime_multiscale_median_filter_max_scale",
      json_object_new_int(
          config->nighttime_multiscale_median_filter_max_scale));
  json_object_object_add(
      jroot, "nighttime_multiscale_median_filter_amount",
      json_object_new_double(
          config->nighttime_multiscale_median_filter_amount));
  json_object_object_add(
      jroot, "measure_brightness_area",
      json_object_new_double(config->measure_brightness_area));
  json_object_object_add(jroot, "target_mean",
                         json_object_new_double(config->target_mean));
  json_object_object_add(jroot, "black_clipping_point",
                         json_object_new_double(config->black_clipping_point));
  json_object_object_add(jroot, "white_clipping_point",
                         json_object_new_double(config->white_clipping_point));
  json_object_object_add(jroot, "hdr", json_object_new_int(config->hdr));
  json_object_object_add(
      jroot, "hdr_clipping_threshold",
      json_object_new_double(config->hdr_clipping_threshold));
  json_object_object_add(
      jroot, "hdr_min_clipped_pixels",
      json_object_new_double(config->hdr_min_clipped_pixels));
  json_object_object_add(jroot, "hdr_y_max_expected",
                         json_object_new_double(config->hdr_y_max_expected));
  json_object_object_add(jroot, "hdr_weight_sigma",
                         json_object_new_double(config->hdr_weight_sigma));
  json_object_object_add(
      jroot, "hdr_weight_clip_factor",
      json_object_new_double(config->hdr_weight_clip_factor));
  json_object_object_add(
      jroot, "hdr_contrast_weight_strength",
      json_object_new_double(config->hdr_contrast_weight_strength));
  json_object_object_add(
      jroot, "hdr_noise_fusion_strength",
      json_object_new_double(config->hdr_noise_fusion_strength));
  json_object_object_add(
      jroot, "hdr_pyramid_levels_override",
      json_object_new_int(config->hdr_pyramid_levels_override));
  json_object_object_add(jroot, "hdr_weight_maps",
                         json_object_new_int(config->hdr_weight_maps));
  json_object_object_add(jroot, "hdr_weight_stats",
                         json_object_new_int(config->hdr_weight_stats));
  json_object_object_add(jroot, "hdr_save_exposures",
                         json_object_new_int(config->hdr_save_exposures));
  json_object_object_add(jroot, "hdrmt", json_object_new_int(config->hdrmt));
  json_object_object_add(jroot, "spektrum_enabled",
                         json_object_new_int(config->spektrum_enabled));
  json_object_object_add(jroot, "spektrum_cx",
                         json_object_new_int(config->spektrum_cx));
  json_object_object_add(jroot, "spektrum_cy",
                         json_object_new_int(config->spektrum_cy));
  json_object_object_add(jroot, "spektrum_radius",
                         json_object_new_int(config->spektrum_radius));
  json_object_object_add(
      jroot, "spektrum_regularization",
      json_object_new_double(config->spektrum_regularization));
  json_object_object_add(jroot, "spektrum_output_path",
                         json_object_new_string(config->spektrum_output_path));
  json_object_object_add(jroot, "autostretch",
                         json_object_new_int(config->autostretch));
  json_object_object_add(jroot, "autostretch_min_val",
                         json_object_new_double(config->autostretch_min_val));
  json_object_object_add(jroot, "autostretch_max_val",
                         json_object_new_double(config->autostretch_max_val));
  json_object_object_add(jroot, "acdnr_filter",
                         json_object_new_int(config->acdnr_filter));
  json_object_object_add(
      jroot, "acdnr_filter_lum_stddev",
      json_object_new_double(config->acdnr_filter_lum_stddev));
  json_object_object_add(
      jroot, "acdnr_filter_lum_amount",
      json_object_new_double(config->acdnr_filter_lum_amount));
  json_object_object_add(
      jroot, "acdnr_filter_lum_iterations",
      json_object_new_int(config->acdnr_filter_lum_iterations));
  json_object_object_add(
      jroot, "acdnr_filter_lum_kernel_size",
      json_object_new_int(config->acdnr_filter_lum_kernel_size));
  json_object_object_add(
      jroot, "acdnr_filter_chrom_stddev",
      json_object_new_double(config->acdnr_filter_chrom_stddev));
  json_object_object_add(
      jroot, "acdnr_filter_chrom_amount",
      json_object_new_double(config->acdnr_filter_chrom_amount));
  json_object_object_add(
      jroot, "acdnr_filter_chrom_iterations",
      json_object_new_int(config->acdnr_filter_chrom_iterations));
  json_object_object_add(
      jroot, "acdnr_filter_chrom_kernel_size",
      json_object_new_int(config->acdnr_filter_chrom_kernel_size));
  json_object_object_add(jroot, "gamma", json_object_new_double(config->gamma));
  json_object_object_add(jroot, "auto_gamma",
                         json_object_new_int(config->auto_gamma));
  json_object_object_add(jroot, "auto_gamma_target",
                         json_object_new_double(config->auto_gamma_target));
  json_object_object_add(jroot, "auto_gamma_min",
                         json_object_new_double(config->auto_gamma_min));
  json_object_object_add(jroot, "auto_gamma_max",
                         json_object_new_double(config->auto_gamma_max));
  json_object_object_add(jroot, "reinhard_tonemap",
                         json_object_new_int(config->reinhard_tonemap));
  json_object_object_add(jroot, "reinhard_white_point",
                         json_object_new_double(config->reinhard_white_point));
  json_object_object_add(jroot, "tone_curve",
                         json_object_new_int(config->tone_curve));
  json_object_object_add(jroot, "tone_curve_shadow_lift",
                         json_object_new_double(config->tone_curve_shadow_lift));
  json_object_object_add(jroot, "tone_curve_midtone_contrast",
                         json_object_new_double(config->tone_curve_midtone_contrast));
  json_object_object_add(jroot, "saturation",
                         json_object_new_double(config->saturation));
  json_object_object_add(jroot, "dehaze_amount",
                         json_object_new_double(config->dehaze_amount));
  json_object_object_add(jroot, "dehaze_estimate",
                         json_object_new_double(config->dehaze_estimate));
  json_object_object_add(jroot, "dehaze_gamma",
                         json_object_new_double(config->dehaze_gamma));
  json_object_object_add(jroot, "dehaze_saturation",
                         json_object_new_double(config->dehaze_saturation));
  json_object_object_add(jroot, "clarity_filter",
                         json_object_new_int(config->clarity_filter));
  json_object_object_add(
      jroot, "clarity_filter_strength",
      json_object_new_double(config->clarity_filter_strength));
  json_object_object_add(jroot, "clarity_filter_radius",
                         json_object_new_int(config->clarity_filter_radius));
  json_object_object_add(
      jroot, "clarity_filter_midtone_width",
      json_object_new_double(config->clarity_filter_midtone_width));
  json_object_object_add(
      jroot, "clarity_filter_preserve_highlights",
      json_object_new_int(config->clarity_filter_preserve_highlights));
  json_object_object_add(jroot, "clarity_filter_mask_mode",
                         json_object_new_int(config->clarity_filter_mask_mode));
  json_object_object_add(
      jroot, "auto_color_calibration_area",
      json_object_new_double(config->auto_color_calibration_area));
  json_object_object_add(
      jroot, "local_histogram_normalization",
      json_object_new_int(config->local_histogram_normalization));
  json_object_object_add(
      jroot, "local_histogram_normalization_kernel_radius",
      json_object_new_double(
          config->local_histogram_normalization_kernel_radius));
  json_object_object_add(
      jroot, "local_histogram_normalization_contrast_limit",
      json_object_new_double(
          config->local_histogram_normalization_contrast_limit));
  json_object_object_add(jroot, "scnr_filter",
                         json_object_new_int(config->scnr_filter));
  json_object_object_add(jroot, "scnr_filter_amount",
                         json_object_new_double(config->scnr_filter_amount));
  json_object_object_add(
      jroot, "scnr_filter_protection",
      json_object_new_string(config->scnr_filter_protection));
  json_object_object_add(jroot, "wavelet_sharpen",
                         json_object_new_int(config->wavelet_sharpen));
  json_object_object_add(
      jroot, "wavelet_sharpen_gain_small",
      json_object_new_double(config->wavelet_sharpen_gain_small));
  json_object_object_add(
      jroot, "wavelet_sharpen_gain_medium",
      json_object_new_double(config->wavelet_sharpen_gain_medium));
  json_object_object_add(
      jroot, "wavelet_sharpen_gain_large",
      json_object_new_double(config->wavelet_sharpen_gain_large));
  json_object_object_add(jroot, "sqm", json_object_new_int(config->sqm));
  json_object_object_add(jroot, "sqm_max_stars",
                         json_object_new_int(config->sqm_max_stars));
  json_object_object_add(jroot, "sqm_radius",
                         json_object_new_int(config->sqm_radius));
  json_object_object_add(jroot, "sqm_center_x",
                         json_object_new_int(config->sqm_center_x));
  json_object_object_add(jroot, "sqm_center_y",
                         json_object_new_int(config->sqm_center_y));
  json_object_object_add(jroot, "sqm_star_sigma",
                         json_object_new_double(config->sqm_star_sigma));
  json_object_object_add(jroot, "sqm_detect_sigma",
                         json_object_new_double(config->sqm_detect_sigma));
  json_object_object_add(jroot, "sqm_max_fwhm",
                         json_object_new_double(config->sqm_max_fwhm));
  json_object_object_add(jroot, "sqm_max_elongation",
                         json_object_new_double(config->sqm_max_elongation));
  json_object_object_add(jroot, "sqm_intercept",
                         json_object_new_double(config->sqm_intercept));
  json_object_object_add(jroot, "sqm_slope",
                         json_object_new_double(config->sqm_slope));
  json_object_object_add(jroot, "debug_raw_exposures",
                         json_object_new_int(config->debug_raw_exposures));
  json_object_object_add(jroot, "camera_monochrome",
                         json_object_new_int(config->camera_monochrome));
  json_object_object_add(jroot, "camera_max_exposure",
                         json_object_new_double(config->camera_max_exposure));
  json_object_object_add(jroot, "camera_min_exposure",
                         json_object_new_double(config->camera_min_exposure));
  json_object_object_add(jroot, "camera_max_gain",
                         json_object_new_int(config->camera_max_gain));
  json_object_object_add(jroot, "camera_min_gain",
                         json_object_new_int(config->camera_min_gain));
  json_object_object_add(jroot, "camera_binning2x2",
                         json_object_new_int(config->camera_binning2x2));
  json_object_object_add(jroot, "camera_cooling",
                         json_object_new_int(config->camera_cooling));
  json_object_object_add(
      jroot, "camera_cooling_temperature",
      json_object_new_int(config->camera_cooling_temperature));
  json_object_object_add(jroot, "camera_r_factor",
                         json_object_new_double(config->camera_r_factor));
  json_object_object_add(jroot, "camera_g_factor",
                         json_object_new_double(config->camera_g_factor));
  json_object_object_add(jroot, "camera_b_factor",
                         json_object_new_double(config->camera_b_factor));
  json_object_object_add(jroot, "camera_debayer",
                         json_object_new_string(config->camera_debayer));
  json_object_object_add(jroot, "jpeg_quality",
                         json_object_new_int(config->jpeg_quality));
  json_object_object_add(jroot, "lens_name",
                         json_object_new_string(config->lens_name));
  json_object_object_add(jroot, "lens_focallength",
                         json_object_new_double(config->lens_focallength));
  json_object_object_add(jroot, "lens_focalratio",
                         json_object_new_double(config->lens_focalratio));
  json_object_object_add(jroot, "image_center_x",
                         json_object_new_int(config->image_center_x));
  json_object_object_add(jroot, "image_center_y",
                         json_object_new_int(config->image_center_y));
  json_object_object_add(jroot, "image_mask_radius",
                         json_object_new_int(config->image_mask_radius));
  json_object_object_add(jroot, "image_horizon_radius",
                         json_object_new_int(config->image_horizon_radius));
  json_object_object_add(jroot, "image_north_angle",
                         json_object_new_double(config->image_north_angle));
  json_object_object_add(jroot, "image_zenith_x",
                         json_object_new_int(config->image_zenith_x));
  json_object_object_add(jroot, "image_zenith_y",
                         json_object_new_int(config->image_zenith_y));
  json_object_object_add(jroot, "focus_radius",
                         json_object_new_int(config->focus_radius));
  json_object_object_add(jroot, "font_path",
                         json_object_new_string(config->font_path));
  json_object_object_add(jroot, "font_size",
                         json_object_new_int(config->font_size));
  json_object_object_add(jroot, "crop_image",
                         json_object_new_int(config->crop_image));
  json_object_object_add(jroot, "crop_width",
                         json_object_new_int(config->crop_width));
  json_object_object_add(jroot, "crop_height",
                         json_object_new_int(config->crop_height));
  json_object_object_add(jroot, "crop_x_offset",
                         json_object_new_int(config->crop_x_offset));
  json_object_object_add(jroot, "crop_y_offset",
                         json_object_new_int(config->crop_y_offset));
  json_object_object_add(jroot, "rotate_image",
                         json_object_new_int(config->rotate_image));
  json_object_object_add(jroot, "thumbnail",
                         json_object_new_int(config->thumbnail));
  json_object_object_add(jroot, "thumbnail_width",
                         json_object_new_int(config->thumbnail_width));
  json_object_object_add(jroot, "panorama",
                         json_object_new_int(config->panorama));
  json_object_object_add(jroot, "panorama_center_x",
                         json_object_new_int(config->panorama_center_x));
  json_object_object_add(jroot, "panorama_center_y",
                         json_object_new_int(config->panorama_center_y));
  json_object_object_add(jroot, "panorama_width",
                         json_object_new_int(config->panorama_width));
  json_object_object_add(jroot, "panorama_height",
                         json_object_new_int(config->panorama_height));
  json_object_object_add(
      jroot, "panorama_horizontal_start",
      json_object_new_int(config->panorama_horizontal_start));
  /* config.json is served unauthenticated through /api/config. The dashboard
   * needs the coordinates (sun/moon altitude, aircraft distances), so they stay
   * - but publish_exact_location=0 rounds them to about 1 km, which is plenty
   * for those calculations and drops house-level precision (review.md, SEC-04). */
  double published_latitude = config->location_latitude;
  double published_longitude = config->location_longitude;
  if (!config->publish_exact_location) {
    published_latitude = round(published_latitude * 100.0) / 100.0;
    published_longitude = round(published_longitude * 100.0) / 100.0;
  }

  json_object_object_add(jroot, "location_latitude",
                         json_object_new_double(published_latitude));
  json_object_object_add(jroot, "location_longitude",
                         json_object_new_double(published_longitude));
  json_object_object_add(jroot, "location_altitude",
                         json_object_new_int(config->location_altitude));
  json_object_object_add(jroot, "location_city",
                         json_object_new_string(config->location_city));
  /* location_owner is a person's name - deliberately not published (SEC-04) */
  json_object_object_add(jroot, "working_directory",
                         json_object_new_string(config->working_directory));
  json_object_object_add(jroot, "image_directory",
                         json_object_new_string(config->image_directory));
  json_object_object_add(jroot, "raw_dump_directory",
                         json_object_new_string(config->raw_dump_directory));
  json_object_object_add(jroot, "video_directory",
                         json_object_new_string(config->video_directory));
  json_object_object_add(jroot, "database_directory",
                         json_object_new_string(config->database_directory));
  json_object_object_add(jroot, "capture_interval",
                         json_object_new_int(config->capture_interval));
  /* Host, port and mount of the INDIGO server describe the internal network
   * and are deliberately not published (SEC-04). The camera name stays: it is
   * hardware information the dashboard shows. */
  json_object_object_add(jroot, "indigocamera",
                         json_object_new_string(config->indigocamera));
  json_object_object_add(jroot, "indigo_capture_timeout",
                         json_object_new_int(config->indigo_capture_timeout));

  const char *json_str =
      json_object_to_json_string_ext(jroot, JSON_C_TO_STRING_SPACED);

  /* config.json sits in the web docroot and is polled by the dashboard -
   * publish it atomically (review.md, REL-05). */
  if (write_text_file_atomic(filename, json_str) != 0) {
    json_object_put(jroot);
    return 3;
  }

  json_object_put(jroot);
  return 0;
}

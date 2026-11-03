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
#ifndef CONFIG_H
#define CONFIG_H

#include <limits.h>

typedef struct {
    int daytime_capture;
    double daytime_startexposure;
    int daytime_startgain;
    double daytime_meanbrightness;
    double daytime_move_blackpoint;
    double daytime_move_blackpoint_min_shift_pct;
    double daytime_move_blackpoint_max_shift_pct;
    double daytime_move_blackpoint_dark_threshold;
    double daytime_hdr_exposure_factor;
    double daytime_white_balance_red;
    double daytime_white_balance_blue;
    double daytime_white_balance_light_protect;

    double nighttime_sunangle;
    double nighttime_startexposure;
    int nighttime_startgain;
    double nighttime_meanbrightness;
    int nighttime_fullmoongain;
    double nighttime_move_blackpoint;
    double nighttime_move_blackpoint_min_shift_pct;
    double nighttime_move_blackpoint_max_shift_pct;
    double nighttime_move_blackpoint_dark_threshold;
    double nighttime_hdr_exposure_factor;
    double nighttime_white_balance_red;
    double nighttime_white_balance_blue;
    double nighttime_white_balance_light_protect;
    int nighttime_measure_noise;
    int nighttime_measure_noise_radius;

    double measure_brightness_area;
    double target_mean;
    double black_clipping_point;
    double white_clipping_point;

    int nighttime_multiscale_median_filter;
    int nighttime_multiscale_median_filter_max_scale;
    double nighttime_multiscale_median_filter_amount;

    int hdr;
    double hdr_clipping_threshold;
    double hdr_min_clipped_pixels;
    double hdr_y_max_expected;
    double hdr_weight_sigma;
    double hdr_weight_clip_factor;

    double hdr_contrast_weight_strength;
    double hdr_noise_fusion_strength;
    int hdr_pyramid_levels_override;
    int hdr_weight_maps;
    int hdr_weight_stats;
    int hdr_save_exposures;
    int hdrmt;

    int   spektrum_enabled;
    int   spektrum_cx;
    int   spektrum_cy;
    int   spektrum_radius;
    double spektrum_regularization;
    char  spektrum_output_path[NAME_MAX + 1];

    int autostretch;
    double autostretch_min_val;
    double autostretch_max_val;

    int acdnr_filter;
    double acdnr_filter_lum_stddev;
    double acdnr_filter_lum_amount;
    int acdnr_filter_lum_iterations;
    int acdnr_filter_lum_kernel_size;
    double acdnr_filter_chrom_stddev;
    double acdnr_filter_chrom_amount;
    int acdnr_filter_chrom_iterations;
    int acdnr_filter_chrom_kernel_size;
    double gamma;
    int auto_gamma;
    double auto_gamma_target;
    double auto_gamma_min;
    double auto_gamma_max;

    int reinhard_tonemap;
    double reinhard_white_point;

    int tone_curve;
    double tone_curve_shadow_lift;
    double tone_curve_midtone_contrast;

    double saturation;

    double dehaze_amount;
    double dehaze_estimate;
    double dehaze_gamma;
    double dehaze_saturation;

    int clarity_filter;
    double clarity_filter_strength;
    int clarity_filter_radius;
    double clarity_filter_midtone_width;
    int clarity_filter_preserve_highlights;
    int clarity_filter_mask_mode;

    double auto_color_calibration_area;

    int scnr_filter;
    double scnr_filter_amount;
    char scnr_filter_protection[100];

    int wavelet_sharpen;
    double wavelet_sharpen_gain_small;
    double wavelet_sharpen_gain_medium;
    double wavelet_sharpen_gain_large;

    int sqm;
    int sqm_max_stars;
    int sqm_radius;
    int sqm_center_x;
    int sqm_center_y;
    double sqm_star_sigma;       /* Gaussian sigma of a star [image px] */
    double sqm_detect_sigma;     /* detection threshold [noise sigma] */
    double sqm_max_fwhm;         /* reject wider sources [image px] */
    double sqm_max_elongation;   /* reject more elongated sources (0..1) */
    double sqm_intercept;
    double sqm_slope;

    int debug_raw_exposures;
    int camera_monochrome;
    double camera_max_exposure;
    double camera_min_exposure;
    int camera_max_gain;
    int camera_min_gain;
    int camera_binning2x2;
    int camera_cooling;
    int camera_cooling_temperature;
    double camera_r_factor;
    double camera_g_factor;
    double camera_b_factor;
    char camera_debayer[50];
    int jpeg_quality;
    char lens_name[100];
    double lens_focallength;
    double lens_focalratio;

    int image_center_x;
    int image_center_y;
    int image_mask_radius;
    int image_horizon_radius;
    double image_north_angle;
    int image_zenith_x;
    int image_zenith_y;
    int focus_radius;


    char font_path[100];
    int font_size;

    int crop_image;
    int crop_width;
    int crop_height;
    int crop_x_offset;
    int crop_y_offset;
    int rotate_image;
    int thumbnail;
    int thumbnail_width;
    int panorama;
    int panorama_center_x;
    int panorama_center_y;
    int panorama_width;
    int panorama_height;
    int panorama_horizontal_start;
    double location_latitude;
    double location_longitude;
    int location_altitude;
    char location_city[100];
    char location_owner[100];
    /* 1 = publish exact coordinates in config.json, 0 = round to 2 decimals */
    int publish_exact_location;

    /* Hook scripts run by the daemon (empty = disabled) */
    char postprocessing_script[NAME_MAX+1];
    char endofnight_script[NAME_MAX+1];

    /* Archive retention (see src/retention.c) */
    int    retention;
    int    retention_max_days;
    double retention_max_size_gb;
    int    retention_keep_days;
    int    retention_dry_run;

    char working_directory[NAME_MAX+1];
    char image_directory[NAME_MAX+1];
    char raw_dump_directory[NAME_MAX+1];
    char video_directory[NAME_MAX+1];
    char database_directory[NAME_MAX+1];
    int capture_interval;
    char indigoservername[100];
    char indigoserver[100];
    int indigoport;
    char indigocamera[100];
    char indigomount[100];
    int indigo_capture_timeout;
} config_t;

/**
 *  Load configuration
 * @param filename: path to the configuration file
 * @param config: pointer to the configuration structure
 * @return: 0 on success, >0 on error
 */
int load_config(const char *filename, config_t *config);

/**
 *  Show configuration
 * @param config: pointer to the configuration structure
 * @return: void
 */
int show_config(config_t *config);

/**
 * @brief Converts the given config_t structure into a JSON file.
 * 
 * @param filename: path to the output JSON file.
 * @param config: pointer to the configuration structure to serialize.
 * @return: 0 on success, >0 on error.
 */
int config_to_json(const char *filename, const config_t *config);

#endif /* CONFIG_H */

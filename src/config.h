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
    int median_brightness;
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
    int hdr_chroma_mode;
    double hdr_contrast_weight_strength;
    int hdr_pyramid_levels_override;
    int hdr_weight_maps;
    int hdr_weight_stats;
    int hdr_save_exposures;
    int hdrmt;

    int autostretch;
    double autostretch_min_val;
    double autostretch_max_val;


    int ambience_color_calibration;

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
    int local_histogram_normalization;
    double local_histogram_normalization_kernel_radius;
    double local_histogram_normalization_contrast_limit;

    int scnr_filter;
    double scnr_filter_amount;
    char scnr_filter_protection[100];

    int wavelet_sharpen;
    double wavelet_sharpen_gain_small;
    double wavelet_sharpen_gain_medium;
    double wavelet_sharpen_gain_large;

    int sqm;
    double sqm_threshold;
    int sqm_max_stars;
    int sqm_radius;
    char sqm_template[500];
    int sqm_star_size;
    double sqm_star_sigma;
    double sqm_intercept;
    double sqm_slope;

    int debug_raw_exposures;
    double camera_max_exposure;
    double camera_min_exposure;
    int camera_max_gain;
    int camera_min_gain;
    int camera_binning2x2;
    int camera_cooling;
    int camera_cooling_temperature;
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
    int focus_center_x;
    int focus_center_y;
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
    double latitude;
    double longitude;
    int altitude;
    char city[100];
    double sunangle;
    char owner[100];
    char image_directory[NAME_MAX+1];
    char video_directory[NAME_MAX+1];
    char database_directory[NAME_MAX+1];
    int weather_sensor;
    char weather_sensor_type[50];
    int weather_sensor_i2c_bus;
    int capture_interval;
    char indigoservername[100];
    char indigoserver[100];
    int indigoport;
    char indigocamera[100];
    char debayer[50];
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

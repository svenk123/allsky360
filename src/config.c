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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <json-c/json.h>

#include "config.h"

int load_config(const char *filename, config_t *config) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return 1;
    }

    // Set default values
    config->daytime_startexposure = 0.05;
    config->daytime_startgain = 0.0;
    config->daytime_move_blackpoint = 0.05;
    config->daytime_hdr_exposure_factor = 12.0;
    config->daytime_white_balance_red = 1.0;
    config->daytime_white_balance_blue = 1.0;
    config->nighttime_startexposure = 1.0;
    config->nighttime_startgain = 100.0;
    config->nighttime_move_blackpoint = 0.05;
    config->nighttime_hdr_exposure_factor = 6.0;
    config->nighttime_white_balance_red = 1.0;
    config->nighttime_white_balance_blue = 1.0;
    config->nighttime_measure_noise = 1.0;
    config->nighttime_measure_noise_radius = 250;
    config->nighttime_measure_noise_cx = 1794;
    config->nighttime_measure_noise_cy = 693;

    config->target_mean = 0.5;
    config->median_brightness = 0;
    config->black_clipping_point = 0.02;
    config->white_clipping_point = 0.98;

    config->multiscale_median_filter = 0;
    config->multiscale_median_filter_max_scale = 2;
    config->multiscale_median_filter_amount = 0.5f;

    config->hdr = 1;
    config->hdr_clipping_threshold = 0.98f;
    config->hdr_y_max_expected = 1.0;
    config->hdr_chroma_mode = 2;
    config->hdr_contrast_weight_strength = 0.5;
    config->hdr_pyramid_levels_override = 6;
    config->hdr_weight_maps = 0;
    config->hdr_weight_stats = 0;
    config->hdr_save_exposures = 1;

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
    config->saturation = 2.8;

    config->dehaze_amount = 0.8;
    config->dehaze_estimate = 0.01;
    config->dehaze_gamma = 1.0;
    config->dehaze_saturation = 1.2;

    config->auto_color_calibration_area = 80.0;

    config->local_histogram_normalization = 1;
    config->local_histogram_normalization_kernel_radius = 64;
    config->local_histogram_normalization_contrast_limit = 1.5;

    config->scnr_filter = 1;
    config->scnr_filter_amount = 1.0f;
    snprintf(config->scnr_filter_protection, sizeof(config->scnr_filter_protection), "average_neutral");

    config->sqm = 0;
    config->sqm_max_stars = 1000;
    config->sqm_threshold = 0.7f;
    config->sqm_radius = 0;
    config->sqm_center_x = 1000;
    config->sqm_center_y = 500;
    config->sqm_template[0] = '\0';
    config->sqm_intercept = 23.0f;
    config->sqm_slope = 1.5f;

    config->debug_pipeline_images = 0;
    config->debug_raw_exposures = 0;

    config->image_center_x = 1000;
    config->image_center_y = 500;
    config->image_mask_radius = 0;
    config->image_horizon_radius = 1000;
    config->image_north_angle = 10.0f;
    config->image_zenith_x = 1000;
    config->image_zenith_y = 500;

    config->camera_max_exposure = 60.0;
    config->camera_min_exposure = 0.000032;
    config->camera_max_gain = 350;
    config->camera_min_gain = 0;
    config->camera_cooling=0;
    config->camera_cooling_temperature = -5;
    config->thumbnail=1;
    config->thumbnail_width=100;

    config->panorama = 0;
    config->panorama_center_x = 1155;
    config->panorama_center_y = 1155;
    config->panorama_width = 2048;
    config->panorama_height = 1090;
    config->panorama_horizontal_start = 90;

    config->altitude=100;
    snprintf(config->image_directory, sizeof(config->image_directory), "/home/allskyuser/images");
    snprintf(config->video_directory, sizeof(config->video_directory), "/home/allskyuser/videos");
    snprintf(config->database_directory, sizeof(config->database_directory), "/home/allskyuser/database");


    config->weather_sensor = 0;
    config->weather_sensor_type[0] = '\0';
    config->weather_sensor_i2c_bus = 0;

    snprintf(config->indigoserver, sizeof(config->indigoserver), "127.0.0.1");
    config->indigoport = 7624;
    snprintf(config->indigocamera, sizeof(config->indigocamera), "Uranus-C @ localhost");
    snprintf(config->debayer, sizeof(config->debayer), "vng");
    config->indigo_capture_timeout = 20;

    char line[256], key[50], value[100];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || strlen(line) < 3) 
	    continue;
        
        // Remove spaces
        char *trim_line = strtok(line, "\n");
        
        if (sscanf(trim_line, "%49[^=] = \"%99[^\"]\"", key, value) == 2 ||
            sscanf(trim_line, "%49[^=] = %99s", key, value) == 2) {
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
            else if (strcmp(key, "daytime_hdr_exposure_factor") == 0)
                config->daytime_hdr_exposure_factor = atof(value);
            else if (strcmp(key, "daytime_white_balance_red") == 0)
                config->daytime_white_balance_red = atof(value);
            else if (strcmp(key, "daytime_white_balance_blue") == 0)
                config->daytime_white_balance_blue = atof(value);

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
            else if (strcmp(key, "nighttime_hdr_exposure_factor") == 0)
                config->nighttime_hdr_exposure_factor = atof(value);
            else if (strcmp(key, "nighttime_white_balance_red") == 0)
                config->nighttime_white_balance_red = atof(value);
            else if (strcmp(key, "nighttime_white_balance_blue") == 0)
                config->nighttime_white_balance_blue = atof(value);
            else if (strcmp(key, "nighttime_measure_noise") == 0)
                config->nighttime_measure_noise = atoi(value);
            else if (strcmp(key, "nighttime_measure_noise_radius") == 0)
                config->nighttime_measure_noise_radius = atoi(value);
            else if (strcmp(key, "nighttime_measure_noise_cx") == 0)
                config->nighttime_measure_noise_cx = atoi(value);
            else if (strcmp(key, "nighttime_measure_noise_cy") == 0)
                config->nighttime_measure_noise_cy = atoi(value);

            else if (strcmp(key, "measure_brightness_area") == 0)
                config->measure_brightness_area = atof(value);
            else if (strcmp(key, "median_brightness") == 0)
                config->median_brightness = atoi(value);

            else if (strcmp(key, "target_mean") == 0)
                config->target_mean = atof(value);
            else if (strcmp(key, "black_clipping_point") == 0)
                config->black_clipping_point = atof(value);
            else if (strcmp(key, "white_clipping_point") == 0)
                config->white_clipping_point = atof(value);

            else if (strcmp(key, "multiscale_median_filter") == 0)
                config->multiscale_median_filter = atoi(value);
            else if (strcmp(key, "multiscale_median_filter_max_scale") == 0)
                config->multiscale_median_filter_max_scale = atoi(value);
            else if (strcmp(key, "multiscale_median_filter_amount") == 0)
                config->multiscale_median_filter_amount = atof(value);

            else if (strcmp(key, "hdr") == 0)
                config->hdr = atoi(value);
            else if (strcmp(key, "hdr_clipping_threshold") == 0)
                config->hdr_clipping_threshold = atof(value);
            else if (strcmp(key, "hdr_y_max_expected") == 0)
                config->hdr_y_max_expected = atof(value);
            else if (strcmp(key, "hdr_chroma_mode") == 0)
                config->hdr_chroma_mode = atoi(value);
            else if (strcmp(key, "hdr_contrast_weight_strength") == 0)
                config->hdr_contrast_weight_strength = atof(value);
            else if (strcmp(key, "hdr_pyramid_levels_override") == 0)
                config->hdr_pyramid_levels_override = atoi(value);            
            else if (strcmp(key, "hdr_weight_maps") == 0)
                config->hdr_weight_maps = atoi(value);
            else if (strcmp(key, "hdr_weight_stats") == 0)
                config->hdr_weight_stats = atoi(value);
            else if (strcmp(key, "hdr_save_exposures") == 0)
                config->hdr_save_exposures = atoi(value);
                
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
                snprintf(config->scnr_filter_protection, sizeof(config->scnr_filter_protection), "%.*s", (int)sizeof(config->scnr_filter_protection)-1, value);

            else if (strcmp(key, "sqm") == 0)
                config->sqm = atoi(value);
            else if (strcmp(key, "sqm_max_stars") == 0)
                config->sqm_max_stars = atoi(value);
            else if (strcmp(key, "sqm_threshold") == 0)
                config->sqm_threshold = atof(value);
            else if (strcmp(key, "sqm_radius") == 0)
                config->sqm_radius = atoi(value);
            else if (strcmp(key, "sqm_center_x") == 0)
                config->sqm_center_x = atoi(value);
            else if (strcmp(key, "sqm_center_y") == 0)
                config->sqm_center_y = atoi(value);
            else if (strcmp(key, "sqm_template") == 0)
                snprintf(config->sqm_template, sizeof(config->sqm_template), "%.*s", (int)sizeof(config->sqm_template)-1, value);
            else if (strcmp(key, "sqm_intercept") == 0)
                config->sqm_intercept = atof(value);
            else if (strcmp(key, "sqm_slope") == 0)
                config->sqm_slope = atof(value);


            else if (strcmp(key, "debug_pipeline_images") == 0)
                config->debug_pipeline_images = atoi(value);
            else if (strcmp(key, "debug_raw_exposures") == 0)
                config->debug_raw_exposures = atoi(value);

            else if (strcmp(key, "camera_max_exposure") == 0)
                config->camera_max_exposure = atof(value);
            else if (strcmp(key, "camera_min_exposure") == 0)
                config->camera_min_exposure = atof(value);
            else if (strcmp(key, "camera_max_gain") == 0)
                config->camera_max_gain = atoi(value);
            else if (strcmp(key, "camera_min_gain") == 0)
                config->camera_min_gain = atoi(value);
            else if (strcmp(key, "camera_binning") == 0)
                snprintf(config->camera_binning, sizeof(config->camera_binning), "%.*s", (int)sizeof(config->camera_binning)-1, value);
            else if (strcmp(key, "camera_cooling") == 0)
                config->camera_cooling = atoi(value);
            else if (strcmp(key, "camera_cooling_temperature") == 0)
                config->camera_cooling_temperature = atoi(value);

            else if (strcmp(key, "png_output") == 0)
                config->png_output = atoi(value);
            else if (strcmp(key, "png_compression") == 0)
                config->png_compression = atoi(value);
            else if (strcmp(key, "jpeg_quality") == 0)
                config->jpeg_quality = atoi(value);
            else if (strcmp(key, "lens_name") == 0)
                snprintf(config->lens_name, sizeof(config->lens_name), "%.*s", (int)sizeof(config->lens_name)-1, value);
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

            else if (strcmp(key, "latitude") == 0)
                config->latitude = atof(value);
            else if (strcmp(key, "longitude") == 0)
                config->longitude = atof(value);
            else if (strcmp(key, "altitude") == 0)
                config->altitude = atoi(value);
            else if (strcmp(key, "city") == 0)
                snprintf(config->city, sizeof(config->city), "%.*s", (int)sizeof(config->city)-1, value);
            else if (strcmp(key, "sunangle") == 0)
                config->sunangle = atof(value);
            else if (strcmp(key, "owner") == 0)
                snprintf(config->owner, sizeof(config->owner), "%.*s", (int)sizeof(config->owner)-1, value);
            else if (strcmp(key, "image_directory") == 0)
                snprintf(config->image_directory, sizeof(config->image_directory), "%.*s", (int)sizeof(config->image_directory)-1, value);
            else if (strcmp(key, "video_directory") == 0)
                snprintf(config->video_directory, sizeof(config->video_directory), "%.*s", (int)sizeof(config->video_directory)-1, value);
            else if (strcmp(key, "database_directory") == 0)
                snprintf(config->database_directory, sizeof(config->database_directory), "%.*s", (int)sizeof(config->database_directory)-1, value);

            else if (strcmp(key, "weather_sensor") == 0)
                config->weather_sensor = atoi(value);
            else if (strcmp(key, "weather_sensor_type") == 0)
                snprintf(config->weather_sensor_type, sizeof(config->weather_sensor_type), "%.*s", (int)sizeof(config->weather_sensor_type)-1, value);

            else if (strcmp(key, "capture_interval") == 0)
                config->capture_interval = atoi(value);
            else if (strcmp(key, "indigo_server") == 0)
                snprintf(config->indigoserver, sizeof(config->indigoserver), "%.*s", (int)sizeof(config->indigoserver)-1, value);
            else if (strcmp(key, "indigo_servername") == 0)
                snprintf(config->indigoservername, sizeof(config->indigoservername), "%.*s", (int)sizeof(config->indigoservername)-1, value);
            else if (strcmp(key, "indigo_port") == 0)
                config->indigoport = atoi(value);
            else if (strcmp(key, "indigo_camera") == 0)
                snprintf(config->indigocamera, sizeof(config->indigocamera), "%.*s", (int)sizeof(config->indigocamera)-1, value);
            else if (strcmp(key, "debayer") == 0)
                snprintf(config->debayer, sizeof(config->debayer), "%.*s", (int)sizeof(config->debayer)-1, value);
            else if (strcmp(key, "indigo_capture_timeout") == 0)
                config->indigo_capture_timeout = atoi(value);

        }
    }

    fclose(file);

    return 0;
}

int show_config(config_t *config) {
    printf("\nLoaded configuration:\n");
    printf("daytime_capture: %d (%s)\n", config->daytime_capture, config->daytime_capture == 1 ? "yes" : "no");
    printf("daytime_startexposure: %.6f\n", config->daytime_startexposure);
    printf("daytime_startgain: %d\n", config->daytime_startgain);
    printf("daytime_meanbrightness: %.2f\n", config->daytime_meanbrightness);
    printf("daytime_move_blackpoint: %.2f\n", config->daytime_move_blackpoint);
    printf("daytime_hdr_exposure_factor: %.1f\n", config->daytime_hdr_exposure_factor);
    printf("daytime_white_balance_red: %.1f\n", config->daytime_white_balance_red);
    printf("daytime_white_balance_blue: %.1f\n", config->daytime_white_balance_blue);

    printf("nighttime_startexposure: %.6f\n", config->nighttime_startexposure);
    printf("nighttime_startgain: %d\n", config->nighttime_startgain);
    printf("nighttime_meanbrightness: %.2f\n", config->nighttime_meanbrightness);
    printf("nighttime_fullmoongain: %d\n", config->nighttime_fullmoongain);
    printf("nighttime_move_blackpoint: %.2f\n", config->nighttime_move_blackpoint);
    printf("nighttime_hdr_exposure_factor: %.1f\n", config->nighttime_hdr_exposure_factor);
    printf("nighttime_white_balance_red: %.1f\n", config->nighttime_white_balance_red);
    printf("nighttime_white_balance_blue: %.1f\n", config->nighttime_white_balance_blue);
    printf("nighttime_measure_noise: %d (%s)\n", config->nighttime_measure_noise, config->nighttime_measure_noise == 1 ? "yes": "no");
    printf("nighttime_measure_noise_radius: %dpx\n", config->nighttime_measure_noise_radius);
    printf("nighttime_measure_noise_cx: %d\n", config->nighttime_measure_noise_cx);
    printf("nighttime_measure_noise_cy: %d\n", config->nighttime_measure_noise_cy);

    printf("measure_brightness_area: %.1f\n", config->measure_brightness_area);
    printf("median_brightness: %d (%s)\n", config->median_brightness, config->median_brightness == 1 ? "yes": "no");
    printf("target_mean: %.2f\n", config->target_mean);
    printf("black_clipping_point: %.2f\n", config->black_clipping_point);
    printf("white_clipping_point: %.2f\n", config->white_clipping_point);

    printf("multiscale_median_filter: %d (%s)\n", config->multiscale_median_filter, config->multiscale_median_filter == 1 ? "yes" : "no");
    printf("multiscale_median_filter_max_scale: %d\n", config->multiscale_median_filter_max_scale);
    printf("multiscale_median_filter_amount: %.1f\n", config->multiscale_median_filter_amount);

    printf("hdr: %d (%s)\n", config->hdr, config->hdr == 1 ? "yes" : "no");
    printf("hdr_clipping_threshold: %.2f\n", config->hdr_clipping_threshold);
    printf("hdr_y_max_expected: %.1f\n", config->hdr_y_max_expected);
    printf("hdr_chroma_mode: %d\n", config->hdr_chroma_mode);
    printf("hdr_contrast_weight_strength: %.1f\n", config->hdr_contrast_weight_strength);
    printf("hdr_pyramid_levels_override: %d\n", config->hdr_pyramid_levels_override);
    printf("hdr_weight_maps: %d (%s)\n", config->hdr_weight_maps, config->hdr_weight_maps == 1 ? "yes" : "no");
    printf("hdr_weight_stats: %d (%s)\n", config->hdr_weight_stats, config->hdr_weight_stats == 1 ? "yes" : "no");
    printf("hdr_save_exposures: %d (%s)\n", config->hdr_save_exposures, config->hdr_save_exposures == 1 ? "yes" : "no");

    printf("acdnr_filter: %d (%s)\n", config->acdnr_filter, config->acdnr_filter == 1 ? "yes" : "no");
    printf("acdnr_filter_lum_stddev: %.4f\n", config->acdnr_filter_lum_stddev);
    printf("acdnr_filter_lum_amount: %.4f\n", config->acdnr_filter_lum_amount);
    printf("acdnr_filter_lum_iterations: %d\n", config->acdnr_filter_lum_iterations);
    printf("acdnr_filter_lum_kernel_size: (%dx%d)\n", config->acdnr_filter_lum_kernel_size, config->acdnr_filter_lum_kernel_size);

    printf("acdnr_filter_chrom_stddev: %.4f\n", config->acdnr_filter_chrom_stddev);
    printf("acdnr_filter_chrom_amount: %.4f\n", config->acdnr_filter_chrom_amount);
    printf("acdnr_filter_chrom_iterations: %d\n", config->acdnr_filter_chrom_iterations);
    printf("acdnr_filter_chrom_kernel_size: (%dx%d)\n", config->acdnr_filter_chrom_kernel_size, config->acdnr_filter_chrom_kernel_size);

    printf("gamma: %.1f\n", config->gamma);
    printf("saturation: %.1f\n", config->saturation);

    printf("dehaze_amount: %.1f\n", config->dehaze_amount);
    printf("dehaze_estimate: %.2f\n", config->dehaze_estimate);
    printf("dehaze_gamma: %.1f\n", config->dehaze_gamma);
    printf("dehaze_saturation: %.1f\n", config->dehaze_saturation);

    printf("auto_color_calibration_area: %.1f%s\n", config->auto_color_calibration_area, config->auto_color_calibration_area == 0.0 ? " (off)" : " ");

    printf("local_histogram_normalization: %d (%s)\n", config->local_histogram_normalization, config->local_histogram_normalization == 1 ? "yes" : "no");
    printf("local_histogram_normalization_kernel_radius: %.1f\n", config->local_histogram_normalization_kernel_radius);
    printf("local_histogram_normalization_contrast_limit: %.1f\n", config->local_histogram_normalization_contrast_limit);

    printf("scnr_filter: %d (%s)\n", config->scnr_filter, config->scnr_filter == 1 ? "yes" : "no");
    printf("scnr_filter_amount: %.1f\n", config->scnr_filter_amount);
    printf("scnr_filter_protection: %s\n", config->scnr_filter_protection);

    printf("sqm: %d (%s)\n", config->sqm, config->sqm == 1 ? "yes" : "no");
    printf("sqm_threshold: %.1f\n", config->sqm_threshold);
    printf("sqm_max_stars: %d\n", config->sqm_max_stars);
    printf("sqm_radius: %dpx\n", config->sqm_radius);
    printf("sqm_center_x: %d\n", config->sqm_center_x);
    printf("sqm_center_y: %d\n", config->sqm_center_y);
    printf("sqm_template: %s\n", config->sqm_template);
    printf("sqm_intercept: %.1f\n", config->sqm_intercept);
    printf("sqm_slope: %.1f\n", config->sqm_slope);

    printf("debug_pipeline_images: %d (%s)\n", config->debug_pipeline_images, config->debug_pipeline_images == 1 ? "yes" : "no");
    printf("debug_raw_exposures: %d (%s)\n", config->debug_raw_exposures, config->debug_raw_exposures == 1 ? "yes" : "no");

    printf("camera_max_exposure: %.6fs\n", config->camera_max_exposure);
    printf("camera_min_exposure: %.6fs\n", config->camera_min_exposure);
    printf("camera_max_gain: %d\n", config->camera_max_gain);
    printf("camera_min_gain: %d\n", config->camera_min_gain);
    printf("camera_binning: %s\n", config->camera_binning);
    printf("camera_cooling: %d (%s)\n", config->camera_cooling, config->camera_cooling == 1 ? "yes" : "no");
    printf("camera_cooling_temperature: %d °C\n", config->camera_cooling_temperature);
    printf("png_output: %d (%s)\n", config->png_output, config->png_output == 1 ? "yes" : "no");
    printf("png_compression: %d\n", config->png_compression);
    printf("jpeg_quality: %d%%\n", config->jpeg_quality);
    printf("lensname: %s\n", config->lens_name);
    printf("lensfocallength: %.2f\n", config->lens_focallength);
    printf("lensfocalratio: %.2f\n", config->lens_focalratio);

    printf("image_center_x: %d\n", config->image_center_x);
    printf("image_center_y: %d\n", config->image_center_y);
    printf("image_mask_radius: %d px (%s)\n", config->image_mask_radius, config->image_mask_radius > 0 ? "enabled" : "disabled");
    printf("image_horizon_radius: %d px\n", config->image_horizon_radius);
    printf("image_north_angle: %.1f degrees\n", config->image_north_angle);
    printf("image_zenith_x: %d\n", config->image_zenith_x);
    printf("image_zenith_y: %d\n", config->image_zenith_y);
    printf("crop_image: %d\n", config->crop_image);
    printf("crop_width: %d px\n", config->crop_width);
    printf("crop_height: %d px\n", config->crop_height);
    printf("crop_x_offset: %d px\n", config->crop_x_offset);
    printf("crop_y_offset: %d px\n", config->crop_y_offset);

    printf("rotate_image: %d\n", config->rotate_image);

    printf("thumbnail: %d (%s)\n", config->thumbnail, config->thumbnail == 1 ? "yes" : "no");
    printf("thumbnail_width: %dpx\n", config->thumbnail_width);

    printf("panorama: %d (%s)\n", config->panorama, config->panorama == 1 ? "yes" : "no");
    printf("panorama_center_x: %d\n", config->panorama_center_x);
    printf("panorama_center_y: %d\n", config->panorama_center_y);
    printf("panorama_width: %d\n", config->panorama_width);
    printf("panorama_height: %d\n", config->panorama_height);
    printf("panorama_horizontal_start: %d deg.\n", config->panorama_horizontal_start);

    printf("latitude: %.6f\n", config->latitude);
    printf("longitude: %.6f\n", config->longitude);
    printf("altitude: %dm\n", config->altitude);
    printf("city: %s\n", config->city);
    printf("sunangle: %.2f\n", config->sunangle);
    printf("owner: %s\n", config->owner);
    printf("image_directory: %s\n", config->image_directory);
    printf("video_directory: %s\n", config->video_directory);
    printf("database_directory: %s\n", config->database_directory);

    printf("weather_sensor: %d (%s)\n", config->weather_sensor, config->weather_sensor == 1 ? "yes" : "no");
    printf("weather_sensor_type: %s\n", config->weather_sensor_type);
    printf("weather_sensor_i2c_bus: %d\n", config->weather_sensor_i2c_bus);

    printf("capture_interval: %d\n", config->capture_interval);
    printf("indigo_server: %s\n", config->indigoserver);
    printf("indigo_servername: %s\n", config->indigoservername);
    printf("indigo_port: %d\n", config->indigoport);
    printf("indigo_camera: %s\n", config->indigocamera);
    printf("debayer: %s\n", config->debayer);
    printf("indigo_capture_timeout: %ds\n", config->indigo_capture_timeout);

    return 0;
}

int config_to_json(const char *filename, const config_t *config) {
    if (!filename || !config) return 1;

    FILE *file = fopen(filename, "w");
    if (!file) return 2;

    struct json_object *jroot = json_object_new_object();

    // Generated input from generate_config_to_json.py
json_object_object_add(jroot, "daytime_capture", json_object_new_int(config->daytime_capture));
json_object_object_add(jroot, "daytime_startexposure", json_object_new_double(config->daytime_startexposure));
json_object_object_add(jroot, "daytime_startgain", json_object_new_int(config->daytime_startgain));
json_object_object_add(jroot, "daytime_meanbrightness", json_object_new_double(config->daytime_meanbrightness));
json_object_object_add(jroot, "daytime_move_blackpoint", json_object_new_double(config->daytime_move_blackpoint));
json_object_object_add(jroot, "daytime_hdr_exposure_factor", json_object_new_double(config->daytime_hdr_exposure_factor));
json_object_object_add(jroot, "daytime_white_balance_red", json_object_new_double(config->daytime_white_balance_red));
json_object_object_add(jroot, "daytime_white_balance_blue", json_object_new_double(config->daytime_white_balance_blue));
json_object_object_add(jroot, "nighttime_startexposure", json_object_new_double(config->nighttime_startexposure));
json_object_object_add(jroot, "nighttime_startgain", json_object_new_int(config->nighttime_startgain));
json_object_object_add(jroot, "nighttime_meanbrightness", json_object_new_double(config->nighttime_meanbrightness));
json_object_object_add(jroot, "nighttime_fullmoongain", json_object_new_int(config->nighttime_fullmoongain));
json_object_object_add(jroot, "nighttime_move_blackpoint", json_object_new_double(config->nighttime_move_blackpoint));
json_object_object_add(jroot, "nighttime_hdr_exposure_factor", json_object_new_double(config->nighttime_hdr_exposure_factor));
json_object_object_add(jroot, "nighttime_white_balance_red", json_object_new_double(config->nighttime_white_balance_red));
json_object_object_add(jroot, "nighttime_white_balance_blue", json_object_new_double(config->nighttime_white_balance_blue));
json_object_object_add(jroot, "nighttime_measure_noise", json_object_new_int(config->nighttime_measure_noise));
json_object_object_add(jroot, "nighttime_measure_noise_radius", json_object_new_int(config->nighttime_measure_noise_radius));
json_object_object_add(jroot, "nighttime_measure_noise_cx", json_object_new_int(config->nighttime_measure_noise_cx));
json_object_object_add(jroot, "nighttime_measure_noise_cy", json_object_new_int(config->nighttime_measure_noise_cy));
json_object_object_add(jroot, "measure_brightness_area", json_object_new_double(config->measure_brightness_area));
json_object_object_add(jroot, "median_brightness", json_object_new_int(config->median_brightness));
json_object_object_add(jroot, "target_mean", json_object_new_double(config->target_mean));
json_object_object_add(jroot, "black_clipping_point", json_object_new_double(config->black_clipping_point));
json_object_object_add(jroot, "white_clipping_point", json_object_new_double(config->white_clipping_point));
json_object_object_add(jroot, "multiscale_median_filter", json_object_new_int(config->multiscale_median_filter));
json_object_object_add(jroot, "multiscale_median_filter_max_scale", json_object_new_int(config->multiscale_median_filter_max_scale));
json_object_object_add(jroot, "multiscale_median_filter_amount", json_object_new_double(config->multiscale_median_filter_amount));
json_object_object_add(jroot, "hdr", json_object_new_int(config->hdr));
json_object_object_add(jroot, "hdr_clipping_threshold", json_object_new_double(config->hdr_clipping_threshold));
json_object_object_add(jroot, "hdr_y_max_expected", json_object_new_double(config->hdr_y_max_expected));
json_object_object_add(jroot, "hdr_chroma_mode", json_object_new_int(config->hdr_chroma_mode));
json_object_object_add(jroot, "hdr_contrast_weight_strength", json_object_new_double(config->hdr_contrast_weight_strength));
json_object_object_add(jroot, "hdr_pyramid_levels_override", json_object_new_int(config->hdr_pyramid_levels_override));
json_object_object_add(jroot, "hdr_weight_maps", json_object_new_int(config->hdr_weight_maps));
json_object_object_add(jroot, "hdr_weight_stats", json_object_new_int(config->hdr_weight_stats));
json_object_object_add(jroot, "hdr_save_exposures", json_object_new_int(config->hdr_save_exposures));
json_object_object_add(jroot, "acdnr_filter", json_object_new_int(config->acdnr_filter));
json_object_object_add(jroot, "acdnr_filter_lum_stddev", json_object_new_double(config->acdnr_filter_lum_stddev));
json_object_object_add(jroot, "acdnr_filter_lum_amount", json_object_new_double(config->acdnr_filter_lum_amount));
json_object_object_add(jroot, "acdnr_filter_lum_iterations", json_object_new_int(config->acdnr_filter_lum_iterations));
json_object_object_add(jroot, "acdnr_filter_lum_kernel_size", json_object_new_int(config->acdnr_filter_lum_kernel_size));
json_object_object_add(jroot, "acdnr_filter_chrom_stddev", json_object_new_double(config->acdnr_filter_chrom_stddev));
json_object_object_add(jroot, "acdnr_filter_chrom_amount", json_object_new_double(config->acdnr_filter_chrom_amount));
json_object_object_add(jroot, "acdnr_filter_chrom_iterations", json_object_new_int(config->acdnr_filter_chrom_iterations));
json_object_object_add(jroot, "acdnr_filter_chrom_kernel_size", json_object_new_int(config->acdnr_filter_chrom_kernel_size));
json_object_object_add(jroot, "gamma", json_object_new_double(config->gamma));
json_object_object_add(jroot, "saturation", json_object_new_double(config->saturation));
json_object_object_add(jroot, "dehaze_amount", json_object_new_double(config->dehaze_amount));
json_object_object_add(jroot, "dehaze_estimate", json_object_new_double(config->dehaze_estimate));
json_object_object_add(jroot, "dehaze_gamma", json_object_new_double(config->dehaze_gamma));
json_object_object_add(jroot, "dehaze_saturation", json_object_new_double(config->dehaze_saturation));
json_object_object_add(jroot, "auto_color_calibration_area", json_object_new_double(config->auto_color_calibration_area));
json_object_object_add(jroot, "local_histogram_normalization", json_object_new_int(config->local_histogram_normalization));
json_object_object_add(jroot, "local_histogram_normalization_kernel_radius", json_object_new_double(config->local_histogram_normalization_kernel_radius));
json_object_object_add(jroot, "local_histogram_normalization_contrast_limit", json_object_new_double(config->local_histogram_normalization_contrast_limit));
json_object_object_add(jroot, "scnr_filter", json_object_new_int(config->scnr_filter));
json_object_object_add(jroot, "scnr_filter_amount", json_object_new_double(config->scnr_filter_amount));
json_object_object_add(jroot, "sqm", json_object_new_int(config->sqm));
json_object_object_add(jroot, "sqm_threshold", json_object_new_double(config->sqm_threshold));
json_object_object_add(jroot, "sqm_max_stars", json_object_new_int(config->sqm_max_stars));
json_object_object_add(jroot, "sqm_radius", json_object_new_int(config->sqm_radius));
json_object_object_add(jroot, "sqm_center_x", json_object_new_int(config->sqm_center_x));
json_object_object_add(jroot, "sqm_center_y", json_object_new_int(config->sqm_center_y));
json_object_object_add(jroot, "sqm_intercept", json_object_new_double(config->sqm_intercept));
json_object_object_add(jroot, "sqm_slope", json_object_new_double(config->sqm_slope));
json_object_object_add(jroot, "debug_pipeline_images", json_object_new_int(config->debug_pipeline_images));
json_object_object_add(jroot, "debug_raw_exposures", json_object_new_int(config->debug_raw_exposures));
json_object_object_add(jroot, "camera_max_exposure", json_object_new_double(config->camera_max_exposure));
json_object_object_add(jroot, "camera_min_exposure", json_object_new_double(config->camera_min_exposure));
json_object_object_add(jroot, "camera_max_gain", json_object_new_int(config->camera_max_gain));
json_object_object_add(jroot, "camera_min_gain", json_object_new_int(config->camera_min_gain));
json_object_object_add(jroot, "camera_cooling", json_object_new_int(config->camera_cooling));
json_object_object_add(jroot, "camera_cooling_temperature", json_object_new_int(config->camera_cooling_temperature));
json_object_object_add(jroot, "png_output", json_object_new_int(config->png_output));
json_object_object_add(jroot, "png_compression", json_object_new_int(config->png_compression));
json_object_object_add(jroot, "jpeg_quality", json_object_new_int(config->jpeg_quality));
json_object_object_add(jroot, "lens_focallength", json_object_new_double(config->lens_focallength));
json_object_object_add(jroot, "lens_focalratio", json_object_new_double(config->lens_focalratio));
json_object_object_add(jroot, "image_center_x", json_object_new_int(config->image_center_x));
json_object_object_add(jroot, "image_center_y", json_object_new_int(config->image_center_y));
json_object_object_add(jroot, "image_mask_radius", json_object_new_int(config->image_mask_radius));
json_object_object_add(jroot, "image_horizon_radius", json_object_new_int(config->image_horizon_radius));
json_object_object_add(jroot, "image_north_angle", json_object_new_double(config->image_north_angle));
json_object_object_add(jroot, "image_zenith_x", json_object_new_int(config->image_zenith_x));
json_object_object_add(jroot, "image_zenith_y", json_object_new_int(config->image_zenith_y));
json_object_object_add(jroot, "crop_image", json_object_new_int(config->crop_image));
json_object_object_add(jroot, "crop_width", json_object_new_int(config->crop_width));
json_object_object_add(jroot, "crop_height", json_object_new_int(config->crop_height));
json_object_object_add(jroot, "crop_x_offset", json_object_new_int(config->crop_x_offset));
json_object_object_add(jroot, "crop_y_offset", json_object_new_int(config->crop_y_offset));
json_object_object_add(jroot, "rotate_image", json_object_new_int(config->rotate_image));
json_object_object_add(jroot, "thumbnail", json_object_new_int(config->thumbnail));
json_object_object_add(jroot, "thumbnail_width", json_object_new_int(config->thumbnail_width));
json_object_object_add(jroot, "panorama", json_object_new_int(config->panorama));
json_object_object_add(jroot, "panorama_center_x", json_object_new_int(config->panorama_center_x));
json_object_object_add(jroot, "panorama_center_y", json_object_new_int(config->panorama_center_y));
json_object_object_add(jroot, "panorama_width", json_object_new_int(config->panorama_width));
json_object_object_add(jroot, "panorama_height", json_object_new_int(config->panorama_height));
json_object_object_add(jroot, "panorama_horizontal_start", json_object_new_int(config->panorama_horizontal_start));
json_object_object_add(jroot, "latitude", json_object_new_double(config->latitude));
json_object_object_add(jroot, "longitude", json_object_new_double(config->longitude));
json_object_object_add(jroot, "altitude", json_object_new_int(config->altitude));
json_object_object_add(jroot, "sunangle", json_object_new_double(config->sunangle));
json_object_object_add(jroot, "weather_sensor", json_object_new_int(config->weather_sensor));
json_object_object_add(jroot, "weather_sensor_i2c_bus", json_object_new_int(config->weather_sensor_i2c_bus));
json_object_object_add(jroot, "capture_interval", json_object_new_int(config->capture_interval));
json_object_object_add(jroot, "indigoport", json_object_new_int(config->indigoport));
json_object_object_add(jroot, "indigo_capture_timeout", json_object_new_int(config->indigo_capture_timeout));

    const char *json_str = json_object_to_json_string_ext(jroot, JSON_C_TO_STRING_SPACED);
    //const char *json_str = json_object_to_json_string_ext(jroot, JSON_C_TO_STRING_PRETTY);
    if (fprintf(file, "%s\n", json_str) < 0) {
        fclose(file);
        json_object_put(jroot);
        return 3;
    }

    fclose(file);
    json_object_put(jroot);
    return 0;
}

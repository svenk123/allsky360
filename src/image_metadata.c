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
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "image_metadata.h"

static double round_to_precision(double value, int precision) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.*f", precision, value);  // Round to precision
    return strtod(buffer, NULL);
}

int save_image_metadata_json(const char *image_filename, const image_metadata_t *metadata) {
    if (!metadata) 
	return 1;

    // Create json object
    struct json_object *root = json_object_new_object();
    if (!root) {
        fprintf(stderr, "ERROR: Can not create JSON object (root)!\n");
        return 1;
    }

    // Fill json data
    json_object_object_add(root, "timestamp", json_object_new_int(metadata->timestamp));
    json_object_object_add(root, "timezone_offset", json_object_new_int(metadata->timezone_offset));
    json_object_object_add(root, "width", json_object_new_int(metadata->width));
    json_object_object_add(root, "height", json_object_new_int(metadata->height));
    json_object_object_add(root, "exposure_t0", json_object_new_double(round_to_precision(metadata->exposure_t0, 6)));
    json_object_object_add(root, "exposure_t1", json_object_new_double(round_to_precision(metadata->exposure_t1, 6)));
    json_object_object_add(root, "exposure_t2", json_object_new_double(round_to_precision(metadata->exposure_t2, 6)));
    json_object_object_add(root, "exposure_t3", json_object_new_double(round_to_precision(metadata->exposure_t3, 6)));
    json_object_object_add(root, "exposure_t4", json_object_new_double(round_to_precision(metadata->exposure_t4, 6)));
    json_object_object_add(root, "sigma_noise_t0", json_object_new_double(round_to_precision(metadata->sigma_noise_t0, 6)));
    json_object_object_add(root, "sigma_noise_t1", json_object_new_double(round_to_precision(metadata->sigma_noise_t1, 6)));
    json_object_object_add(root, "sigma_noise_t2", json_object_new_double(round_to_precision(metadata->sigma_noise_t2, 6)));
    json_object_object_add(root, "sigma_noise_t3", json_object_new_double(round_to_precision(metadata->sigma_noise_t3, 6)));
    json_object_object_add(root, "sigma_noise_t4", json_object_new_double(round_to_precision(metadata->sigma_noise_t4, 6)));
    json_object_object_add(root, "gain", json_object_new_double(round_to_precision(metadata->gain, 0)));
    json_object_object_add(root, "focus", json_object_new_double(round_to_precision(metadata->focus, 2)));
    json_object_object_add(root, "capture_interval", json_object_new_int(metadata->capture_interval));
    json_object_object_add(root, "night_mode", json_object_new_int(metadata->night_mode));
    json_object_object_add(root, "hdr", json_object_new_int(metadata->hdr));
    json_object_object_add(root, "sensor_temperature", json_object_new_double(round_to_precision(metadata->sensor_temperature, 1)));
    json_object_object_add(root, "temperature", json_object_new_double(round_to_precision(metadata->temperature, 1)));
    json_object_object_add(root, "humidity", json_object_new_double(round_to_precision(metadata->humidity, 0)));
    json_object_object_add(root, "pressure", json_object_new_double(round_to_precision(metadata->pressure, 2)));
    json_object_object_add(root, "mean_brightness", json_object_new_double(round_to_precision(metadata->mean_brightness, 2)));
    json_object_object_add(root, "target_brightness", json_object_new_double(round_to_precision(metadata->target_brightness, 2)));
    json_object_object_add(root, "stars", json_object_new_int(metadata->stars));
    json_object_object_add(root, "sqm", json_object_new_double(round_to_precision(metadata->sqm, 1)));
    json_object_object_add(root, "sun_altitude", json_object_new_double(round_to_precision(metadata->sun_altitude, 1)));
    json_object_object_add(root, "moon_altitude", json_object_new_double(round_to_precision(metadata->moon_altitude, 1)));
    json_object_object_add(root, "moon_phase_percentage", json_object_new_double(round_to_precision(metadata->moon_phase_percentage, 0)));

    char json_filename[1024];
    strncpy(json_filename, image_filename, sizeof(json_filename) - 5);
    json_filename[sizeof(json_filename) - 5] = '\0';
    char *ext = strrchr(json_filename, '.');
    if (ext) {
        strcpy(ext, ".json");
    } else {
        strcat(json_filename, ".json");
    }

    // Save json file
    FILE *file = fopen(json_filename, "w");
    if (!file) {
        fprintf(stderr, "ERROR: Can not write image %s!\n", json_filename);
        json_object_put(root);
        return 1;
    }

    fprintf(file, "%s\n", json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY));
    fclose(file);

    printf("JSON meta data file saved: %s\n", json_filename);
    json_object_put(root);

    return 0;
}

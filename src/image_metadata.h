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
#ifndef IMAGE_METADATA_H
#define IMAGE_METADATA_H

#include <json-c/json.h>

typedef struct {
    int timestamp;
    int timezone_offset;
    int width;
    int height;
    double exposure_t0;
    double exposure_t1;
    double exposure_t2;
    double exposure_t3;
    double exposure_t4;
    double sigma_noise_t0;
    double sigma_noise_t1;
    double sigma_noise_t2;
    double sigma_noise_t3;
    double sigma_noise_t4;
    double gain;
    double focus;
    int capture_interval;
    int night_mode;
    int hdr;
    double sensor_temperature;
    double temperature;
    double humidity;
    double pressure;
    double mean_brightness;
    double target_brightness;
    int stars;
    int sqm;
    double sun_altitude;
    double moon_altitude;
    double moon_phase_percentage;
} image_metadata_t;

/**
 * Saves image metadata as a JSON file with the same base filename as the image.
 * The JSON file will contain various metadata fields such as timestamp, exposures, gain,
 * sensor readings, and astronomical data.
 *
 * @param image_filename: filename of the associated image (used to derive the JSON filename).
 * @param metadata: pointer to the image metadata structure to save.
 * @return: 0 on success, 1 on error (invalid parameters, file I/O error, etc.).
 */
int save_image_metadata_json(const char *image_filename, const image_metadata_t *metadata);

#endif // IMAGE_METADATA_H

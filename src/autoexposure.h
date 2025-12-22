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
#ifndef AUTOEXPOSURE_H
#define AUTOEXPOSURE_H

#define CHANNELS	4

/**
 * Automatic exposure control calculates the exposure time and gain value based on image brightness.
 * 
 * @param config     Pointer to the configuration
 * @param is_night   Indicates whether it's day or night
 * @param exposure   Pointer to the exposure value
 * @param gain       Pointer to the gain value
 * @param brightness Average image brightness
 * @return
 */
int adjust_exposure_gain(double brightness, double target_brightness, double max_exposure,
	double min_exposure, double max_gain, double min_gain, double *exposure, double *gain);

#define HISTORY_SIZE 10

typedef struct {
    float shutter;          // current exposure time [s]
    float gain;             // current gain factor
    float shutter_min;
    float shutter_max;
    float gain_min;
    float gain_max;
    float response;         // damping factor (0–1)

    float median_history[HISTORY_SIZE];
    int history_index;
    int history_count;

    float hysteresis_threshold; // e.g. 0.05
    int lights_on;              // output: 1 = light should be on
} ExposureController;

/**
 * Updates the exposure controller based on the current median.
 *
 * @param ctrl: pointer to controller struct (stateful)
 * @param current_median: current median brightness (0.0–1.0)
 * @return: 0 on success, 1 on error
 */
int update_exposure_controller(ExposureController *ctrl, float current_median, float target_median);

/**
 * Smart calculation of median brightness with outlier filtering (ignores extreme brightness values).
 * Computes the median brightness of an RGBA float image for all three RGB channels.
 * Only pixels with RGB > 0.0 are included in the calculation. Masked pixels are ignored!
 *
 * @param rgba: pointer to the float RGB data array (format: RGBA, 4 values per pixel).
 * @param width: image width.
 * @param height: image height.
 * @param center_pct: portion of the image to be analyzed (0.0 - 1.0, e.g., 0.3 for 30%).
 * @param red_brightness: pointer receiving the filtered median for the red channel.
 * @param green_brightness: pointer receiving the filtered median for the green channel.
 * @param blue_brightness: pointer receiving the filtered median for the blue channel.
 * @return: 0 on success, 1 on error.
 */
int compute_filtered_median_brightness_rgbf1(const float *rgba, int width, int height,
                                              double center_pct, double *red_brightness,
                                              double *green_brightness, double *blue_brightness);

/**
 * Backwards-compatible helper returning only the green median brightness.
 */
int compute_filtered_median_brightness_green_rgbf1(const float *rgba, int width, int height,
                                                   double center_pct, double *brightness);

#endif // AUTOEXPOSURE_H

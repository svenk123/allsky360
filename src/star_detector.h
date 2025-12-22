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
#ifndef STAR_DETECTOR_H
#define STAR_DETECTOR_H

#define CHANNELS	4

typedef struct {
    int x;
    int y;
} star_position_t;

/**
 * @param size: size of the template (width and height in pixels; must be square).
 * @param sigma: standard deviation of the Gaussian function, controlling the spread.
 * @return: pointer to a float array of size (size × size), values normalized to 0.0–1.0.
 *          Memory must be freed using allsky_safe_free(). Returns NULL on failure.
 */
float *generate_gaussian_template(int size, float sigma);

/**
 * Load a PNG image file and convert it to a grayscale float array.
 *
 * Supports:
 *   - 8-bit or 16-bit PNG
 *   - RGB, grayscale, palette
 *   - Automatic conversion to grayscale using luminance weights
 *
 * @param filename: path to the PNG file to load.
 * @param width_out: pointer to store the image width (in pixels).
 * @param height_out: pointer to store the image height (in pixels).
 * @return: pointer to a float array of size width*height (0.0 – 1.0),
 *          or NULL on error. Memory must be freed with allsky_safe_free().
 */
float *load_template_image(const char *filename, int *width_out, int *height_out);

/**
 * @param rgba: pointer to RGBA float image data (4 channels per pixel, range 0.0–1.0).
 * @param width: width of the input image in pixels.
 * @param height: height of the input image in pixels.
 * @param template_data: pointer to grayscale float template data (0.0–1.0).
 * @param tpl_width: width of the template image in pixels.
 * @param tpl_height: height of the template image in pixels.
 * @param cx: X-coordinate of the circular ROI center.
 * @param cy: Y-coordinate of the circular ROI center.
 * @param radius: radius of the circular ROI; stars outside are ignored.
 * @param threshold: correlation threshold (0.0–1.0); only matches above this are counted.
 * @param max_stars: maximum number of stars to detect and return.
 * @param stars_out: pointer to store the resulting array of star positions (allocated inside).
 * @param num_stars_out: pointer to store the number of stars found (≤ max_stars).
 * @return: 0 on success, >0 on failure (e.g. allocation failure).
 */
int find_stars_template(const float *rgba, int width, int height,
                        const float *template_data, int tpl_width, int tpl_height,
                        int cx, int cy, int radius, float threshold,
                        int max_stars, star_position_t **stars_out, int *num_stars_out);

/**
 * Write a list of star positions to a JSON file.
 *
 * @param stars: pointer to array of star_position_t elements.
 * @param num_stars: number of elements in the array.
 * @param filename: path to the output JSON file.
 * @return: 0 on success, >0 on error.
 */
int stars_to_json(const star_position_t *stars, int num_stars, const char *filename);


/**
 * Estimate sky quality (SQM) from number of detected stars using a logarithmic model.
 *
 * @param num_stars: number of detected stars in the defined ROI.
 * @param a: calibration parameter (e.g. 23.0) – maximum SQM for dark sky.
 * @param b: calibration slope (e.g. 1.5) – sensitivity of the star count to SQM.
 * @param sqm_out: pointer to float where the estimated SQM value will be stored.
 * @return: 0 on success, >0 on error (e.g. invalid input).
 */
int estimate_sqm_from_star_count(int num_stars, float a, float b, float *sqm_out);

/**
 * Estimate sky quality (SQM) from star count, corrected for exposure and gain.
 *
 * @param num_stars: number of detected stars in the ROI.
 * @param exposure_s: exposure time in seconds (e.g. 1.5 for 1500 ms).
 * @param gain_db: camera gain in dB (e.g. 0–30).
 * @param a: SQM model parameter (e.g. 23.0).
 * @param b: SQM slope parameter (e.g. 1.5).
 * @param sqm_out: pointer to float where the SQM result will be stored.
 * @return: 0 on success, >0 on invalid input or division by zero.
 */
int estimate_sqm_corrected(int num_stars, float exposure_s, float gain_db,
                           float a, float b, float *sqm_out);

#endif // STAR_DETECTOR_H

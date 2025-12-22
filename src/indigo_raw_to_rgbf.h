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
#ifndef RAW_TO_RGB16_H
#define RAW_TO_RGB16_H

#define CHANNELS 4

/** Get the width of a Indigo raw array
 *
 * @param raw_data: the raw data.
 * @param width_out: the width of the raw data.
 * @return: 0 on success, 1 on error.
 */
int indigo_raw_get_width(const unsigned char *raw_data, int *width_out);

/** Get the height of a Indigo raw array
 *
 * @param raw_data: the raw data.
 * @param height_out: the height of the raw data.
 * @return: 0 on success, 1 on error.
 */
int indigo_raw_get_height(const unsigned char *raw_data, int *height_out);

/** Load a Indigo raw array as rgb16
 *
 * @param rgb16_out: the output rgb16 data.
 * @param width_out: the width of the raw data.
 * @param height_out: the height of the raw data.
 * @param raw_data: the raw data.
 * @param debayer_alg: the debayer algorithm.
 * @return: 0 on success, 1 on error.
 */
int indigo_raw_to_rgb16(unsigned short **rgb16_out, int *width_out,
                        int *height_out, const unsigned char *raw_data,
                        int debayer_alg);

/** Load a Indigo raw array as rgbf
 *
 * @param rgba: the output rgbf data.
 * @param width: the width of the raw data.
 * @param height: the height of the raw data.
 * @param raw_data: the raw data.
 * @param debayer_alg: the debayer algorithm.
 * @param crop_width: the width of the crop.
 * @param crop_height: the height of the crop.
 * @param crop_offset_x: the x offset of the crop.
 * @param crop_offset_y: the y offset of the crop.
 * @return: 0 on success, 1 on error.
 */
int indigo_raw_to_rgbf1(float *rgba, int width, int height,
                        const unsigned char *raw_data, int debayer_alg,
                        int crop_width, int crop_height, int crop_offset_x,
                        int crop_offset_y);

#endif // RAW_TO_RGB16_H

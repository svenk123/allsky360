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

#include <stddef.h>

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

/** Bytes per pixel of an Indigo raw array, derived from the header signature
 *
 * @param raw_data: the raw data.
 * @return: 1 (MONO8), 2 (MONO16), 3 (RGB24), 4 (RGBA32/ABGR32), 6 (RGB48)
 *          or 0 for an unsupported format.
 */
int indigo_raw_bytes_per_pixel(const unsigned char *raw_data);

/** Minimum size of a complete Indigo raw array (header + pixel data)
 *
 * @param raw_data: the raw data.
 * @return: size in bytes, 0 for an unsupported format or invalid size.
 */
size_t indigo_raw_expected_size(const unsigned char *raw_data);

/** Load a Indigo raw array as rgbf
 *
 * @param rgba: the output rgbf data.
 * @param width: the width of the raw data.
 * @param height: the height of the raw data.
 * @param raw_data: the raw data.
 * @param is_monochrome: whether the image is monochrome.
 * @param debayer_alg: the debayer algorithm.
 * @param crop_width: the width of the crop.
 * @param crop_height: the height of the crop.
 * @param crop_offset_x: the x offset of the crop.
 * @param crop_offset_y: the y offset of the crop.
 * @return: 0 on success, 1 on error.
 */
int indigo_raw_to_rgbf1(float *rgba, int width, int height,
                        const unsigned char *raw_data, int is_monochrome, int debayer_alg,
                        int crop_width, int crop_height, int crop_offset_x,
                        int crop_offset_y);

#endif // RAW_TO_RGB16_H

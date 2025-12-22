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
#ifndef PNG_TO_RGBF_H
#define PNG_TO_RGBF_H

#define CHANNELS 4

/**
 * Saves an RGB float image (16-bit per channel) as PNG file.
 * The input is an RGBA float array, but only RGB is used.
 *
 * @param img_array: pointer to float[width * height * 4], range 0.0–65535.0 or 0.0–1.0
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param compression: PNG compression level (0 = no compression, 9 = max compression)
 * @param filename: output filename (path to PNG file)
 * @param scale_to_16bit: if 1 → input floats are 0.0–1.0 → scaled to 0–65535
 *                       if 0 → input floats are already 0.0–65535.0 → direct mapping
 * @param scale_percent: scaling factor in percent (100 = original size, 50 = half size, etc.)
 *                       The aspect ratio is preserved. If <= 0 or >= 10000, no scaling is performed.
 * @return: 0 on success, 1 on error
 */
int save_rgbf16_as_png(const float *img_array, int width, int height,
                       int compression, const char *filename,
                           int scale_to_16bit, int scale_percent);

#endif // PNG_TO_RGBF_H
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
#ifndef RGB16_TO_CAIRO_H
#define RGB16_TO_CAIRO_H

#define CHANNELS 4

/**
 * Converts 16-bit RGB data to 8-bit RGB data with automatic normalization.
 * The min/max values in the 16-bit input are mapped to 0 and 255 in the 8-bit output.
 * The alpha channel is always set to 255.
 *
 * @param data: pointer to the input 16-bit RGBA data (4 channels per pixel).
 * @param output: pointer to the output 8-bit RGBA data (4 channels per pixel), suitable for Cairo.
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 */
int rgb16_to_cairo(const unsigned short *data, unsigned char *output, int width, int height);

/**
 * Converts float-based RGB data to 8-bit RGBA data with automatic normalization.
 * The min/max values in the float input are mapped to 0 and 255 in the 8-bit output.
 * The alpha channel is always set to 255.
 *
 * @param data: pointer to the input float RGBA data (4 channels per pixel).
 * @param output: pointer to the output 8-bit RGBA data (4 channels per pixel), suitable for Cairo.
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 */
int rgbf_to_cairo(const float *data, unsigned char *output, int width, int height);

#endif // RGB16_TO_CAIRO_H

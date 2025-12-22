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
#ifndef IMAGE_OVERLAY_H
#define IMAGE_OVERLAY_H

#include <stdio.h>
#include <time.h>
#include <cairo/cairo.h>

#define CHANNELS	4

/**
 * Checks whether a Cairo image surface is a monochrome (grayscale) image.
 * Supports CAIRO_FORMAT_A8 (grayscale) and CAIRO_FORMAT_RGB24 (checks if R = G = B for all pixels).
 *
 * @param surface: pointer to the Cairo surface to check.
 * @return: 1 if the image is monochrome, 0 otherwise.
 */
int is_monochrome_image(cairo_surface_t *surface);

/**
 * Overlays the current date and time (derived from a timestamp) onto an image surface.
 *
 * @param surface: pointer to the Cairo surface pointer (image will be drawn in-place).
 * @param timestamp: timestamp to format and overlay as a date/time string.
 * @return: 0 on success, 1 on error (invalid surface pointer).
 */
int overlay_datetime_on_image(cairo_surface_t **surface, time_t timestamp);


/**
 * Draws a text string onto an image surface using FreeType.
 *
 * @param rgbf: pointer to the image data (RGBA format, float values 0.0 to 1.0 per channel).
 * @param width: image width.
 * @param height: image height.
 * @param text: text string to draw.
 * @param pixel_height: font height in pixels.
 * @param xpos: x-coordinate of the text position.
 * @param ypos: y-coordinate of the text position.
 * @return: 0 on success, 1 on error.
 */
int draw_text_red_freetype_rgbf1(float *rgbf,
    int width, int height,
    const char *text,
    int pixel_height,
    int xpos, int ypos, const char *font_path);

#endif // IMAGE_OVERLAY_H

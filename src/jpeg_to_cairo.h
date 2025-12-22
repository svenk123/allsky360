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
#ifndef JPEG_TO_CAIRO_H
#define JPEG_TO_CAIRO_H

#include <cairo.h>

#define CHANNELS	4

/**
 * Loads a JPEG image into a Cairo surface.
 *
 * @param surface: pointer to a Cairo surface pointer (output surface will be created).
 * @param filename: filename of the JPEG image to load.
 * @return: 1 on success, 0 on error.
 */
int load_jpeg_as_cairo_surface(cairo_surface_t **surface, const char *filename);

/**
 * Saves a Cairo surface as a JPEG image.
 *
 * @param surface: pointer to the Cairo surface to save.
 * @param output_filename: filename for the output JPEG image.
 * @param quality: JPEG quality (0–100).
 * @return: 1 on success, 0 on error.
 */
int save_cairo_surface_as_jpeg(cairo_surface_t *surface, const char *output_filename, int quality);

#endif // JPEG_TO_CAIRO_H
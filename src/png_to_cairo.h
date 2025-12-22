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
#ifndef PNG_TO_CAIRO_H
#define PNG_TO_CAIRO_H

#include <cairo.h>

/**
 * Loads a PNG image into a Cairo surface.
 *
 * @param surface: pointer to a Cairo surface pointer (output surface will be created).
 * @param filename: filename of the PNG image to load.
 * @return: 1 on success, 0 on error.
 */
int load_png_as_cairo_surface(cairo_surface_t **surface, const char *filename);

/**
 * Saves a Cairo surface as a PNG image with adjustable compression.
 *
 * @param surface: pointer to the Cairo surface to save.
 * @param filename: filename for the output PNG image.
 * @param compression: PNG compression level (0–9).
 * @return: 1 on success, 0 on error.
 */
int save_cairo_surface_as_png(cairo_surface_t *surface, const char *filename, int compression);

#endif // PNG_TO_CAIRO_H
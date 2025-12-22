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
#ifndef THUMBNAIL_H
#define THUMBNAIL_H

/**
 * Create a thumbnail from an existing cairo_surface_t while preserving aspect ratio.
 *
 * @param source: pointer to the input Cairo surface (must not be NULL)
 * @param output_path: path where the PNG thumbnail will be saved
 * @param thumb_width: desired width of the thumbnail in pixels
 * @return: 0 on success, 1 on error
 */
int create_thumbnail_from_surface(cairo_surface_t *source, const char *output_path, int thumb_width);

#endif // THUMBNAIL_H

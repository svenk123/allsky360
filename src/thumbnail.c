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
#include <stdio.h>
#include <stdlib.h>
#include <cairo.h>
#include "thumbnail.h"

int create_thumbnail_from_surface(cairo_surface_t *source, const char *output_path, int thumb_width) {
    if (!source || !output_path || thumb_width <= 0) {
        fprintf(stderr, "Invalid parameters.\n");
        return 1;
    }

    int orig_width = cairo_image_surface_get_width(source);
    int orig_height = cairo_image_surface_get_height(source);

    if (orig_width <= 0 || orig_height <= 0) {
        fprintf(stderr, "Invalid source dimensions.\n");
        return 1;
    }

    // Calculate thumbnail height preserving the aspect ratio
    double aspect_ratio = (double)orig_height / (double)orig_width;
    int thumb_height = (int)(thumb_width * aspect_ratio + 0.5);  // Round to nearest integer

    // Create new surface for the thumbnail
    cairo_surface_t *thumb_surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, thumb_width, thumb_height);
    if (cairo_surface_status(thumb_surface) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create thumbnail surface.\n");
        return 1;
    }

    cairo_t *cr = cairo_create(thumb_surface);
    if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create Cairo context.\n");
        cairo_surface_destroy(thumb_surface);
        return 1;
    }

    // Apply scaling (bilinear filtering for better quality)
    cairo_scale(cr, (double)thumb_width / orig_width, (double)thumb_width / orig_width);
    cairo_set_source_surface(cr, source, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BILINEAR);
    cairo_paint(cr);

    cairo_destroy(cr);

    // Save thumbnail as PNG
    cairo_status_t status = cairo_surface_write_to_png(thumb_surface, output_path);
    cairo_surface_destroy(thumb_surface);

    if (status != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to save thumbnail: %s\n", output_path);
        return 1;
    }

    printf("Thumbnail saved: %s (%dx%d)\n", output_path, thumb_width, thumb_height);
    return 0;
}

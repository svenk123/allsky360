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
#include <string.h>
#include <cairo.h>
#include <math.h>
#include "rotate_image.h"

int rotate_surface(cairo_surface_t **surface, int angle) {
    if (!surface || !(*surface)) {
        return 1;
    }

    if (angle != 90 && angle != 180 && angle != 270) {
        fprintf(stderr, "ERROR: Invalid rotation angle (%d°). Valid: 90, 180, 270\n", angle);
        return 1;
    }

    int orig_width = cairo_image_surface_get_width(*surface);
    int orig_height = cairo_image_surface_get_height(*surface);
    int new_width = (angle == 180) ? orig_width : orig_height;
    int new_height = (angle == 180) ? orig_height : orig_width;

    cairo_surface_t *rotated_surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, new_width, new_height);
    cairo_t *cr = cairo_create(rotated_surface);

    /* Rotation transformation */
    cairo_translate(cr, new_width / 2, new_height / 2);
    cairo_rotate(cr, angle * (M_PI / 180.0));  // Conversion to radians

    cairo_translate(cr, -orig_width / 2, -orig_height / 2);
    cairo_set_source_surface(cr, *surface, 0, 0);
    cairo_paint(cr);

    cairo_destroy(cr);
    cairo_surface_destroy(*surface);
    *surface = rotated_surface;

    printf("Image rotated %d degrees\n", angle);

    return 0;
}

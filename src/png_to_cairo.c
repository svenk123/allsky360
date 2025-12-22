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
#include <png.h>
#include "png_to_cairo.h"

int load_png_as_cairo_surface(cairo_surface_t **surface, const char *filename) {
    *surface = cairo_image_surface_create_from_png(filename);
    if (cairo_surface_status(*surface) != CAIRO_STATUS_SUCCESS) {
        return 1;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Can not open file %s\n", filename);
        return 1;
    }

    // Read PNG-Header
    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8) {
        fprintf(stderr, "ERROR: Can not read PNG header (file: %s)!\n", filename);
        fclose(fp);
        return 1;
    }

    // Check signature
    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "ERROR: Invalid PNG file: %s\n", filename);
        fclose(fp);
        return 1;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return 1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return 1;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return 1;
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    int bit_depth = png_get_bit_depth(png, info);
    int color_type = png_get_color_type(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);  // Reduce 16 bit to 8 bit
        fprintf(stderr, "WARNING: 16 bit PNG reduced 8 bit.\n");
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    png_read_update_info(png, info);

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return 0;
}

int save_cairo_surface_as_png(cairo_surface_t *surface, const char *filename, int compression) {
    if (!surface)
        return 1;

    if (compression < 0 || compression > 9) {
        fprintf(stderr, "ERROR: PNG compression factor must be in range 0..9!\n");
        return 1;
    }

    cairo_status_t status = cairo_surface_write_to_png(surface, filename);
    if (status != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Can not save png file %s\n", filename);
        return 1;
    }

    FILE *fp = fopen(filename, "rb+");
    if (!fp) {
        fprintf(stderr, "ERROR: Can not write file %s\n", filename);
        return 1;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return 1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return 1;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 1;
    }

    png_set_compression_level(png, compression);
    png_destroy_write_struct(&png, &info);
    fclose(fp);

    return 0;
}

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
#include <stdint.h>
#include <png.h>
#include "allsky.h"
#include "png_to_rgb8.h"

int save_rgb8_as_png(const uint8_t *rgb, int width, int height,
                     int compression, const char *filename) {
    if (!rgb || width <= 0 || height <= 0 || !filename)
        return 1;

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing.\n", filename);
        return 1;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return 1;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return 1;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return 1;
    }

    png_init_io(png_ptr, fp);
    png_set_compression_level(png_ptr, compression);

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGB,  // 8-Bit RGB
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    png_bytep *row_pointers = (png_bytep *)allsky_safe_malloc(sizeof(png_bytep) * height);
    if (!row_pointers) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return 1;
    }

    for (int y = 0; y < height; ++y) {
        row_pointers[y] = (png_bytep)&rgb[y * width * 3];
    }

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    allsky_safe_free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    return 0;
}

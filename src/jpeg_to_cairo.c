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
#include <jpeglib.h>
#include <cairo.h>
#include "allsky.h"
#include "jpeg_to_cairo.h"
#include "jpeg_error.h"

int load_jpeg_as_cairo_surface(cairo_surface_t **surface, const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct allsky_jpeg_err_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;

    if (!surface)
        return 1;
    *surface = NULL;

    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "ERROR: Can not open file %s\n", filename);
        return 1;
    }

    // Initialize jpeg: longjmp instead of exit() on errors (e.g. a
    // truncated or corrupt file)
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = allsky_jpeg_error_exit;
    if (setjmp(jerr.setjmp_buf)) {
        fprintf(stderr, "ERROR: Can not decode JPEG file %s\n", filename);
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        if (*surface) {
            cairo_surface_destroy(*surface);
            *surface = NULL;
        }
        return 1;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    unsigned int width = cinfo.output_width;
    unsigned int height = cinfo.output_height;
    int channels = cinfo.output_components;

    if (channels != 3) {
        fprintf(stderr, "ERROR: Only RGB jpeg files are supported!\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return 1;
    }

    // Create cairo surface
    *surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, width, height);
    if (cairo_surface_status(*surface) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to create cairo surface!\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return 1;
    }

    // Initialize image buffer
    unsigned char *data = cairo_image_surface_get_data(*surface);
    row_stride = width * channels;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    while (cinfo.output_scanline < height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        unsigned char *dst = data + (cinfo.output_scanline - 1) * width * CHANNELS;

        for (unsigned int i = 0, j = 0; i < width; i++, j += 3) {
            dst[i * CHANNELS + 2] = buffer[0][j];     // R
            dst[i * CHANNELS + 1] = buffer[0][j + 1]; // G
            dst[i * CHANNELS + 0] = buffer[0][j + 2]; // B
            dst[i * CHANNELS + 3] = 255;              // Alpha
        }
    }

    cairo_surface_mark_dirty(*surface);

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    printf("Load JPEG as Cairo surface: ok, filename: %s\n", filename);

    return 0;
}

int save_cairo_surface_as_jpeg(cairo_surface_t *surface, const char *output_filename, int quality) {
    if (!surface) {
        return 1;
    }

    if (quality < 0 || quality > 100) {
        fprintf(stderr, "ERROR: JPEG quality must be in range of 0..100!\n");
        return 1;
    }

    FILE *outfile = fopen(output_filename, "wb");
    if (!outfile) {
        fprintf(stderr, "ERROR: Can not write file: %s\n", output_filename);
        return 1;
    }

    struct jpeg_compress_struct cinfo;
    struct allsky_jpeg_err_mgr jerr;

    unsigned int width = cairo_image_surface_get_width(surface);
    unsigned int height = cairo_image_surface_get_height(surface);
    unsigned char *data = cairo_image_surface_get_data(surface);
    unsigned char *row_data = (unsigned char *)allsky_safe_malloc(width * 3);
    if (!row_data) {
        fclose(outfile);
        remove(output_filename);
        return 1;
    }

    // longjmp instead of exit() on errors, e.g. a write error on a full disk
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = allsky_jpeg_error_exit;
    if (setjmp(jerr.setjmp_buf)) {
        fprintf(stderr, "ERROR: Can not write JPEG file %s\n", output_filename);
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
        remove(output_filename);
        allsky_safe_free(row_data);
        return 1;
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;  // RGB
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer;

    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char *src = data + cinfo.next_scanline * width * CHANNELS;
        for (unsigned int i = 0; i < width; i++) {
            row_data[i * 3] = src[i * CHANNELS + 2];     // R
            row_data[i * 3 + 1] = src[i * CHANNELS + 1]; // G
            row_data[i * 3 + 2] = src[i * CHANNELS + 0]; // B
        }
        row_pointer = row_data;
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    allsky_safe_free(row_data);
    if (fclose(outfile) != 0) {
        fprintf(stderr, "ERROR: Write error on JPEG file %s\n", output_filename);
        remove(output_filename);
        return 1;
    }

    printf("Save Cairo surface as JPEG: ok, filename: %s (%dx%d)\n",output_filename, width, height);

    return 0;
}

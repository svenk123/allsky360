/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>

#include "jpeg_io.h"

/* libjpeg calls exit() on errors by default: jump back instead, so one
 * broken image only drops that image */
struct err_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
};

static void err_exit(j_common_ptr cinfo)
{
    struct err_mgr *err = (struct err_mgr *)cinfo->err;
    longjmp(err->jump, 1);
}

/* Corrupt-data warnings (e.g. a truncated file) are errors for us */
static void err_emit(j_common_ptr cinfo, int msg_level)
{
    if (msg_level < 0)
        err_exit(cinfo);
}

static void err_silent(j_common_ptr cinfo) { (void)cinfo; }

uint8_t *jpeg_read_rgb_scaled(const char *path, int scale_denom, int *width, int *height)
{
    struct jpeg_decompress_struct cinfo;
    struct err_mgr jerr;
    /* volatile: modified between setjmp and longjmp */
    uint8_t *volatile rgb = NULL;
    FILE *volatile fp = fopen(path, "rb");

    if (!fp)
        return NULL;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = err_exit;
    jerr.pub.emit_message = err_emit;
    jerr.pub.output_message = err_silent;
    if (setjmp(jerr.jump)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        free(rgb);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    cinfo.scale_num = 1;
    cinfo.scale_denom = (unsigned int)(scale_denom > 0 ? scale_denom : 1);
    cinfo.dct_method = JDCT_IFAST;
    jpeg_start_decompress(&cinfo);

    const int w = (int)cinfo.output_width, h = (int)cinfo.output_height;
    const size_t stride = (size_t)w * 3;
    rgb = malloc(stride * (size_t)h);
    if (!rgb)
        longjmp(jerr.jump, 1);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = (JSAMPROW)(rgb + stride * cinfo.output_scanline);
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    *width = w;
    *height = h;
    return rgb;
}

int jpeg_read_size(const char *path, int *width, int *height)
{
    struct jpeg_decompress_struct cinfo;
    struct err_mgr jerr;
    FILE *volatile fp = fopen(path, "rb");

    if (!fp)
        return 1;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = err_exit;
    jerr.pub.output_message = err_silent;
    if (setjmp(jerr.jump)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return 2;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    *width = (int)cinfo.image_width;
    *height = (int)cinfo.image_height;
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
    return 0;
}

int jpeg_write_rgb(const char *path, const uint8_t *rgb, int width, int height, int quality)
{
    struct jpeg_compress_struct cinfo;
    struct err_mgr jerr;
    char tmp[4096];
    FILE *volatile fp;

    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp))
        return 1;
    fp = fopen(tmp, "wb");
    if (!fp)
        return 2;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = err_exit;
    jerr.pub.output_message = err_silent;
    if (setjmp(jerr.jump)) {
        jpeg_destroy_compress(&cinfo);
        fclose(fp);
        remove(tmp);
        return 3;
    }
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);
    cinfo.image_width = (JDIMENSION)width;
    cinfo.image_height = (JDIMENSION)height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    const size_t stride = (size_t)width * 3;
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row = (JSAMPROW)(rgb + stride * cinfo.next_scanline);
        jpeg_write_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    if (fclose(fp) != 0) {
        remove(tmp);
        return 4;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        return 5;
    }
    return 0;
}

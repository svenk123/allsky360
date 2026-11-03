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
#include "image_check.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <setjmp.h>

struct jpeg_check_error {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
};

/* libjpeg's default error_exit() terminates the process: a single truncated
 * file (camera still writing it) would abort the whole timelapse. */
static void jpeg_check_error_exit(j_common_ptr cinfo) {
    struct jpeg_check_error *err = (struct jpeg_check_error *)cinfo->err;
    longjmp(err->jump, 1);
}

/**
 * Reads the dimensions (width and height) of a JPEG image without fully decoding it.
 *
 * @param filename  Path to the JPEG file.
 * @param width     Output: pointer to an int that will receive the image width.
 * @param height    Output: pointer to an int that will receive the image height.
 * @return          0 on success, -1 on file access or JPEG header error.
 */
int get_jpeg_dimensions(const char *filename, int *width, int *height) {
    FILE *infile = fopen(filename, "rb");
    if (!infile) return -1;

    struct jpeg_decompress_struct cinfo;
    struct jpeg_check_error jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_check_error_exit;
    if (setjmp(jerr.jump)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return -1;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    *width = cinfo.image_width;
    *height = cinfo.image_height;

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
}

void fit_rect(int src_w, int src_h, int dst_w, int dst_h,
              int *out_x, int *out_y, int *out_w, int *out_h) {
    int w = dst_w, h = dst_h;

    /* Compare src_w/src_h with dst_w/dst_h without rounding errors */
    if ((long long)src_w * dst_h > (long long)dst_w * src_h) {
        /* wider than the frame: full width, bars top and bottom */
        h = (int)(((long long)dst_w * src_h + src_w / 2) / src_w);
    } else if ((long long)src_w * dst_h < (long long)dst_w * src_h) {
        /* taller than the frame: full height, bars left and right */
        w = (int)(((long long)dst_h * src_w + src_h / 2) / src_h);
    }
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    *out_w = w;
    *out_h = h;
    *out_x = (dst_w - w) / 2;
    *out_y = (dst_h - h) / 2;
}

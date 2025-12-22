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

/**
 * Reads the dimensions (width and height) of a JPEG image without fully decoding it.
 *
 * @param filename  Path to the JPEG file.
 * @param width     Output: pointer to an int that will receive the image width.
 * @param height    Output: pointer to an int that will receive the image height.
 * @return          0 on success, -1 on file access error.
 */
int get_jpeg_dimensions(const char *filename, int *width, int *height) {
    FILE *infile = fopen(filename, "rb");
    if (!infile) return -1;

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    *width = cinfo.image_width;
    *height = cinfo.image_height;

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
}

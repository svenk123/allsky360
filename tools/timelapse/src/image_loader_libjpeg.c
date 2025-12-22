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
#include "image_loader.h"
#include "allsky.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

uint8_t *load_and_resize_jpeg(const char *filename, int target_width,
                              int target_height, int expected_src_width,
                              int expected_src_height) {
  FILE *infile = fopen(filename, "rb");
  if (!infile)
    return NULL;

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  int row_stride = cinfo.output_width * cinfo.output_components;
  int src_width = cinfo.output_width;
  int src_height = cinfo.output_height;

  if (cinfo.output_width != expected_src_width ||
      cinfo.output_height != expected_src_height) {
    fprintf(
        stderr,
        "[WARN] Skipping image %s: size %dx%d doesn't match expected %dx%d\n",
        filename, cinfo.output_width, cinfo.output_height, expected_src_width,
        expected_src_height);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return NULL;
  }

  uint8_t *raw = allsky_safe_malloc(src_height * row_stride);
  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo,
                                                 JPOOL_IMAGE, row_stride, 1);

  for (int y = 0; cinfo.output_scanline < src_height; y++) {
    jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(&raw[y * row_stride], buffer[0], row_stride);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);

  // CPU-based resizing using bilinear interpolation
  size_t dst_stride = target_width * 3;
  uint8_t *h_dst = allsky_safe_malloc(target_height * dst_stride);
  if (!h_dst) {
    fprintf(stderr, "[ERROR] malloc failed for h_dst\n");
    allsky_safe_free(raw);
    return NULL;
  }

  // Bilinear interpolation for RGB images
  // Handle edge case where target dimensions are 1
  float x_ratio = (target_width > 1)
                      ? (float)(src_width - 1) / (float)(target_width - 1)
                      : 0.0f;
  float y_ratio = (target_height > 1)
                      ? (float)(src_height - 1) / (float)(target_height - 1)
                      : 0.0f;

  // Parallelize the outer loop - each row can be processed independently
#pragma omp parallel for
  for (int y = 0; y < target_height; y++) {
    for (int x = 0; x < target_width; x++) {
      float src_x = x * x_ratio;
      float src_y = y * y_ratio;

      int x1 = (int)src_x;
      int y1 = (int)src_y;
      int x2 = (x1 < src_width - 1) ? x1 + 1 : x1;
      int y2 = (y1 < src_height - 1) ? y1 + 1 : y1;

      float x_diff = src_x - x1;
      float y_diff = src_y - y1;

      // Get the four neighboring pixels
      uint8_t *p11 = &raw[(y1 * row_stride) + (x1 * 3)];
      uint8_t *p12 = &raw[(y1 * row_stride) + (x2 * 3)];
      uint8_t *p21 = &raw[(y2 * row_stride) + (x1 * 3)];
      uint8_t *p22 = &raw[(y2 * row_stride) + (x2 * 3)];

      // Interpolate for each RGB channel
      for (int c = 0; c < 3; c++) {
        float val = p11[c] * (1.0f - x_diff) * (1.0f - y_diff) +
                    p12[c] * x_diff * (1.0f - y_diff) +
                    p21[c] * (1.0f - x_diff) * y_diff +
                    p22[c] * x_diff * y_diff;
        h_dst[(y * dst_stride) + (x * 3) + c] = (uint8_t)(val + 0.5f);
      }
    }
  }

  allsky_safe_free(raw);
  return h_dst;
}

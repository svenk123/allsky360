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
#include "image_check.h"
#include "allsky.h"
#include <setjmp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

struct loader_error {
  struct jpeg_error_mgr pub;
  jmp_buf jump;
};

/* Default error_exit() would terminate the whole timelapse on one bad file */
static void loader_error_exit(j_common_ptr cinfo) {
  struct loader_error *err = (struct loader_error *)cinfo->err;
  (*cinfo->err->output_message)(cinfo);
  longjmp(err->jump, 1);
}

uint8_t *load_and_resize_jpeg(const char *filename, int target_width,
                              int target_height, int expected_src_width,
                              int expected_src_height) {
  struct jpeg_decompress_struct cinfo;
  struct loader_error jerr;
  /* volatile: modified between setjmp() and a possible longjmp() */
  uint8_t *volatile raw = NULL;
  FILE *infile = fopen(filename, "rb");
  if (!infile)
    return NULL;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = loader_error_exit;
  if (setjmp(jerr.jump)) {
    fprintf(stderr, "[WARN] Cannot decode %s\n", filename);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    allsky_safe_free(raw);
    return NULL;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_RGB; /* also for greyscale JPEGs: always 3 bytes */
  jpeg_start_decompress(&cinfo);

  int src_width = cinfo.output_width;
  int src_height = cinfo.output_height;
  size_t row_stride = (size_t)src_width * 3;

  if (expected_src_width > 0 && expected_src_height > 0 &&
      (src_width != expected_src_width || src_height != expected_src_height)) {
    fprintf(stderr,
            "[WARN] Skipping image %s: size %dx%d doesn't match expected %dx%d\n",
            filename, src_width, src_height, expected_src_width,
            expected_src_height);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return NULL;
  }

  raw = allsky_safe_malloc((size_t)src_height * row_stride);
  while (cinfo.output_scanline < (JDIMENSION)src_height) {
    JSAMPROW row = (JSAMPROW)&raw[cinfo.output_scanline * row_stride];
    jpeg_read_scanlines(&cinfo, &row, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);

  /* Letterbox: scale into fit_w x fit_h at (off_x, off_y), the rest stays
   * black (allsky_safe_malloc zeroes the buffer). */
  int off_x, off_y, fit_w, fit_h;
  fit_rect(src_width, src_height, target_width, target_height, &off_x, &off_y,
           &fit_w, &fit_h);

  size_t dst_stride = (size_t)target_width * 3;
  uint8_t *h_dst = allsky_safe_malloc((size_t)target_height * dst_stride);
  const uint8_t *src = raw;

  // Bilinear interpolation, edge case: fit dimension 1
  float x_ratio =
      (fit_w > 1) ? (float)(src_width - 1) / (float)(fit_w - 1) : 0.0f;
  float y_ratio =
      (fit_h > 1) ? (float)(src_height - 1) / (float)(fit_h - 1) : 0.0f;

  // Each row can be processed independently
#pragma omp parallel for
  for (int y = 0; y < fit_h; y++) {
    float src_y = y * y_ratio;
    int y1 = (int)src_y;
    int y2 = (y1 < src_height - 1) ? y1 + 1 : y1;
    float y_diff = src_y - y1;
    uint8_t *dst_row = &h_dst[(size_t)(y + off_y) * dst_stride + (size_t)off_x * 3];

    for (int x = 0; x < fit_w; x++) {
      float src_x = x * x_ratio;
      int x1 = (int)src_x;
      int x2 = (x1 < src_width - 1) ? x1 + 1 : x1;
      float x_diff = src_x - x1;

      // The four neighbouring pixels
      const uint8_t *p11 = &src[y1 * row_stride + (size_t)x1 * 3];
      const uint8_t *p12 = &src[y1 * row_stride + (size_t)x2 * 3];
      const uint8_t *p21 = &src[y2 * row_stride + (size_t)x1 * 3];
      const uint8_t *p22 = &src[y2 * row_stride + (size_t)x2 * 3];

      for (int c = 0; c < 3; c++) {
        float val = p11[c] * (1.0f - x_diff) * (1.0f - y_diff) +
                    p12[c] * x_diff * (1.0f - y_diff) +
                    p21[c] * (1.0f - x_diff) * y_diff +
                    p22[c] * x_diff * y_diff;
        dst_row[x * 3 + c] = (uint8_t)(val + 0.5f);
      }
    }
  }

  allsky_safe_free(raw);
  return h_dst;
}

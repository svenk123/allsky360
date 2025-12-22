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
#include "jpeg_functions.h"
#include "allsky.h"
#include <jpeglib.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int save_jpeg_rgbf1(const float *rgba, int width, int height,
                    int compression_ratio, float scale, const char *filename) {
  if (!rgba || width <= 0 || height <= 0 || scale <= 0.0f ||
      compression_ratio <= 0)
    return 1;

  int new_w = (int)(width * scale);
  int new_h = (int)(height * scale);
  if (new_w < 1 || new_h < 1)
    return 2;

  unsigned char *rgb8 = allsky_safe_malloc(new_w * new_h * 3);
  if (!rgb8)
    return 3;

// Simple nearest-neighbor downscale (fast)
#pragma omp parallel for if (scale < 1.0f)
  for (int y = 0; y < new_h; y++) {
    for (int x = 0; x < new_w; x++) {
      int src_x = (int)(x / scale);
      int src_y = (int)(y / scale);

      if (src_x >= width)
        src_x = width - 1;
      if (src_y >= height)
        src_y = height - 1;
      int src_idx = (src_y * width + src_x) * 4;
      int dst_idx = (y * new_w + x) * 3;
      float r = rgba[src_idx + 0];
      float g = rgba[src_idx + 1];
      float b = rgba[src_idx + 2];

      rgb8[dst_idx + 0] =
          (unsigned char)(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
      rgb8[dst_idx + 1] =
          (unsigned char)(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
      rgb8[dst_idx + 2] =
          (unsigned char)(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
    }
  }

  // JPEG encoding
  FILE *outfile = fopen(filename, "wb");
  if (!outfile) {
    allsky_safe_free(rgb8);
    return 4;
  }

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = new_w;
  cinfo.image_height = new_h;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, compression_ratio, TRUE);
  jpeg_start_compress(&cinfo, TRUE);

  JSAMPROW row_pointer[1];
  int row_stride = new_w * 3;

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &rgb8[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
  allsky_safe_free(rgb8);

  printf("Save JPEG: ok, filename: %s\n", filename);

  return 0;
}

#ifndef USE_CUDA

int save_jpeg_rgbf16(const float *rgba, int width, int height,
                     int compression_ratio, float scale, const char *filename) {
  if (!rgba || width <= 0 || height <= 0 || scale <= 0.0f ||
      compression_ratio <= 0)
    return 1;

  int new_w = (int)(width * scale);
  int new_h = (int)(height * scale);
  if (new_w < 1 || new_h < 1)
    return 2;

  unsigned char *rgb8 = allsky_safe_malloc(new_w * new_h * 3);
  if (!rgb8)
    return 3;

// Simple nearest-neighbor downscale (fast)
// Input range: 0.0-65535.0 (16-bit), output range: 0-255 (8-bit)
#pragma omp parallel for if (scale < 1.0f)
  for (int y = 0; y < new_h; y++) {
    for (int x = 0; x < new_w; x++) {
      int src_x = (int)(x / scale);
      int src_y = (int)(y / scale);

      if (src_x >= width)
        src_x = width - 1;
      if (src_y >= height)
        src_y = height - 1;
      int src_idx = (src_y * width + src_x) * 4;
      int dst_idx = (y * new_w + x) * 3;
      float r = rgba[src_idx + 0];
      float g = rgba[src_idx + 1];
      float b = rgba[src_idx + 2];

      // Scale from 0.0-65535.0 to 0-255
      rgb8[dst_idx + 0] =
          (unsigned char)(fminf(fmaxf(r * 255.0f / 65535.0f, 0.0f), 255.0f));
      rgb8[dst_idx + 1] =
          (unsigned char)(fminf(fmaxf(g * 255.0f / 65535.0f, 0.0f), 255.0f));
      rgb8[dst_idx + 2] =
          (unsigned char)(fminf(fmaxf(b * 255.0f / 65535.0f, 0.0f), 255.0f));
    }
  }

  // JPEG encoding
  FILE *outfile = fopen(filename, "wb");
  if (!outfile) {
    allsky_safe_free(rgb8);
    return 4;
  }

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = new_w;
  cinfo.image_height = new_h;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, compression_ratio, TRUE);
  jpeg_start_compress(&cinfo, TRUE);

  JSAMPROW row_pointer[1];
  int row_stride = new_w * 3;

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &rgb8[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
  allsky_safe_free(rgb8);

  printf("Save JPEG: ok, filename: %s\n", filename);

  return 0;
}

int tonemap_rgbf1_to_rgbf16(float *rgbf, int width, int height) {
  if (!rgbf || width <= 0 || height <= 0)
    return 1;

  int pixel_count = width * height;

  float min_r = rgbf[0], max_r = rgbf[0];
  float min_g = rgbf[1], max_g = rgbf[1];
  float min_b = rgbf[2], max_b = rgbf[2];

  for (int i = 1; i < pixel_count; i++) {
    int idx = i * CHANNELS;
    float r = rgbf[idx + 0];
    float g = rgbf[idx + 1];
    float b = rgbf[idx + 2];

    if (r < min_r)
      min_r = r;
    if (r > max_r)
      max_r = r;
    if (g < min_g)
      min_g = g;
    if (g > max_g)
      max_g = g;
    if (b < min_b)
      min_b = b;
    if (b > max_b)
      max_b = b;
  }

  float range_r = fmaxf(max_r - min_r, 1.0f);
  float range_g = fmaxf(max_g - min_g, 1.0f);
  float range_b = fmaxf(max_b - min_b, 1.0f);

// In-place mormalization + scaling (backwards)
#pragma omp parallel for
  for (int i = pixel_count - 1; i >= 0; i--) {
    int idx = i * CHANNELS;
    rgbf[idx + 0] = ((rgbf[idx + 0] - min_r) / range_r) * 65535.0f;
    rgbf[idx + 1] = ((rgbf[idx + 1] - min_g) / range_g) * 65535.0f;
    rgbf[idx + 2] = ((rgbf[idx + 2] - min_b) / range_b) * 65535.0f;
    rgbf[idx + 3] = 65535.0f; // Alpha
  }

  printf("Tonemapping: ok, float RGB (0.0–1.0) to RGB float 16-bit "
         "(0.0–65535.0).\n");
  return 0;
}

#endif

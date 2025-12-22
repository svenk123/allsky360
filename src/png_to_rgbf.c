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
#include "png_to_rgbf.h"
#include "allsky.h"
#include <png.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper: Bilinear interpolation for RGBA float images */
static void resize_bilinear_rgbf(const float *src, int sw, int sh,
                                  float *dst, int dw, int dh) {
  if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
    return;

  if (dw == sw && dh == sh) {
    // Identity copy
    for (int i = 0; i < sw * sh * CHANNELS; ++i)
      dst[i] = src[i];
    return;
  }

  for (int y = 0; y < dh; ++y) {
    float gy = (sh == 1) ? 0.0f : ((float)y * (float)(sh - 1)) / (float)(dh - 1);
    int y0 = (int)gy;
    int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
    float wy = gy - (float)y0;

    for (int x = 0; x < dw; ++x) {
      float gx = (sw == 1) ? 0.0f : ((float)x * (float)(sw - 1)) / (float)(dw - 1);
      int x0 = (int)gx;
      int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
      float wx = gx - (float)x0;

      for (int c = 0; c < CHANNELS; ++c) {
        float v00 = src[(y0 * sw + x0) * CHANNELS + c];
        float v01 = src[(y0 * sw + x1) * CHANNELS + c];
        float v10 = src[(y1 * sw + x0) * CHANNELS + c];
        float v11 = src[(y1 * sw + x1) * CHANNELS + c];

        float v0 = v00 * (1.0f - wx) + v01 * wx;
        float v1 = v10 * (1.0f - wx) + v11 * wx;
        dst[(y * dw + x) * CHANNELS + c] = v0 * (1.0f - wy) + v1 * wy;
      }
    }
  }
}

int save_rgbf16_as_png(const float *img_array, int width, int height,
                       int compression, const char *filename,
                       int scale_to_16bit, int scale_percent) {
  if (!img_array || width <= 0 || height <= 0 || !filename)
    return 1;

  /* Create volatile copy to avoid clobbering warning with setjmp */
  const float * volatile img_array_volatile = img_array;

  int output_width = width;
  int output_height = height;
  const float * volatile image_data = img_array_volatile;
  float * volatile scaled_image = NULL;

  // Apply scaling if requested
  if (scale_percent > 0 && scale_percent < 10000 && scale_percent != 100) {
    output_width = (int)((long long)width * scale_percent / 100);
    output_height = (int)((long long)height * scale_percent / 100);

    // Ensure minimum size of 1x1
    if (output_width < 1) 
      output_width = 1;
    
    if (output_height < 1) 
      output_height = 1;

    // Allocate scaled image buffer
    scaled_image = (float *)allsky_safe_malloc((size_t)output_width * (size_t)output_height * CHANNELS * sizeof(float));
    if (!scaled_image) {
      fprintf(stderr, "Error: Failed to allocate memory for scaled image.\n");
      return 1;
    }

    // Resize image using bilinear interpolation
    resize_bilinear_rgbf(img_array_volatile, width, height, scaled_image, output_width, output_height);
    image_data = scaled_image;
  }

  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open %s for writing.\n", filename);
    if (scaled_image) allsky_safe_free(scaled_image);
    return 1;
  }

  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
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

  /* Set header: RGB, 16-bit per channel */
  png_set_IHDR(png_ptr, info_ptr, output_width, output_height, 16, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);

  /* Allocate one row buffer for all rows */
  png_bytep *row_pointers =
      (png_bytep *)allsky_safe_malloc(sizeof(png_bytep) * output_height);
  if (!row_pointers) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    if (scaled_image) allsky_safe_free(scaled_image);
    return 1;
  }

  png_bytep row_buffer = (png_bytep)allsky_safe_malloc(6 * output_width * output_height);
  if (!row_buffer) {
    allsky_safe_free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    if (scaled_image) allsky_safe_free(scaled_image);
    return 1;
  }

  float scale_factor = (scale_to_16bit ? 65535.0f : 1.0f);

#pragma omp parallel for
  for (int y = 0; y < output_height; ++y) {
    png_bytep row = row_buffer + y * 6 * output_width;
    row_pointers[y] = row;

    for (int x = 0; x < output_width; ++x) {
      int pixel_idx = (y * output_width + x) * CHANNELS;

      for (int c = 0; c < 3; ++c) { // R, G, B only
        float val = image_data[pixel_idx + c] * scale_factor;
        if (val < 0.0f)
          val = 0.0f;
        if (val > 65535.0f)
          val = 65535.0f;
        uint16_t out_val = (uint16_t)(val + 0.5f); // round to nearest integer

        row[x * 6 + c * 2 + 0] = (out_val >> 8) & 0xFF;
        row[x * 6 + c * 2 + 1] = out_val & 0xFF;
      }
    }
  }

  /* Write image */
  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);

  /* Clean up */
  allsky_safe_free(row_buffer);
  allsky_safe_free(row_pointers);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  if (scaled_image) allsky_safe_free(scaled_image);

  return 0;
}

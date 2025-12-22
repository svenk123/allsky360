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
#include "star_detector.h"
#include "allsky.h"
#include <json-c/json.h>
#include <math.h>
#include <omp.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>

/* Generate a circular Gaussian template (normalized to 0.0–1.0) */
float *generate_gaussian_template(int size, float sigma) {
  float *tpl = allsky_safe_malloc(size * size * sizeof(float));
  if (!tpl)
    return NULL;

  float sum = 0.0f;
  int center = size / 2;

  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      int dx = x - center;
      int dy = y - center;
      float r2 = dx * dx + dy * dy;
      float val = expf(-r2 / (2.0f * sigma * sigma));
      tpl[y * size + x] = val;
      sum += val;
    }
  }

  /* Normalize to 0.0 – 1.0 */
  for (int i = 0; i < size * size; i++) {
    tpl[i] /= sum;
  }

  return tpl;
}

float *load_template_image(const char *filename, int *width_out,
                           int *height_out) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "[ERROR] Cannot open file: %s\n", filename);
    return NULL;
  }

  /* Read and validate PNG signature */
  png_byte header[8];
  size_t bytes_read = fread(header, 1, 8, fp);
  if (bytes_read != sizeof(header)) {
    fprintf(stderr, "[ERROR] Failed to read PNG header from: %s\n", filename);
    fclose(fp);

    return NULL;
  }

  if (png_sig_cmp(header, 0, sizeof(header))) {
    fprintf(stderr, "[ERROR] Not a valid PNG file: %s\n", filename);
    fclose(fp);

    return NULL;
  }

  // Initialize PNG read structures
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr) {
    fclose(fp);

    return NULL;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);

    return NULL;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    return NULL;
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  int width = png_get_image_width(png_ptr, info_ptr);
  int height = png_get_image_height(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);

  /* Convert palette to RGB */
  if (color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png_ptr);

  /* Expand grayscale images with <8 bits to 8-bit */
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png_ptr);

  //   Convert transparency to full alpha channel
  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png_ptr);

  /* Convert grayscale+alpha to RGB */
  if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png_ptr);

  /* Convert grayscale to RGB */
  if (color_type == PNG_COLOR_TYPE_GRAY)
    png_set_gray_to_rgb(png_ptr);

  /* Swap 16-bit byte order if needed (endianness) */
  if (bit_depth == 16)
    png_set_swap(png_ptr);

  png_read_update_info(png_ptr, info_ptr);

  int channels = png_get_channels(png_ptr, info_ptr);
  png_bytepp row_pointers = allsky_safe_malloc(sizeof(png_bytep) * height);
  int rowbytes = png_get_rowbytes(png_ptr, info_ptr);

  png_bytep image_data = allsky_safe_malloc(rowbytes * height);
  for (int i = 0; i < height; i++)
    row_pointers[i] = image_data + i * rowbytes;

  png_read_image(png_ptr, row_pointers);

  fclose(fp);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  allsky_safe_free(row_pointers);

  /* Allocate output float buffer */
  float *tpl = allsky_safe_malloc(sizeof(float) * width * height);
  if (!tpl) {
    allsky_safe_free(image_data);
    return NULL;
  }

  /* Convert image data to normalized grayscale float */
  for (int y = 0; y < height; y++) {
    png_bytep row = image_data + y * rowbytes;
    for (int x = 0; x < width; x++) {
      float gray;
      if (bit_depth == 16) {
        png_uint_16 *pix = (png_uint_16 *)row;
        png_uint_16 r = pix[x * channels + 0];
        png_uint_16 g = pix[x * channels + 1];
        png_uint_16 b = pix[x * channels + 2];
        gray = (0.2126f * r + 0.7152f * g + 0.0722f * b) / 65535.0f;
      } else {
        png_bytep pix = &row[x * channels];
        png_byte r = pix[0];
        png_byte g = pix[1];
        png_byte b = pix[2];
        gray = (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255.0f;
      }
      tpl[y * width + x] = gray;
    }
  }

  allsky_safe_free(image_data);
  if (width_out)
    *width_out = width;
  if (height_out)
    *height_out = height;

  printf("Loaded star template from file: %s (%dx%d, %d-bit)\n", filename,
         width, height, bit_depth);

  return tpl;
}

/* Compute grayscale intensity from RGB (Rec.709) */
static inline float intensity(const float *rgba) {
  return 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2];
}

/* Check if (x, y) is inside a circle */
static inline int in_circle(int x, int y, int cx, int cy, int r) {
  int dx = x - cx;
  int dy = y - cy;

  return dx * dx + dy * dy <= r * r;
}

/* Template matching using normalized cross-correlation (TM_CCOEFF_NORMED) */
int find_stars_template(const float *rgba, int width, int height,
                        const float *template_data, int tpl_width,
                        int tpl_height, int cx, int cy, int radius,
                        float threshold, int max_stars,
                        star_position_t **stars_out, int *num_stars_out) {
  if (!rgba || !template_data || !stars_out || !num_stars_out)
    return 1;

  star_position_t *stars =
      allsky_safe_malloc(sizeof(star_position_t) * max_stars);
  if (!stars)
    return 2;

  int count = 0;
  int half_tw = tpl_width / 2;
  int half_th = tpl_height / 2;
  int min_dist = 8; // duplicate threshold

  /* Compute template mean and stddev once */
  float tpl_sum = 0, tpl_sqsum = 0;
  for (int j = 0; j < tpl_height; j++) {
    for (int i = 0; i < tpl_width; i++) {
      float v = template_data[j * tpl_width + i];
      tpl_sum += v;
      tpl_sqsum += v * v;
    }
  }

  /* Compute template mean and stddev once */
  float tpl_mean = tpl_sum / (tpl_width * tpl_height);
  float tpl_var = tpl_sqsum - tpl_mean * tpl_sum;
  float tpl_std = sqrtf(tpl_var);

#pragma omp parallel for collapse(2)
  for (int y = half_th; y < height - half_th; y++) {
    for (int x = half_tw; x < width - half_tw; x++) {

      /* Skip if not in detection circle */
      if (!in_circle(x, y, cx, cy, radius))
        continue;

      /* Compute local patch statistics */
      float img_sum = 0, img_sqsum = 0, cross = 0;
      for (int j = 0; j < tpl_height; j++) {
        for (int i = 0; i < tpl_width; i++) {
          int ix = x + i - half_tw;
          int iy = y + j - half_th;
          int idx = (iy * width + ix) * CHANNELS;
          float I = intensity(&rgba[idx]);
          float T = template_data[j * tpl_width + i];

          img_sum += I;
          img_sqsum += I * I;
          cross += (I * T);
        }
      }

      float img_mean = img_sum / (tpl_width * tpl_height);
      float img_var = img_sqsum - img_mean * img_sum;
      float img_std = sqrtf(img_var);

      float denom = img_std * tpl_std;
      float corr = 0;
      if (denom > 1e-5f)
        corr = (cross - img_mean * tpl_sum) / denom;

      if (corr < threshold)
        continue;

      /* Critical section for writing star */
#pragma omp critical
      {
        int too_close = 0;
        for (int i = 0; i < count; i++) {
          int dx = stars[i].x - x;
          int dy = stars[i].y - y;

          /* Check if star is too close to another star */
          if (dx * dx + dy * dy < min_dist * min_dist) {
            too_close = 1;
            break;
          }
        }

        if (!too_close && count < max_stars) {
          stars[count++] = (star_position_t){x, y};
#ifdef DEBUG
          printf("[DEBUG] Star match at (%d,%d): corr=%.3f\n", x, y, corr);
#endif
        }
      }
    }
  }

  *stars_out = stars;
  *num_stars_out = count;

  printf("Found %d stars using template matching.\n", count);
  return 0;
}

int stars_to_json(const star_position_t *stars, int num_stars,
                  const char *filename) {
  if (!stars || num_stars < 0 || !filename)
    return 1;

  /* Create the outer object */
  struct json_object *root = json_object_new_object();
  if (!root)
    return 2;

  /* Create the "stars" array */
  struct json_object *star_array = json_object_new_array();
  if (!star_array) {
    json_object_put(root);
    return 3;
  }

  /* Fill the array with x/y star objects */
  for (int i = 0; i < num_stars; i++) {
    struct json_object *obj = json_object_new_object();
    json_object_object_add(obj, "x", json_object_new_int(stars[i].x));
    json_object_object_add(obj, "y", json_object_new_int(stars[i].y));
    json_object_array_add(star_array, obj);
  }

  /* Attach array to root */
  json_object_object_add(root, "stars", star_array);

  /* Write to file */
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    json_object_put(root); // free memory
    return 4;
  }

  const char *json_str =
      json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
  fprintf(fp, "%s\n", json_str);
  fclose(fp);

  json_object_put(root); // free all JSON memory

  printf("Written %d stars to file: %s\n", num_stars, filename);
  return 0;
}

int estimate_sqm_from_star_count(int num_stars, float a, float b,
                                 float *sqm_out) {
  if (num_stars < 1 || !sqm_out)
    return 1;

  *sqm_out = a - b * log10f((float)num_stars);
  return 0;
}

int estimate_sqm_corrected(int num_stars, float exposure_s, float gain_db,
                           float a, float b, float *sqm_out) {
  if (num_stars < 1 || exposure_s <= 0.0f || !sqm_out)
    return 1;

  // Convert gain in dB to linear amplification (6 dB ≈ 2×)
  // float gain_factor = powf(2.0f, gain_db / 6.0f);

  // Combined correction factor: exposure (s) × gain factor
  // float correction_factor = exposure_s * gain_factor;
  float correction_factor = pow(exposure_s, 0.5f) * pow(2.0f, gain_db / 12.0f);
  if (correction_factor <= 0.0f)
    return 2;

  float corrected_count = (float)num_stars / correction_factor;

  /* Compute corrected SQM value */
  *sqm_out = a - b * log10f(corrected_count);
  return 0;
}

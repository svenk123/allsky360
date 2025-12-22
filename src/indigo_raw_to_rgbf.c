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
#include "indigo_raw_to_rgbf.h"
#include "allsky.h"
#include "debayer_bilinear.h"
#include "debayer_nni.h"
#include "debayer_vng.h"

#include <indigo/indigo_bus.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int indigo_raw_get_width(const unsigned char *raw_data, int *width_out) {
  if (!raw_data || !width_out)
    return 1;

  const indigo_raw_header *header = (const indigo_raw_header *)raw_data;
  int width = header->width;

  *width_out = width;

  return 0;
}

int indigo_raw_get_height(const unsigned char *raw_data, int *height_out) {
  if (!raw_data || !height_out)
    return 1;

  const indigo_raw_header *header = (const indigo_raw_header *)raw_data;
  int height = header->height;

  *height_out = height;

  return 0;
}

int indigo_raw_to_rgb16(unsigned short **rgb16_out, int *width_out,
                        int *height_out, const unsigned char *raw_data,
                        int debayer_alg) {
  if (!raw_data || !rgb16_out || !width_out || !height_out)
    return 1;

  const indigo_raw_header *header = (const indigo_raw_header *)raw_data;
  int width = header->width;
  int height = header->height;
  uint32_t type = header->signature;
  int pixel_count = width * height;

  const unsigned char *data_ptr = raw_data + sizeof(indigo_raw_header);
  unsigned short *image_data = NULL;
  unsigned char *image_u8 = NULL;
  char bayer_pattern[10] = "";
  int x_offset = 0, y_offset = 0;

  switch (type) {
  case INDIGO_RAW_MONO8: // MONO8
    image_u8 = (unsigned char *)allsky_safe_malloc(pixel_count);
    memcpy(image_u8, data_ptr, pixel_count);
    data_ptr += pixel_count;
    break;
  case INDIGO_RAW_MONO16: // MONO16
    image_data = (unsigned short *)allsky_safe_malloc(pixel_count *
                                                      sizeof(unsigned short));
    memcpy(image_data, data_ptr, pixel_count * 2);
    data_ptr += pixel_count * 2;
    break;
  case INDIGO_RAW_RGB24: // RGB24
    image_u8 = (unsigned char *)allsky_safe_malloc(pixel_count * 3);
    memcpy(image_u8, data_ptr, pixel_count * 3);
    data_ptr += pixel_count * 3;
    break;
  case INDIGO_RAW_RGBA32: // RGBA32
  case INDIGO_RAW_ABGR32: // ABGR32
    image_u8 = (unsigned char *)allsky_safe_malloc(pixel_count * 4);
    memcpy(image_u8, data_ptr, pixel_count * 4);
    data_ptr += pixel_count * 4;
    break;
  case INDIGO_RAW_RGB48: // RGB48
    image_data = (unsigned short *)allsky_safe_malloc(pixel_count * 3 *
                                                      sizeof(unsigned short));
    memcpy(image_data, data_ptr, pixel_count * 3 * 2);
    data_ptr += pixel_count * 3 * 2;
    break;
  default:
    fprintf(stderr, "Unsupported RAW signature: 0x%08X\n", type);
    return 1;
  }

  const char *meta = (const char *)data_ptr;
  if (strstr(meta, "SIMPLE=T")) {
    char *bayer = strstr(meta, "BAYERPAT=");
    if (bayer) {
      printf("BAYERPAT gefunden!\n");
      sscanf(bayer, "BAYERPAT='%4[^']'", bayer_pattern);
    }

    char *xoff = strstr(meta, "XBAYROFF=");
    if (xoff) {
      printf("XBAYROFF gefunden!\n");
      sscanf(xoff, "XBAYROFF=%d", &x_offset);
    }

    char *yoff = strstr(meta, "YBAYROFF=");
    if (yoff) {
      printf("YBAYROFF gefunden\n");
      sscanf(yoff, "YBAYROFF=%d", &y_offset);
    }
  }

  printf("Indigo raw image header: BAYERPAT=%s, XBAYROFF=%d, YBAYROFF=%d\n",
         bayer_pattern, x_offset, y_offset);

  unsigned short *rgb16 = (unsigned short *)allsky_safe_calloc(
      pixel_count * 4, sizeof(unsigned short));
  if (!rgb16) {
    if (image_data != NULL)
      allsky_safe_free(image_data);
    if (image_u8 != NULL)
      allsky_safe_free(image_u8);

    return 1;
  }

  if (strlen(bayer_pattern) > 0) {
    if (type == INDIGO_RAW_MONO8) {
      image_data = (unsigned short *)allsky_safe_malloc(pixel_count *
                                                        sizeof(unsigned short));
      for (int i = 0; i < pixel_count; i++)
        image_data[i] = image_u8[i] << 8;
      allsky_safe_free(image_u8);
    }

    if (debayer_alg == 1)
      debayer_bilinear_rgb16(image_data, width, height, rgb16, x_offset,
                             y_offset, bayer_pattern);
    else if (debayer_alg == 2)
      debayer_vng_rgb16(image_data, width, height, rgb16, x_offset, y_offset,
                        bayer_pattern);
    else
      debayer_nearest_neighbor_rgb16(image_data, width, height, rgb16, x_offset,
                                     y_offset, bayer_pattern);
  } else {
    for (int i = 0; i < pixel_count; i++) {
      if (type == INDIGO_RAW_MONO8) { // MONO8
        unsigned char val = image_u8[i];
        rgb16[i * 4 + 0] = val << 8;
        rgb16[i * 4 + 1] = val << 8;
        rgb16[i * 4 + 2] = val << 8;
        rgb16[i * 4 + 3] = 65535;
      } else if (type == INDIGO_RAW_MONO16) { // MONO16
        unsigned short val = image_data[i];
        rgb16[i * 4 + 0] = val;
        rgb16[i * 4 + 1] = val;
        rgb16[i * 4 + 2] = val;
        rgb16[i * 4 + 3] = 65535;
      } else if (type == INDIGO_RAW_RGB24) { // RGB24
        rgb16[i * 4 + 0] = image_u8[i * 3 + 2] << 8;
        rgb16[i * 4 + 1] = image_u8[i * 3 + 1] << 8;
        rgb16[i * 4 + 2] = image_u8[i * 3 + 0] << 8;
        rgb16[i * 4 + 3] = 65535;
      } else if (type == INDIGO_RAW_RGBA32) { // RGBA32
        rgb16[i * 4 + 0] = image_u8[i * 4 + 2] << 8;
        rgb16[i * 4 + 1] = image_u8[i * 4 + 1] << 8;
        rgb16[i * 4 + 2] = image_u8[i * 4 + 0] << 8;
        rgb16[i * 4 + 3] = image_u8[i * 4 + 3] << 8;
      } else if (type == INDIGO_RAW_ABGR32) { // ABGR32
        rgb16[i * 4 + 0] = image_u8[i * 4 + 1] << 8;
        rgb16[i * 4 + 1] = image_u8[i * 4 + 2] << 8;
        rgb16[i * 4 + 2] = image_u8[i * 4 + 3] << 8;
        rgb16[i * 4 + 3] = image_u8[i * 4 + 0] << 8;
      } else if (type == INDIGO_RAW_RGB48) { // RGB48
        rgb16[i * 4 + 0] = image_data[i * 3 + 2];
        rgb16[i * 4 + 1] = image_data[i * 3 + 1];
        rgb16[i * 4 + 2] = image_data[i * 3 + 0];
        rgb16[i * 4 + 3] = 65535;
      }
    }
  }

  *rgb16_out = rgb16;
  *width_out = width;
  *height_out = height;

  allsky_safe_free(image_data);
  allsky_safe_free(image_u8);
  return 0;
}

#define MAX_U16 65535.0f

static void normalize_rgbf(float *rgba, int pixel_count) {
  if (!rgba || pixel_count <= 0)
    return;

#pragma omp parallel for
  for (int i = 0; i < pixel_count; ++i) {
    int idx = i * 4;
    rgba[idx + 0] = clampf1(rgba[idx + 0] / MAX_U16); // R
    rgba[idx + 1] = clampf1(rgba[idx + 1] / MAX_U16); // G
    rgba[idx + 2] = clampf1(rgba[idx + 2] / MAX_U16); // B
    rgba[idx + 3] = 1.0f;                             // set alpha to 1.0f
  }
}

int indigo_raw_to_rgbf1(float *rgba, int width, int height,
                        const unsigned char *raw_data, int debayer_alg,
                        int crop_width, int crop_height, int crop_offset_x,
                        int crop_offset_y) {
  if (!raw_data || !rgba || width <= 0 || height <= 0)
    return 1;

  /* Validate crop parameters */
  if (crop_width <= 0 || crop_height <= 0 || crop_offset_x < 0 ||
      crop_offset_y < 0) {
    fprintf(stderr, "Error: Invalid crop parameters - width=%d, height=%d, offset_x=%d, offset_y=%d\n",
            crop_width, crop_height, crop_offset_x, crop_offset_y);
    return 1;
  }

  /* Check if the crop region is within the original image */
  if (crop_offset_x + crop_width > width ||
      crop_offset_y + crop_height > height) {
    fprintf(stderr, "Error: Crop region exceeds image boundaries - "
            "image size: %dx%d, crop region: %dx%d at offset (%d,%d)\n",
            width, height, crop_width, crop_height, crop_offset_x, crop_offset_y);
    return 1;
  }

  /* Assumption: rgba-Array must have at least crop_pixel_count * 4 float
   values. We cannot directly check the size of the array, so we trust that the
   caller provides a sufficiently large array */

  const indigo_raw_header *header = (const indigo_raw_header *)raw_data;
  uint32_t type = header->signature;
  int pixel_count = width * height;
  int crop_pixel_count = crop_width * crop_height;

  const unsigned char *data_ptr = raw_data + sizeof(indigo_raw_header);
  unsigned short *image_data = NULL;
  unsigned char *image_u8 = NULL;
  char bayer_pattern[10] = "";
  int x_offset = 0, y_offset = 0;

  /* Temporary arrays for the entire image */
  unsigned short *full_image_data = NULL;
  unsigned char *full_image_u8 = NULL;

  switch (type) {
  case INDIGO_RAW_MONO8:
    full_image_u8 = (unsigned char *)allsky_safe_malloc(pixel_count);
    memcpy(full_image_u8, data_ptr, pixel_count);
    data_ptr += pixel_count;
    break;
  case INDIGO_RAW_MONO16:
    full_image_data = (unsigned short *)allsky_safe_malloc(
        pixel_count * sizeof(unsigned short));
    memcpy(full_image_data, data_ptr, pixel_count * 2);
    data_ptr += pixel_count * 2;
    break;
  case INDIGO_RAW_RGB24: // RGB24
    full_image_u8 = (unsigned char *)allsky_safe_malloc(pixel_count * 3);
    memcpy(full_image_u8, data_ptr, pixel_count * 3);
    data_ptr += pixel_count * 3;
    break;
  case INDIGO_RAW_RGBA32: // RGBA32
  case INDIGO_RAW_ABGR32: // ABGR32
    full_image_u8 = (unsigned char *)allsky_safe_malloc(pixel_count * 4);
    memcpy(full_image_u8, data_ptr, pixel_count * 4);
    data_ptr += pixel_count * 4;
    break;
  case INDIGO_RAW_RGB48: // RGB48
    full_image_data = (unsigned short *)allsky_safe_malloc(
        pixel_count * 3 * sizeof(unsigned short));
    memcpy(full_image_data, data_ptr, pixel_count * 3 * 2);
    data_ptr += pixel_count * 3 * 2;
    break;
  default:
    fprintf(stderr, "Unsupported RAW signature: 0x%08X\n", type);
    return 1;
  }

  const char *meta = (const char *)data_ptr;
  if (strstr(meta, "SIMPLE=T")) {
    char *bayer = strstr(meta, "BAYERPAT=");
    if (bayer)
      sscanf(bayer, "BAYERPAT='%4[^']'", bayer_pattern);

    char *xoff = strstr(meta, "XBAYROFF=");
    if (xoff)
      sscanf(xoff, "XBAYROFF=%d", &x_offset);

    char *yoff = strstr(meta, "YBAYROFF=");
    if (yoff)
      sscanf(yoff, "YBAYROFF=%d", &y_offset);
  }

  printf("Indigo raw image header: BAYERPAT=%s, XBAYROFF=%d, YBAYROFF=%d\n",
         bayer_pattern, x_offset, y_offset);

  /* Extract the crop region from the entire image */
  switch (type) {
  case INDIGO_RAW_MONO8:
    image_u8 = (unsigned char *)allsky_safe_malloc(crop_pixel_count);
    for (int y = 0; y < crop_height; y++) {
      for (int x = 0; x < crop_width; x++) {
        int src_idx = (crop_offset_y + y) * width + (crop_offset_x + x);
        int dst_idx = y * crop_width + x;
        image_u8[dst_idx] = full_image_u8[src_idx];
      }
    }
    allsky_safe_free(full_image_u8);
    break;
  case INDIGO_RAW_MONO16:
    image_data = (unsigned short *)allsky_safe_malloc(crop_pixel_count *
                                                      sizeof(unsigned short));
    for (int y = 0; y < crop_height; y++) {
      for (int x = 0; x < crop_width; x++) {
        int src_idx = (crop_offset_y + y) * width + (crop_offset_x + x);
        int dst_idx = y * crop_width + x;
        image_data[dst_idx] = full_image_data[src_idx];
      }
    }
    allsky_safe_free(full_image_data);
    break;
  case INDIGO_RAW_RGB24:
    image_u8 = (unsigned char *)allsky_safe_malloc(crop_pixel_count * 3);
    for (int y = 0; y < crop_height; y++) {
      for (int x = 0; x < crop_width; x++) {
        int src_idx = ((crop_offset_y + y) * width + (crop_offset_x + x)) * 3;
        int dst_idx = (y * crop_width + x) * 3;
        image_u8[dst_idx + 0] = full_image_u8[src_idx + 0];
        image_u8[dst_idx + 1] = full_image_u8[src_idx + 1];
        image_u8[dst_idx + 2] = full_image_u8[src_idx + 2];
      }
    }
    allsky_safe_free(full_image_u8);
    break;
  case INDIGO_RAW_RGBA32:
  case INDIGO_RAW_ABGR32:
    image_u8 = (unsigned char *)allsky_safe_malloc(crop_pixel_count * 4);
    for (int y = 0; y < crop_height; y++) {
      for (int x = 0; x < crop_width; x++) {
        int src_idx = ((crop_offset_y + y) * width + (crop_offset_x + x)) * 4;
        int dst_idx = (y * crop_width + x) * 4;
        image_u8[dst_idx + 0] = full_image_u8[src_idx + 0];
        image_u8[dst_idx + 1] = full_image_u8[src_idx + 1];
        image_u8[dst_idx + 2] = full_image_u8[src_idx + 2];
        image_u8[dst_idx + 3] = full_image_u8[src_idx + 3];
      }
    }
    allsky_safe_free(full_image_u8);
    break;
  case INDIGO_RAW_RGB48:
    image_data = (unsigned short *)allsky_safe_malloc(crop_pixel_count * 3 *
                                                      sizeof(unsigned short));
    for (int y = 0; y < crop_height; y++) {
      for (int x = 0; x < crop_width; x++) {
        int src_idx = ((crop_offset_y + y) * width + (crop_offset_x + x)) * 3;
        int dst_idx = (y * crop_width + x) * 3;
        image_data[dst_idx + 0] = full_image_data[src_idx + 0];
        image_data[dst_idx + 1] = full_image_data[src_idx + 1];
        image_data[dst_idx + 2] = full_image_data[src_idx + 2];
      }
    }
    allsky_safe_free(full_image_data);
    break;
  }

#if 0
    float *rgbf = (float *)allsky_safe_calloc(pixel_count * 4, sizeof(float));
    if (!rgbf) {
	allsky_safe_free(image_data);
	allsky_safe_free(image_u8);

	return 1;
    }
#endif

  if (strlen(bayer_pattern) > 0) {
    if (type == INDIGO_RAW_MONO8) {
      unsigned short *temp_image_data = (unsigned short *)allsky_safe_malloc(
          crop_pixel_count * sizeof(unsigned short));
      for (int i = 0; i < crop_pixel_count; i++)
        temp_image_data[i] = image_u8[i] << 8;
      allsky_safe_free(image_u8);
      image_data = temp_image_data;
    }

    if (debayer_alg == 1)
      debayer_bilinear_rgbf(image_data, crop_width, crop_height, rgba, x_offset,
                            y_offset, bayer_pattern);
    else if (debayer_alg == 2)
      debayer_vng_rgbf(image_data, crop_width, crop_height, rgba, x_offset,
                       y_offset, bayer_pattern);
    else
      debayer_nearest_neighbor_rgbf(image_data, crop_width, crop_height, rgba,
                                    x_offset, y_offset, bayer_pattern);

    /* Normalize to 0.0-1.0f */
    normalize_rgbf(rgba, crop_pixel_count);
  } else {
#pragma omp parallel for
    for (int i = 0; i < crop_pixel_count; i++) {
      if (type == INDIGO_RAW_MONO8) {
        float val = (float)(image_u8[i] << 8);
        rgba[i * 4 + 0] = val;
        rgba[i * 4 + 1] = val;
        rgba[i * 4 + 2] = val;
        rgba[i * 4 + 3] = MAX_U16;
      } else if (type == INDIGO_RAW_MONO16) {
        float val = (float)(image_data[i]);
        rgba[i * 4 + 0] = val;
        rgba[i * 4 + 1] = val;
        rgba[i * 4 + 2] = val;
        rgba[i * 4 + 3] = MAX_U16;
      } else if (type == INDIGO_RAW_RGB24) { // RGB24
        rgba[i * 4 + 0] = (float)(image_u8[i * 3 + 2] << 8);
        rgba[i * 4 + 1] = (float)(image_u8[i * 3 + 1] << 8);
        rgba[i * 4 + 2] = (float)(image_u8[i * 3 + 0] << 8);
        rgba[i * 4 + 3] = MAX_U16;
      } else if (type == INDIGO_RAW_RGBA32) { // RGBA32
        rgba[i * 4 + 0] = (float)(image_u8[i * 4 + 2] << 8);
        rgba[i * 4 + 1] = (float)(image_u8[i * 4 + 1] << 8);
        rgba[i * 4 + 2] = (float)(image_u8[i * 4 + 0] << 8);
        rgba[i * 4 + 3] = (float)(image_u8[i * 4 + 3] << 8);
      } else if (type == INDIGO_RAW_ABGR32) { // ABGR32
        rgba[i * 4 + 0] = (float)(image_u8[i * 4 + 1] << 8);
        rgba[i * 4 + 1] = (float)(image_u8[i * 4 + 2] << 8);
        rgba[i * 4 + 2] = (float)(image_u8[i * 4 + 3] << 8);
        rgba[i * 4 + 3] = (float)(image_u8[i * 4 + 0] << 8);
      } else if (type == INDIGO_RAW_RGB48) { // RGB48
        rgba[i * 4 + 0] = (float)(image_data[i * 3 + 2]);
        rgba[i * 4 + 1] = (float)(image_data[i * 3 + 1]);
        rgba[i * 4 + 2] = (float)(image_data[i * 3 + 0]);
        rgba[i * 4 + 3] = MAX_U16;
      }
    }
  }

  allsky_safe_free(image_data);
  allsky_safe_free(image_u8);
  return 0;
}

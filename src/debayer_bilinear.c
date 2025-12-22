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
#include "debayer_bilinear.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int debayer_bilinear_rgb16(const unsigned short *raw, int width, int height,
                           unsigned short *rgb, int x_offset, int y_offset,
                           const char *bayer_pattern) {
  int offsets = 0;

  // Bayer pattern as offset value
  if (strcmp(bayer_pattern, "RGGB") == 0)
    offsets = 0x00;
  else if (strcmp(bayer_pattern, "GBRG") == 0)
    offsets = 0x01;
  else if (strcmp(bayer_pattern, "GRBG") == 0)
    offsets = 0x10;
  else if (strcmp(bayer_pattern, "BGGR") == 0)
    offsets = 0x11;
  else {
    fprintf(stderr, "ERROR: Unknoww bayer pattern '%s'\n", bayer_pattern);
    return 1;
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      int rgb_idx = (y * width + x) * CHANNELS;

      float r = 0, g = 0, b = 0;

      // Determine bayer pattern for this pixerl
      int local_offset =
          offsets ^ (((x + x_offset) & 1) << 4 | ((y + y_offset) & 1));

      switch (local_offset) {
      case 0x00: // RGGB (R)
        r = raw[idx];
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 3.0
                                     : raw[idx];
        b = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0
                : raw[idx];
        break;

      case 0x10: // GRBG (G)
        r = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0
                                     : raw[idx];
        g = raw[idx];
        b = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0
                : raw[idx];
        break;

      case 0x01: // GBRG (G)
        r = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0
                : raw[idx];
        g = raw[idx];
        b = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0
                                     : raw[idx];
        break;

      case 0x11: // BGGR (B)
        r = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0
                : raw[idx];
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0
                                     : raw[idx];
        b = raw[idx];
        break;
      }

      rgb[rgb_idx + 0] = (unsigned short)r;
      rgb[rgb_idx + 1] = (unsigned short)g;
      rgb[rgb_idx + 2] = (unsigned short)b;
      rgb[rgb_idx + 3] = 65535; // Alpha
    }
  }

  printf("Bilinear debayering (pattern: %s) ok.\n", bayer_pattern);

  return 0;
}

int debayer_bilinear_rgbf(const uint16_t *raw, int width, int height,
                          float *rgbf, int x_offset, int y_offset,
                          const char *bayer_pattern) {
  int offsets = 0;

  // Bayer pattern as offset value
  if (strcmp(bayer_pattern, "RGGB") == 0)
    offsets = 0x00;
  else if (strcmp(bayer_pattern, "GBRG") == 0)
    offsets = 0x01;
  else if (strcmp(bayer_pattern, "GRBG") == 0)
    offsets = 0x10;
  else if (strcmp(bayer_pattern, "BGGR") == 0)
    offsets = 0x11;
  else {
    fprintf(stderr, "ERROR: Unknown bayer pattern '%s'\n", bayer_pattern);
    return 1;
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      int rgb_idx = idx * CHANNELS;

      float r = 0.0f, g = 0.0f, b = 0.0f;

      // Determine local bayer pattern
      int local_offset =
          offsets ^ (((x + x_offset) & 1) << 4 | ((y + y_offset) & 1));

      switch (local_offset) {
      case 0x00: // RGGB (R)
        r = raw[idx];
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 3.0f
                                     : (float)raw[idx];
        b = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0f
                : (float)raw[idx];
        break;

      case 0x10: // GRBG (G)
        r = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0f
                                     : (float)raw[idx];
        g = raw[idx];
        b = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0f
                : (float)raw[idx];
        break;

      case 0x01: // GBRG (G)
        r = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0f
                : (float)raw[idx];
        g = raw[idx];
        b = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0f
                                     : (float)raw[idx];
        break;

      case 0x11: // BGGR (B)
        r = (y > 0 && y < height - 1)
                ? (raw[idx - width] + raw[idx + width]) / 2.0f
                : (float)raw[idx];
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0f
                                     : (float)raw[idx];
        b = raw[idx];
        break;
      }

      rgbf[rgb_idx + 0] = r;
      rgbf[rgb_idx + 1] = g;
      rgbf[rgb_idx + 2] = b;
      rgbf[rgb_idx + 3] = 65535.0f; // Alpha
    }
  }

  printf("Bilinear debayering (float) (pattern: %s) ok.\n", bayer_pattern);

  return 0;
}

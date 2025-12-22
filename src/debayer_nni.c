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
#include "debayer_nni.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int debayer_nearest_neighbor_rgb16(const uint16_t *raw, int width, int height,
                                   uint16_t *rgb, int x_offset, int y_offset,
                                   const char *bayer_pattern) {
  // Set black border to the right bottom border
  for (int y = 0; y < height; y++) {
    int idx = (y * width + (width - 1)) * CHANNELS;
    rgb[idx + 0] = 0;
    rgb[idx + 1] = 0;
    rgb[idx + 2] = 0;
    rgb[idx + 3] = 65535;
  }

  for (int x = 0; x < width; x++) {
    int idx = ((height - 1) * width + x) * CHANNELS;
    rgb[idx + 0] = 0;
    rgb[idx + 1] = 0;
    rgb[idx + 2] = 0;
    rgb[idx + 3] = 65535;
  }

  for (int y = 0; y < height - 1; y++) {
    for (int x = 0; x < width - 1; x++) {
      int idx = y * width + x;
      int rgb_idx = (y * width + x) * CHANNELS;

      int x_mod = (x + x_offset) % 2;
      int y_mod = (y + y_offset) % 2;

      uint16_t R = 0, G = 0, B = 0;

      if (strcmp(bayer_pattern, "RGGB") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          R = raw[idx];
          G = raw[idx + 1];
          B = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          R = raw[idx - 1];
          G = raw[idx];
          B = raw[idx + width];
        } else if (x_mod == 0 && y_mod == 1) {
          R = raw[idx - width];
          G = raw[idx];
          B = raw[idx + 1];
        } else {
          R = raw[idx - width];
          G = raw[idx];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "BGGR") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          B = raw[idx];
          G = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          B = raw[idx - 1];
          G = raw[idx];
          R = raw[idx + width];
        } else if (x_mod == 0 && y_mod == 1) {
          B = raw[idx - width];
          G = raw[idx];
          R = raw[idx + 1];
        } else {
          B = raw[idx - width];
          G = raw[idx];
          R = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "GBRG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          G = raw[idx];
          R = raw[idx + 1];
          B = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx - width];
          R = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx + 1];
        } else {
          G = raw[idx];
          B = raw[idx + width];
          R = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "GRBG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          G = raw[idx];
          B = raw[idx - width];
          R = raw[idx + 1];
        } else {
          G = raw[idx];
          R = raw[idx + width];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "GBGR") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          B = raw[idx];
          G = raw[idx - width];
          R = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx + 1];
        } else {
          R = raw[idx];
          G = raw[idx + width];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "RGBG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          R = raw[idx];
          G = raw[idx + 1];
          B = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx - width];
          R = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          B = raw[idx];
          G = raw[idx + 1];
          R = raw[idx - width];
        } else {
          G = raw[idx];
          R = raw[idx + width];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "BGRG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          B = raw[idx];
          G = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          R = raw[idx];
          G = raw[idx + 1];
          B = raw[idx - width];
        } else {
          G = raw[idx];
          B = raw[idx + width];
          R = raw[idx - 1];
        }
      } else {
        printf("ERROR: Unknown bayer pattern %s\n", bayer_pattern);
        return 1;
      }

      rgb[rgb_idx + 0] = R;
      rgb[rgb_idx + 1] = G;
      rgb[rgb_idx + 2] = B;
      rgb[rgb_idx + 3] = 65535;
    }
  }

  printf("Nearest-Neighbor-Debayering (bayer pattern: %s) ok!", bayer_pattern);

  return 0;
}

int debayer_nearest_neighbor_rgbf(const uint16_t *raw, int width, int height,
                                  float *rgbf, int x_offset, int y_offset,
                                  const char *bayer_pattern) {
  // Set black border to right bottom border
  for (int y = 0; y < height; y++) {
    int idx = (y * width + (width - 1)) * CHANNELS;
    rgbf[idx + 0] = 0.0f;
    rgbf[idx + 1] = 0.0f;
    rgbf[idx + 2] = 0.0f;
    rgbf[idx + 3] = 65535.0f;
  }

  for (int x = 0; x < width; x++) {
    int idx = ((height - 1) * width + x) * CHANNELS;
    rgbf[idx + 0] = 0.0f;
    rgbf[idx + 1] = 0.0f;
    rgbf[idx + 2] = 0.0f;
    rgbf[idx + 3] = 65535.0f;
  }

  for (int y = 0; y < height - 1; y++) {
    for (int x = 0; x < width - 1; x++) {
      int idx = y * width + x;
      int rgb_idx = (y * width + x) * CHANNELS;

      int x_mod = (x + x_offset) % 2;
      int y_mod = (y + y_offset) % 2;

      float R = 0.0f, G = 0.0f, B = 0.0f;

      if (strcmp(bayer_pattern, "RGGB") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          R = raw[idx];
          G = raw[idx + 1];
          B = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          R = raw[idx - 1];
          G = raw[idx];
          B = raw[idx + width];
        } else if (x_mod == 0 && y_mod == 1) {
          R = raw[idx - width];
          G = raw[idx];
          B = raw[idx + 1];
        } else {
          R = raw[idx - width];
          G = raw[idx];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "BGGR") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          B = raw[idx];
          G = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          B = raw[idx - 1];
          G = raw[idx];
          R = raw[idx + width];
        } else if (x_mod == 0 && y_mod == 1) {
          B = raw[idx - width];
          G = raw[idx];
          R = raw[idx + 1];
        } else {
          B = raw[idx - width];
          G = raw[idx];
          R = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "GBRG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          G = raw[idx];
          R = raw[idx + 1];
          B = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx - width];
          R = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx + 1];
        } else {
          G = raw[idx];
          B = raw[idx + width];
          R = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "GRBG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          G = raw[idx];
          B = raw[idx - width];
          R = raw[idx + 1];
        } else {
          G = raw[idx];
          R = raw[idx + width];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "GBGR") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          B = raw[idx];
          G = raw[idx - width];
          R = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx + 1];
        } else {
          R = raw[idx];
          G = raw[idx + width];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "RGBG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          R = raw[idx];
          G = raw[idx + 1];
          B = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          B = raw[idx - width];
          R = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          B = raw[idx];
          G = raw[idx + 1];
          R = raw[idx - width];
        } else {
          G = raw[idx];
          R = raw[idx + width];
          B = raw[idx - 1];
        }
      } else if (strcmp(bayer_pattern, "BGRG") == 0) {
        if (x_mod == 0 && y_mod == 0) {
          B = raw[idx];
          G = raw[idx + 1];
          R = raw[idx + width];
        } else if (x_mod == 1 && y_mod == 0) {
          G = raw[idx];
          R = raw[idx - width];
          B = raw[idx - 1];
        } else if (x_mod == 0 && y_mod == 1) {
          R = raw[idx];
          G = raw[idx + 1];
          B = raw[idx - width];
        } else {
          G = raw[idx];
          B = raw[idx + width];
          R = raw[idx - 1];
        }
      } else {
        printf("ERROR: Unknown bayer pattern %s\n", bayer_pattern);
        return 1;
      }

      rgbf[rgb_idx + 0] = R;
      rgbf[rgb_idx + 1] = G;
      rgbf[rgb_idx + 2] = B;
      rgbf[rgb_idx + 3] = 65535.0f;
    }
  }

  printf("Nearest-Neighbor-Debayering (float) (bayer pattern: %s) "
         "abgeschlossen!\n",
         bayer_pattern);

  return 0;
}

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
#include <string.h>

/* Channel (0=R, 1=G, 2=B) of the raw pixel x,y. bayer_pattern gives the
 * 2x2 cell row by row ("RGGB": R G / G B), shifted by x_offset/y_offset. */
static inline int bayer_channel(const char *bayer_pattern, int x, int y,
                                int x_offset, int y_offset) {
  char c = bayer_pattern[((y + y_offset) & 1) * 2 + ((x + x_offset) & 1)];
  return c == 'R' ? 0 : (c == 'B' ? 2 : 1);
}

/* A usable pattern: four characters, one R, one B and two G */
static int bayer_pattern_valid(const char *bayer_pattern) {
  if (!bayer_pattern || strlen(bayer_pattern) != 4)
    return 0;
  int r = 0, g = 0, b = 0;
  for (int i = 0; i < 4; i++) {
    if (bayer_pattern[i] == 'R')
      r++;
    else if (bayer_pattern[i] == 'G')
      g++;
    else if (bayer_pattern[i] == 'B')
      b++;
  }
  return r == 1 && g == 2 && b == 1;
}

/* Nearest-neighbor debayering of the pixel x,y: the pixel keeps its own
 * value, the two missing colors come from its 2x2 Bayer cell (green
 * averaged). At the image border the cell is moved inside the image, so
 * no read leaves the buffer. Requires width, height >= 3. */
static void debayer_nni_pixel(const uint16_t *raw, int width, int height,
                              int x, int y, int x_offset, int y_offset,
                              const char *bayer_pattern, float out[3]) {
  int bx = x - ((x + x_offset) & 1);
  int by = y - ((y + y_offset) & 1);
  if (bx < 0)
    bx += 2;
  if (bx + 1 >= width)
    bx -= 2;
  if (by < 0)
    by += 2;
  if (by + 1 >= height)
    by -= 2;

  float sum[3] = {0.0f, 0.0f, 0.0f};
  int count[3] = {0, 0, 0};
  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      int c = bayer_channel(bayer_pattern, bx + dx, by + dy, x_offset,
                            y_offset);
      sum[c] += raw[(by + dy) * width + (bx + dx)];
      count[c]++;
    }
  }
  for (int c = 0; c < 3; c++)
    out[c] = count[c] ? sum[c] / count[c] : 0.0f;

  out[bayer_channel(bayer_pattern, x, y, x_offset, y_offset)] =
      raw[y * width + x];
}

int debayer_nearest_neighbor_rgb16(const uint16_t *raw, int width, int height,
                                   uint16_t *rgb, int x_offset, int y_offset,
                                   const char *bayer_pattern) {
  if (!raw || !rgb || width < 3 || height < 3 ||
      !bayer_pattern_valid(bayer_pattern)) {
    printf("ERROR: NNI debayering: invalid image (%dx%d) or bayer pattern "
           "%s\n",
           width, height, bayer_pattern ? bayer_pattern : "(null)");
    return 1;
  }

#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float c[3];
      debayer_nni_pixel(raw, width, height, x, y, x_offset, y_offset,
                        bayer_pattern, c);
      size_t idx = ((size_t)y * width + x) * CHANNELS;
      rgb[idx + 0] = (uint16_t)(c[0] + 0.5f);
      rgb[idx + 1] = (uint16_t)(c[1] + 0.5f);
      rgb[idx + 2] = (uint16_t)(c[2] + 0.5f);
      rgb[idx + 3] = 65535;
    }
  }

  printf("Nearest-Neighbor-Debayering (bayer pattern: %s) ok.\n",
         bayer_pattern);
  return 0;
}

int debayer_nearest_neighbor_rgbf(const uint16_t *raw, int width, int height,
                                  float *rgbf, int x_offset, int y_offset,
                                  const char *bayer_pattern) {
  if (!raw || !rgbf || width < 3 || height < 3 ||
      !bayer_pattern_valid(bayer_pattern)) {
    printf("ERROR: NNI debayering: invalid image (%dx%d) or bayer pattern "
           "%s\n",
           width, height, bayer_pattern ? bayer_pattern : "(null)");
    return 1;
  }

#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float c[3];
      debayer_nni_pixel(raw, width, height, x, y, x_offset, y_offset,
                        bayer_pattern, c);
      size_t idx = ((size_t)y * width + x) * CHANNELS;
      rgbf[idx + 0] = c[0];
      rgbf[idx + 1] = c[1];
      rgbf[idx + 2] = c[2];
      rgbf[idx + 3] = 65535.0f;
    }
  }

  printf("Nearest-Neighbor-Debayering (float) (bayer pattern: %s) ok.\n",
         bayer_pattern);
  return 0;
}

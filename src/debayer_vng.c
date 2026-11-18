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
#include <stdio.h>
#include <string.h>
#include "debayer_vng.h"

/* Channel (0=R, 1=G, 2=B) per Bayer parity [y & 1][x & 1] of the pattern
 * string (2x2 cell row by row, "RGGB": R G / G B), shifted by the offsets.
 * Returns 0 for a usable pattern (one R, one B, two G). */
static int vng_channel_map(const char *bayer_pattern, int x_offset,
                           int y_offset, int ch[2][2]) {
  if (!bayer_pattern || strlen(bayer_pattern) != 4)
    return 1;
  int count[3] = {0, 0, 0};
  for (int py = 0; py < 2; py++) {
    for (int px = 0; px < 2; px++) {
      char c = bayer_pattern[((py + y_offset) & 1) * 2 + ((px + x_offset) & 1)];
      if (c != 'R' && c != 'G' && c != 'B')
        return 1;
      ch[py][px] = c == 'R' ? 0 : (c == 'B' ? 2 : 1);
      count[ch[py][px]]++;
    }
  }
  return (count[0] == 1 && count[1] == 2 && count[2] == 1) ? 0 : 1;
}

/* Reflect a coordinate at the border by 2 pixels (-1 -> 1, n -> n - 2):
 * the replacement keeps the Bayer parity, i.e. the color */
static inline int vng_reflect(int v, int n) {
  if (v < 0)
    return -v;
  if (v >= n)
    return 2 * (n - 1) - v;
  return v;
}

/* Weighted 3x3 mean per color (center weight 5, neighbors 1) of pixel x,y.
 * Neighbors outside the image are reflected, so every pixel is computed
 * and no read leaves the buffer. Requires width, height >= 2. */
static inline void vng_pixel(const unsigned short *raw, int width, int height,
                             int x, int y, const int ch[2][2], float out[3]) {
  float sum[3] = {0.0f, 0.0f, 0.0f};
  float count[3] = {0.0f, 0.0f, 0.0f};

  for (int ky = -1; ky <= 1; ky++) {
    int ny = vng_reflect(y + ky, height);
    for (int kx = -1; kx <= 1; kx++) {
      int nx = vng_reflect(x + kx, width);
      float weight = (kx == 0 && ky == 0) ? 5.0f : 1.0f;
      int c = ch[ny & 1][nx & 1];
      sum[c] += (float)raw[ny * width + nx] * weight;
      count[c] += weight;
    }
  }

  for (int c = 0; c < 3; c++)
    out[c] = (count[c] > 0.0f) ? (sum[c] / count[c]) : 0.0f;
}

int debayer_vng_rgb16(const unsigned short *raw, int width, int height,
                      unsigned short *rgb, int x_offset, int y_offset,
                      const char *bayer_pattern) {
  int ch[2][2];
  if (!raw || !rgb || width < 2 || height < 2 ||
      vng_channel_map(bayer_pattern, x_offset, y_offset, ch) != 0) {
    fprintf(stderr,
            "ERROR: VNG debayering: invalid image (%dx%d) or bayer pattern "
            "%s\n",
            width, height, bayer_pattern ? bayer_pattern : "(null)");
    return 1;
  }

  printf("VNG-Debayering: Pattern=%s, Offsets=(%d,%d)\n", bayer_pattern,
         x_offset, y_offset);

#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float c[3];
      vng_pixel(raw, width, height, x, y, ch, c);
      size_t rgb_idx = ((size_t)y * width + x) * CHANNELS;
      rgb[rgb_idx + 0] = (unsigned short)(c[0] + 0.5f);
      rgb[rgb_idx + 1] = (unsigned short)(c[1] + 0.5f);
      rgb[rgb_idx + 2] = (unsigned short)(c[2] + 0.5f);
      rgb[rgb_idx + 3] = 65535; // Alpha
    }
  }

  printf("VNG-Debayering (bayer pattern: %s) ok.\n", bayer_pattern);
  return 0;
}

int debayer_vng_rgbf(const unsigned short *raw, int width, int height,
                     float *rgbf, int x_offset, int y_offset,
                     const char *bayer_pattern) {
  int ch[2][2];
  if (!raw || !rgbf || width < 2 || height < 2 ||
      vng_channel_map(bayer_pattern, x_offset, y_offset, ch) != 0) {
    fprintf(stderr,
            "ERROR: VNG debayering: invalid image (%dx%d) or bayer pattern "
            "%s\n",
            width, height, bayer_pattern ? bayer_pattern : "(null)");
    return 1;
  }

  printf("VNG-Debayering: Pattern=%s, Offsets=(%d,%d)\n", bayer_pattern,
         x_offset, y_offset);

#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float c[3];
      vng_pixel(raw, width, height, x, y, ch, c);
      size_t rgb_idx = ((size_t)y * width + x) * CHANNELS;
      rgbf[rgb_idx + 0] = c[0];
      rgbf[rgb_idx + 1] = c[1];
      rgbf[rgb_idx + 2] = c[2];
      rgbf[rgb_idx + 3] = 65535.0f; // Alpha
    }
  }

  printf("VNG-Debayering ok.\n");
  return 0;
}

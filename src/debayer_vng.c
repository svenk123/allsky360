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
#include <stdlib.h>
#include <string.h>
#include "debayer_vng.h"

int debayer_vng_rgb16(const unsigned short *raw, int width, int height, unsigned short *rgb,
                 int x_offset, int y_offset, const char *bayer_pattern) {
    const int kernel_size = 5;
    const int half_kernel = kernel_size / 2;

    printf("VNG-Debayering: Pattern=%s, Offsets=(%d,%d)\n", bayer_pattern, x_offset, y_offset);

    for (int y = half_kernel; y < height - half_kernel; y++) {
        for (int x = half_kernel; x < width - half_kernel; x++) {
            int rgb_idx = (y * width + x) * CHANNELS;
#if 0
            int idx = y * width + x;
            int x_mod = (x + x_offset) % 2;
            int y_mod = (y + y_offset) % 2;

            int is_red = 0, is_blue = 0, is_green = 0;
            if (strcmp(bayer_pattern, "RGGB") == 0) {
                is_red   = (y_mod == 0 && x_mod == 0);
                is_green = (y_mod + x_mod == 1);
                is_blue  = (y_mod == 1 && x_mod == 1);
            } else if (strcmp(bayer_pattern, "BGGR") == 0) {
                is_red   = (y_mod == 1 && x_mod == 1);
                is_green = (y_mod + x_mod == 1);
                is_blue  = (y_mod == 0 && x_mod == 0);
            } else if (strcmp(bayer_pattern, "GRBG") == 0) {
                is_red   = (y_mod == 0 && x_mod == 1);
                is_green = (y_mod + x_mod == 1);
                is_blue  = (y_mod == 1 && x_mod == 0);
            } else if (strcmp(bayer_pattern, "GBRG") == 0) {
                is_red   = (y_mod == 1 && x_mod == 0);
                is_green = (y_mod + x_mod == 1);
                is_blue  = (y_mod == 0 && x_mod == 1);
            } else {
                fprintf(stderr, "ERROR: Unknown bayer pattern: %s\n", bayer_pattern);
                return 1;
            }
#endif
            int sum_r = 0, sum_g = 0, sum_b = 0;
            int count_r = 0, count_g = 0, count_b = 0;

            for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    int n_idx = ny * width + nx;

                    int dx = abs(kx);
                    int dy = abs(ky);
                    int weight = (dx == 0 && dy == 0) ? 5 : 1;

                    int nx_mod = (nx + x_offset) % 2;
                    int ny_mod = (ny + y_offset) % 2;

                    int neighbor_is_red = 0, neighbor_is_blue = 0, neighbor_is_green = 0;

                    if (strcmp(bayer_pattern, "RGGB") == 0) {
                        neighbor_is_red   = (ny_mod == 0 && nx_mod == 0);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 1 && nx_mod == 1);
                    } else if (strcmp(bayer_pattern, "BGGR") == 0) {
                        neighbor_is_red   = (ny_mod == 1 && nx_mod == 1);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 0 && nx_mod == 0);
                    } else if (strcmp(bayer_pattern, "GRBG") == 0) {
                        neighbor_is_red   = (ny_mod == 0 && nx_mod == 1);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 1 && nx_mod == 0);
                    } else if (strcmp(bayer_pattern, "GBRG") == 0) {
                        neighbor_is_red   = (ny_mod == 1 && nx_mod == 0);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 0 && nx_mod == 1);
                    }

                    if (neighbor_is_red) {
                        sum_r += raw[n_idx] * weight;
                        count_r += weight;
                    } else if (neighbor_is_green) {
                        sum_g += raw[n_idx] * weight;
                        count_g += weight;
                    } else if (neighbor_is_blue) {
                        sum_b += raw[n_idx] * weight;
                        count_b += weight;
                    }
                }
            }

            unsigned short r = (count_r > 0) ? (sum_r / count_r) : 0;
            unsigned short g = (count_g > 0) ? (sum_g / count_g) : 0;
            unsigned short b = (count_b > 0) ? (sum_b / count_b) : 0;

            rgb[rgb_idx + 0] = r;
            rgb[rgb_idx + 1] = g;
            rgb[rgb_idx + 2] = b;
            rgb[rgb_idx + 3] = 65535; // Alpha
        }
    }

    printf("VNG-Debayering (bayer pattern: %s) ok.\n", bayer_pattern);

    return 0;
}

int debayer_vng_rgbf(const unsigned short *raw, int width, int height, float *rgbf,
                       int x_offset, int y_offset, const char *bayer_pattern) {
    const int kernel_size = 3;
    const int half_kernel = kernel_size / 2;

    printf("VNG-Debayering: Pattern=%s, Offsets=(%d,%d)\n", bayer_pattern, x_offset, y_offset);

    for (int y = half_kernel; y < height - half_kernel; y++) {
        for (int x = half_kernel; x < width - half_kernel; x++) {
            int rgb_idx = (y * width + x) * CHANNELS;

            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            float count_r = 0.0f, count_g = 0.0f, count_b = 0.0f;

            for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    int n_idx = ny * width + nx;

                    int dx = abs(kx);
                    int dy = abs(ky);
                    float weight = (dx == 0 && dy == 0) ? 5.0f : 1.0f;

                    int nx_mod = (nx + x_offset) % 2;
                    int ny_mod = (ny + y_offset) % 2;

                    int neighbor_is_red = 0, neighbor_is_blue = 0, neighbor_is_green = 0;

                    if (strcmp(bayer_pattern, "RGGB") == 0) {
                        neighbor_is_red   = (ny_mod == 0 && nx_mod == 0);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 1 && nx_mod == 1);
                    } else if (strcmp(bayer_pattern, "BGGR") == 0) {
                        neighbor_is_red   = (ny_mod == 1 && nx_mod == 1);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 0 && nx_mod == 0);
                    } else if (strcmp(bayer_pattern, "GRBG") == 0) {
                        neighbor_is_red   = (ny_mod == 0 && nx_mod == 1);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 1 && nx_mod == 0);
                    } else if (strcmp(bayer_pattern, "GBRG") == 0) {
                        neighbor_is_red   = (ny_mod == 1 && nx_mod == 0);
                        neighbor_is_green = (ny_mod + nx_mod == 1);
                        neighbor_is_blue  = (ny_mod == 0 && nx_mod == 1);
                    } else {
                        fprintf(stderr, "ERROR: Unknown bayer pattern: %s\n", bayer_pattern);
                        return 1;
                    }

                    float val = (float)raw[n_idx];
                    if (neighbor_is_red) {
                        sum_r += val * weight;
                        count_r += weight;
                    } else if (neighbor_is_green) {
                        sum_g += val * weight;
                        count_g += weight;
                    } else if (neighbor_is_blue) {
                        sum_b += val * weight;
                        count_b += weight;
                    }
                }
            }

            rgbf[rgb_idx + 0] = (count_r > 0.0f) ? (sum_r / count_r) : 0.0f;
            rgbf[rgb_idx + 1] = (count_g > 0.0f) ? (sum_g / count_g) : 0.0f;
            rgbf[rgb_idx + 2] = (count_b > 0.0f) ? (sum_b / count_b) : 0.0f;
            rgbf[rgb_idx + 3] = 65535.0f;  // Alpha
        }
    }

    printf("VNG-Debayering ok.\n");

    return 0;
}

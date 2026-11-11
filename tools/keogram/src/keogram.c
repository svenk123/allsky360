/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
#include <math.h>
#include <stddef.h>

#include "keogram.h"

/* Bilinear sample of channel values at (x, y); 0 when outside the image */
static int sample(const uint8_t *rgb, int width, int height, double x, double y, double out[3])
{
    if (x < 0.0 || y < 0.0 || x > width - 1.0 || y > height - 1.0)
        return 0;
    int ix0 = (int)x, iy0 = (int)y;
    int ix1 = ix0 + 1 < width ? ix0 + 1 : ix0;
    int iy1 = iy0 + 1 < height ? iy0 + 1 : iy0;
    double fx = x - ix0, fy = y - iy0;
    const uint8_t *p00 = rgb + ((size_t)iy0 * width + ix0) * 3;
    const uint8_t *p10 = rgb + ((size_t)iy0 * width + ix1) * 3;
    const uint8_t *p01 = rgb + ((size_t)iy1 * width + ix0) * 3;
    const uint8_t *p11 = rgb + ((size_t)iy1 * width + ix1) * 3;
    for (int c = 0; c < 3; c++) {
        double top = p00[c] + (p10[c] - p00[c]) * fx;
        double bot = p01[c] + (p11[c] - p01[c]) * fx;
        out[c] = top + (bot - top) * fy;
    }
    return 1;
}

void keogram_slice(const uint8_t *rgb, int width, int height, double scale,
                   const keogram_geometry_t *geo, int out_h, int half_width,
                   uint8_t *out)
{
    const double a = geo->north_deg * M_PI / 180.0;
    /* unit vector to the north (image y grows downwards) and across it */
    const double nx = sin(a), ny = -cos(a);
    const double px = cos(a), py = sin(a);
    const double cx = geo->zenith_x / scale, cy = geo->zenith_y / scale;
    const double r = geo->radius / scale;

    for (int row = 0; row < out_h; row++) {
        /* +r at the northern horizon, 0 at the zenith, -r in the south */
        const double d = r * (1.0 - 2.0 * (row + 0.5) / out_h);
        double sum[3] = {0.0, 0.0, 0.0};
        int n = 0;
        for (int k = -half_width; k <= half_width; k++) {
            double v[3];
            if (sample(rgb, width, height, cx + d * nx + k * px, cy + d * ny + k * py, v)) {
                sum[0] += v[0];
                sum[1] += v[1];
                sum[2] += v[2];
                n++;
            }
        }
        for (int c = 0; c < 3; c++)
            out[row * 3 + c] = n ? (uint8_t)lround(sum[c] / n) : 0;
    }
}

/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
#ifndef KEOGRAM_KEOGRAM_H
#define KEOGRAM_KEOGRAM_H

#include <stdint.h>

/* Geometry of the allsky image in full-size pixels, as in config.ini
 * (image_zenith_x/y, image_horizon_radius, image_north_angle). North lies
 * north_deg clockwise from "up" in the image, like the azimuth grid of the
 * dashboard: point = zenith + r * (sin a, -cos a). */
typedef struct {
    double zenith_x;
    double zenith_y;
    double radius;
    double north_deg;
} keogram_geometry_t;

/**
 * Sample the north-south meridian through the zenith from one image.
 *
 * rgb/width/height: the image as decoded, scale: full size / decoded size
 * (the JPEG scale_denom). out_h rows go from the northern horizon (row 0)
 * through the zenith to the southern horizon (last row). Each row averages
 * 2 * half_width + 1 samples across the meridian (decoded pixels), bilinear.
 * Points outside the image are skipped; a row without any sample is black.
 * out: out_h * 3 bytes RGB.
 */
void keogram_slice(const uint8_t *rgb, int width, int height, double scale,
                   const keogram_geometry_t *geo, int out_h, int half_width,
                   uint8_t *out);

#endif /* KEOGRAM_KEOGRAM_H */

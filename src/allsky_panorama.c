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
#include <stdint.h>
#include <math.h>
#include <cairo.h>
#include "allsky.h"
#include "allsky_panorama.h"

#define PI 3.14159265358979323846

cairo_surface_t *convert_allsky_to_panorama(cairo_surface_t *fisheye_surface, int panorama_width, int panorama_height, int cx, int cy, double theta_start_deg) {
    if (!fisheye_surface) {
        fprintf(stderr, "ERROR: Invalid fisheye image!\n");
        return NULL;
    }

    int fisheye_width = cairo_image_surface_get_width(fisheye_surface);
    int fisheye_height = cairo_image_surface_get_height(fisheye_surface);
    unsigned char *fisheye_data = cairo_image_surface_get_data(fisheye_surface);
    int fisheye_stride = cairo_image_surface_get_stride(fisheye_surface);

    int fisheye_radius = (fisheye_width < fisheye_height ? fisheye_width : fisheye_height) / 2;

    // Create new cairo surface
    cairo_surface_t *panorama_surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, panorama_width, panorama_height);
    unsigned char *panorama_data = cairo_image_surface_get_data(panorama_surface);
    int panorama_stride = cairo_image_surface_get_stride(panorama_surface);

    printf("Creating optimized panorama: %dx%d from fisheye %dx%d (center: %d, %d)\n", 
            panorama_width, panorama_height, fisheye_width, fisheye_height, cx, cy);

    // Angles pre-calculation
    double *theta_values = (double *)allsky_safe_malloc(panorama_width * sizeof(double));
    double *phi_values = (double *)allsky_safe_malloc(panorama_height * sizeof(double));
    double theta_offset_rad = fmod(theta_start_deg * PI / 180.0, 2.0 * PI);

    for (int x = 0; x < panorama_width; x++) {
//      theta_values[x] = (double)x / panorama_width * 2.0 * PI;
	theta_values[x] = fmod((double)(panorama_width - 1 - x) / panorama_width * 2.0 * PI + theta_offset_rad, 2.0 * PI);
    }
    for (int y = 0; y < panorama_height; y++) {
        //phi_values[y] = (double)y / panorama_height * PI;
	// Top aera up to 90 degrees
	phi_values[y] = ((double)y / (panorama_height - 1)) * (PI / 2);
    }

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < panorama_height; y++) {
        for (int x = 0; x < panorama_width; x++) {
            double theta = theta_values[x];
            double phi = phi_values[y];

            double x3D = sin(phi) * cos(theta);
            double y3D = cos(phi);
            double z3D = sin(phi) * sin(theta);

            double r = fisheye_radius * acos(y3D) / (PI / 2);
            double src_x = cx + r * x3D;
            double src_y = cy + r * z3D;

            if (src_x >= 0 && src_x < fisheye_width - 1 && src_y >= 0 && src_y < fisheye_height - 1) {
                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                double dx = src_x - x0;
                double dy = src_y - y0;

                int idx00 = y0 * fisheye_stride + x0 * 4;
                int idx01 = y1 * fisheye_stride + x0 * 4;
                int idx10 = y0 * fisheye_stride + x1 * 4;
                int idx11 = y1 * fisheye_stride + x1 * 4;
                int dest_idx = y * panorama_stride + x * 4;

                // Bilinear interpolation with direct mapping
                for (int c = 0; c < 3; c++) {
                    double v00 = fisheye_data[idx00 + c];
                    double v01 = fisheye_data[idx01 + c];
                    double v10 = fisheye_data[idx10 + c];
                    double v11 = fisheye_data[idx11 + c];

                    double v0 = v00 * (1 - dx) + v10 * dx;
                    double v1 = v01 * (1 - dx) + v11 * dx;
                    panorama_data[dest_idx + c] = (unsigned char)(v0 * (1 - dy) + v1 * dy);
                }

		// Alpha channel
                panorama_data[dest_idx + 3] = 255;
            }
        }
    }

    // Release pre-calculation mem
    allsky_safe_free(theta_values);
    allsky_safe_free(phi_values);

    cairo_surface_mark_dirty(panorama_surface);

    printf("Generated panorama image with direct pixel mapping.\n");
    return panorama_surface;
}

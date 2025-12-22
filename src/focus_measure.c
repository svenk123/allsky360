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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "focus_measure.h"

/* Macro to access grayscale value from RGBA (average of RGB) */
#define GRAY(r, g, b) (((r) + (g) + (b)) / 3.0f)

int measure_focus_laplacian_rgba(const float *rgba, int width, int height,
                                 int cx, int cy, int radius, float *sharpness) {
    if (!rgba || width <= 2 || height <= 2 || radius <= 0 || !sharpness)
        return 1;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;

    #pragma omp parallel for reduction(+:sum, sum_sq, count)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            int dx = x - cx;
            int dy = y - cy;
            if (dx * dx + dy * dy > radius * radius)
                continue;

            int idx = (y * width + x) * CHANNELS;
            float center = GRAY(rgba[idx], rgba[idx + 1], rgba[idx + 2]);

            float sum_neighbors = 0.0f;
            int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (int i = 0; i < 4; i++) {
                int nx = x + offsets[i][0];
                int ny = y + offsets[i][1];
                int nidx = (ny * width + nx) * CHANNELS;
                sum_neighbors += GRAY(rgba[nidx], rgba[nidx + 1], rgba[nidx + 2]);
            }

            float lap = center * 4.0f - sum_neighbors;

            sum += lap;
            sum_sq += lap * lap;
            count++;
        }
    }

    if (count == 0)
        return 2;

    /* Compute variance of Laplacian */
    float mean = sum / count;
    float var = (sum_sq / count) - (mean * mean);

    *sharpness = var;

    printf("Focus: %.6f (laplacian)\n", *sharpness);

    return 0;
}

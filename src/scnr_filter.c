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
#include <math.h>
#include <omp.h>
#include "scnr_filter.h"

int scnr_green_filter_rgbf1(float *rgbf, int width, int height,
                                float amount, scnr_protection_t protection) {
 
    if (!rgbf || width <= 0 || height <= 0 || amount < 0.0f || amount > 1.0f) {
        printf("SCNR filter: invalid parameters\n");
        return 1;
    }

    printf("SCNR filter: amount: %.2f\n", amount);
    printf("SCNR filter: protection: %d\n", protection);

    if (protection == SCNR_PROTECT_AVERAGE_NEUTRAL) {
        printf("SCNR filter: protection: average neutral\n");
    } else if (protection == SCNR_PROTECT_MAXIMUM_NEUTRAL) {
        printf("SCNR filter: protection: maximum neutral\n");
    } else {
        printf("SCNR filter: protection: none\n");
    }

    const int pixel_count = width * height;

    #pragma omp parallel for
    for (int i = 0; i < pixel_count; ++i) {
        float *p = &rgbf[i * CHANNELS];
        float r = p[0];
        float g = p[1];
        float b = p[2];

        float neutral = 0.0f;

        switch (protection) {
            case SCNR_PROTECT_AVERAGE_NEUTRAL:
                neutral = (r + b) / 2.0f;
                break;
            case SCNR_PROTECT_MAXIMUM_NEUTRAL:
                neutral = fmaxf(r, b);
                break;
            case SCNR_PROTECT_NONE:
            default:
                neutral = 0.0f;  // No protection
                break;
        }

        float excess = g - neutral;
        if (excess > 0.0f) {
            g -= amount * excess;
            if (g < 0.0f) g = 0.0f;
            p[1] = g;
        }
    }

    printf("SCNR filter: ok, amount: %.2f\n", amount);

    return 0;
}

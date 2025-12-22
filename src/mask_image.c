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
#include <math.h>
#include <stdio.h>
#include "mask_image.h"

/**
 * Applies a circular mask to a float RGB image (in-place).
 *
 * @param rgb_image Pointer to float array of size width * height * 3 (RGB only, no alpha)
 * @param width     Image width in pixels
 * @param height    Image height in pixels
 * @param cx        X-coordinate of the circular image center (in pixels)
 * @param cy        Y-coordinate of the circular image center (in pixels)
 * @param radius    Radius of valid circular image area (in pixels); if 0 → no masking
 * @return          0 on success, 1 on invalid parameters
 */
int mask_image_circle_rgbf1(float *rgba_image, int width, int height,
                            int cx, int cy, int radius) {
    if (!rgba_image || width <= 0 || height <= 0) 
        return 1;

    /* No masking requested */
    if (radius <= 0)
        return 0;

    float r2 = radius * radius;  // compare with squared distance

    for (int y = 0; y < height; ++y) {
        float dy = (float)y - (float)cy;

        for (int x = 0; x < width; ++x) {
            float dx = (float)x - (float)cx;
            float dist2 = dx * dx + dy * dy;

            int index = (y * width + x) * CHANNELS;

            if (dist2 > r2) {
                rgba_image[index + 0] = 0.0f;  // R
                rgba_image[index + 1] = 0.0f;  // G
                rgba_image[index + 2] = 0.0f;  // B
                // A channel (index + 3) stays unchanged
            }
            // else: leave pixel untouched
        }
    }

    printf("Add circular mask: radius: %d, x: %d, y: %d\n", radius, cx, cy);

    return 0;
}

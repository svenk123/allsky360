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
#include <stdint.h>

#include <math.h>
#include <limits.h>  // UINT16_MAX (65535)
#include <float.h>  // FLT_MAX
#include "rgb16_to_cairo.h"

int rgb16_to_cairo(const unsigned short *data, unsigned char *output, int width, int height) {
    int pixel_count = width * height * CHANNELS;
    unsigned short min_val = UINT16_MAX;
    unsigned short max_val = 0;

    /* Find min/max values */
    for (int i = 0; i < pixel_count; i += CHANNELS) {
        if (data[i + 0] < min_val) 
	    min_val = data[i + 0]; // R
        if (data[i + 1] < min_val) 
	    min_val = data[i + 1]; // G
        if (data[i + 2] < min_val) 
	    min_val = data[i + 2]; // B

        if (data[i + 0] > max_val) 
	    max_val = data[i + 0]; // R
        if (data[i + 1] > max_val) 
	    max_val = data[i + 1]; // G
        if (data[i + 2] > max_val) 
	    max_val = data[i + 2]; // B
    }
    printf("Normalization: min=%d, max=%d\n", min_val, max_val);

    /* Avoid division by 0 */
    double scale = (max_val > min_val) ? (255.0 / (max_val - min_val)) : 1.0;

    for (int i = 0; i < pixel_count; i += CHANNELS) {
        output[i + 2] = (unsigned char) fmin((data[i + 0] - min_val) * scale, 255); // R
        output[i + 1] = (unsigned char) fmin((data[i + 1] - min_val) * scale, 255); // G
        output[i + 0] = (unsigned char) fmin((data[i + 2] - min_val) * scale, 255); // B
        output[i + 3] = 255; // Alpha
    }

    printf("Normalized 16 bit (int) to 8 bit (int).\n");

    return 0;
}


int rgbf_to_cairo(const float *data, unsigned char *output, int width, int height) {
    if (!data || !output || width <= 0 || height <= 0) 
	return 1;

    int pixel_count = width * height * CHANNELS;
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    /* Find min/max values */
    for (int i = 0; i < pixel_count; i += CHANNELS) {
        if (data[i + 0] < min_val)
	    min_val = data[i + 0]; // R
        if (data[i + 1] < min_val)
	    min_val = data[i + 1]; // G
        if (data[i + 2] < min_val) 
	    min_val = data[i + 2]; // B

        if (data[i + 0] > max_val)
	    max_val = data[i + 0];
        if (data[i + 1] > max_val)
	    max_val = data[i + 1];
        if (data[i + 2] > max_val)
	    max_val = data[i + 2];
    }

    printf("Normalization: min=%.2f, max=%.2f\n", min_val, max_val);

    float scale = (max_val > min_val) ? (255.0f / (max_val - min_val)) : 1.0f;

    for (int i = 0; i < pixel_count; i += CHANNELS) {
        output[i + 2] = (unsigned char)fminf((data[i + 0] - min_val) * scale, 255.0f); // R
        output[i + 1] = (unsigned char)fminf((data[i + 1] - min_val) * scale, 255.0f); // G
        output[i + 0] = (unsigned char)fminf((data[i + 2] - min_val) * scale, 255.0f); // B
        output[i + 3] = 255; // Alpha
    }

    printf("Normalized 32 bit (float) to 8 bit (int).\n");

    return 0;
}

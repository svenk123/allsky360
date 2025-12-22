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
#ifndef CLARITY_FILTER_H
#define CLARITY_FILTER_H

#include <stddef.h>

/* Mask modes for clarity_filter_rgbf_masked():
 *  CLARITY_MASK_NONE         : ignore masking (default old behavior)
 *  CLARITY_MASK_ALPHA_ZERO   : mask if alpha == 0
 *  CLARITY_MASK_RGB_ALL_ZERO : mask if r==g==b==0
 *  CLARITY_MASK_RGB_OR_ALPHA : mask if (alpha==0) OR (r==g==b==0)
 */
enum {
    CLARITY_MASK_NONE = 0,
    CLARITY_MASK_ALPHA_ZERO = 1,
    CLARITY_MASK_RGB_ALL_ZERO = 2,
    CLARITY_MASK_RGB_OR_ALPHA = 3
};

#ifdef USE_CUDA
#include "clarity_filter_cuda.h"
#define clarity_filter_rgbf_masked(rgba, width, height, strength, radius, midtone_width, preserve_highlights, mask_mode) clarity_filter_rgbf_masked_cuda_Wrapper(rgba, width, height, strength, radius, midtone_width, preserve_highlights, mask_mode)
#else

/**
 * Clarity filter (RGBA float 0..1) with masking support.
 * Behavior:
 *  - Pixels considered "masked" (per mask_mode) are excluded from blur and
 *    detail computation (kernel renormalization on valid neighbors).
 *  - Masked pixels are left unchanged in the output.
 *
 * Parameters:
 *  @param rgba: pointer to interleaved RGBA floats, size width*height*4
 *  @param width: image width
 *  @param height: image height
 *  @param strength:  amount in [-1..+1], typical 0.2..0.6
 *  @param radius:    Gaussian radius >=1
 *  @param midtone_width: 0.2..0.6 (bell width around midtones, ~0.35 default)
 *  @param preserve_highlights: if !=0, rolloff near highlights to reduce halos
 *  @param mask_mode: see enum above
 *
 * @return 0 on success, >0 on error.
 */
int clarity_filter_rgbf_masked(
    float *rgba,
    int width,
    int height,
    float strength,
    int radius,
    float midtone_width,
    int preserve_highlights,
    int mask_mode
);

#endif

#endif // CLARITY_FILTER_H
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
#include "darkframe_cuda.h"

#ifdef USE_CUDA
int subtract_darkframe_rgb16_cuda_Wrapper(unsigned short *rgb_image, unsigned short *darkframe,
                             int width, int height, int dark_width, int dark_height) {
    return subtract_darkframe_rgb16_cuda(rgb_image, darkframe, width, height, dark_width, dark_height);
}
#else
// Dummy function
int subtract_darkframe_rgb16_cuda_Wrapper(unsigned short *rgb_image, unsigned short *darkframe, int width, int height, int dark_width, int dark_height) {
    (void)rgb_image;
    (void)darkframe;
    (void)width;
    (void)height;
    (void)dark_width;
    (void)dark_height;
    fprintf(stderr, "subtract darkframe CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}
#endif

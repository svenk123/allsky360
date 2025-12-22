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
#ifndef DARKFRAME_CUDA_H
#define DARKFRAME_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

int subtract_darkframe_rgb16_cuda(unsigned short *rgb_image, unsigned short *darkframe,
                             int width, int height, int dark_width, int dark_height);
int subtract_darkframe_rgb16_cuda_Wrapper(unsigned short *rgb_image, unsigned short *darkframe,
                             int width, int height, int dark_width, int dark_height);

#ifdef __cplusplus
}
#endif

#endif // DARKFRAME_CUDA_H

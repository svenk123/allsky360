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
#ifndef COLOR_CALIBRATION_CUDA_H
#define COLOR_CALIBRATION_CUDA_H

#define CHANNELS 4

#ifdef __cplusplus
extern "C" {
#endif

int adjust_saturation_rgbf1_cuda(float *rgba, int width, int height, float saturation_factor);
int adjust_saturation_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float saturation_factor);

int apply_gamma_correction_rgbf1_cuda(float *rgba, int width, int height, float gamma);
int apply_gamma_correction_rgbf1_cuda_Wrapper(float *rgba, int width, int height, float gamma);

#ifdef __cplusplus
}
#endif

#endif // COLOR_CALIBRATION_CUDA_H

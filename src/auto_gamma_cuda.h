/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#ifndef AUTO_GAMMA_CUDA_H
#define AUTO_GAMMA_CUDA_H

/* CUDA Auto-Gamma / Reinhard / Tone-Curve functions (C-compatible) */
#ifdef __cplusplus
extern "C" {
#endif

int reinhard_tonemap_rgbf1_cuda(float *rgbf, int width, int height,
                                float white_point);
int reinhard_tonemap_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                        float white_point);

int apply_tone_curve_rgbf1_cuda(float *rgbf, int width, int height,
                                float shadow_lift, float midtone_contrast);
int apply_tone_curve_rgbf1_cuda_Wrapper(float *rgbf, int width, int height,
                                        float shadow_lift, float midtone_contrast);

#ifdef __cplusplus
}
#endif

#endif /* AUTO_GAMMA_CUDA_H */

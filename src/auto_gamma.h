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
#ifndef AUTO_GAMMA_H
#define AUTO_GAMMA_H

/* If CUDA is enabled, use CUDA versions of Reinhard TMO and Tone Curve */
#ifdef USE_CUDA
    #include "auto_gamma_cuda.h"

    #define reinhard_tonemap_rgbf1(rgbf, width, height, white_point) \
        reinhard_tonemap_rgbf1_cuda_Wrapper(rgbf, width, height, white_point)

    #define apply_tone_curve_rgbf1(rgbf, width, height, shadow_lift, midtone_contrast) \
        apply_tone_curve_rgbf1_cuda_Wrapper(rgbf, width, height, shadow_lift, midtone_contrast)
#else

/**
 * Applies a Reinhard global tone mapping operator to a float RGBA image
 * in-place.  The operator works in luminance space (Rec.709) to preserve
 * colour hue while compressing the dynamic range.
 *
 * Simple mode  (white_point == 0):
 *   Y_out = Y_in / (1 + Y_in)
 *
 * Extended mode (white_point > 0):
 *   Y_out = Y_in * (1 + Y_in / white_point²) / (1 + Y_in)
 *
 * In extended mode highlights near @p white_point are reproduced at full
 * brightness (1.0), while the simple mode asymptotically approaches 1.0.
 *
 * @param rgbf        Float RGBA image (interleaved R,G,B,A, range 0.0–∞).
 * @param width       Image width in pixels.
 * @param height      Image height in pixels.
 * @param white_point Scene white point in linear light.  Pass 0 for the
 *                    simple operator (no white-point anchoring).
 * @return            0 on success, 1 on invalid input.
 */
int reinhard_tonemap_rgbf1(float *rgbf, int width, int height,
                           float white_point);

/**
 * Applies a parametric S-curve tone mapping to a float RGBA image in-place.
 * The curve operates in luminance space (Rec.709) and scales RGB channels
 * proportionally to preserve colour hue.
 *
 * Two independent correction terms are summed on top of the identity:
 *
 *   Shadow lift  (shadow_lift > 0):
 *     delta = shadow_lift × 4 × Y × (1−Y)²
 *     Peaks at Y ≈ 0.33 (lower third of the tonal range).
 *     Lifts shadows and lower midtones; blacks and whites are unaffected.
 *
 *   S-curve contrast  (midtone_contrast > 0):
 *     delta = midtone_contrast × 4 × Y × (1−Y) × (1−2Y)
 *     Positive for Y < 0.5 (shadows up), negative for Y > 0.5 (highlights
 *     down).  Creates a classic S-curve with the pivot at 0.5.
 *
 * Both terms can be combined freely.  The result is clamped to [0, 1].
 *
 * Recommended starting values:
 *   shadow_lift      0.10 – 0.30
 *   midtone_contrast 0.15 – 0.40
 *
 * @param rgbf            Float RGBA image (interleaved R,G,B,A, range 0–1).
 * @param width           Image width in pixels.
 * @param height          Image height in pixels.
 * @param shadow_lift     Shadow lift strength ≥ 0 (0 = disabled).
 * @param midtone_contrast S-curve strength ≥ 0 (0 = disabled).
 * @return                0 on success, 1 on invalid input.
 */
int apply_tone_curve_rgbf1(float *rgbf, int width, int height,
                           float shadow_lift, float midtone_contrast);

#endif /* !USE_CUDA */

/* compute_auto_gamma_rgbf1 has no CUDA version — always use CPU */
float compute_auto_gamma_rgbf1(const float *rgbf, int width, int height,
                               float target_mean, float gamma_min,
                               float gamma_max);

#endif // AUTO_GAMMA_H

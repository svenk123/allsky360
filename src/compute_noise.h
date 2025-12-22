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
#ifndef COMPUTE_NOISE_H
#define COMPUTE_NOISE_H

#define CHANNELS	4

/**
 * Compute background noise as MAD (Median Absolute Deviation)
 * scaled to sigma equivalent (sigma ≈ 1.4826 * MAD)
 *
 * @param image: pointer to RGBA float image data (0.0–1.0)
 * @param width: width of the image
 * @param height: height of the image
 * @param median: background median value of the green channel
 * @param sigma_out: pointer to float where the noise value will be stored
 * @param cx: center x position of the circle
 * @param cy: center y position of the circle
 * @param radius: radius of the circle
 * @return: 0 on success, 1 on error
 */
int compute_background_noise_mad_rgbf1(
    const float *image,
    int width,
    int height,
    float median,
    float *sigma_out,
    int cx,
    int cy,
    int radius
);

#endif // COMPUTE_NOISE_H

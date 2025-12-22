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

#ifndef HDRMT_H
#define HDRMT_H

#define CHANNELS 4

/**
 * HDR Multiscale Transform on linear RGB float images (0..1).
 * - In-place on interleaved RGB (RGBRGB...).
 * - Works on BT.709 luminance only, then rescales RGB by Y'/Y.
 * - You should still normalize / tone-map to 0..1 afterwards.
 *
 * @param rgb: Pointer to the image data (RGBA format, float values 0.0 to 1.0 per channel)
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param levels: number of à trous scales (1..10), e.g. 5..7
 * @param start_level: first level to compress (0 = finest). Typical: 3 or 4
 * @param strength: base compression strength (0..1). Typical: 0.25..0.6
 * @param strength_boost: extra factor for coarser levels (0..1). Typical: 0.2
 * @param midtones: midtones balance in [0,1]; 0.5 = neutral. Shift target brightness
 * @param shadow_protect protect darks [0..1], 0=off, 1=protect fully (less compression in shadows)
 * @param highlight_protect: Protect brights [0..1], 0=off, 1=protect fully (less compression in highlights)
 * @param epsilon: small stabilizer, e.g. 1e-6
 * @param gain_cap: limit RGB gain (>=1). e.g. 4.0; set <1 to disable
 * @return: 0 on success, >0 on error.
 */
int hdrmt_rgbf1(float *rgb, int width, int height, int levels, int start_level,
                float strength, float strength_boost, float midtones,
                float shadow_protect, float highlight_protect, float epsilon,
                float gain_cap);

#endif /* HDRMT_H */

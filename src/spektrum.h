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

#ifndef SPEKTRUM_H
#define SPEKTRUM_H

/* Return codes */
#define SPEKTRUM_OK              0
#define SPEKTRUM_ERR_NULL_INPUT  1   /* rgb pointer is NULL */
#define SPEKTRUM_ERR_INVALID_DIMS 2  /* width or height <= 0 */
#define SPEKTRUM_ERR_INVALID_ROI 3   /* ROI entirely outside the image */
#define SPEKTRUM_ERR_NO_PIXELS   4   /* no pixels inside ROI after bounds check */
#define SPEKTRUM_ERR_WRITE       5   /* JSON output file could not be written */
#define SPEKTRUM_ERR_ALLOC       6   /* memory allocation failed */

/* Number of spectral bins: 400 nm – 700 nm in 10 nm steps */
#define SPEKTRUM_BINS  31

/**
 * Estimate a coarse spectrum from a circular region of an RGB image.
 *
 * The reconstruction has effectively 3 degrees of freedom (one per colour
 * channel) and should be treated as a rough spectral fingerprint only, not
 * a calibrated measurement.
 *
 * @param rgb            Interleaved float RGB image (R,G,B,...), values 0–1,
 *                       linear, after debayering, before stretching/gamma.
 * @param width          Image width in pixels.
 * @param height         Image height in pixels.
 * @param cx             X-coordinate of the measurement circle centre.
 * @param cy             Y-coordinate of the measurement circle centre.
 * @param radius         Radius of the measurement circle (max 100 px).
 * @param lambda         Tikhonov regularisation parameter (e.g. 0.01).
 * @param output_path    Path for the JSON output file.
 * @return               SPEKTRUM_OK on success, one of the SPEKTRUM_ERR_*
 *                       codes otherwise.
 */
int spektrum_analyze(
    const float *rgb,
    int          width,
    int          height,
    int          cx,
    int          cy,
    int          radius,
    float        lambda,
    const char  *output_path
);

#endif /* SPEKTRUM_H */

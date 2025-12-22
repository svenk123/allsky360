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
#ifndef SCNR_FILTER_H
#define SCNR_FILTER_H

#define CHANNELS	4

typedef enum {
    SCNR_PROTECT_NONE,
    SCNR_PROTECT_AVERAGE_NEUTRAL,
    SCNR_PROTECT_MAXIMUM_NEUTRAL
} scnr_protection_t;

/**
 * Apply an advanced SCNR (Subtractive Chromatic Noise Reduction) filter to an RGBA float image.
 *
 * @param rgbf : pointer to the image data (RGBA format, float values 0.0 to 1.0 per channel).
 *                    The array must have (width * height * 4) elements. The operation is performed in-place.
 * @param width: width of the image in pixels.
 * @param height: height of the image in pixels.
 * @param amount: strength of the SCNR filter (0.0 = no effect, 1.0 = full strength).
 * @param protection: protection method to preserve neutral colors:
 *                    SCNR_PROTECT_NONE: No protection, apply SCNR fully.
 *                    SCNR_PROTECT_AVERAGE_NEUTRAL: Protect pixels where the green channel is close to the average of red and blue.
 *                    SCNR_PROTECT_MAXIMUM_NEUTRAL: Protect pixels where the green channel is close to the maximum of red and blue.
 */
int scnr_green_filter_rgbf1(float *rgbf, int width, int height,
                                float amount, scnr_protection_t protection);

#endif // SCNR_FILTER_H

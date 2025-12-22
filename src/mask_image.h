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
#ifndef MASK_IMAGE_H
#define MASK_IMAGE_H

#define CHANNELS	4

/**
 * Applies a circular mask to a float RGB image (in-place).
 *
 * @param rgb_image: pointer to float array of size width * height * 3 (RGB only, no alpha)
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param cx: X-coordinate of the circular image center (in pixels)
 * @param cy: Y-coordinate of the circular image center (in pixels)
 * @param radius: radius of valid circular image area (in pixels); if 0 → no masking
 * @return: 0 on success, 1 on invalid parameters
 */
int mask_image_circle_rgbf1(float *rgb_image, int width, int height,
                          int cx, int cy, int radius);

#endif // MASK_IMAGE_H

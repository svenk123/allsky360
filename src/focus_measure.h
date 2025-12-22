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
#ifndef FOCUS_MEASURE_H
#define FOCUS_MEASURE_H

#define CHANNELS	4

/**
 * @brief Measure image sharpness using Laplacian in a circular region of a float RGBA image.
 *
 * @param rgba: pointer to RGBA float image data (range 0.0 - 1.0, 4 floats per pixel).
 * @param width: image width in pixels.
 * @param height: image height in pixels.
 * @param cx: X-coordinate of the circular region center.
 * @param cy: Y-coordinate of the circular region center.
 * @param radius: radius of the circular region to evaluate.
 * @param sharpness: output pointer to store computed sharpness value (variance of Laplacian).
 * @return 0 on success, >0 on error (e.g. invalid input).
 */
int measure_focus_laplacian_rgba(const float *rgba, int width, int height,
    int cx, int cy, int radius, float *sharpness);

#endif // FOCUS_MEASURE_H

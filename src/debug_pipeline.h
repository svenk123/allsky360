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

#ifndef DEBUG_PIPELINE_H
#define DEBUG_PIPELINE_H

#include <limits.h>

/**
 * Saves a debug pipeline image as a PNG file
 * 
 * @param image: the image data array (float RGBA)
 * @param width: image width in pixels
 * @param height: image height in pixels
 * @param pipeline_number: the number for the filename (e.g. 1 for "pipeline_01.png")
 * @param images_dir: the directory to save the image
 * @param scale_to_16bit: 1 for 16-bit PNG, 0 for 8-bit PNG
 * @return: 0 on success, -1 on error
 */
int save_debug_pipeline_image(const float *image, int width, int height, 
                             int pipeline_number, const char *images_dir, 
                             int scale_to_16bit);

#endif /* DEBUG_PIPELINE_H */

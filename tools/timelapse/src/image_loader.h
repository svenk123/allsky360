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
#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Loads a JPEG image, optionally verifies the original dimensions, and resizes it to the target dimensions.
 *
 * @param filename            Path to the JPEG file.
 * @param target_width        Desired output width in pixels.
 * @param target_height       Desired output height in pixels.
 * @param expected_src_width  Optional: expected source width (set to 0 to skip check).
 * @param expected_src_height Optional: expected source height (set to 0 to skip check).
 * @return                    Pointer to resized RGB image buffer (3 bytes per pixel), or NULL on error.
 */
uint8_t* load_and_resize_jpeg(const char *filename,
                               int width, int height,
                               int expected_src_width, int expected_src_height);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_LOADER_H

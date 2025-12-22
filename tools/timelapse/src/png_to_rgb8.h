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
#ifndef PNG_TO_RGB8_H
#define PNG_TO_RGB8_H

/**
 * Saves an RGB image (8-bit per channel) as PNG file.
 *
 * @param rgb            Pointer to uint8_t[width * height], range 0-255
 * @param width          Image width in pixels
 * @param height         Image height in pixels
 * @param compression    PNG compression level (0 = no compression, 9 = max compression)
 * @param filename       Output filename (path to PNG file)
 * @return               0 on success, 1 on error
 */
int save_rgb8_as_png(const uint8_t *rgb, int width, int height,
                         int compression, const char *filename);


#endif // PNG_TO_RGB8_H
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
#ifndef VIDEO_ENCODER_H
#define VIDEO_ENCODER_H

#include <stdint.h>

/**
 * Initializes the NVENC-based video encoder.
 *
 * @param filename  Output filename for the encoded video (e.g., .mp4).
 * @param width     Width of the video frames in pixels.
 * @param height    Height of the video frames in pixels.
 * @param fps       Frame rate (frames per second).
 * @return          0 on success, 1 on failure.
 */
int init_encoder_nvenc(const char *filename, int width, int height, int fps);

/**
 * Encodes a single video frame using the initialized NVENC encoder.
 * The input must be an RGB image matching the encoder's resolution.
 *
 * @param rgb_data  Pointer to the RGB frame data (width × height × 3 bytes).
 */
void encode_frame(uint8_t *rgb_data);

/**
 * Finalizes the video encoder and releases all associated resources.
 * This function must be called after all frames have been encoded.
 */
void finalize_encoder(void);

#endif

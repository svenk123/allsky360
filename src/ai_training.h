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
#ifndef AI_TRAINING_H
#define AI_TRAINING_H

#define CHANNELS	4

/**
 * Save extracted patches from an RGBA float image to PNG (16-bit) and JSON metadata.
 * 
 * @param rgba: pointer to input image data (RGBA float, 0.0-1.0)
 * @param width: width of input image
 * @param height: height of input image
 * @param patch_size: size of square patches (e.g., 128)
 * @param positions: array of (x,y) pairs indicating patch top-left positions
 * @param n_patches: number of patch positions
 * @param out_dir: output directory (e.g., "input" or "target")
 * @param prefix: filename prefix (e.g., "scene01")
 * @param meta_data: JSON object with metadata (provided by caller)
 * @return: 0 on success, 1 on failure
 */
int save_patches_with_metadata(const float *rgba, int width, int height, int patch_size,
                               const int (*positions)[2], int n_patches,
                               const char *out_dir, const char *prefix,
                               struct json_object *meta_data);

#endif // AI_TRAINING_H

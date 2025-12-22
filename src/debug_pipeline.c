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

#include <limits.h>
#include <stdio.h>

#include "allsky.h"
#include "debug_pipeline.h"
#include "jpeg_functions.h"

int save_debug_pipeline_image(const float *image, int width, int height,
                              int pipeline_number, const char *images_dir,
                              int scale_to_16bit) {
  if (!image || !images_dir || pipeline_number < 0) {
    return -1;
  }

  char pipeline_path[PATH_MAX + 1];
  if (allsky_safe_snprintf(pipeline_path, sizeof(pipeline_path),
                           "%s/pipeline_%02d.jpg", images_dir,
                           pipeline_number)) {
    fprintf(stderr, "WARNING: String %s truncated\n", pipeline_path);
  }

  save_jpeg_rgbf1(image, width, height, 9, 0.25f, pipeline_path);

  return 0;
}

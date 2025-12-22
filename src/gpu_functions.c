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
#include "gpu_functions.h"
#include <stdlib.h>
#include <stddef.h>
#include "allsky.h"

#ifndef USE_CUDA

int upload_rgbf_to_device(const float *rgba_host,
    int width, int height,
    float **rgba_device)
{
    if (!rgba_host || !rgba_device || width <= 0 || height <= 0) {
        return 1;
    }

    // Cast away const here because signature expects float** out
    *rgba_device = (float*)rgba_host;
    return 0;
}

int download_rgbf_from_device(const float *rgba_device,
    int width, int height,
    float *rgba_host)
{
    if (!rgba_device || !rgba_host || width <= 0 || height <= 0) {
        return 1;
    }

    // Nothing to do: rgba_device == rgba_host in CPU-only mode
    return 0;
}

int free_rgbf_on_device(float *rgba_device) {
    // Nothing to do: rgba_device is just a pointer to rgba_host in CPU-only mode
    (void)rgba_device; // unused parameter
    return 0;
}

void* alloc_rgbf_on_host(size_t size, int pinned) {
    (void)pinned;
    return allsky_safe_malloc(size);
}

void free_rgbf_on_host(void* ptr, int pinned) {
    (void)pinned;
    allsky_safe_free(ptr);
}

#endif

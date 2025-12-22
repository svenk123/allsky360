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
#include <stdio.h>
#include "gpu_functions_cuda.h"

#ifdef USE_CUDA
int upload_rgbf_to_device_Wrapper(const float *rgba_host, int width, int height, float **rgba_device) {
    return upload_rgbf_to_device(rgba_host, width, height, rgba_device);
}

int download_rgbf_from_device_Wrapper(const float *rgba_device, int width, int height, float *rgba_host) {
    return download_rgbf_from_device(rgba_device, width, height, rgba_host);
}

int free_rgbf_on_device_Wrapper(float *rgba_device) {
    return free_rgbf_on_device(rgba_device);
}

void alloc_rgbf_on_host_Wrapper(size_t size, int pinned) {
    return alloc_rgbf_on_host(size, pinned);
}

void free_rgbf_on_host_Wrapper(void* ptr, int pinned) {
    return free_rgbf_on_host(ptr, pinned);
}

#else
// Use dummy function
int upload_rgbf_to_device_Wrapper(const float *rgba_host, int width, int height, float **rgba_device) {
    (void)rgba_host;
    (void)width;
    (void)height;
    (void)rgba_device;
    fprintf(stderr, "Upload RGBF to device CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

int download_rgbf_from_device_Wrapper(const float *rgba_device, int width, int height, float *rgba_host) {
    (void)rgba_device;
    (void)width;
    (void)height;
    (void)rgba_host;
    fprintf(stderr, "Download RGBF from device CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

int free_rgbf_on_device_Wrapper(float *rgba_device) {
    (void)rgba_device;
    fprintf(stderr, "Free RGBF on device CUDA is disabled! Falling back to CPU implementation.\n");
    return 1;
}

void alloc_rgbf_on_host_Wrapper(size_t size, int pinned) {
    (void)size;
    (void)pinned;
    fprintf(stderr, "Alloc RGBF on host CUDA is disabled! Falling back to CPU implementation.\n");
    return;
}

void free_rgbf_on_host_Wrapper(void* ptr, int pinned) {
    (void)ptr;
    (void)pinned;
    fprintf(stderr, "Free RGBF on host CUDA is disabled! Falling back to CPU implementation.\n");
    return;
}
#endif

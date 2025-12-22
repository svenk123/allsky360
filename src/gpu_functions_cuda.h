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
#ifndef GPU_FUNCTIONS_CUDA_H
#define GPU_FUNCTIONS_CUDA_H

// CUDA-GPU-Functions (C-kompatibel)
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Upload an RGBA float image from host to device memory.
 *
 * @param rgba_host   Host pointer (RGBA float interleaved, size = width*height*4 floats)
 * @param width   Image width
 * @param height  Image height
 * @param rgba_device   [out] Device pointer (allocated inside, must be freed with cudaFree)
 * @return 0 on success, 1 on error
 *
 * Note:
 * - The caller owns *rgba_device after success and must call cudaFree(*rgba_device).
 * - Function must be compiled with nvcc (since it uses CUDA runtime APIs).
 */
int upload_rgbf_to_device(const float *rgba_host,
        int width, int height,
        float **rgba_device);

int upload_rgbf_to_device_Wrapper(const float *rgba_host,
        int width, int height,
        float **rgba_device);

/**
 * Download an RGBA float image from device memory back to host.
 *
 * @param rgba_device   Device pointer (RGBA float interleaved, size = width*height*4 floats)
 * @param width   Image width
 * @param height  Image height
 * @param rgba_host   Host pointer (already allocated, must have capacity width*height*4 floats)
 * @return 0 on success, 1 on error
 */        
int download_rgbf_from_device(const float *rgba_device,
        int width, int height,
        float *rgba_host);

int download_rgbf_from_device_Wrapper(const float *rgba_device,
        int width, int height,
        float *rgba_host);

/**
 * Free RGBA float buffer on device.
 *
 * @param rgba_device Device pointer (must have been allocated with cudaMalloc)
 * @return 0 on success, 1 on error
 */
int free_rgbf_on_device(float *rgba_device);

int free_rgbf_on_device_Wrapper(float *rgba_device);

/**
 * Allocate RGBA float buffer on host.
 *
 * @param size Size of the memory to allocate
 * @param pinned 1 if the memory is pinned, 0 otherwise
 * @return pointer to the allocated memory
 */
void* alloc_rgbf_on_host(size_t size, int pinned);
void alloc_rgbf_on_host_Wrapper(size_t size, int pinned);

/**
 * Free pinned memory on host.
 *
 * @param ptr Pointer to the memory to free
 * @param pinned 1 if the memory is pinned, 0 otherwise
 * @return 0 on success, 1 on error
 */
void free_rgbf_on_host(void* ptr, int pinned);

void free_rgbf_on_host_Wrapper(void* ptr, int pinned);

#ifdef __cplusplus
}
#endif

#endif // GPU_FUNCTIONS_CUDA_H

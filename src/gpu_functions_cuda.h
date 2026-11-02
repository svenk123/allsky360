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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocation modes for alloc_rgbf_on_host / free_rgbf_on_host.
 *
 * ALLSKY_HOST_ALLOC_CPU    - Standard malloc (pageable, no GPU access)
 * ALLSKY_HOST_ALLOC_PINNED - cudaMallocHost (page-locked, faster DMA)
 * ALLSKY_HOST_ALLOC_MAPPED - cudaHostAllocMapped (zero-copy, GPU gets device
 *                            pointer via cudaHostGetDevicePointer; ideal on
 *                            Jetson Orin where CPU and GPU share the same DDR)
 */
#define ALLSKY_HOST_ALLOC_CPU    0
#define ALLSKY_HOST_ALLOC_PINNED 1
#define ALLSKY_HOST_ALLOC_MAPPED 2

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
 * @param size Size of the memory to allocate in bytes
 * @param mode ALLSKY_HOST_ALLOC_CPU / _PINNED / _MAPPED
 * @return pointer to the allocated memory, NULL on error
 */
void* alloc_rgbf_on_host(size_t size, int mode);
void* alloc_rgbf_on_host_Wrapper(size_t size, int mode);

/**
 * Free host buffer previously allocated with alloc_rgbf_on_host.
 *
 * @param ptr  Pointer to the memory to free
 * @param mode Must match the mode used during allocation
 */
void free_rgbf_on_host(void* ptr, int mode);
void free_rgbf_on_host_Wrapper(void* ptr, int mode);

/**
 * Get a CUDA device pointer for a zero-copy mapped host buffer.
 * The host buffer must have been allocated with ALLSKY_HOST_ALLOC_MAPPED.
 * On Jetson Orin the returned device pointer addresses the same physical
 * memory — no H2D/D2H transfer is required.
 *
 * @param host_ptr   Host pointer (from alloc_rgbf_on_host with MAPPED mode)
 * @param device_ptr [out] Corresponding CUDA device pointer
 * @return 0 on success, 1 on error
 */
int get_device_ptr_from_mapped(float *host_ptr, float **device_ptr);
int get_device_ptr_from_mapped_Wrapper(float *host_ptr, float **device_ptr);

/**
 * One-time CUDA device initialisation. Must be called before any other
 * CUDA function. Sets cudaDeviceMapHost so that mapped/zero-copy host
 * allocations work, verifies canMapHostMemory, and warms up the context.
 *
 * @return 0 on success, 1 on error
 */
int allsky_cuda_init(void);
int allsky_cuda_init_Wrapper(void);

#ifdef __cplusplus
}
#endif

#endif // GPU_FUNCTIONS_CUDA_H

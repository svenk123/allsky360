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
#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

#include <stddef.h>

// Falls CUDA aktiviert ist, verwende CUDA-Version
#ifdef USE_CUDA
    #include "gpu_functions_cuda.h"

    #define upload_rgbf_to_device(rgba_host, width, height, rgba_device) upload_rgbf_to_device_Wrapper(rgba_host, width, height, rgba_device)
    #define download_rgbf_from_device(rgba_device, width, height, rgba_host) download_rgbf_from_device_Wrapper(rgba_device, width, height, rgba_host)
    #define free_rgbf_on_device(rgba_device) free_rgbf_on_device_Wrapper(rgba_device)
    #define alloc_rgbf_on_host(size, pinned) alloc_rgbf_on_host_Wrapper(size, pinned)
    #define free_rgbf_on_host(ptr, pinned) free_rgbf_on_host_Wrapper(ptr, pinned)
#else

    #define CHANNELS	4

    /**
     *  Upload RGBA image to device
     * @param rgba_host: pointer to the image data (RGBA format, float values 0.0 to 1.0 per channel)
     * @param width: image width
     * @param height: image height
     * @param rgba_device: pointer to the device image data
     * @return: 0 on success, >0 on error
     */
    int upload_rgbf_to_device(const float *rgba_host, int width, int height, float **rgba_device);
    
    /**
     *  Download RGBA image from device
     * @param rgba_device: pointer to the device image data
     * @param width: image width
     * @param height: image height
     * @param rgba_host: pointer to the image data
     * @return: 0 on success, >0 on error
     */
    int download_rgbf_from_device(const float *rgba_device, int width, int height, float *rgba_host);

    /**
     *  Free RGBA image on device
     * @param rgba_device: pointer to the device image data
     * @return: 0 on success, >0 on error
     */
    int free_rgbf_on_device(float *rgba_device);

    /**
     *  Allocate RGBA image on host
     * @param width: image width
     * @param height: image height
     * @param pinned: 1 if the memory is pinned, 0 otherwise
     * @return: pointer to the image data
     */
    void* alloc_rgbf_on_host(size_t size, int pinned);

    /**
     *  Free pinned memory on host
     * @param ptr: pointer to the memory to free
     * @return: 0 on success, >0 on error
     */
    void free_rgbf_on_host(void* ptr, int pinned);
#endif

#endif // GPU_FUNCTIONS_H

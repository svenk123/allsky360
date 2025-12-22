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
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#define CHANNELS 4

extern "C" int upload_rgbf_to_device(const float *rgba_host,
    int width, int height,
    float **rgba_device)
{
    if (!rgba_host || width <= 0 || height <= 0) {
        printf("upload_rgbf_to_device: Invalid parameters\n");
        return 1;
    }

    size_t nbytes = (size_t)width * (size_t)height * CHANNELS * sizeof(float);
    
    // Debug: Print memory requirements
    printf("Requesting %zu bytes (%dx%d pixels, %d channels)\n", 
           nbytes, width, height, CHANNELS);

    float *dev_ptr = NULL;
    cudaError_t st;

    // Initialize CUDA context (Jetson-specific)
    st = cudaFree(0); // This initializes the CUDA context
    if (st != cudaSuccess) {
        printf("upload_rgbf_to_device: CUDA context initialization failed: %s\n", cudaGetErrorString(st));
        printf("upload_rgbf_to_device: Falling back to CPU implementation\n");
        return 1;
    }
    
    // Ensure we're using the correct device for current user
    st = cudaSetDevice(0);
    if (st != cudaSuccess) {
        printf("upload_rgbf_to_device: Failed to set device for current user: %s\n", cudaGetErrorString(st));
        printf("upload_rgbf_to_device: Falling back to CPU implementation\n");
        return 1;
    }
    
    // Check CUDA device availability
    int device_count = 0;
    st = cudaGetDeviceCount(&device_count);
    if (st != cudaSuccess) {
        printf("upload_rgbf_to_device: CUDA not available: %s\n", cudaGetErrorString(st));
        printf("upload_rgbf_to_device: This might be a Jetson system without CUDA support\n");
        printf("upload_rgbf_to_device: Falling back to CPU implementation\n");
        return 1; // Signal to use CPU fallback
    }
    if (device_count == 0) {
        printf("upload_rgbf_to_device: No CUDA devices found - falling back to CPU\n");
        return 1; // Signal to use CPU fallback
    }
    
    // Set device 0 and get device info
    st = cudaSetDevice(0);
    if (st != cudaSuccess) {
        printf("upload_rgbf_to_device: Failed to set CUDA device 0: %s\n", cudaGetErrorString(st));
        printf("upload_rgbf_to_device: Falling back to CPU implementation\n");
        return 1;
    }
    
    // Get device properties for debugging
    cudaDeviceProp prop;
    st = cudaGetDeviceProperties(&prop, 0);
    if (st == cudaSuccess) {
        printf("upload_rgbf_to_device: CUDA Device: %s\n", prop.name);
        printf("upload_rgbf_to_device: Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("upload_rgbf_to_device: Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
    }
    
    // Get current device memory info
    size_t free_mem, total_mem;
    st = cudaMemGetInfo(&free_mem, &total_mem);
    if (st == cudaSuccess) {
        printf("upload_rgbf_to_device: GPU memory - Free: %zu MB, Total: %zu MB, Requested: %zu MB\n",
               free_mem / (1024*1024), total_mem / (1024*1024), nbytes / (1024*1024));
        if (free_mem < nbytes) {
            printf("upload_rgbf_to_device: ERROR - Not enough GPU memory available!\n");
            return 1;
        }
    }

    st = cudaMalloc((void**)&dev_ptr, nbytes);
    if (st != cudaSuccess) {
        printf("upload_rgbf_to_device: CUDA malloc failed: %s\n", cudaGetErrorString(st));
        printf("upload_rgbf_to_device: Failed to allocate %zu bytes on GPU\n", nbytes);
        return 1;
    }   

    st = cudaMemcpy(dev_ptr, rgba_host, nbytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        printf("upload_rgbf_to_device: CUDA memcpy H2D failed: %s\n", cudaGetErrorString(st));
        cudaFree(dev_ptr);
        return 1;
    }

    *rgba_device = dev_ptr;

    printf("Upload image buffer to Device (GPU): ok\n");

    return 0;
}

extern "C" int download_rgbf_from_device(const float *rgba_device,
    int width, int height,
    float *rgba_host)
{
    if (!rgba_device || !rgba_host || width <= 0 || height <= 0) {
        printf("download_rgbf_from_device: Invalid parameters\n");
        return 1;
    }

    size_t nbytes = (size_t)width * (size_t)height * CHANNELS * sizeof(float);

    cudaError_t st = cudaMemcpy(rgba_host, rgba_device, nbytes, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        printf("download_rgbf_from_device: CUDA memcpy D2H failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    printf("Download image buffer from Device (GPU): ok\n");

    return 0;
}

extern "C" int free_rgbf_on_device(float *rgba_device) {
    if (!rgba_device) {
        printf("free_rgbf_on_device: Invalid parameter (NULL pointer)\n");
        return 1;
    }

    cudaError_t st = cudaFree(rgba_device);
    if (st != cudaSuccess) {
        printf("free_rgbf_on_device: CUDA free failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    printf("Free image buffer on Device (GPU): ok\n");

    return 0;
}

extern "C" void* alloc_rgbf_on_host(size_t size, int pinned) {
    if (pinned) {
        float *ptr = NULL;
        cudaError_t st = cudaMallocHost((void**)&ptr, size);
        if (st != cudaSuccess) {
            printf("alloc_rgbf_on_host: CUDA malloc failed: %s\n", cudaGetErrorString(st));
            return NULL;
        }
        return ptr;
    }
    return (void*)allsky_safe_malloc(size);
}

extern "C" void free_rgbf_on_host(void* ptr, int pinned) {
    if (!ptr) {
        printf("free_rgbf_on_host: Invalid parameter (NULL pointer)\n");
        return;
    }

    if (pinned) {
        cudaFreeHost(ptr);
        return;
    }
    allsky_safe_free((void*)ptr);
}

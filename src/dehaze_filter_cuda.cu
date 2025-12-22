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
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CLAMPF(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))
#define CHANNELS 4  // RGBA

// Comparison function for qsort
static int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

__global__ void compute_luminance_kernel(const float *rgba_image, float *luma, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    int idx = i * CHANNELS;
    float r = rgba_image[idx + 0];
    float g = rgba_image[idx + 1];
    float b = rgba_image[idx + 2];
    
    luma[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__global__ void laplacian_filter_kernel(const float *src, float *dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float center = src[y * width + x];
    float sum = 0.0f;
    int count = 0;
    
    // 4-neighbour Laplacian
    if (x > 0) {
        sum += src[y * width + (x - 1)];
        count++;
    }
    if (x < width - 1) {
        sum += src[y * width + (x + 1)];
        count++;
    }
    if (y > 0) {
        sum += src[(y - 1) * width + x];
        count++;
    }
    if (y < height - 1) {
        sum += src[(y + 1) * width + x];
        count++;
    }
    
    dst[y * width + x] = center * (float)count - sum;
}

__global__ void apply_dehaze_kernel(float *rgba_image, const float *lap, int pixel_count, 
                                   float haze_level, float amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    int idx = i * CHANNELS;
    float delta = lap[i];
    float grad = fabsf(delta);
    const float epsilon = 5000.0f;
    float adapt_amount = amount * (grad / (grad + epsilon));
    
    for (int c = 0; c < 3; ++c) {
        float I = rgba_image[idx + c];
        float J = (I - haze_level) * (1.0f + adapt_amount) + adapt_amount * delta;
        rgba_image[idx + c] = CLAMPF(J);
    }
    rgba_image[idx + 3] = 1.0f; // Alpha stays 1.0f
}

// Simple reduction kernel for finding haze level
__global__ void collect_valid_luma_kernel(const float *rgba_image, const float *luma, 
                                         float *valid_luma, int *valid_count, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    int idx = i * CHANNELS;
    float r = rgba_image[idx + 0];
    float g = rgba_image[idx + 1];
    float b = rgba_image[idx + 2];
    
    // Check if pixel is not fully black
    if (!(r == 0.0f && g == 0.0f && b == 0.0f)) {
        int local_idx = atomicAdd(valid_count, 1);
        if (local_idx < pixel_count) { // Safety check to prevent buffer overflow
            valid_luma[local_idx] = luma[i];
        }
    }
}

// Simple sorting kernel for haze level calculation
__global__ void sort_and_calculate_haze_kernel(const float *valid_luma, int count, 
                                              float haze_percent, float *haze_level) {
    // This is a simplified version - in practice, you might want to use
    // a more sophisticated sorting algorithm or do this on the host
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int num_top = (int)(count * haze_percent);
        if (num_top <= 0) num_top = 1;
        
        float haze_sum = 0.0f;
        for (int i = 0; i < num_top && i < count; ++i) {
            haze_sum += valid_luma[i];
        }
        *haze_level = haze_sum / num_top;
    }
}

extern "C" int perceptual_dehaze_rgbf1_multiscale_full_cuda(float *rgba_image, int width, int height, 
                                                             float amount, float haze_percent) {
    if (!rgba_image || width <= 0 || height <= 0 || amount <= 0.0f ||
        haze_percent <= 0.0f || haze_percent > 1.0f)
        return 1;

    printf("Perceptual Dehaze (GPU): amount=%f, haze_percent=%f\n", amount, haze_percent);

    int pixel_count = width * height;
    size_t image_bytes = pixel_count * sizeof(float);
    
    cudaError_t err;
    
    // Allocate device memory
    float *d_luma, *d_lap, *d_valid_luma, *d_haze_level;
    int *d_valid_count;
    
    // Declare all variables at the beginning to avoid goto issues
    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    float *h_valid_luma = nullptr;
    int num_top = 0;
    float haze_sum = 0.0f;
    float haze_level = 0.0f;
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    int valid_count = 0;
    
    err = cudaMalloc(&d_luma, image_bytes);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_luma failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_lap, image_bytes);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_lap failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_luma);
        return 1;
    }
    
    err = cudaMalloc(&d_valid_luma, image_bytes);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_valid_luma failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_luma);
        cudaFree(d_lap);
        return 1;
    }
    
    err = cudaMalloc(&d_haze_level, sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc d_haze_level failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_luma);
        cudaFree(d_lap);
        cudaFree(d_valid_luma);
        return 1;
    }
    
    err = cudaMalloc(&d_valid_count, sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc d_valid_count failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_luma);
        cudaFree(d_lap);
        cudaFree(d_valid_luma);
        cudaFree(d_haze_level);
        return 1;
    }
    
    // Initialize valid count
    err = cudaMemset(d_valid_count, 0, sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA memset d_valid_count failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Step 1: Compute Luminance
    
    compute_luminance_kernel<<<blocks, threads>>>(rgba_image, d_luma, pixel_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("compute_luminance_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Step 2: Collect valid luma values for haze level calculation
    collect_valid_luma_kernel<<<blocks, threads>>>(rgba_image, d_luma, d_valid_luma, d_valid_count, pixel_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("collect_valid_luma_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Copy valid count to host
    err = cudaMemcpy(&valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy valid_count failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    if (valid_count == 0) {
        printf("Warning: No valid pixels found for haze calculation\n");
        goto cleanup;
    }
    
    // Safety check: prevent processing of extremely large arrays
    if (valid_count > 10000000) { // 10M pixels limit
        printf("Warning: Too many valid pixels (%d), limiting to 10M\n", valid_count);
        valid_count = 10000000;
    }
    
    // Copy valid luma to host for sorting (simplified approach)
    h_valid_luma = (float*)malloc(valid_count * sizeof(float));
    if (!h_valid_luma) {
        printf("Host memory allocation failed\n");
        goto cleanup;
    }
    
    err = cudaMemcpy(h_valid_luma, d_valid_luma, valid_count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy h_valid_luma failed: %s\n", cudaGetErrorString(err));
        free(h_valid_luma);
        goto cleanup;
    }
    
    // Sort on host using qsort (much faster than bubble sort)
    qsort(h_valid_luma, valid_count, sizeof(float), compare_floats);
    
    // Calculate haze level
    num_top = (int)(valid_count * haze_percent);
    if (num_top <= 0) num_top = 1;
    
    haze_sum = 0.0f;
    for (int i = 0; i < num_top; ++i) {
        haze_sum += h_valid_luma[i];
    }
    haze_level = haze_sum / num_top;
    
    // Step 3: Laplacian filter
    
    laplacian_filter_kernel<<<gridSize, blockSize>>>(d_luma, d_lap, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("laplacian_filter_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Step 4: Apply dehaze
    apply_dehaze_kernel<<<blocks, threads>>>(rgba_image, d_lap, pixel_count, haze_level, amount);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("apply_dehaze_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Synchronize to ensure all kernels have completed
    cudaDeviceSynchronize();
    
    printf("Perceptual Dehaze (GPU): ok, haze_level=%.4f\n", haze_level);
    
cleanup:
    if (h_valid_luma) {
        free(h_valid_luma);
        h_valid_luma = nullptr;
    }
    cudaFree(d_luma);
    cudaFree(d_lap);
    cudaFree(d_valid_luma);
    cudaFree(d_haze_level);
    cudaFree(d_valid_count);
    
    return (err == cudaSuccess) ? 0 : 1;
}

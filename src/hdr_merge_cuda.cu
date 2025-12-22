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
#define EPS 1e-6f

__global__ void find_minmax_kernel(const float *rgba, float *min_val, float *max_val, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    const float *px = &rgba[i * CHANNELS];
    
    // Skip masked pixels (all RGB channels are 0 or very close to 0)
    if (px[0] <= EPS && px[1] <= EPS && px[2] <= EPS)
        return;
    
    // Use shared memory for reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Initialize shared memory
    sdata[tid] = 1e30f;  // min
    sdata[tid + blockDim.x] = -1e30f;  // max
    
    __syncthreads();
    
    // Find min/max for this thread
    float thread_min = 1e30f;
    float thread_max = -1e30f;
    
    for (int c = 0; c < 3; c++) {
        float v = px[c];
        if (v < thread_min) thread_min = v;
        if (v > thread_max) thread_max = v;
    }
    
    sdata[tid] = thread_min;
    sdata[tid + blockDim.x] = thread_max;
    
    __syncthreads();
    
    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
            sdata[tid + blockDim.x] = fmaxf(sdata[tid + blockDim.x], sdata[tid + blockDim.x + s]);
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        min_val[bid] = sdata[0];
        max_val[bid] = sdata[blockDim.x];
    }
}

// CUDA kernel for finding min/max values in RGB channels (for normalization)
__global__ void find_rgb_minmax_kernel(const float *rgba, float *min_val, float *max_val, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    const float *px = &rgba[i * CHANNELS];
    
    // Use shared memory for reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Initialize shared memory
    sdata[tid] = 1e30f;  // min
    sdata[tid + blockDim.x] = -1e30f;  // max
    
    __syncthreads();
    
    // Find min/max for RGB channels only
    float thread_min = 1e30f;
    float thread_max = -1e30f;
    
    for (int c = 0; c < 3; c++) { // R, G, B only
        float v = px[c];
        if (v < thread_min) thread_min = v;
        if (v > thread_max) thread_max = v;
    }
    
    sdata[tid] = thread_min;
    sdata[tid + blockDim.x] = thread_max;
    
    __syncthreads();
    
    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
            sdata[tid + blockDim.x] = fmaxf(sdata[tid + blockDim.x], sdata[tid + blockDim.x + s]);
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        min_val[bid] = sdata[0];
        max_val[bid] = sdata[blockDim.x];
    }
}

// CUDA kernel for normalizing RGB values to target range
__global__ void normalize_range_kernel(float *rgba, int pixel_count, float min_val, float max_val, 
                                      float target_min, float target_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    int idx = i * CHANNELS;
    
    // Avoid division by zero
    float range_in = max_val - min_val;
    if (range_in < 1e-6f)
        range_in = 1.0f;
    
    float range_out = target_max - target_min;
    
    // Normalize RGB channels
    for (int c = 0; c < 3; c++) {
        float v = rgba[idx + c];
        v = (v - min_val) / range_in;   // [0..1]
        v = target_min + v * range_out; // [target_min..target_max]
        
        // Clamp to target range
        if (v < target_min)
            v = target_min;
        if (v > target_max)
            v = target_max;
        
        rgba[idx + c] = v;
    }
    
    // Set alpha to 1.0
    rgba[idx + 3] = 1.0f;
}

extern "C" int hdr_normalize_range_rgbf1_cuda(float *rgba, int width, int height, float target_min, float target_max) {
    if (!rgba || width <= 0 || height <= 0 || target_max <= target_min)
        return 1;

    printf("HDR Normalize Range (GPU): target %.3f .. %.3f\n", target_min, target_max);

    int pixel_count = width * height;
    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    
    // Allocate device memory for min/max reduction
    float *d_min_vals, *d_max_vals;
    cudaError_t err;
    
    err = cudaMalloc(&d_min_vals, blocks * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc d_min_vals failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_max_vals, blocks * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc d_max_vals failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_min_vals);
        return 1;
    }
    
    // Find min/max values for RGB channels
    size_t shared_mem_size = 2 * threads * sizeof(float);
    find_rgb_minmax_kernel<<<blocks, threads, shared_mem_size>>>(rgba, d_min_vals, d_max_vals, pixel_count);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("find_rgb_minmax_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_min_vals);
        cudaFree(d_max_vals);
        return 1;
    }
    
    // Copy results back to host for final reduction
    float *h_min_vals = (float*)malloc(blocks * sizeof(float));
    float *h_max_vals = (float*)malloc(blocks * sizeof(float));
    
    if (!h_min_vals || !h_max_vals) {
        printf("Host memory allocation failed\n");
        cudaFree(d_min_vals);
        cudaFree(d_max_vals);
        free(h_min_vals);
        free(h_max_vals);
        return 1;
    }
    
    err = cudaMemcpy(h_min_vals, d_min_vals, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy h_min_vals failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_min_vals);
        cudaFree(d_max_vals);
        free(h_min_vals);
        free(h_max_vals);
        return 1;
    }
    
    err = cudaMemcpy(h_max_vals, d_max_vals, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy h_max_vals failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_min_vals);
        cudaFree(d_max_vals);
        free(h_min_vals);
        free(h_max_vals);
        return 1;
    }
    
    // Final reduction on host
    float min_val = 1e30f;
    float max_val = -1e30f;
    
    for (int i = 0; i < blocks; i++) {
        if (h_min_vals[i] < min_val) min_val = h_min_vals[i];
        if (h_max_vals[i] > max_val) max_val = h_max_vals[i];
    }
    
    printf("HDR Normalize (GPU): min=%.6f max=%.6f -> target %.3f .. %.3f\n", 
           min_val, max_val, target_min, target_max);
    
    // Apply normalization
    normalize_range_kernel<<<blocks, threads>>>(rgba, pixel_count, min_val, max_val, target_min, target_max);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("normalize_range_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_min_vals);
        cudaFree(d_max_vals);
        free(h_min_vals);
        free(h_max_vals);
        return 1;
    }
    
    // Synchronize to ensure all kernels have completed
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_min_vals);
    cudaFree(d_max_vals);
    free(h_min_vals);
    free(h_max_vals);
    
    printf("HDR Normalize (GPU): ok\n");
    
    return 0;
}

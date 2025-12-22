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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CLAMPF(x) ((x) < 0.0f ? 0.0f : ((x) > 65535.0f ? 65535.0f : (x)))

__device__ float quick_median(float *v, int len) {
    // Naive bubble sort for small arrays
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - i - 1; j++) {
            if (v[j] > v[j + 1]) {
                float tmp = v[j];
                v[j] = v[j + 1];
                v[j + 1] = tmp;
            }
        }
    }
    return v[len / 2];
}

__global__ void median_filter_rgbf_kernel(const float *input, float *output, int width, int height, int kernel_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < kernel_radius || x >= width - kernel_radius || y < kernel_radius || y >= height - kernel_radius) return;

    int idx = (y * width + x) * 4;
    int size = (2 * kernel_radius + 1) * (2 * kernel_radius + 1);

    extern __shared__ float buffer[];
    float *r_vals = buffer;
    float *g_vals = buffer + size;
    float *b_vals = buffer + 2 * size;

    int k = 0;
    for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
        for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
            int ix = x + dx;
            int iy = y + dy;
            int iidx = (iy * width + ix) * 4;
            r_vals[k] = input[iidx + 0];
            g_vals[k] = input[iidx + 1];
            b_vals[k] = input[iidx + 2];
            k++;
        }
    }
    output[idx + 0] = quick_median(r_vals, size);
    output[idx + 1] = quick_median(g_vals, size);
    output[idx + 2] = quick_median(b_vals, size);
    output[idx + 3] = 65535.0f;
}

extern "C" int median_filter_rgbf_cuda(float *rgba, int width, int height, int kernel_radius) {
    if (!rgba || width <= 0 || height <= 0 || kernel_radius < 1 || kernel_radius > 10) return 1;

    int pixel_count = width * height;
    size_t bufsize = pixel_count * 4 * sizeof(float);

    float *dev_input = nullptr;
    float *dev_output = nullptr;
    
    cudaError_t err;
    err = cudaMalloc(&dev_input, bufsize);
    if (err != cudaSuccess) { printf("CUDA malloc dev_input failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&dev_output, bufsize);
    if (err != cudaSuccess) { printf("CUDA malloc dev_output failed: %s\n", cudaGetErrorString(err)); return 1; }
    
    err = cudaMemcpy(dev_input, rgba, bufsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); return 1; }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    int kernel_size = 2 * kernel_radius + 1;
    int shared_mem_size = 3 * kernel_size * kernel_size * sizeof(float);

    median_filter_rgbf_kernel<<<gridSize, blockSize, shared_mem_size>>>(dev_input, dev_output, width, height, kernel_radius);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("median_filter_rgbf_kernel failed: %s\n", cudaGetErrorString(err)); return 1; }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    err = cudaMemcpy(rgba, dev_output, bufsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err)); return 1; }

    cudaFree(dev_input);
    cudaFree(dev_output);

    printf("Median filter (GPU): ok\n");
    return 0;
}

/*************************************/

#define CLAMP(v, min, max) ((v) < (min) ? (min) : ((v) > (max) ? (max) : (v)))

// Device swap
__device__ inline void swapf(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}

// Device median-of-three pivot
__device__ inline float median_of_three(float *arr, int a, int b, int c) {
    if (arr[a] < arr[b]) {
        if (arr[b] < arr[c]) return arr[b];
        else if (arr[a] < arr[c]) return arr[c];
        else return arr[a];
    } else {
        if (arr[a] < arr[c]) return arr[a];
        else if (arr[b] < arr[c]) return arr[c];
        else return arr[b];
    }
}

// Device quickselect
__device__ float quickselect(float *arr, int n, int k) {
    int left = 0, right = n - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        float pivot = median_of_three(arr, left, mid, right);

        int i = left, j = right;
        while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
                swapf(arr[i], arr[j]);
                i++; j--;
            }
        }

        if (k <= j) right = j;
        else if (k >= i) left = i;
        else return arr[k];
    }
    return arr[left];
}

// CUDA kernel
__global__ void multiscale_median_kernel(
    float *rgba, float *temp, int width, int height,
    int radius, float blend_factor, int use_quickselect)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int max_window = 225; // radius <= 7
    float r_vals[max_window], g_vals[max_window], b_vals[max_window];

    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        int yy = CLAMP(y + dy, 0, height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = CLAMP(x + dx, 0, width - 1);
            int idx = (yy * width + xx) * 4;
            r_vals[count] = rgba[idx + 0];
            g_vals[count] = rgba[idx + 1];
            b_vals[count] = rgba[idx + 2];
            count++;
        }
    }

    int k = count / 2;
    float r_med, g_med, b_med;
    if (use_quickselect) {
        r_med = quickselect(r_vals, count, k);
        g_med = quickselect(g_vals, count, k);
        b_med = quickselect(b_vals, count, k);
    } else {
        // Partial sort selection sort
        for (int i = 0; i <= k; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < count; ++j)
                if (r_vals[j] < r_vals[min_idx]) min_idx = j;
            swapf(r_vals[i], r_vals[min_idx]);
        }
        for (int i = 0; i <= k; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < count; ++j)
                if (g_vals[j] < g_vals[min_idx]) min_idx = j;
            swapf(g_vals[i], g_vals[min_idx]);
        }
        for (int i = 0; i <= k; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < count; ++j)
                if (b_vals[j] < b_vals[min_idx]) min_idx = j;
            swapf(b_vals[i], b_vals[min_idx]);
        }
        r_med = r_vals[k];
        g_med = g_vals[k];
        b_med = b_vals[k];
    }

    int out_idx = (y * width + x) * 4;
    temp[out_idx + 0] = (1.0f - blend_factor) * rgba[out_idx + 0] + blend_factor * r_med;
    temp[out_idx + 1] = (1.0f - blend_factor) * rgba[out_idx + 1] + blend_factor * g_med;
    temp[out_idx + 2] = (1.0f - blend_factor) * rgba[out_idx + 2] + blend_factor * b_med;
    temp[out_idx + 3] = rgba[out_idx + 3];
}

extern "C" int multiscale_median_filter_rgbf1_cuda(
    float *host_rgba, int width, int height, int max_scale,
    float blend_factor)
{
    if (!host_rgba || width <= 0 || height <= 0 || max_scale < 1 || max_scale > 10 || blend_factor < 0.0f || blend_factor > 1.0f) {
        printf("Multiscale median filter (GPU): invalid parameters\n");
        return 1;
    }

    int use_quickselect = 1;
    float *dev_rgba = NULL, *dev_temp = NULL;
    size_t bytes = sizeof(float) * 4 * width * height;

    cudaError_t err;
    err = cudaMalloc(&dev_rgba, bytes);
    if (err != cudaSuccess) { printf("CUDA malloc dev_rgba failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&dev_temp, bytes);
    if (err != cudaSuccess) { printf("CUDA malloc dev_temp failed: %s\n", cudaGetErrorString(err)); return 1; }
    
    err = cudaMemcpy(dev_rgba, host_rgba, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); return 1; }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    for (int scale = 1; scale <= max_scale; ++scale) {
        int radius = scale;
        int ksize = 2 * radius + 1;
        int window_area = ksize * ksize;
        if (window_area > 225) {
            printf("Warning: kernel too large (%d x %d), skipping scale %d\n", ksize, ksize, scale);
            continue;
        }

        printf("Launching CUDA kernel: scale %d, radius %d, blend %.2f, quickselect %s\n",
               scale, radius, blend_factor, use_quickselect ? "ON" : "OFF");

        multiscale_median_kernel<<<grid, block>>>(
            dev_rgba, dev_temp, width, height, radius, blend_factor, use_quickselect);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("multiscale_median_kernel failed: %s\n", cudaGetErrorString(err)); return 1; }
        
        cudaDeviceSynchronize();

        // swap pointers
        float *tmp = dev_rgba;
        dev_rgba = dev_temp;
        dev_temp = tmp;
    }

    err = cudaMemcpy(host_rgba, dev_rgba, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err)); return 1; }
    
    cudaFree(dev_rgba);
    cudaFree(dev_temp);

    printf("Multi-scale median filter (GPU): ok\n");
    return 0;
}







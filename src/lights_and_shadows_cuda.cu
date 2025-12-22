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
#include <stdlib.h>

#define CHANNELS 4
#define CLAMPF1(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))

// CUDA kernel for autostretch stretch operation
__global__ void autostretch_stretch_kernel(float *rgba, int pixel_count, float black, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    
    int base = idx * CHANNELS;
    for (int c = 0; c < 3; ++c) {
        float val = (rgba[base + c] - black) * scale;
        rgba[base + c] = CLAMPF1(val);
    }
    // Alpha channel remains unchanged
}

// CUDA kernel for collecting valid RGB values
__global__ void collect_valid_values_kernel(const float *rgba, float *valid_values, int *valid_count, int pixel_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    
    int base = idx * CHANNELS;
    for (int c = 0; c < 3; ++c) {
        float val = rgba[base + c];
        if (isfinite(val) && val >= 0.0f) {
            int pos = atomicAdd(valid_count, 1);
            if (pos < pixel_count * 3) {
                valid_values[pos] = val;
            }
        }
    }
}

// CUDA kernel for histogram computation (for adjust_black_point)
__global__ void compute_histogram_kernel(const float *rgba, int *hist, int pixel_count, int hist_bins, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    
    const float *px = &rgba[idx * CHANNELS];
    
    // Skip masked pixels
    if (px[0] <= eps && px[1] <= eps && px[2] <= eps) return;
    
    // Compute luminance (Rec.709)
    float L = 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
    L = fmaxf(0.0f, fminf(1.0f, L));
    
    int bin = (int)(L * (hist_bins - 1));
    if (bin >= 0 && bin < hist_bins) {
        atomicAdd(&hist[bin], 1);
    }
}

// CUDA kernel for black point adjustment
__global__ void apply_black_point_kernel(float *rgba, int pixel_count, float black_point, float denom, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    
    float *px = &rgba[idx * CHANNELS];
    if (px[0] <= eps && px[1] <= eps && px[2] <= eps) return;
    
    for (int c = 0; c < 3; c++) {
        float v = (px[c] - black_point) / denom;
        px[c] = CLAMPF1(v);
    }
}

// Helper comparison function for qsort
static int compare_floats_cuda(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

extern "C" int autostretch_rgbf1_cuda(float *hdr_image, int width, int height, float min_val, float max_val) {
    if (!hdr_image || width <= 0 || height <= 0 || 
        min_val < 0.0f || min_val > 1.0f || 
        max_val < 0.0f || max_val > 1.0f || 
        max_val <= min_val)
        return 1;

    int pixel_count = width * height;
    
    // Collect valid values on GPU (hdr_image is already on device)
    cudaError_t err;
    float *dev_valid_values = nullptr;
    int *dev_valid_count = nullptr;
    int *h_valid_count = (int*)malloc(sizeof(int));
    *h_valid_count = 0;
    
    err = cudaMalloc(&dev_valid_values, pixel_count * 3 * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc dev_valid_values failed: %s\n", cudaGetErrorString(err));
        free(h_valid_count);
        return 1;
    }
    
    err = cudaMalloc(&dev_valid_count, sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc dev_valid_count failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_valid_values);
        free(h_valid_count);
        return 1;
    }
    
    err = cudaMemset(dev_valid_count, 0, sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA memset failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_valid_values);
        cudaFree(dev_valid_count);
        free(h_valid_count);
        return 1;
    }
    
    // Launch kernel to collect valid values (hdr_image is already on device)
    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    collect_valid_values_kernel<<<blocks, threads>>>(hdr_image, dev_valid_values, dev_valid_count, pixel_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("collect_valid_values_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_valid_values);
        cudaFree(dev_valid_count);
        free(h_valid_count);
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // Copy valid count back
    err = cudaMemcpy(h_valid_count, dev_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy D2H (valid_count) failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_valid_values);
        cudaFree(dev_valid_count);
        free(h_valid_count);
        return 1;
    }
    
    int k = *h_valid_count;
    free(h_valid_count);
    
    if (k < 2) {
        printf("Autostretch: not enough valid values: k=%d\n", k);
        cudaFree(dev_valid_values);
        cudaFree(dev_valid_count);
        return 1;
    }
    
    // Copy valid values to host for percentile calculation (CPU)
    float *h_valid_values = (float*)malloc(k * sizeof(float));
    if (!h_valid_values) {
        cudaFree(dev_valid_values);
        cudaFree(dev_valid_count);
        return 1;
    }
    
    err = cudaMemcpy(h_valid_values, dev_valid_values, k * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy D2H (valid_values) failed: %s\n", cudaGetErrorString(err));
        free(h_valid_values);
        cudaFree(dev_valid_values);
        cudaFree(dev_valid_count);
        return 1;
    }
    
    cudaFree(dev_valid_values);
    cudaFree(dev_valid_count);
    
    // Calculate percentiles on CPU (using qsort for simplicity)
    // Note: For better performance, could use Quickselect, but qsort is simpler here
    if (k == 2) {
        printf("Found only 2 valid values: simple Min/Max stretch\n");
        float black = h_valid_values[0] < h_valid_values[1] ? h_valid_values[0] : h_valid_values[1];
        float white = h_valid_values[0] > h_valid_values[1] ? h_valid_values[0] : h_valid_values[1];
        free(h_valid_values);
        
        float range = white - black;
        if (range < 1e-6f) {
            return 1;
        }
        
        float scale = 1.0f / range;
        
        // Apply stretch on GPU (hdr_image is already on device, modify in-place)
        autostretch_stretch_kernel<<<blocks, threads>>>(hdr_image, pixel_count, black, scale);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("autostretch_stretch_kernel failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        cudaDeviceSynchronize();
        printf("Autostretch (GPU): ok, %.2f . %.2f (%.3f . %.3f)\n",
               min_val * 100.0f, max_val * 100.0f, black, white);
        return 0;
    }
    
    // Sort for percentile calculation
    qsort(h_valid_values, k, sizeof(float), compare_floats_cuda);
    
    int idx_min = (int)(min_val * (k - 1));
    int idx_max = (int)(max_val * (k - 1));
    if (idx_min < 0) idx_min = 0;
    if (idx_max >= k) idx_max = k - 1;
    
    float black = h_valid_values[idx_min];
    float white = h_valid_values[idx_max];
    free(h_valid_values);
    
    float range = white - black;
    if (range < 1e-6f) {
        return 1;
    }
    
    float scale = 1.0f / range;
    
    // Apply stretch on GPU (hdr_image is already on device, modify in-place)
    autostretch_stretch_kernel<<<blocks, threads>>>(hdr_image, pixel_count, black, scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("autostretch_stretch_kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    printf("Autostretch (GPU): ok, %.2f . %.2f (%.3f . %.3f)\n",
           min_val * 100.0f, max_val * 100.0f, black, white);
    return 0;
}

extern "C" int adjust_black_point_rgbf1_cuda(float *rgbf, int width, int height,
                                               double min_shift_pct, double max_shift_pct,
                                               double dark_threshold) {
    if (!rgbf || width <= 0 || height <= 0)
        return 1;

    const int pixel_count = width * height;
    const int hist_bins = 2048;
    const float eps = 1e-6f;
    
    // Allocate device memory for histogram only (rgbf is already on device)
    int *dev_hist = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&dev_hist, hist_bins * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc dev_hist failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Initialize histogram to zero
    err = cudaMemset(dev_hist, 0, hist_bins * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA memset failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_hist);
        return 1;
    }
    
    // Compute histogram on GPU (rgbf is already on device)
    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    compute_histogram_kernel<<<blocks, threads>>>(rgbf, dev_hist, pixel_count, hist_bins, eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("compute_histogram_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_hist);
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // Copy histogram back to host
    int *hist = (int*)calloc(hist_bins, sizeof(int));
    if (!hist) {
        cudaFree(dev_hist);
        return 1;
    }
    
    err = cudaMemcpy(hist, dev_hist, hist_bins * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy D2H (hist) failed: %s\n", cudaGetErrorString(err));
        free(hist);
        cudaFree(dev_hist);
        return 1;
    }
    
    cudaFree(dev_hist);
    
    // Compute statistics on CPU
    long valid_pixels = 0;
    double sum_L = 0.0;
    double min_val = 1.0, max_val = 0.0;
    
    // Calculate statistics from histogram
    for (int b = 0; b < hist_bins; b++) {
        if (hist[b] > 0) {
            float L = (float)b / (float)(hist_bins - 1);
            valid_pixels += hist[b];
            sum_L += L * hist[b];
            if (L < min_val) min_val = L;
            if (L > max_val) max_val = L;
        }
    }
    
    if (valid_pixels == 0) {
        free(hist);
        printf("Warning: No valid pixels found for black point adjustment.\n");
        return 1;
    }
    
    printf("Merged local histograms into global histogram\n");

    // Compute quantiles
    long target01 = (long)(valid_pixels * 0.01);
    long target99 = (long)(valid_pixels * 0.99);
    long cumulative = 0;
    float p01 = 0.0f, p99 = 1.0f;
    for (int b = 0; b < hist_bins; b++) {
        cumulative += hist[b];
        if (cumulative >= target01 && p01 == 0.0f)
            p01 = (float)b / (float)(hist_bins - 1);
        if (cumulative >= target99) {
            p99 = (float)b / (float)(hist_bins - 1);
            break;
        }
    }
    
    free(hist);
    
    double avg_L = sum_L / (double)valid_pixels;
    
    printf("Computed quantiles for global histogram\n");

    // Decide shift adaptively
    double black_shift_pct = 0.0;
    
    if (avg_L > dark_threshold) {
        black_shift_pct = min_shift_pct;
    } else if (p01 > 0.02 && (p99 - p01) > 0.05) {
        black_shift_pct = max_shift_pct;
    } else if (p01 > 0.005) {
        black_shift_pct = (min_shift_pct + max_shift_pct) * 0.5;
    } else {
        black_shift_pct = 0.0;
    }
    
    printf("[BlackPoint] valid=%ld  avgL=%.4f  p01=%.4f  p99=%.4f  "
           "shift=%.4f  (%.2f%%)\n",
           valid_pixels, avg_L, p01, p99, black_shift_pct,
           black_shift_pct * 100.0);
    
    if (black_shift_pct <= 0.0) {
        return 0;
    }
    
    // Apply shift on GPU (rgbf is already on device, modify in-place)
    const float range = (float)(max_val - min_val);
    const float black_point = (float)(min_val + black_shift_pct * range);
    const float denom = (float)(range * (1.0 - black_shift_pct));
    
    apply_black_point_kernel<<<blocks, threads>>>(rgbf, pixel_count, black_point, denom, eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("apply_black_point_kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    printf("Blackpoint adjustment (GPU): ok, %.1f%% (%.4f . %.4f)\n", 
           black_shift_pct * 100.0, min_val, black_point);
    
    return 0;
}


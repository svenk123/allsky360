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
#define EPSILON 1e-8f
#define TINY 1e-8f

// Mask modes (from clarity_filter.h)
#define CLARITY_MASK_NONE 0
#define CLARITY_MASK_ALPHA_ZERO 1
#define CLARITY_MASK_RGB_ALL_ZERO 2
#define CLARITY_MASK_RGB_OR_ALPHA 3

/* Compute Rec.709 linear luminance from RGB */
__device__ inline float rgb_to_luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

/* Safe ratio (avoid division by zero) */
__device__ inline float safe_ratio(float num, float den) {
    float d = (fabsf(den) < EPSILON) ? (den >= 0.f ? EPSILON : -EPSILON) : den;
    return num / d;
}

/* Decide whether a pixel is masked according to mask_mode */
__device__ inline int is_masked(const float *rgba, int mask_mode) {
    if (mask_mode == CLARITY_MASK_NONE)
        return 0;
    
    const float r = rgba[0], g = rgba[1], b = rgba[2], a = rgba[3];
    const float eps = 1e-12f;
    int alpha_zero = (fabsf(a) <= eps);
    int rgb_zero = (fabsf(r) <= eps) && (fabsf(g) <= eps) && (fabsf(b) <= eps);

    switch (mask_mode) {
    case CLARITY_MASK_ALPHA_ZERO:
        return alpha_zero;
    case CLARITY_MASK_RGB_ALL_ZERO:
        return rgb_zero;
    case CLARITY_MASK_RGB_OR_ALPHA:
        return alpha_zero || rgb_zero;
    default:
        return 0;
    }
}

/* Soft limiter to reduce halos */
__device__ inline float soft_limiter(float x) { 
    return tanhf(x); 
}

/* Midtone weighting (bell around 0.5) */
__device__ inline float midtone_bell(float y, float width) {
    float w = (width <= 0.05f) ? 0.05f : (width > 1.0f ? 1.0f : width);
    float sigma = 0.25f * w;
    float d = y - 0.5f;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    return expf(-(d * d) * inv2s2);
}

/* Optional highlight rolloff */
__device__ inline float highlight_rolloff(float y) {
    if (y <= 0.8f)
        return 1.0f;
    float t = (y - 0.8f) / 0.2f;
    if (t < 0.f) t = 0.f;
    if (t > 1.f) t = 1.f;
    return 0.5f * (1.0f + cosf((float)M_PI * t));
}

/* Kernel to compute luminance and validity mask */
__global__ void compute_luma_and_mask_kernel(const float *rgba_image, float *luma, 
                                           unsigned char *valid, int pixel_count, int mask_mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    const float *p = rgba_image + i * CHANNELS;
    int masked = is_masked(p, mask_mode);
    valid[i] = masked ? 0 : 1;
    
    float y = rgb_to_luma(p[0], p[1], p[2]);
    luma[i] = CLAMPF(y);
}

/* Horizontal Gaussian blur with masking */
__global__ void gauss_blur_h_masked_kernel(const float *src, const unsigned char *valid,
                                          float *dst, int width, int height,
                                          const float *kernel, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float num = 0.0f, den = 0.0f;
    
    for (int i = -radius; i <= radius; ++i) {
        int xi = x + i;
        // Mirror border handling
        if (xi < 0) xi = -xi;
        if (xi >= width) xi = 2 * width - 2 - xi;
        
        int kernel_idx = i + radius;
        float w = kernel[kernel_idx];
        int valid_idx = y * width + xi;
        
        if (valid[valid_idx]) {
            num += src[valid_idx] * w;
            den += w;
        }
    }
    
    dst[idx] = (den > TINY) ? (num / den) : src[idx];
}

/* Vertical Gaussian blur with masking */
__global__ void gauss_blur_v_masked_kernel(const float *src, const unsigned char *valid,
                                          float *dst, int width, int height,
                                          const float *kernel, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float num = 0.0f, den = 0.0f;
    
    for (int i = -radius; i <= radius; ++i) {
        int yi = y + i;
        // Mirror border handling
        if (yi < 0) yi = -yi;
        if (yi >= height) yi = 2 * height - 2 - yi;
        
        int kernel_idx = i + radius;
        float w = kernel[kernel_idx];
        int valid_idx = yi * width + x;
        
        if (valid[valid_idx]) {
            num += src[valid_idx] * w;
            den += w;
        }
    }
    
    dst[idx] = (den > TINY) ? (num / den) : src[idx];
}

/* Apply clarity enhancement */
__global__ void apply_clarity_kernel(float *rgba_image, float *luma, const float *luma_blur,
                                   const unsigned char *valid, int pixel_count,
                                   float strength, float midtone_width, int preserve_highlights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    if (!valid[i]) return; // Skip masked pixels
    
    float y = luma[i];
    float yb = luma_blur[i];
    float detail = soft_limiter((y - yb) * 1.0f); // pre_gain = 1.0f
    
    float w = midtone_bell(y, midtone_width);
    if (preserve_highlights)
        w *= highlight_rolloff(y);
    
    float y_new = CLAMPF(y + strength * w * detail);
    luma[i] = y_new;
}

/* Remap RGB by luma ratio */
__global__ void remap_rgb_kernel(float *rgba_image, const float *luma_new, 
                                const unsigned char *valid, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    
    if (!valid[i]) return; // Skip masked pixels
    
    float *p = rgba_image + i * CHANNELS;
    float y_old = rgb_to_luma(p[0], p[1], p[2]);
    float scale = safe_ratio(luma_new[i], y_old);
    
    if (y_old < 1e-4f) {
        float t = y_old / 1e-4f; // fade-in to avoid noise pop
        scale = t * scale + (1.0f - t);
    }
    
    p[0] = CLAMPF(p[0] * scale);
    p[1] = CLAMPF(p[1] * scale);
    p[2] = CLAMPF(p[2] * scale);
    // alpha unchanged
}

extern "C" int clarity_filter_rgbf_masked_cuda(float *rgba, int width, int height,
                                             float strength, int radius, float midtone_width,
                                             int preserve_highlights, int mask_mode) {
    if (!rgba || width <= 0 || height <= 0)
        return 1;

    if (fabsf(strength) < 1e-6f)
        return 0;

    if (radius < 1)
        radius = 1;

    if (midtone_width <= 0.f)
        midtone_width = 0.35f;

    if (midtone_width > 1.5f)
        midtone_width = 1.5f;

    printf("Clarity Filter (GPU): strength=%f, radius=%d, midtone_width=%f, preserve_highlights=%d, mask_mode=%d\n", 
           strength, radius, midtone_width, preserve_highlights, mask_mode);

    int pixel_count = width * height;
    size_t image_bytes = pixel_count * sizeof(float);
    size_t mask_bytes = pixel_count * sizeof(unsigned char);
    
    cudaError_t err = cudaSuccess;
    
    // Allocate device memory
    float *d_luma = NULL, *d_luma_tmp = NULL, *d_luma_blur = NULL, *d_kernel = NULL;
    unsigned char *d_valid = NULL;
    
    // Declare variables that might be used after goto cleanup
    float *h_kernel = NULL;
    float sigma = 0.0f;
    float inv2s2 = 0.0f;
    float sum = 0.0f;
    int threads = 256;
    int blocks = 0;
    dim3 blockSize(16, 16);
    dim3 gridSize(0, 0);
    
    err = cudaMalloc(&d_luma, image_bytes);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_luma failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_luma_tmp, image_bytes);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_luma_tmp failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_luma);
        return 1;
    }
    
    err = cudaMalloc(&d_luma_blur, image_bytes);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_luma_blur failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_luma);
        cudaFree(d_luma_tmp);
        return 1;
    }
    
    err = cudaMalloc(&d_valid, mask_bytes);
    if (err != cudaSuccess) {
        printf("CUDA malloc d_valid failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_luma);
        cudaFree(d_luma_tmp);
        cudaFree(d_luma_blur);
        return 1;
    }
    
    // Build Gaussian kernel
    int klen = 2 * radius + 1;
    err = cudaMalloc(&d_kernel, klen * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc d_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Generate kernel on host
    h_kernel = (float*)malloc(klen * sizeof(float));
    if (!h_kernel) {
        printf("Host kernel allocation failed\n");
        goto cleanup;
    }
    
    sigma = (float)radius / 3.0f;
    if (sigma < 0.5f) sigma = 0.5f;
    inv2s2 = 1.0f / (2.0f * sigma * sigma);
    
    sum = 0.0f;
    for (int i = 0; i < klen; ++i) {
        int x = i - radius;
        float v = expf(-(x * x) * inv2s2);
        h_kernel[i] = v;
        sum += v;
    }
    
    for (int i = 0; i < klen; ++i)
        h_kernel[i] /= sum;
    
    // Copy kernel to device
    err = cudaMemcpy(d_kernel, h_kernel, klen * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy d_kernel failed: %s\n", cudaGetErrorString(err));
        free(h_kernel);
        goto cleanup;
    }
    
    free(h_kernel);
    h_kernel = NULL;
    
    // Setup kernel launch parameters
    threads = 256;
    blocks = (pixel_count + threads - 1) / threads;
    blockSize = dim3(16, 16);
    gridSize = dim3((width + 15) / 16, (height + 15) / 16);
    
    // Step 1: Compute luminance and validity mask
    compute_luma_and_mask_kernel<<<blocks, threads>>>(rgba, d_luma, d_valid, pixel_count, mask_mode);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("compute_luma_and_mask_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Step 2: Horizontal Gaussian blur
    gauss_blur_h_masked_kernel<<<gridSize, blockSize>>>(d_luma, d_valid, d_luma_tmp, width, height, d_kernel, radius);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("gauss_blur_h_masked_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Step 3: Vertical Gaussian blur
    gauss_blur_v_masked_kernel<<<gridSize, blockSize>>>(d_luma_tmp, d_valid, d_luma_blur, width, height, d_kernel, radius);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("gauss_blur_v_masked_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Step 4: Apply clarity enhancement
    apply_clarity_kernel<<<blocks, threads>>>(rgba, d_luma, d_luma_blur, d_valid, pixel_count, 
                                             strength, midtone_width, preserve_highlights);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("apply_clarity_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Step 5: Remap RGB by luma ratio
    remap_rgb_kernel<<<blocks, threads>>>(rgba, d_luma, d_valid, pixel_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("remap_rgb_kernel failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Synchronize to ensure all kernels have completed
    cudaDeviceSynchronize();
    
    printf("Clarity Filter (GPU): ok\n");
    
cleanup:
    if (d_luma) cudaFree(d_luma);
    if (d_luma_tmp) cudaFree(d_luma_tmp);
    if (d_luma_blur) cudaFree(d_luma_blur);
    if (d_valid) cudaFree(d_valid);
    if (d_kernel) cudaFree(d_kernel);
    if (h_kernel) free(h_kernel);
    
    return (err == cudaSuccess) ? 0 : 1;
}


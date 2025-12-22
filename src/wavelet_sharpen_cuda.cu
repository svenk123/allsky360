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

#define CHANNELS 4

/* Gaussian blur kernels (separable) */
__global__ void gaussian_blur_horizontal(
    const float *src, float *dst, int width, int height,
    const float *kernel, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) 
        return;

    int idx = (y * width + x) * CHANNELS;
    float sum[CHANNELS] = {0};

    for (int k = -radius; k <= radius; k++) {
        int xx = x + k;

        if (xx < 0) 
            xx = 0;

        if (xx >= width) 
            xx = width - 1;

        int sidx = (y * width + xx) * CHANNELS;
        float w = kernel[k + radius];
        sum[0] += src[sidx + 0] * w;
        sum[1] += src[sidx + 1] * w;
        sum[2] += src[sidx + 2] * w;
        sum[3] += src[sidx + 3] * w;
    }

    dst[idx + 0] = sum[0];
    dst[idx + 1] = sum[1];
    dst[idx + 2] = sum[2];
    dst[idx + 3] = sum[3];
}

/* vertical Gaussian blur */
__global__ void gaussian_blur_vertical(
    const float *src, float *dst, int width, int height,
    const float *kernel, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) 
        return;

    int idx = (y * width + x) * CHANNELS;
    float sum[CHANNELS] = {0};

    for (int k = -radius; k <= radius; k++) {
        int yy = y + k;

        if (yy < 0) 
            yy = 0;

        if (yy >= height) 
            yy = height - 1;

        int sidx = (yy * width + x) * CHANNELS;
        float w = kernel[k + radius];
        sum[0] += src[sidx + 0] * w;
        sum[1] += src[sidx + 1] * w;
        sum[2] += src[sidx + 2] * w;
        sum[3] += src[sidx + 3] * w;
    }

    dst[idx + 0] = sum[0];
    dst[idx + 1] = sum[1];
    dst[idx + 2] = sum[2];
    dst[idx + 3] = sum[3];
}

/* combine blurred images into sharpened result */
__global__ void wavelet_combine(
    float *rgba, const float *blur1, const float *blur2, const float *blur3,
    int width, int height,
    float gain_small, float gain_medium, float gain_large)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) 
        return;

    int idx = (y * width + x) * CHANNELS;

    for (int c = 0; c < 3; c++) { // only RGB
        float orig   = rgba[idx + c];
        float small  = orig - blur1[idx + c];
        float medium = blur1[idx + c] - blur2[idx + c];
        float large  = blur2[idx + c] - blur3[idx + c];
        float val = orig + gain_small * small + gain_medium * medium + gain_large * large;
        rgba[idx + c] = fminf(fmaxf(val, 0.0f), 1.0f);
    }
}

/* build Gaussian kernel on host */
static void make_gaussian_kernel(float sigma, float **outKernel, int *outRadius)
{
    int radius = (int)ceilf(3.0f * sigma);
    int size = 2 * radius + 1;
    float *kernel = (float *)malloc(size * sizeof(float));
    float sum = 0.0f;

    for (int i = -radius; i <= radius; i++) {
        float v = expf(-(i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = v;
        sum += v;
    }

    for (int i = 0; i < size; i++) 
        kernel[i] /= sum;

    *outKernel = kernel;
    *outRadius = radius;
}

extern "C"
int wavelet_sharpen_rgbf1_cuda(float *d_rgba, int width, int height,
                              float gain_small, float gain_medium, float gain_large)
{
    if (!d_rgba || width <= 0 || height <= 0)
        return 1;

    cudaError_t err;
    size_t sz = (size_t)width * height * CHANNELS * sizeof(float);
    float *d_tmp1 = NULL, *d_tmp2 = NULL, *d_tmp3 = NULL, *d_tmp4 = NULL;

    // Allocate temporary buffers with error checking
    err = cudaMalloc(&d_tmp1, sz);
    if (err != cudaSuccess || !d_tmp1) {
        fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: cudaMalloc failed for d_tmp1: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&d_tmp2, sz);
    if (err != cudaSuccess || !d_tmp2) {
        fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: cudaMalloc failed for d_tmp2: %s\n", cudaGetErrorString(err));
        cudaFree(d_tmp1);
        return 1;
    }

    err = cudaMalloc(&d_tmp3, sz);
    if (err != cudaSuccess || !d_tmp3) {
        fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: cudaMalloc failed for d_tmp3: %s\n", cudaGetErrorString(err));
        cudaFree(d_tmp1);
        cudaFree(d_tmp2);
        return 1;
    }

    err = cudaMalloc(&d_tmp4, sz);
    if (err != cudaSuccess || !d_tmp4) {
        fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: cudaMalloc failed for d_tmp4: %s\n", cudaGetErrorString(err));
        cudaFree(d_tmp1);
        cudaFree(d_tmp2);
        cudaFree(d_tmp3);
        return 1;
    }

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // --- Three Gaussian blurs ---
    float sigmas[3] = {1.0f, 3.0f, 8.0f};
    float *blurred[3] = {d_tmp2, d_tmp3, d_tmp4};
    printf("Multi-scale Gaussian blurs:\n");

    for (int s = 0; s < 3; s++) {
        float *h_kernel;
        int radius;
        make_gaussian_kernel(sigmas[s], &h_kernel, &radius);
        if (!h_kernel) {
            fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: malloc failed for kernel (sigma=%.1f)\n", sigmas[s]);
            // Cleanup
            for (int i = 0; i < s; i++) {
                // Free kernels from previous iterations would be needed, but they're already freed
            }
            cudaFree(d_tmp1);
            cudaFree(d_tmp2);
            cudaFree(d_tmp3);
            cudaFree(d_tmp4);
            return 1;
        }

        int ksize = 2 * radius + 1;
        float *d_kernel = NULL;

        err = cudaMalloc(&d_kernel, ksize * sizeof(float));
        if (err != cudaSuccess || !d_kernel) {
            fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: cudaMalloc failed for d_kernel (sigma=%.1f): %s\n", 
                    sigmas[s], cudaGetErrorString(err));
            free(h_kernel);
            cudaFree(d_tmp1);
            cudaFree(d_tmp2);
            cudaFree(d_tmp3);
            cudaFree(d_tmp4);
            return 1;
        }

        err = cudaMemcpy(d_kernel, h_kernel, ksize * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: cudaMemcpy failed for kernel (sigma=%.1f): %s\n", 
                    sigmas[s], cudaGetErrorString(err));
            cudaFree(d_kernel);
            free(h_kernel);
            cudaFree(d_tmp1);
            cudaFree(d_tmp2);
            cudaFree(d_tmp3);
            cudaFree(d_tmp4);
            return 1;
        }

        // Horizontal pass
        gaussian_blur_horizontal<<<grid, block>>>(d_rgba, d_tmp1, width, height, d_kernel, radius);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: gaussian_blur_horizontal kernel launch failed (sigma=%.1f): %s\n", 
                    sigmas[s], cudaGetErrorString(err));
            cudaFree(d_kernel);
            free(h_kernel);
            cudaFree(d_tmp1);
            cudaFree(d_tmp2);
            cudaFree(d_tmp3);
            cudaFree(d_tmp4);
            return 1;
        }

        // Vertical pass
        gaussian_blur_vertical<<<grid, block>>>(d_tmp1, blurred[s], width, height, d_kernel, radius);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: gaussian_blur_vertical kernel launch failed (sigma=%.1f): %s\n", 
                    sigmas[s], cudaGetErrorString(err));
            cudaFree(d_kernel);
            free(h_kernel);
            cudaFree(d_tmp1);
            cudaFree(d_tmp2);
            cudaFree(d_tmp3);
            cudaFree(d_tmp4);
            return 1;
        }

        // Synchronize after each blur to ensure completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: cudaDeviceSynchronize failed after blur (sigma=%.1f): %s\n", 
                    sigmas[s], cudaGetErrorString(err));
            cudaFree(d_kernel);
            free(h_kernel);
            cudaFree(d_tmp1);
            cudaFree(d_tmp2);
            cudaFree(d_tmp3);
            cudaFree(d_tmp4);
            return 1;
        }

        cudaFree(d_kernel);
        free(h_kernel);

        printf("Scale %d: sigma: %.1f\n", s + 1, sigmas[s]);
    }

    // Combine
    wavelet_combine<<<grid, block>>>(d_rgba, blurred[0], blurred[1], blurred[2],
                                     width, height, gain_small, gain_medium, gain_large);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: wavelet_combine kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_tmp1);
        cudaFree(d_tmp2);
        cudaFree(d_tmp3);
        cudaFree(d_tmp4);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "wavelet_sharpen_rgbf1_cuda: final cudaDeviceSynchronize failed: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_tmp1);
        cudaFree(d_tmp2);
        cudaFree(d_tmp3);
        cudaFree(d_tmp4);
        return 1;
    }

    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    cudaFree(d_tmp3);
    cudaFree(d_tmp4);

    printf("Sharpened wavelet coefficients:\n");
    printf("Scale 1: gain: %.2f\nScale 2: gain: %.2f\nScale 3: gain: %.2f\n", gain_small, gain_medium, gain_large);
    printf("Wavelet sharpen (GPU): ok\n");

    return 0;
}
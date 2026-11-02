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

#define CLAMPF(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))

/* Rec.709 luminance */
__device__ static inline float luma709(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

/* -------------------------------------------------------------------------
 * Reinhard TMO
 * ------------------------------------------------------------------------- */

/* Pass 1: find maximum luminance via parallel reduction */
__global__ void reinhard_max_luma_kernel(const float *rgbf, int pixel_count,
                                         float *block_max) {
    extern __shared__ float smax[];

    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float local_max = 0.0f;
    const float eps = 1e-6f;

    if (i < pixel_count) {
        const float *px = &rgbf[i * 4];
        if (!(px[0] <= eps && px[1] <= eps && px[2] <= eps)) {
            float Y = luma709(px[0], px[1], px[2]);
            if (Y > local_max)
                local_max = Y;
        }
    }

    smax[tid] = local_max;
    __syncthreads();

    /* Reduction within block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid])
                smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        block_max[blockIdx.x] = smax[0];
}

/* Pass 2: apply extended Reinhard per pixel */
__global__ void reinhard_apply_kernel(float *rgbf, int pixel_count, float lw2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count)
        return;

    const float eps = 1e-6f;
    float *px = &rgbf[i * 4];

    float Y_in = luma709(px[0], px[1], px[2]);
    if (Y_in <= eps)
        return;

    /* Extended Reinhard: Y_out = Y_in * (1 + Y_in/lw²) / (1 + Y_in) */
    float Y_out = Y_in * (1.0f + Y_in / lw2) / (1.0f + Y_in);
    float scale = Y_out / Y_in;

    px[0] = CLAMPF(px[0] * scale);
    px[1] = CLAMPF(px[1] * scale);
    px[2] = CLAMPF(px[2] * scale);
    /* px[3] (alpha) unchanged */
}

extern "C" int reinhard_tonemap_rgbf1_cuda(float *rgbf, int width, int height,
                                           float white_point) {
    if (!rgbf || width <= 0 || height <= 0)
        return 1;

    const int pixel_count = width * height;
    const float eps = 1e-6f;

    float *d_rgbf = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_rgbf, (size_t)pixel_count * 4 * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Reinhard CUDA: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_rgbf, rgbf,
                     (size_t)pixel_count * 4 * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Reinhard CUDA: cudaMemcpy H2D failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_rgbf);
        return 1;
    }

    float lw = white_point;

    if (lw <= eps) {
        /* Auto-detect maximum luminance on GPU */
        const int block_size = 256;
        const int grid_size  = (pixel_count + block_size - 1) / block_size;

        float *d_block_max = NULL;
        err = cudaMalloc(&d_block_max, grid_size * sizeof(float));
        if (err != cudaSuccess) {
            fprintf(stderr, "Reinhard CUDA: cudaMalloc block_max failed: %s\n",
                    cudaGetErrorString(err));
            cudaFree(d_rgbf);
            return 1;
        }

        reinhard_max_luma_kernel<<<grid_size, block_size,
                                   block_size * sizeof(float)>>>(
            d_rgbf, pixel_count, d_block_max);
        cudaDeviceSynchronize();

        float *h_block_max = (float *)malloc(grid_size * sizeof(float));
        if (!h_block_max) {
            cudaFree(d_block_max);
            cudaFree(d_rgbf);
            return 1;
        }
        cudaMemcpy(h_block_max, d_block_max,
                   grid_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_block_max);

        float max_Y = 0.0f;
        for (int b = 0; b < grid_size; b++) {
            if (h_block_max[b] > max_Y)
                max_Y = h_block_max[b];
        }
        free(h_block_max);

        lw = (max_Y > eps) ? max_Y : 1.0f;
        printf("Reinhard TMO (CUDA): white_point=%.4f (auto-detected)\n", lw);
    } else {
        printf("Reinhard TMO (CUDA): white_point=%.4f (user-defined)\n", lw);
    }

    const float lw2 = lw * lw;

    const int block_size = 256;
    const int grid_size  = (pixel_count + block_size - 1) / block_size;
    reinhard_apply_kernel<<<grid_size, block_size>>>(d_rgbf, pixel_count, lw2);
    cudaDeviceSynchronize();

    err = cudaMemcpy(rgbf, d_rgbf,
                     (size_t)pixel_count * 4 * sizeof(float),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_rgbf);

    if (err != cudaSuccess) {
        fprintf(stderr, "Reinhard CUDA: cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Reinhard TMO (CUDA): ok\n");
    return 0;
}

/* -------------------------------------------------------------------------
 * Tone Curve (Shadow Lift + S-Curve)
 * ------------------------------------------------------------------------- */

__global__ void tone_curve_kernel(float *rgbf, int pixel_count,
                                  float shadow_lift, float midtone_contrast) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count)
        return;

    const float eps = 1e-6f;
    float *px = &rgbf[i * 4];

    if (px[0] <= eps && px[1] <= eps && px[2] <= eps)
        return;

    float Y_in = luma709(px[0], px[1], px[2]);
    if (Y_in <= eps)
        return;

    float x = Y_in;

    /* Shadow lift: 4·x·(1−x)²  — bell curve peaking at x≈0.33 */
    float sl = shadow_lift * 4.0f * x * (1.0f - x) * (1.0f - x);

    /* S-curve: 4·x·(1−x)·(1−2x)  — antisymmetric around 0.5 */
    float sc = midtone_contrast * 4.0f * x * (1.0f - x) * (1.0f - 2.0f * x);

    float Y_out = CLAMPF(x + sl + sc);
    float scale = Y_out / Y_in;

    px[0] = CLAMPF(px[0] * scale);
    px[1] = CLAMPF(px[1] * scale);
    px[2] = CLAMPF(px[2] * scale);
    /* px[3] (alpha) unchanged */
}

extern "C" int apply_tone_curve_rgbf1_cuda(float *rgbf, int width, int height,
                                           float shadow_lift,
                                           float midtone_contrast) {
    if (!rgbf || width <= 0 || height <= 0)
        return 1;

    if (shadow_lift < 0.0f)
        shadow_lift = 0.0f;
    if (midtone_contrast < 0.0f)
        midtone_contrast = 0.0f;

    printf("Tone Curve (CUDA): shadow_lift=%.3f  midtone_contrast=%.3f\n",
           shadow_lift, midtone_contrast);

    const int pixel_count = width * height;

    float *d_rgbf = NULL;
    cudaError_t err;

    err = cudaMalloc(&d_rgbf, (size_t)pixel_count * 4 * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Tone Curve CUDA: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_rgbf, rgbf,
                     (size_t)pixel_count * 4 * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Tone Curve CUDA: cudaMemcpy H2D failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_rgbf);
        return 1;
    }

    const int block_size = 256;
    const int grid_size  = (pixel_count + block_size - 1) / block_size;
    tone_curve_kernel<<<grid_size, block_size>>>(d_rgbf, pixel_count,
                                                 shadow_lift, midtone_contrast);
    cudaDeviceSynchronize();

    err = cudaMemcpy(rgbf, d_rgbf,
                     (size_t)pixel_count * 4 * sizeof(float),
                     cudaMemcpyDeviceToHost);
    cudaFree(d_rgbf);

    if (err != cudaSuccess) {
        fprintf(stderr, "Tone Curve CUDA: cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Tone Curve (CUDA): ok\n");
    return 0;
}

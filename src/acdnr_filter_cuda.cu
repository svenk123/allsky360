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
#include <string.h>
#include <stdlib.h>

#define CLAMPF(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))

__device__ float laplace3x3(const float *luminance, int x, int y, int width, int height) {
    const int k[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};
    float result = 0.0f;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                result += luminance[iy * width + ix] * k[ky + 1][kx + 1];
            }
        }
    }
    return fabsf(result);
}

__global__ void rgb_to_yuv_kernel(const float *rgba, float *Y, float *U, float *V, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;
    float r = rgba[i * 4 + 0];
    float g = rgba[i * 4 + 1];
    float b = rgba[i * 4 + 2];
    /* Rec.709 YUV conversion */
    Y[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    U[i] = -0.114572f * r - 0.385428f * g + 0.5f * b;
    V[i] =  0.5f * r - 0.454153f * g - 0.045847f * b;
}

__global__ void yuv_to_rgb_kernel(float *rgba, const float *Y, const float *U, const float *V, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;

    float y = Y[i], u = U[i], v = V[i];
    float r = y + 1.13983f * v;
    float g = y - 0.39465f * u - 0.58060f * v;
    float b = y + 2.03211f * u;

    rgba[i * 4 + 0] = CLAMPF(r);
    rgba[i * 4 + 1] = CLAMPF(g);
    rgba[i * 4 + 2] = CLAMPF(b);
    rgba[i * 4 + 3] = 1.0f;
}

// NOTE: Hier einfache Boxblur-Prototypen (nicht optimal)
__global__ void box_blur_kernel(float *channel, float *temp, int width, int height, int radius, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int ix = x + dx;
            int iy = y + dy;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                sum += temp[iy * width + ix];
                count++;
            }
        }
    }
    int idx = y * width + x;
    float blurred = sum / count;
    channel[idx] = (1.0f - amount) * temp[idx] + amount * blurred;
}

extern "C" int acdnr_filter_rgbf1_cuda(float *rgba, int width, int height,
                                       float stddev_l, float amount_l, int iterations_l, int structure_size_l,
                                       float stddev_c, float amount_c, int iterations_c, int structure_size_c) {
    if (!rgba || width <= 0 || height <= 0) 
        return 1;

    printf("ACDNR Filter (GPU): stddev_l=%f, amount_l=%f, iterations_l=%d, structure_size_l=%d, stddev_c=%f, amount_c=%f, iterations_c=%d, structure_size_c=%d\n", stddev_l, amount_l, iterations_l, structure_size_l, stddev_c, amount_c, iterations_c, structure_size_c);

    int pixel_count = width * height;
    size_t image_bytes = pixel_count * sizeof(float);

    float *Y, *U, *V;
    float *Y_temp, *U_temp, *V_temp;
    
    cudaError_t err;
    err = cudaMalloc(&Y, image_bytes);
    if (err != cudaSuccess) { 
        printf("CUDA malloc Y failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }
    err = cudaMalloc(&U, image_bytes);
    if (err != cudaSuccess) { 
        printf("CUDA malloc U failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }
    err = cudaMalloc(&V, image_bytes);
    if (err != cudaSuccess) { 
        printf("CUDA malloc V failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }
    err = cudaMalloc(&Y_temp, image_bytes);
    if (err != cudaSuccess) { 
        printf("CUDA malloc Y_temp failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }
    err = cudaMalloc(&U_temp, image_bytes);
    if (err != cudaSuccess) { 
        printf("CUDA malloc U_temp failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }
    err = cudaMalloc(&V_temp, image_bytes);
    if (err != cudaSuccess) { 
        printf("CUDA malloc V_temp failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }

    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;

    rgb_to_yuv_kernel<<<blocks, threads>>>(rgba, Y, U, V, pixel_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) { 
        printf("rgb_to_yuv_kernel failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    for (int i = 0; i < iterations_l; i++) {
        err = cudaMemcpy(Y_temp, Y, image_bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) { 
            printf("cudaMemcpy Y_temp failed: %s\n", cudaGetErrorString(err)); 
            return 1; 
        }
        box_blur_kernel<<<gridSize, blockSize>>>(Y, Y_temp, width, height, structure_size_l, amount_l);
        err = cudaGetLastError();
        if (err != cudaSuccess) { 
            printf("box_blur_kernel Y failed: %s\n", cudaGetErrorString(err)); 
            return 1; 
        }
    }

    for (int i = 0; i < iterations_c; i++) {
        err = cudaMemcpy(U_temp, U, image_bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) { 
            printf("cudaMemcpy U_temp failed: %s\n", cudaGetErrorString(err)); 
            return 1; 
        }
        err = cudaMemcpy(V_temp, V, image_bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) { 
            printf("cudaMemcpy V_temp failed: %s\n", cudaGetErrorString(err)); 
            return 1; 
        }
        box_blur_kernel<<<gridSize, blockSize>>>(U, U_temp, width, height, structure_size_c, amount_c);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("box_blur_kernel U failed: %s\n", cudaGetErrorString(err)); 
            return 1; 
        }
        box_blur_kernel<<<gridSize, blockSize>>>(V, V_temp, width, height, structure_size_c, amount_c);
        err = cudaGetLastError();
        if (err != cudaSuccess) { 
            printf("box_blur_kernel V failed: %s\n", cudaGetErrorString(err)); 
            return 1; 
        }
    }

    yuv_to_rgb_kernel<<<blocks, threads>>>(rgba, Y, U, V, pixel_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) { 
        printf("yuv_to_rgb_kernel failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }

    // Synchronize to ensure all kernels have completed
    cudaDeviceSynchronize();

    cudaFree(Y);
    cudaFree(U);
    cudaFree(V);
    cudaFree(Y_temp); 
    cudaFree(U_temp);
    cudaFree(V_temp);

    printf("ACDNR Filter (GPU): ok\n");

    return 0;
}

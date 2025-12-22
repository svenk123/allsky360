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
#include <string.h>
#include <math.h>

#include "color_calibration_cuda.h"

#define CLAMPF(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))

// RGB to HSV (0.0..1.0)
__device__ static void rgb_to_hsv_f(float r, float g, float b, float *h, float *s, float *v) {
    float max = fmaxf(r, fmaxf(g, b));
    float min = fminf(r, fminf(g, b));
    *v = max;

    float d = max - min;
    *s = (max == 0.0f) ? 0.0f : d / max;

    if (max == min) {
        *h = 0.0f;
    } else {
        if (max == r) {
            *h = (g - b) / d + (g < b ? 6.0f : 0.0f);
        } else if (max == g) {
            *h = (b - r) / d + 2.0f;
        } else {
            *h = (r - g) / d + 4.0f;
        }

        *h /= 6.0f;
    }
}

// HSV to RGB
__device__ void hsv_to_rgb_f(float h, float s, float v, float *r, float *g, float *b) {
    int i = (int)(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (i % 6) {
        case 0: *r = v; *g = t; *b = p; break;
        case 1: *r = q; *g = v; *b = p; break;
        case 2: *r = p; *g = v; *b = t; break;
        case 3: *r = p; *g = q; *b = v; break;
        case 4: *r = t; *g = p; *b = v; break;
        case 5: *r = v; *g = p; *b = q; break;
    }
}

// CUDA Kernel for saturation (0.0..1.0)
__global__ void adjust_saturation_kernel(float *rgba, int pixel_count, float saturation_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) 
        return;

    int idx = i * 4;
    float r = rgba[idx + 0];  // already in 0..1
    float g = rgba[idx + 1];
    float b = rgba[idx + 2];

    float h, s, v;
    rgb_to_hsv_f(r, g, b, &h, &s, &v);

    s *= saturation_factor;
    if (s > 1.0f) 
        s = 1.0f;

    hsv_to_rgb_f(h, s, v, &r, &g, &b);

    rgba[idx + 0] = CLAMPF(r);
    rgba[idx + 1] = CLAMPF(g);
    rgba[idx + 2] = CLAMPF(b);
    rgba[idx + 3] = 1.0f;  // Alpha in 0..1
}

extern "C" int adjust_saturation_rgbf1_cuda(float *rgba, int width, int height, float saturation_factor) {
    if (!rgba || width <= 0 || height <= 0 || saturation_factor <= 0.0f)
        return 1;

    int pixel_count = width * height;

    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    adjust_saturation_kernel<<<blocks, threads>>>(rgba, pixel_count, saturation_factor);
    cudaDeviceSynchronize();

    printf("Saturation (GPU): ok. Factor: %.2f\n", saturation_factor);

    return 0;
}

#define LUT_SIZE 4096  // 4096 reicht perfekt für Gamma-Korrektur 0..1

__device__ float gamma_lut[LUT_SIZE];  // LUT in GPU memory

__global__ void gamma_lut_kernel(float *rgba, int pixel_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixel_count) return;

    int idx = i * CHANNELS;

    for (int c = 0; c < 3; c++) {  // R, G, B
        float x = rgba[idx + c];
        x = CLAMPF(x);

        float lut_pos = x * (LUT_SIZE - 1);
        int idx0 = (int)floorf(lut_pos);
        int idx1 = min(idx0 + 1, LUT_SIZE - 1);
        float frac = lut_pos - (float)idx0;

        float lut_val0 = gamma_lut[idx0];
        float lut_val1 = gamma_lut[idx1];

        // Linear interpolation
        rgba[idx + c] = lut_val0 * (1.0f - frac) + lut_val1 * frac;
    }

    rgba[idx + 3] = 1.0f;  // Alpha bleibt 1.0
}

extern "C" int apply_gamma_correction_rgbf1_cuda(float *rgba, int width, int height, float gamma) {
    if (!rgba || width <= 0 || height <= 0 || gamma <= 0.0f) {
        return 1;
    }

    // LUT (CPU)
    static float lut_host[LUT_SIZE];
    static float last_gamma = -1.0f;

    if (gamma != last_gamma) {
        for (int i = 0; i < LUT_SIZE; i++) {
            float normalized = (float)i / (float)(LUT_SIZE - 1);
            lut_host[i] = powf(normalized, gamma);  // stays in 0..1
        }
        cudaMemcpyToSymbol(gamma_lut, lut_host, sizeof(lut_host));
        last_gamma = gamma;
        printf(" Gamma-LUT loaded (Gamma=%.3f)\n", gamma);
    }

    // Copy to GPU memory
    int pixel_count = width * height;

    // Start GPU kernel
    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    gamma_lut_kernel<<<blocks, threads>>>(rgba, pixel_count);
    cudaDeviceSynchronize();

    printf("Gamma correction (GPU + LUT): %.3f\n", gamma);
    return 0;
}

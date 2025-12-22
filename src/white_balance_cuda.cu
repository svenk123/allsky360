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
#include "white_balance.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CLAMPF(x) ((x) < 0.0f ? 0.0f : ((x) > 1.0f ? 1.0f : (x)))
#define CHANNELS 4 // RGBA

// CUDA kernel for computing statistics in a subframe region
__global__ void compute_stats_kernel(const float *rgba, int width, int height,
                                     int x0, int y0, int x1, int y1,
                                     double *sum_r, double *sum_g,
                                     double *sum_b, double *sum_r2,
                                     double *sum_g2, double *sum_b2,
                                     int *count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = (x1 - x0) * (y1 - y0);

  if (i >= total_pixels)
    return;

  // Convert linear index to 2D coordinates within the subframe
  int local_x = i % (x1 - x0);
  int local_y = i / (x1 - x0);
  int x = x0 + local_x;
  int y = y0 + local_y;

  if (x >= x1 || y >= y1)
    return;

  int idx = (y * width + x) * CHANNELS;
  float r = rgba[idx + 0];
  float g = rgba[idx + 1];
  float b = rgba[idx + 2];

  // Only process non-clipped pixels
  if (r < 1.0f && g < 1.0f && b < 1.0f) {
    atomicAdd(sum_r, (double)r);
    atomicAdd(sum_g, (double)g);
    atomicAdd(sum_b, (double)b);
    atomicAdd(sum_r2, (double)(r * r));
    atomicAdd(sum_g2, (double)(g * g));
    atomicAdd(sum_b2, (double)(b * b));
    atomicAdd(count, 1);
  }
}

// CUDA kernel for applying white balance and background neutralization
__global__ void apply_white_balance_kernel(float *rgba, int pixel_count,
                                           float scale_r, float scale_g,
                                           float scale_b, float max_val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixel_count)
    return;

  int idx = i * CHANNELS;
  float r = rgba[idx + 0];
  float g = rgba[idx + 1];
  float b = rgba[idx + 2];

  // Background neutralization
  r = fminf(r, max_val);
  g = fminf(g, max_val);
  b = fminf(b, max_val);

  // White balance
  r *= scale_r;
  g *= scale_g;
  b *= scale_b;

  // Clamping
  rgba[idx + 0] = CLAMPF(r);
  rgba[idx + 1] = CLAMPF(g);
  rgba[idx + 2] = CLAMPF(b);
  // Alpha remains untouched
}

extern "C" int auto_color_calibration_rgbf1_cuda(float *rgba, int width,
                                                 int height,
                                                 float subframe_percent) {
  if (!rgba || width <= 0 || height <= 0 || subframe_percent <= 0.0f ||
      subframe_percent > 100.0f)
    return 1;

  printf("Auto Color Calibration (GPU): subframe %.1f%%\n", subframe_percent);

  // Calculate subframe boundaries
  int x0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * width);
  int y0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * height);
  int x1 = width - x0;
  int y1 = height - y0;

  int subframe_pixels = (x1 - x0) * (y1 - y0);
  if (subframe_pixels <= 0)
    return 1;

  // Allocate device memory for statistics
  double *d_sum_r, *d_sum_g, *d_sum_b;
  double *d_sum_r2, *d_sum_g2, *d_sum_b2;
  int *d_count;

  cudaError_t err;

  err = cudaMalloc(&d_sum_r, sizeof(double));
  if (err != cudaSuccess) {
    printf("CUDA malloc d_sum_r failed: %s\n", cudaGetErrorString(err));

    return 1;
  }

  err = cudaMalloc(&d_sum_g, sizeof(double));
  if (err != cudaSuccess) {
    printf("CUDA malloc d_sum_g failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);

    return 1;
  }

  err = cudaMalloc(&d_sum_b, sizeof(double));
  if (err != cudaSuccess) {
    printf("CUDA malloc d_sum_b failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);

    return 1;
  }

  err = cudaMalloc(&d_sum_r2, sizeof(double));
  if (err != cudaSuccess) {
    printf("CUDA malloc d_sum_r2 failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);

    return 1;
  }

  err = cudaMalloc(&d_sum_g2, sizeof(double));
  if (err != cudaSuccess) {
    printf("CUDA malloc d_sum_g2 failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);

    return 1;
  }

  err = cudaMalloc(&d_sum_b2, sizeof(double));
  if (err != cudaSuccess) {
    printf("CUDA malloc d_sum_b2 failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);

    return 1;
  }

  err = cudaMalloc(&d_count, sizeof(int));
  if (err != cudaSuccess) {
    printf("CUDA malloc d_count failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);

    return 1;
  }

  // Initialize device memory
  cudaMemset(d_sum_r, 0, sizeof(double));
  cudaMemset(d_sum_g, 0, sizeof(double));
  cudaMemset(d_sum_b, 0, sizeof(double));
  cudaMemset(d_sum_r2, 0, sizeof(double));
  cudaMemset(d_sum_g2, 0, sizeof(double));
  cudaMemset(d_sum_b2, 0, sizeof(double));
  cudaMemset(d_count, 0, sizeof(int));

  // Launch kernel to compute statistics
  int threads = 256;
  int blocks = (subframe_pixels + threads - 1) / threads;

  compute_stats_kernel<<<blocks, threads>>>(rgba, width, height, x0, y0, x1, y1,
                                            d_sum_r, d_sum_g, d_sum_b, d_sum_r2,
                                            d_sum_g2, d_sum_b2, d_count);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("compute_stats_kernel failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  // Copy results back to host
  double sum_r, sum_g, sum_b;
  double sum_r2, sum_g2, sum_b2;
  int count;

  err = cudaMemcpy(&sum_r, d_sum_r, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy sum_r failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  err = cudaMemcpy(&sum_g, d_sum_g, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy sum_g failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  err = cudaMemcpy(&sum_b, d_sum_b, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy sum_b failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  err = cudaMemcpy(&sum_r2, d_sum_r2, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy sum_r2 failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  err = cudaMemcpy(&sum_g2, d_sum_g2, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy sum_g2 failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  err = cudaMemcpy(&sum_b2, d_sum_b2, sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy sum_b2 failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  err = cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cudaMemcpy count failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  if (count == 0) {
    printf("Warning: No valid pixels found in subframe\n");
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);

    return 1;
  }

  // Calculate statistics
  double mean_r = sum_r / count;
  double mean_g = sum_g / count;
  double mean_b = sum_b / count;

  double stddev_r = sqrt(sum_r2 / count - mean_r * mean_r);
  double stddev_g = sqrt(sum_g2 / count - mean_g * mean_g);
  double stddev_b = sqrt(sum_b2 / count - mean_b * mean_b);

  double target = fmax(fmax(mean_r, mean_g), mean_b);

  float scale_r = (float)(target / mean_r);
  float scale_g = (float)(target / mean_g);
  float scale_b = (float)(target / mean_b);

  float nSigma = (float)(3.0 * fmax(fmax(stddev_r, stddev_g), stddev_b));
  float max_val = (float)(target + nSigma);

  printf("Auto Color Calibration (GPU): mean_r=%.6f mean_g=%.6f mean_b=%.6f\n",
         mean_r, mean_g, mean_b);
  printf(
      "Auto Color Calibration (GPU): scale_r=%.6f scale_g=%.6f scale_b=%.6f\n",
      scale_r, scale_g, scale_b);

  // Apply white balance to entire image
  int pixel_count = width * height;
  threads = 256;
  blocks = (pixel_count + threads - 1) / threads;

  apply_white_balance_kernel<<<blocks, threads>>>(rgba, pixel_count, scale_r,
                                                  scale_g, scale_b, max_val);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("apply_white_balance_kernel failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);
    return 1;
  }

  // Synchronize to ensure all kernels have completed
  cudaDeviceSynchronize();

  // Cleanup
  cudaFree(d_sum_r);
  cudaFree(d_sum_g);
  cudaFree(d_sum_b);
  cudaFree(d_sum_r2);
  cudaFree(d_sum_g2);
  cudaFree(d_sum_b2);
  cudaFree(d_count);

  printf("Auto Color Calibration (GPU): ok\n");

  return 0;
}

// Compute Rec.709 luminance
__device__ __forceinline__ float d_luminance709(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Simple saturation metric (HSL-like)
__device__ __forceinline__ float d_saturation_rgb(float r, float g, float b) {
  float maxc = fmaxf(r, fmaxf(g, b));
  float minc = fminf(r, fminf(g, b));
  if (maxc <= 1e-6f)
    return 0.0f;

  return (maxc - minc) / maxc;
}

// Global accumulators: sums and counters
typedef struct {
  double sumY;
  double sumY2;
  unsigned long long bright_pixels;
  unsigned long long aurora_pixels;
  unsigned long long cloudy_pixels;
} SceneAccum;

// Kernel to accumulate basic statistics over whole image.
// d_rgba: device pointer to RGBA float data, 0..1, size = width * height * 4
__global__ void scene_accum_kernel(const float *__restrict__ d_rgba,
                                   int totalPixels, SceneAccum *accum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= totalPixels)
    return;

  int idx = i * 4;
  float r = d_rgba[idx + 0];
  float g = d_rgba[idx + 1];
  float b = d_rgba[idx + 2];

  float Y = d_luminance709(r, g, b);
  float sat = d_saturation_rgb(r, g, b);

  double dY = (double)Y;

  // Use atomic operations to update shared accumulators
  atomicAdd(&accum->sumY, dY);
  atomicAdd(&accum->sumY2, dY * dY);

  if (Y > 0.85f) {
    atomicAdd(&accum->bright_pixels, 1ULL);
  }

  if (sat > 0.40f && Y < 0.85f) {
    atomicAdd(&accum->aurora_pixels, 1ULL);
  }

  if (Y > 0.15f && Y < 0.65f) {
    atomicAdd(&accum->cloudy_pixels, 1ULL);
  }
}

// Host function: runs kernel, reads back statistics, classifies scene.
// d_rgba must point to device memory (width*height*4 floats).
extern "C" SceneType detect_scene_rgbf1_cuda(float *d_rgba, int width,
                                             int height, int *errcode) {
  if (errcode)
    *errcode = 0;

  if (!d_rgba || width <= 0 || height <= 0) {
    fprintf(stderr, "Scene Detection (GPU): invalid image or dimensions\n");
    if (errcode)
      *errcode = 1;

    return SCENE_NIGHT_CLEAR;
  }

  int totalPixels = width * height;

  // Allocate accumulator struct on device, zero-init
  SceneAccum h_accum = {0};
  SceneAccum *d_accum = nullptr;
  cudaError_t cuerr;

  cuerr = cudaMalloc((void **)&d_accum, sizeof(SceneAccum));
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Scene Detection (GPU): cudaMalloc failed: %s\n",
            cudaGetErrorString(cuerr));
    if (errcode)
      *errcode = 2;

    return SCENE_NIGHT_CLEAR;
  }

  cuerr = cudaMemset(d_accum, 0, sizeof(SceneAccum));
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Scene Detection (GPU): cudaMemset failed: %s\n",
            cudaGetErrorString(cuerr));
    cudaFree(d_accum);
    if (errcode)
      *errcode = 3;

    return SCENE_NIGHT_CLEAR;
  }

  // Launch kernel
  int blockSize = 256;
  int gridSize = (totalPixels + blockSize - 1) / blockSize;
  scene_accum_kernel<<<gridSize, blockSize>>>(d_rgba, totalPixels, d_accum);
  cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Scene Detection (GPU): kernel launch failed: %s\n",
            cudaGetErrorString(cuerr));
    cudaFree(d_accum);
    if (errcode)
      *errcode = 4;

    return SCENE_NIGHT_CLEAR;
  }

  // Synchronize
  cuerr = cudaDeviceSynchronize();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Scene Detection (GPU): cudaDeviceSynchronize failed: %s\n",
            cudaGetErrorString(cuerr));
    cudaFree(d_accum);
    if (errcode)
      *errcode = 5;

    return SCENE_NIGHT_CLEAR;
  }

  // Copy accumulators back to host
  cuerr =
      cudaMemcpy(&h_accum, d_accum, sizeof(SceneAccum), cudaMemcpyDeviceToHost);
  cudaFree(d_accum);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, "Scene Detection (GPU): cudaMemcpy failed: %s\n",
            cudaGetErrorString(cuerr));
    if (errcode)
      *errcode = 6;

    return SCENE_NIGHT_CLEAR;
  }

  // Compute stats on host
  double sumY = h_accum.sumY;
  double sumY2 = h_accum.sumY2;
  double meanY = sumY / (double)totalPixels;
  double varY = (sumY2 / (double)totalPixels) - meanY * meanY;
  if (varY < 0.0)
    varY = 0.0;
  double stdY = sqrt(varY);

  double bright_pct = (double)h_accum.bright_pixels / (double)totalPixels;
  double aurora_pct = (double)h_accum.aurora_pixels / (double)totalPixels;
  double cloudy_pct = (double)h_accum.cloudy_pixels / (double)totalPixels;

  printf("Scene Detection (GPU): meanY=%.4f stdY=%.4f bright=%.6f aurora=%.6f "
         "cloudy=%.6f\n",
         meanY, stdY, bright_pct, aurora_pct, cloudy_pct);

  // Classification logic identical/similar zur CPU-Version
  if (meanY > 0.25) {
    // DAYTIME
    if (cloudy_pct > 0.25)
      return SCENE_DAY_CLOUDY;
    else
      return SCENE_DAY_CLEAR;
  }

  if (meanY < 0.02) {
    // NIGHT
    if (bright_pct > 0.0002) // ~0.02% bright pixels = moon/sun
      return SCENE_NIGHT_MOON;
    if (aurora_pct > 0.01) // >1% strong saturated pixels
      return SCENE_NIGHT_AURORA;
    if (cloudy_pct > 0.20)
      return SCENE_NIGHT_CLOUDY;

    return SCENE_NIGHT_CLEAR;
  }

  // TWILIGHT (between night and day)
  if (stdY > 0.08 || fabs(meanY - 0.10) < 0.05)
    return SCENE_TWILIGHT;

  return SCENE_TWILIGHT;
}

// Simple clamp
__device__ __forceinline__ float d_clampf1(float v) {
  if (v < 0.0f)
    return 0.0f;
  if (v > 1.0f)
    return 1.0f;

  return v;
}

__device__ __forceinline__ float d_luma709(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__device__ __forceinline__ float d_sat_rgb(float r, float g, float b) {
  float maxc = fmaxf(r, fmaxf(g, b));
  float minc = fminf(r, fminf(g, b));

  if (maxc <= 1e-6f)
    return 0.0f;

  return (maxc - minc) / maxc;
}

// Accumulator for ambience stats
typedef struct {
  double sumR;
  double sumG;
  double sumB;
  double sumY;
  double sumY2;
  unsigned long long count;
} AmbienceAccum;

// Kernel 1: first-pass statistics over subframe with luma/sat filtering.
__global__ void ambience_first_pass_kernel(const float *__restrict__ d_rgba,
                                           int width, int height, int x0,
                                           int y0, int x1, int y1,
                                           float luma_clip_high,
                                           float sat_clip_high,
                                           AmbienceAccum *acc) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + x0;
  int y = blockIdx.y * blockDim.y + threadIdx.y + y0;

  if (x >= x1 || y >= y1)
    return;

  int idx_pixel = (y * width + x) * 4;
  float r = d_rgba[idx_pixel + 0];
  float g = d_rgba[idx_pixel + 1];
  float b = d_rgba[idx_pixel + 2];

  // Basic range check
  if (r < 0.0f || r > 1.0f || g < 0.0f || g > 1.0f || b < 0.0f || b > 1.0f)
    return;

  float Y = d_luma709(r, g, b);
  float sat = d_sat_rgb(r, g, b);

  if (Y > luma_clip_high)
    return;
  if (sat > sat_clip_high)
    return;

  double dR = (double)r;
  double dG = (double)g;
  double dB = (double)b;
  double dY = (double)Y;

  atomicAdd(&acc->sumR, dR);
  atomicAdd(&acc->sumG, dG);
  atomicAdd(&acc->sumB, dB);
  atomicAdd(&acc->sumY, dY);
  atomicAdd(&acc->sumY2, dY * dY);
  atomicAdd(&acc->count, 1ULL);
}

// Kernel 2: second-pass band-limited stats (Y within [bandLow, bandHigh]).
__global__ void ambience_band_pass_kernel(const float *__restrict__ d_rgba,
                                          int width, int height, int x0, int y0,
                                          int x1, int y1, float luma_clip_high,
                                          float sat_clip_high, float bandLow,
                                          float bandHigh, AmbienceAccum *acc) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + x0;
  int y = blockIdx.y * blockDim.y + threadIdx.y + y0;

  if (x >= x1 || y >= y1)
    return;

  int idx_pixel = (y * width + x) * 4;
  float r = d_rgba[idx_pixel + 0];
  float g = d_rgba[idx_pixel + 1];
  float b = d_rgba[idx_pixel + 2];

  if (r < 0.0f || r > 1.0f || g < 0.0f || g > 1.0f || b < 0.0f || b > 1.0f)
    return;

  float Y = d_luma709(r, g, b);
  float sat = d_sat_rgb(r, g, b);

  if (Y > luma_clip_high)
    return;
  if (sat > sat_clip_high)
    return;

  if (Y < bandLow || Y > bandHigh)
    return;

  double dR = (double)r;
  double dG = (double)g;
  double dB = (double)b;
  double dY = (double)Y;

  atomicAdd(&acc->sumR, dR);
  atomicAdd(&acc->sumG, dG);
  atomicAdd(&acc->sumB, dB);
  atomicAdd(&acc->sumY, dY);
  atomicAdd(&acc->sumY2, dY * dY);
  atomicAdd(&acc->count, 1ULL);
}

// Kernel 3: apply global RGB scales to entire image.
__global__ void ambience_apply_scales_kernel(float *d_rgba, int totalPixels,
                                             float scaleR, float scaleG,
                                             float scaleB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= totalPixels)
    return;

  int idx = i * 4;
  float r = d_rgba[idx + 0];
  float g = d_rgba[idx + 1];
  float b = d_rgba[idx + 2];

  r *= scaleR;
  g *= scaleG;
  b *= scaleB;

  d_rgba[idx + 0] = d_clampf1(r);
  d_rgba[idx + 1] = d_clampf1(g);
  d_rgba[idx + 2] = d_clampf1(b);
  // alpha untouched
}

extern "C" int
ambience_color_calibration_rgbf1_cuda(float *d_rgba, int width, int height,
                                      float subframe_percent,
                                      float luma_clip_high, float sat_clip_high,
                                      float sigma_factor, float mix_factor) {
  if (!d_rgba || width <= 0 || height <= 0) {
    fprintf(stderr, "Ambience-based color calibration (GPU): invalid image or "
                    "dimensions\n");
    return 1;
  }

  if (subframe_percent <= 0.0f || subframe_percent > 100.0f) {
    fprintf(stderr,
            "Ambience-based color calibration (GPU): invalid subframe_percent "
            "%.2f\n",
            subframe_percent);
    return 2;
  }

  if (mix_factor <= 0.0f) {
    fprintf(stderr, "Ambience-based color calibration (GPU): mix_factor <= "
                    "0.0, skipping calibration\n");
    return 0;
  }

  // Compute central subframe region on host
  int x0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * (float)width);
  int y0 = (int)((1.0f - subframe_percent / 100.0f) * 0.5f * (float)height);
  int x1 = width - x0;
  int y1 = height - y0;

  if (x0 < 0)
    x0 = 0;
  if (y0 < 0)
    y0 = 0;
  if (x1 > width)
    x1 = width;
  if (y1 > height)
    y1 = height;

  printf(" subframe %.1f%% -> x=[%d,%d), y=[%d,%d)\n", subframe_percent, x0, x1,
         y0, y1);

  cudaError_t cuerr;

  // Allocate accum struct on device and zero it
  AmbienceAccum h_acc = {0};
  AmbienceAccum *d_acc = nullptr;

  cuerr = cudaMalloc((void **)&d_acc, sizeof(AmbienceAccum));
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " cudaMalloc failed: %s\n", cudaGetErrorString(cuerr));
    return 3;
  }

  // Launch first-pass kernel (2D grid over subframe)
  dim3 block2D(16, 16);
  dim3 grid2D((x1 - x0 + block2D.x - 1) / block2D.x,
              (y1 - y0 + block2D.y - 1) / block2D.y);

  ambience_first_pass_kernel<<<grid2D, block2D>>>(d_rgba, width, height, x0, y0,
                                                  x1, y1, luma_clip_high,
                                                  sat_clip_high, d_acc);
  cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " first_pass kernel failed: %s\n",
            cudaGetErrorString(cuerr));
    cudaFree(d_acc);

    return 4;
  }

  cuerr = cudaDeviceSynchronize();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " first_pass sync failed: %s\n", cudaGetErrorString(cuerr));
    cudaFree(d_acc);

    return 5;
  }

  // Copy first-pass accum back
  cuerr =
      cudaMemcpy(&h_acc, d_acc, sizeof(AmbienceAccum), cudaMemcpyDeviceToHost);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " memcpy first_pass failed: %s\n",
            cudaGetErrorString(cuerr));
    cudaFree(d_acc);

    return 6;
  }

  if (h_acc.count < 100) {
    fprintf(stderr, " not enough valid samples in subframe (count=%llu)\n",
            (unsigned long long)h_acc.count);
    cudaFree(d_acc);

    return 7;
  }

  double meanR = h_acc.sumR / (double)h_acc.count;
  double meanG = h_acc.sumG / (double)h_acc.count;
  double meanB = h_acc.sumB / (double)h_acc.count;
  double meanY = h_acc.sumY / (double)h_acc.count;
  double varY = (h_acc.sumY2 / (double)h_acc.count) - meanY * meanY;
  if (varY < 0.0)
    varY = 0.0;
  double stdY = sqrt(varY);

  printf(" first-pass samples=%llu  meanRGB=(%.4f, %.4f, %.4f) meanY=%.4f "
         "stdY=%.4f\n",
         (unsigned long long)h_acc.count, meanR, meanG, meanB, meanY, stdY);

  // Second-pass band-limited stats
  double bandLow = meanY - sigma_factor * stdY;
  double bandHigh = meanY + sigma_factor * stdY;
  if (stdY <= 0.0) {
    bandLow = -1e9;
    bandHigh = 1e9;
  }

  // Reset device accum for second pass
  cuerr = cudaMemset(d_acc, 0, sizeof(AmbienceAccum));
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " cudaMemset second_pass failed: %s\n",
            cudaGetErrorString(cuerr));
    cudaFree(d_acc);

    return 8;
  }

  ambience_band_pass_kernel<<<grid2D, block2D>>>(
      d_rgba, width, height, x0, y0, x1, y1, luma_clip_high, sat_clip_high,
      (float)bandLow, (float)bandHigh, d_acc);
  cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " band_pass kernel failed: %s\n",
            cudaGetErrorString(cuerr));
    cudaFree(d_acc);

    return 9;
  }

  cuerr = cudaDeviceSynchronize();
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " band_pass sync failed: %s\n", cudaGetErrorString(cuerr));
    cudaFree(d_acc);

    return 10;
  }

  // Read back band-pass accum
  AmbienceAccum h_acc_band;
  cuerr = cudaMemcpy(&h_acc_band, d_acc, sizeof(AmbienceAccum),
                     cudaMemcpyDeviceToHost);
  cudaFree(d_acc);
  if (cuerr != cudaSuccess) {
    fprintf(stderr, " memcpy band_pass failed: %s\n",
            cudaGetErrorString(cuerr));
    return 12;
  }

  if (h_acc_band.count < 50) {
    // Fallback to first-pass means
    fprintf(stderr,
            " band filtering left few samples (count=%llu), fallback to "
            "first-pass\n",
            (unsigned long long)h_acc_band.count);
    h_acc_band = h_acc;
  }

  double ambienceR = h_acc_band.sumR / (double)h_acc_band.count;
  double ambienceG = h_acc_band.sumG / (double)h_acc_band.count;
  double ambienceB = h_acc_band.sumB / (double)h_acc_band.count;
  double ambienceY =
      0.2126 * ambienceR + 0.7152 * ambienceG + 0.0722 * ambienceB;

  printf(" ambienceRGB=(%.4f, %.4f, %.4f) ambienceY=%.4f samples=%llu\n",
         ambienceR, ambienceG, ambienceB, ambienceY,
         (unsigned long long)h_acc_band.count);

  if (ambienceR <= 0.0 || ambienceG <= 0.0 || ambienceB <= 0.0 ||
      ambienceY <= 0.0) {
    fprintf(stderr, " invalid ambience values, skipping\n");
    return 13;
  }

  double fullScaleR = ambienceY / ambienceR;
  double fullScaleG = ambienceY / ambienceG;
  double fullScaleB = ambienceY / ambienceB;

  const double SCALE_MIN = 0.25;
  const double SCALE_MAX = 4.0;
  if (fullScaleR < SCALE_MIN)
    fullScaleR = SCALE_MIN;
  if (fullScaleR > SCALE_MAX)
    fullScaleR = SCALE_MAX;
  if (fullScaleG < SCALE_MIN)
    fullScaleG = SCALE_MIN;
  if (fullScaleG > SCALE_MAX)
    fullScaleG = SCALE_MAX;
  if (fullScaleB < SCALE_MIN)
    fullScaleB = SCALE_MIN;
  if (fullScaleB > SCALE_MAX)
    fullScaleB = SCALE_MAX;

  float scaleR = (float)(1.0 + (fullScaleR - 1.0) * (double)mix_factor);
  float scaleG = (float)(1.0 + (fullScaleG - 1.0) * (double)mix_factor);
  float scaleB = (float)(1.0 + (fullScaleB - 1.0) * (double)mix_factor);

  printf(" fullScale=(%.4f, %.4f, %.4f) mix_factor=%.3f finalScale=(%.4f, "
         "%.4f, %.4f)\n",
         fullScaleR, fullScaleG, fullScaleB, mix_factor, scaleR, scaleG,
         scaleB);

  if (fabsf(scaleR - 1.0f) < 1e-3f && fabsf(scaleG - 1.0f) < 1e-3f &&
      fabsf(scaleB - 1.0f) < 1e-3f) {
    printf(" scales ~1.0, skipping apply\n");

    return 0;
  }

  int totalPixels = width * height;
  int blockSize = 256;
  int gridSize = (totalPixels + blockSize - 1) / blockSize;

  ambience_apply_scales_kernel<<<gridSize, blockSize>>>(d_rgba, totalPixels,
                                                        scaleR, scaleG, scaleB);
  cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess) {
    fprintf(stderr,
            "Ambience-based color calibration (GPU): apply_scales kernel "
            "failed: %s\n",
            cudaGetErrorString(cuerr));

    return 14;
  }

  cuerr = cudaDeviceSynchronize();
  if (cuerr != cudaSuccess) {
    fprintf(stderr,
            "Ambience-based color calibration (GPU): apply_scales sync failed: "
            "%s\n",
            cudaGetErrorString(cuerr));

    return 15;
  }

  printf("Ambience-based color calibration (GPU): ok\n");
  return 0;

}

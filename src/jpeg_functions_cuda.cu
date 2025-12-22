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
#include "allsky.h"
#include "jpeg_functions.h"
#include <cuda_runtime.h>
#include <float.h>
#ifdef USE_GPUJPEG
#include <libgpujpeg/gpujpeg.h>
#endif
#include <jpeglib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHANNELS 4

// ------------------------------------------------------------
// Helper: Save RGB8 buffer (host) as JPEG using libjpeg
// ------------------------------------------------------------
static int save_jpeg_libjpeg_rgb8(const unsigned char *rgb8, int width,
                                  int height, int quality,
                                  const char *filename) {
  if (!rgb8 || !filename || width <= 0 || height <= 0)
    return 1;

  if (quality < 1)
    quality = 1;
  if (quality > 100)
    quality = 100;

  FILE *outfile = fopen(filename, "wb");
  if (!outfile)
    return 2;

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  jpeg_start_compress(&cinfo, TRUE);

  int row_stride = width * 3;
  JSAMPROW row_pointer[1];

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = (unsigned char *)&rgb8[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);

  return 0;
}

// Float[0..65535] RGBA -> uint8 RGB
__global__ void rgba16f_to_rgb8_kernel(const float *src, unsigned char *dst,
                                       int width, int height, int new_w,
                                       int new_h, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= new_w || y >= new_h)
    return;

  int sx = min((int)(x / scale), width - 1);
  int sy = min((int)(y / scale), height - 1);

  int src_idx = (sy * width + sx) * 4;
  int dst_idx = (y * new_w + x) * 3;

  // 0..65535f --> 0..255u
  const float k = 255.0f / 65535.0f;
  float r = src[src_idx + 0] * k;
  float g = src[src_idx + 1] * k;
  float b = src[src_idx + 2] * k;

  dst[dst_idx + 0] = (unsigned char)fminf(fmaxf(r, 0.0f), 255.0f);
  dst[dst_idx + 1] = (unsigned char)fminf(fmaxf(g, 0.0f), 255.0f);
  dst[dst_idx + 2] = (unsigned char)fminf(fmaxf(b, 0.0f), 255.0f);
}

extern "C" int save_jpeg_rgbf16_cuda(const float *rgba, int width, int height,
                                     int compression_ratio, float scale,
                                     const char *filename) {
  if (!rgba || width <= 0 || height <= 0 || scale <= 0.0f ||
      compression_ratio <= 0 || compression_ratio > 100) {
    fprintf(stderr,
            "Invalid parameters: rgba=%p, width=%d, height=%d, scale=%f, "
            "compression_ratio=%d, filename=%s\n",
            rgba, width, height, scale, compression_ratio, filename);
    return 1;
  }

  int new_w = (int)(width * scale);
  int new_h = (int)(height * scale);
  if (new_w < 1 || new_h < 1)
    return 2;

  
  unsigned char *rgb8_d = NULL;
  int rc = 0;

  /* allocate output buffer on device */
  size_t out_bytes = new_w * new_h * 3;
  cudaError_t cerr = cudaMalloc((void **)&rgb8_d, out_bytes);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "CUDA malloc failed for rgb8_d: %s\n",
            cudaGetErrorString(cerr));
    return 5;
  }

  // launch kernel (RGBA float → RGB8)
  dim3 block(16, 16);
  dim3 grid((new_w + 15) / 16, (new_h + 15) / 16);

  rgba16f_to_rgb8_kernel<<<grid, block>>>(rgba, rgb8_d, width, height, new_w,
                                          new_h, scale);

  cerr = cudaGetLastError();
  if (cerr != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch failed: %s\n",
            cudaGetErrorString(cerr));
    cudaFree(rgb8_d);
    return 6;
  }

  cudaDeviceSynchronize();

#ifdef USE_GPUJPEG
  /* GPU input + GPUJPEG (DEVICE to DEVICE, zero-copy) */
  if (gpujpeg_init_device(0, 0) != 0) {
    cudaFree(rgb8_d);
    return 10;
  }

  cudaStream_t stream = 0;
  struct gpujpeg_encoder* enc = gpujpeg_encoder_create(stream);
  if (!enc) {
      cudaFree(rgb8_d);
      return 50;
  }

  /* Set output to pinned memory */
  gpujpeg_encoder_set_option(enc,
    GPUJPEG_ENC_OPT_OUT,
    GPUJPEG_ENC_OUT_VAL_PINNED);

  /* Prepare encoder parameters */
  struct gpujpeg_parameters param;
  //memset(&param, 0, sizeof(param));
  gpujpeg_set_default_parameters(&param); 
  param.quality      = compression_ratio;  // 1..100
  param.restart_interval = 0;              // default
  param.interleaved      = 1;              // standard JPEG

  /* Prepare image parameters */
  struct gpujpeg_image_parameters ip;
  //memset(&ip, 0, sizeof(ip));
  gpujpeg_image_set_default_parameters(&ip);
  ip.width        = new_w;
  ip.height       = new_h;
  ip.color_space  = GPUJPEG_RGB;             // RGB colorspace
  ip.pixel_format = GPUJPEG_444_U8_P012;     // 3 channel RGB 8-bit packed

  /* Prepare input: GPU image buffer */
  struct gpujpeg_encoder_input input;
  gpujpeg_encoder_input_set_gpu_image(&input, rgb8_d);

  /* Encode*/
  uint8_t* jpg_data = NULL;
  size_t   jpg_size = 0;

  int ret = gpujpeg_encoder_encode(
                enc,
                &param,
                &ip,
                &input,
                &jpg_data,
                &jpg_size
            );

  if (ret != 0) {
      gpujpeg_encoder_destroy(enc);
      cudaFree(rgb8_d);
      return 51;
  }


  FILE *f = fopen(filename, "wb");
  if (!f) {
    gpujpeg_encoder_destroy(enc);
    cudaFree(rgb8_d);
    return 13;
  }

  fwrite(jpg_data, 1, jpg_size, f);
  fclose(f);

  gpujpeg_encoder_destroy(enc);
  cudaFree(rgb8_d);

  printf("Save JPEG (GPU + GPUJPEG): ok %s\n", filename);
  return 0;

#else
  /* GPU input + CPU libjpeg */
  unsigned char *rgb8_h = NULL;
  rgb8_h = (unsigned char *)allsky_safe_malloc(out_bytes);
  if (!rgb8_h) {
    cudaFree(rgb8_d);
    return 8;
  }

  // device → host
  cerr = cudaMemcpy(rgb8_h, rgb8_d, out_bytes, cudaMemcpyDeviceToHost);
  cudaFree(rgb8_d);
  if (cerr != cudaSuccess) {
    allsky_safe_free(rgb8_h);
    return 9;
  }

  rc =
      save_jpeg_libjpeg_rgb8(rgb8_h, new_w, new_h, compression_ratio, filename);
  allsky_safe_free(rgb8_h);

  printf("Save JPEG (GPU + libjpeg): ok %s\n", filename);
  return rc;

#endif

  return 99; // Should never hit
}

// Kernel for finding min/max values per block
__global__ void find_minmax_block_kernel(const float *rgbf, int pixel_count,
                                         float *block_minmax) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float my_min_r = (idx < pixel_count) ? rgbf[idx * CHANNELS + 0] : FLT_MAX;
  float my_max_r = (idx < pixel_count) ? rgbf[idx * CHANNELS + 0] : -FLT_MAX;
  float my_min_g = (idx < pixel_count) ? rgbf[idx * CHANNELS + 1] : FLT_MAX;
  float my_max_g = (idx < pixel_count) ? rgbf[idx * CHANNELS + 1] : -FLT_MAX;
  float my_min_b = (idx < pixel_count) ? rgbf[idx * CHANNELS + 2] : FLT_MAX;
  float my_max_b = (idx < pixel_count) ? rgbf[idx * CHANNELS + 2] : -FLT_MAX;

  // Reduction in shared memory
  sdata[tid * 6 + 0] = my_min_r;
  sdata[tid * 6 + 1] = my_max_r;
  sdata[tid * 6 + 2] = my_min_g;
  sdata[tid * 6 + 3] = my_max_g;
  sdata[tid * 6 + 4] = my_min_b;
  sdata[tid * 6 + 5] = my_max_b;
  __syncthreads();

  // Parallel reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid * 6 + 0] = fminf(sdata[tid * 6 + 0], sdata[(tid + s) * 6 + 0]);
      sdata[tid * 6 + 1] = fmaxf(sdata[tid * 6 + 1], sdata[(tid + s) * 6 + 1]);
      sdata[tid * 6 + 2] = fminf(sdata[tid * 6 + 2], sdata[(tid + s) * 6 + 2]);
      sdata[tid * 6 + 3] = fmaxf(sdata[tid * 6 + 3], sdata[(tid + s) * 6 + 3]);
      sdata[tid * 6 + 4] = fminf(sdata[tid * 6 + 4], sdata[(tid + s) * 6 + 4]);
      sdata[tid * 6 + 5] = fmaxf(sdata[tid * 6 + 5], sdata[(tid + s) * 6 + 5]);
    }
    __syncthreads();
  }

  // Write block result to global memory
  if (tid == 0) {
    int block_idx = blockIdx.x * 6;
    block_minmax[block_idx + 0] = sdata[0]; // min_r
    block_minmax[block_idx + 1] = sdata[1]; // max_r
    block_minmax[block_idx + 2] = sdata[2]; // min_g
    block_minmax[block_idx + 3] = sdata[3]; // max_g
    block_minmax[block_idx + 4] = sdata[4]; // min_b
    block_minmax[block_idx + 5] = sdata[5]; // max_b
  }
}

// Kernel for tonemapping normalization and scaling
__global__ void tonemap_normalize_kernel(float *rgbf, int pixel_count,
                                         float min_r, float range_r,
                                         float min_g, float range_g,
                                         float min_b, float range_b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pixel_count)
    return;

  int pixel_idx = idx * CHANNELS;
  rgbf[pixel_idx + 0] = ((rgbf[pixel_idx + 0] - min_r) / range_r) * 65535.0f;
  rgbf[pixel_idx + 1] = ((rgbf[pixel_idx + 1] - min_g) / range_g) * 65535.0f;
  rgbf[pixel_idx + 2] = ((rgbf[pixel_idx + 2] - min_b) / range_b) * 65535.0f;
  rgbf[pixel_idx + 3] = 65535.0f; // Alpha
}

extern "C" int tonemap_rgbf1_to_rgbf16_cuda(float *rgbf_d, int width,
                                            int height) {
  if (!rgbf_d || width <= 0 || height <= 0)
    return 1;

  int pixel_count = width * height;
  cudaError_t err;

  // Setup kernel launch parameters
  const int threads_per_block = 256;
  int num_blocks = (pixel_count + threads_per_block - 1) / threads_per_block;

  // Allocate memory for block-level min/max values
  float *block_minmax_d = NULL;
  size_t block_minmax_size = num_blocks * 6 * sizeof(float);
  err = cudaMalloc(&block_minmax_d, block_minmax_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc failed for block_minmax: %s\n",
            cudaGetErrorString(err));
    return 2;
  }

  // Find min/max per block
  size_t shared_mem_size = threads_per_block * 6 * sizeof(float);
  find_minmax_block_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      rgbf_d, pixel_count, block_minmax_d);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch failed (find_minmax): %s\n",
            cudaGetErrorString(err));
    cudaFree(block_minmax_d);
    return 3;
  }
  cudaDeviceSynchronize();

  // Copy block results to host and find global min/max
  float *block_minmax_h = (float *)malloc(block_minmax_size);
  if (!block_minmax_h) {
    cudaFree(block_minmax_d);
    return 4;
  }
  err = cudaMemcpy(block_minmax_h, block_minmax_d, block_minmax_size,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy failed: %s\n", cudaGetErrorString(err));
    free(block_minmax_h);
    cudaFree(block_minmax_d);
    return 5;
  }

  // Find global min/max from block results
  float min_r = block_minmax_h[0];
  float max_r = block_minmax_h[1];
  float min_g = block_minmax_h[2];
  float max_g = block_minmax_h[3];
  float min_b = block_minmax_h[4];
  float max_b = block_minmax_h[5];

  for (int i = 1; i < num_blocks; i++) {
    int idx = i * 6;
    min_r = fminf(min_r, block_minmax_h[idx + 0]);
    max_r = fmaxf(max_r, block_minmax_h[idx + 1]);
    min_g = fminf(min_g, block_minmax_h[idx + 2]);
    max_g = fmaxf(max_g, block_minmax_h[idx + 3]);
    min_b = fminf(min_b, block_minmax_h[idx + 4]);
    max_b = fmaxf(max_b, block_minmax_h[idx + 5]);
  }

  free(block_minmax_h);
  cudaFree(block_minmax_d);

  // Calculate ranges
  float range_r = fmaxf(max_r - min_r, 1.0f);
  float range_g = fmaxf(max_g - min_g, 1.0f);
  float range_b = fmaxf(max_b - min_b, 1.0f);

  // Normalize and scale
  tonemap_normalize_kernel<<<num_blocks, threads_per_block>>>(
      rgbf_d, pixel_count, min_r, range_r, min_g, range_g, min_b, range_b);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch failed (tonemap_normalize): %s\n",
            cudaGetErrorString(err));
    return 6;
  }
  cudaDeviceSynchronize();

  printf("Tonemapping (GPU): ok, float RGB (0.0–1.0) to RGB float 16-bit "
         "(0.0–65535.0).\n");

  return 0;
}

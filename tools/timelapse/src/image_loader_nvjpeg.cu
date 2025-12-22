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
#include "image_loader.h"
#include "allsky.h"
#include <cuda_runtime.h>
#include <npp.h>
#include <nppcore.h>
#include <nvjpeg.h>
#include <stdio.h>
#include <stdlib.h>

uint8_t *load_and_resize_jpeg(const char *filename, int target_width,
                              int target_height, int expected_src_width,
                              int expected_src_height) {
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return NULL;

  fseek(fp, 0, SEEK_END);
  long length = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  unsigned char *jpeg_buffer = (unsigned char *)allsky_safe_malloc(length);
  size_t bytes_read = fread(jpeg_buffer, 1, length, fp);
  if (bytes_read != length) {
    fprintf(stderr, "[ERROR] Failed to read full JPEG file: %s\n", filename);
    allsky_safe_free(jpeg_buffer);
    return NULL;
  }
  fclose(fp);

  if (!jpeg_buffer || length <= 0) {
    fprintf(stderr, "[ERROR] Invalid JPEG buffer or length\n");
    return NULL;
  }

  nvjpegHandle_t handle;
  nvjpegJpegState_t state;
  nvjpegCreateSimple(&handle);
  nvjpegJpegStateCreate(handle, &state);

  int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  if (nvjpegGetImageInfo(handle, jpeg_buffer, length, &channels, &subsampling,
                         widths, heights) != NVJPEG_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] Failed to get JPEG info for %s\n", filename);
    return NULL;
  }

  int src_width = widths[0], src_height = heights[0];

  // Check image size
  if (src_width != expected_src_width || src_height != expected_src_height) {
    fprintf(stderr, "[WARN] Skipping image %s: expected %dx%d, got %dx%d\n",
            filename, expected_src_width, expected_src_height, src_width,
            src_height);
    nvjpegJpegStateDestroy(state);
    nvjpegDestroy(handle);
    allsky_safe_free(jpeg_buffer);
    return NULL;
  }

  int src_stride = src_width * 3;
  int dst_stride = target_width * 3;

  uint8_t *d_src, *d_dst;
  cudaError_t err;

  cudaError_t err_pre = cudaDeviceSynchronize();
  if (err_pre != cudaSuccess) {
    fprintf(stderr, "[ERROR] CUDA error before malloc: %s\n",
            cudaGetErrorString(err_pre));
    return NULL;
  }

  err = cudaMalloc(&d_src, src_height * src_stride);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaMalloc d_src failed: %s\n",
            cudaGetErrorString(err));
    return NULL;
  }
  err = cudaMalloc(&d_dst, target_height * dst_stride);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaMalloc d_dst failed: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_src);
    return NULL;
  }

  nvjpegImage_t nv_out;
  memset(&nv_out, 0, sizeof(nv_out));

  nv_out.channel[0] = d_src;
  nv_out.pitch[0] = src_stride;

  nvjpegStatus_t decode_status = nvjpegDecode(
      handle, state, jpeg_buffer, length, NVJPEG_OUTPUT_RGBI, &nv_out, 0);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] CUDA error after nvjpegDecode: %s\n",
            cudaGetErrorString(err));
  }
  if (decode_status != NVJPEG_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] nvjpegDecode failed: %d\n", decode_status);
    cudaError_t cerr = cudaGetLastError();
    fprintf(stderr, "[DEBUG] CUDA error after decode: %s\n",
            cudaGetErrorString(cerr));
    cudaFree(d_src);
    allsky_safe_free(jpeg_buffer);
    return NULL;
  }

  // Prepare CUDA stream and NPP context
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess)
    fprintf(stderr, "[ERROR] cudaStreamSynchronize failed: %s\n",
            cudaGetErrorString(sync_status));

  if (decode_status != NVJPEG_STATUS_SUCCESS)
    fprintf(stderr, "[ERROR] nvjpegDecode failed: %d\n", decode_status);

  cudaError_t post_decode = cudaGetLastError();
  if (post_decode != cudaSuccess)
    fprintf(stderr, "[ERROR] CUDA error after decode: %s\n",
            cudaGetErrorString(post_decode));

  NppStreamContext nppCtx = {0};
  nppCtx.hStream = stream;

  // Prepare ROI and sizes
  NppiSize srcSize = {src_width, src_height};
  NppiRect srcROI = {0, 0, src_width, src_height};
  NppiSize dstSize = {target_width, target_height};
  NppiRect dstROI = {0, 0, target_width, target_height};

  NppStatus status = nppiResize_8u_C3R_Ctx(d_src, src_stride, srcSize, srcROI,
                                           d_dst, dst_stride, dstSize, dstROI,
                                           NPPI_INTER_LINEAR, nppCtx);

  if (status != NPP_SUCCESS) {
    fprintf(stderr, "[ERROR] nppiResize_8u_C3R_Ctx failed: %d\n", status);
    cudaFree(d_src);
    cudaFree(d_dst);
    allsky_safe_free(jpeg_buffer);
    nvjpegJpegStateDestroy(state);
    nvjpegDestroy(handle);
    cudaStreamDestroy(stream);
    return NULL;
  }

  // Copy back to host
  uint8_t *h_dst = (uint8_t *)malloc(target_height * dst_stride);
  cudaError_t copy_status = cudaMemcpy(h_dst, d_dst, dst_stride * target_height,
                                       cudaMemcpyDeviceToHost);
  if (copy_status != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaMemcpy failed: %s\n",
            cudaGetErrorString(copy_status));
    allsky_safe_free(h_dst);
    h_dst = NULL;
    // clean up
    cudaFree(d_src);
    cudaFree(d_dst);
    allsky_safe_free(jpeg_buffer);
    nvjpegJpegStateDestroy(state);
    nvjpegDestroy(handle);
    cudaStreamDestroy(stream);

    return NULL;
  }

  // Clean up
  cudaFree(d_src);
  cudaFree(d_dst);
  allsky_safe_free(jpeg_buffer);
  nvjpegJpegStateDestroy(state);
  nvjpegDestroy(handle);
  cudaStreamDestroy(stream);

  return h_dst;
}

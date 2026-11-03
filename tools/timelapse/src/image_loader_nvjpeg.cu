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
#include "image_check.h"
#include "allsky.h"
#include <cuda_runtime.h>
#include <npp.h>
#include <nppcore.h>
#include <nvjpeg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t *load_and_resize_jpeg(const char *filename, int target_width,
                              int target_height, int expected_src_width,
                              int expected_src_height) {
  /* All resources start out empty so every error path can share the cleanup
   * block at the end (nvcc compiles C++: goto must not skip initialisations,
   * hence all declarations up front). */
  unsigned char *jpeg_buffer = NULL;
  uint8_t *d_src = NULL, *d_dst = NULL, *h_dst = NULL;
  nvjpegHandle_t handle = NULL;
  nvjpegJpegState_t state = NULL;
  cudaStream_t stream = NULL;
  int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;
  int src_width, src_height;
  int off_x, off_y, fit_w, fit_h;
  size_t src_stride, dst_stride;
  nvjpegImage_t nv_out;
  nvjpegStatus_t nvj_status;
  cudaError_t err;
  NppStatus npp_status;
  NppStreamContext nppCtx;
  NppiSize srcSize, dstSize;
  NppiRect srcROI, dstROI;
  size_t bytes_read;
  long length;

  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return NULL;

  fseek(fp, 0, SEEK_END);
  length = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (length <= 0) {
    fprintf(stderr, "[ERROR] Empty or unreadable JPEG file: %s\n", filename);
    fclose(fp);
    return NULL;
  }

  jpeg_buffer = (unsigned char *)allsky_safe_malloc(length);
  bytes_read = fread(jpeg_buffer, 1, length, fp);
  fclose(fp);
  if (bytes_read != (size_t)length) {
    fprintf(stderr, "[ERROR] Failed to read full JPEG file: %s\n", filename);
    goto cleanup;
  }

  nvj_status = nvjpegCreateSimple(&handle);
  if (nvj_status != NVJPEG_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] nvjpegCreateSimple failed: %d\n", nvj_status);
    handle = NULL;
    goto cleanup;
  }
  nvj_status = nvjpegJpegStateCreate(handle, &state);
  if (nvj_status != NVJPEG_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] nvjpegJpegStateCreate failed: %d\n", nvj_status);
    state = NULL;
    goto cleanup;
  }

  if (nvjpegGetImageInfo(handle, jpeg_buffer, length, &channels, &subsampling,
                         widths, heights) != NVJPEG_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] Failed to get JPEG info for %s\n", filename);
    goto cleanup;
  }

  src_width = widths[0];
  src_height = heights[0];

  // Optional size check (0 = accept any size)
  if (expected_src_width > 0 && expected_src_height > 0 &&
      (src_width != expected_src_width || src_height != expected_src_height)) {
    fprintf(stderr, "[WARN] Skipping image %s: expected %dx%d, got %dx%d\n",
            filename, expected_src_width, expected_src_height, src_width,
            src_height);
    goto cleanup;
  }

  src_stride = (size_t)src_width * 3;
  dst_stride = (size_t)target_width * 3;

  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaStreamCreate failed: %s\n",
            cudaGetErrorString(err));
    stream = NULL;
    goto cleanup;
  }

  err = cudaMalloc(&d_src, (size_t)src_height * src_stride);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaMalloc d_src failed: %s\n",
            cudaGetErrorString(err));
    d_src = NULL;
    goto cleanup;
  }
  err = cudaMalloc(&d_dst, (size_t)target_height * dst_stride);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaMalloc d_dst failed: %s\n",
            cudaGetErrorString(err));
    d_dst = NULL;
    goto cleanup;
  }

  memset(&nv_out, 0, sizeof(nv_out));
  nv_out.channel[0] = d_src;
  nv_out.pitch[0] = src_stride;

  // Decode and resize on the same stream, synchronised once before the copy
  nvj_status = nvjpegDecode(handle, state, jpeg_buffer, length,
                            NVJPEG_OUTPUT_RGBI, &nv_out, stream);
  if (nvj_status != NVJPEG_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] nvjpegDecode failed for %s: %d (%s)\n", filename,
            nvj_status, cudaGetErrorString(cudaGetLastError()));
    goto cleanup;
  }

  /* Letterbox: black frame, the image scaled into fit_w x fit_h at
   * (off_x, off_y); same aspect ratio as the frame = whole frame. */
  fit_rect(src_width, src_height, target_width, target_height, &off_x, &off_y,
           &fit_w, &fit_h);
  err = cudaMemsetAsync(d_dst, 0, (size_t)target_height * dst_stride, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaMemsetAsync failed: %s\n",
            cudaGetErrorString(err));
    goto cleanup;
  }

  memset(&nppCtx, 0, sizeof(nppCtx));
  nppCtx.hStream = stream;

  srcSize = {src_width, src_height};
  srcROI = {0, 0, src_width, src_height};
  dstSize = {fit_w, fit_h};
  dstROI = {0, 0, fit_w, fit_h};

  npp_status = nppiResize_8u_C3R_Ctx(d_src, src_stride, srcSize, srcROI,
                                     d_dst + (size_t)off_y * dst_stride +
                                         (size_t)off_x * 3,
                                     dst_stride, dstSize, dstROI,
                                     NPPI_INTER_LINEAR, nppCtx);
  if (npp_status != NPP_SUCCESS) {
    fprintf(stderr, "[ERROR] nppiResize_8u_C3R_Ctx failed: %d\n", npp_status);
    goto cleanup;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaStreamSynchronize failed: %s\n",
            cudaGetErrorString(err));
    goto cleanup;
  }

  // Copy back to host
  h_dst = (uint8_t *)allsky_safe_malloc((size_t)target_height * dst_stride);
  err = cudaMemcpy(h_dst, d_dst, (size_t)target_height * dst_stride,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[ERROR] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    allsky_safe_free(h_dst);
    h_dst = NULL;
  }

cleanup:
  if (d_src)
    cudaFree(d_src);
  if (d_dst)
    cudaFree(d_dst);
  if (stream)
    cudaStreamDestroy(stream);
  if (state)
    nvjpegJpegStateDestroy(state);
  if (handle)
    nvjpegDestroy(handle);
  allsky_safe_free(jpeg_buffer);

  return h_dst;
}

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

#define CHANNELS	4

__global__ void subtract_darkframe_kernel(unsigned short *rgb_image, const unsigned short *darkframe, int pixel_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;

    int base = idx * CHANNELS; // RGBA
    for (int c = 0; c < 3; ++c) {
        int value = (int)rgb_image[base + c] - (int)darkframe[base + c];
        rgb_image[base + c] = (value < 0) ? 0 : (unsigned short)value;
    }
}

extern "C" int subtract_darkframe_rgb16_cuda(unsigned short *rgb_image, unsigned short *darkframe,
                                              int width, int height, int dark_width, int dark_height) {
    if (!rgb_image || !darkframe || width <= 0 || height <= 0 || width != dark_width || height != dark_height)
        return 1;

    if (width != dark_width || height != dark_height) {
        fprintf(stderr, "ERROR: Frame witdth/height mismatch\n");
        return 2;
    }

    size_t image_size = width * height * CHANNELS * sizeof(unsigned short);
    unsigned short *d_rgb = nullptr, *d_dark = nullptr;

    cudaError_t err;
    err = cudaMalloc((void**)&d_rgb, image_size);
    if (err != cudaSuccess) return 2;
    err = cudaMalloc((void**)&d_dark, image_size);
    if (err != cudaSuccess) { cudaFree(d_rgb); return 3; }

    cudaMemcpy(d_rgb, rgb_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dark, darkframe, image_size, cudaMemcpyHostToDevice);

    int pixel_count = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (pixel_count + threadsPerBlock - 1) / threadsPerBlock;
    subtract_darkframe_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_dark, pixel_count);
    cudaDeviceSynchronize();

    cudaMemcpy(rgb_image, d_rgb, image_size, cudaMemcpyDeviceToHost);
    cudaFree(d_rgb);
    cudaFree(d_dark);

    printf("CUDA: Darkframe calibration (%dx%d) ok.\n", width, height);
    return 0;
}

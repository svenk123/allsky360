/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 *
 * CUDA-Kernels und Wrapper fuer den Motion-Stacker:
 *   - Histogramm-Update (Pass 1, pro Frame)
 *   - Median-Berechnung (nach Pass 1)
 *   - Motion-Detektion  (Pass 2, pro Frame)
 *   - Dynamische Skalierung fuer den Ausgabe-Frame
 *
 *****************************************************************************/
#include "stacker_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>

// ── Persistente GPU-Puffer ───────────────────────────────────────────────────

static uint16_t *d_hist      = nullptr;  // n_pixels * 256 uint16_t
static uint8_t  *d_median_bg = nullptr;  // n_pixels uint8_t
static float    *d_accum     = nullptr;  // n_pixels float
static uint8_t  *d_rgb       = nullptr;  // n_pixels uint8_t (Frame-Buffer, wiederverwendet)
static uint8_t  *d_out       = nullptr;  // n_pixels uint8_t (skalierter Ausgabe-Frame)
static float    *d_max_val   = nullptr;  // 1 float  (fuer Max-Reduktion)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA] %s:%d – %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                  \
            return;                                                           \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_RET(call, ret)                                            \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA] %s:%d – %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                  \
            return (ret);                                                     \
        }                                                                    \
    } while (0)

// ── Hilfsfunktion: atomicMax fuer float ─────────────────────────────────────
// IEEE 754: positive Floats sind als uint32 korrekt sortierbar.
__device__ static void atomicMaxFloat(float *addr, float val) {
    int *addr_i = reinterpret_cast<int *>(addr);
    int  old_i  = *addr_i;
    int  val_i  = __float_as_int(val);
    while (__int_as_float(old_i) < val) {
        int prev = atomicCAS(addr_i, old_i, val_i);
        if (prev == old_i) break;
        old_i = prev;
    }
}

// ── Kernel: Histogramm-Update ─────────────────────────────────────────────
// Jeder Thread verarbeitet genau einen Pixel → kein Konflikt, kein Atomic noetig.
__global__ static void k_update_histogram(uint16_t *hist, const uint8_t *rgb,
                                           size_t n_pixels) {
    size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    uint16_t *h = hist + p * 256;
    uint8_t   v = rgb[p];
    if (h[v] < 65535u) h[v]++;
}

// ── Kernel: Median aus Histogramm ────────────────────────────────────────────
__global__ static void k_build_median(const uint16_t *hist, uint32_t half,
                                       size_t n_pixels, uint8_t *out) {
    size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    const uint16_t *h   = hist + p * 256;
    uint32_t        cum = 0;
    for (int v = 0; v < 256; ++v) {
        cum += h[v];
        if (cum >= half) { out[p] = (uint8_t)v; return; }
    }
    out[p] = 255u;
}

// ── Kernel: Motion-Akkumulator (Max-Variante) ────────────────────────────────
__global__ static void k_update_motion(float *accum, const uint8_t *rgb,
                                        const uint8_t *median_bg,
                                        float threshold, size_t n_pixels) {
    size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    float diff = (float)rgb[p] - (float)median_bg[p];
    if (diff >= threshold && diff > accum[p])
        accum[p] = diff;
}

// ── Kernel: Max-Reduktion (fuer dynamisches Scaling) ────────────────────────
__global__ static void k_find_max(const float *accum, size_t n_pixels,
                                   float *max_val) {
    size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    atomicMaxFloat(max_val, accum[p]);
}

// ── Kernel: Skalierung des Ausgabe-Frames ────────────────────────────────────
__global__ static void k_scale_output(const float *accum, float scale,
                                       size_t n_pixels, uint8_t *out) {
    size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pixels) return;
    float v = accum[p] * scale;
    if (v > 255.0f) v = 255.0f;
    out[p] = (uint8_t)(v + 0.5f);
}

// ── Grid-Groesse berechnen ───────────────────────────────────────────────────
static inline dim3 grid_for(size_t n, int block) {
    return dim3((unsigned int)((n + block - 1) / block));
}

// ============================================================================
// Oeffentliche Wrapper-Funktionen (extern "C")
// ============================================================================

extern "C" int cuda_stacker_init(size_t n_pixels) {
    size_t hist_bytes = n_pixels * 256 * sizeof(uint16_t);

    CUDA_CHECK_RET(cudaMalloc(&d_hist,      hist_bytes),                -1);
    CUDA_CHECK_RET(cudaMemset(d_hist, 0,    hist_bytes),                -1);
    CUDA_CHECK_RET(cudaMalloc(&d_median_bg, n_pixels * sizeof(uint8_t)), -1);
    CUDA_CHECK_RET(cudaMalloc(&d_accum,     n_pixels * sizeof(float)),   -1);
    CUDA_CHECK_RET(cudaMemset(d_accum, 0,   n_pixels * sizeof(float)),   -1);
    CUDA_CHECK_RET(cudaMalloc(&d_rgb,       n_pixels * sizeof(uint8_t)), -1);
    CUDA_CHECK_RET(cudaMalloc(&d_out,       n_pixels * sizeof(uint8_t)), -1);
    CUDA_CHECK_RET(cudaMalloc(&d_max_val,   sizeof(float)),              -1);

    printf("[CUDA] Stacker-GPU-Puffer allokiert (Histogramm: %.0f MB)\n",
           (double)hist_bytes / (1024.0 * 1024.0));
    return 0;
}

extern "C" void cuda_stacker_free(void) {
    cudaFree(d_hist);      d_hist      = nullptr;
    cudaFree(d_median_bg); d_median_bg = nullptr;
    cudaFree(d_accum);     d_accum     = nullptr;
    cudaFree(d_rgb);       d_rgb       = nullptr;
    cudaFree(d_out);       d_out       = nullptr;
    cudaFree(d_max_val);   d_max_val   = nullptr;
}

extern "C" void cuda_update_pixel_histogram(const uint8_t *rgb_host,
                                             size_t n_pixels) {
    const int BLOCK = 1024;
    CUDA_CHECK(cudaMemcpy(d_rgb, rgb_host, n_pixels, cudaMemcpyHostToDevice));
    k_update_histogram<<<grid_for(n_pixels, BLOCK), BLOCK>>>(d_hist, d_rgb,
                                                              n_pixels);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void cuda_build_median_from_histogram(int n_frames, size_t n_pixels,
                                                  uint8_t *median_out_host) {
    const int BLOCK = 1024;
    uint32_t  half  = ((uint32_t)n_frames + 1u) / 2u;
    k_build_median<<<grid_for(n_pixels, BLOCK), BLOCK>>>(d_hist, half,
                                                          n_pixels, d_median_bg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(median_out_host, d_median_bg, n_pixels,
                          cudaMemcpyDeviceToHost));
}

extern "C" void cuda_update_motion(const uint8_t *rgb_host,
                                    const uint8_t *median_bg_host,
                                    float threshold, size_t n_pixels) {
    const int BLOCK = 1024;
    // median_bg_host nur beim ersten Aufruf hochladen (wenn d_median_bg noch
    // nicht gesetzt ist, kopieren wir ihn explizit).
    CUDA_CHECK(cudaMemcpy(d_rgb,       rgb_host,       n_pixels,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_median_bg, median_bg_host, n_pixels,
                          cudaMemcpyHostToDevice));
    k_update_motion<<<grid_for(n_pixels, BLOCK), BLOCK>>>(d_accum, d_rgb,
                                                           d_median_bg,
                                                           threshold, n_pixels);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void cuda_export_motion(size_t n_pixels, uint8_t *out_host) {
    const int BLOCK = 1024;

    // Max-Wert bestimmen
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_max_val, &zero, sizeof(float),
                          cudaMemcpyHostToDevice));
    k_find_max<<<grid_for(n_pixels, BLOCK), BLOCK>>>(d_accum, n_pixels,
                                                      d_max_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float max_val;
    CUDA_CHECK(cudaMemcpy(&max_val, d_max_val, sizeof(float),
                          cudaMemcpyDeviceToHost));

    if (max_val > 0.0f) {
        float scale = 255.0f / max_val;
        k_scale_output<<<grid_for(n_pixels, BLOCK), BLOCK>>>(d_accum, scale,
                                                               n_pixels, d_out);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemset(d_out, 0, n_pixels));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_host, d_out, n_pixels, cudaMemcpyDeviceToHost));
}

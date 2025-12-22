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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef USE_CUDA

#define CLAMPF(x) ((x) < 0.0f ? 0.0f : ((x) > 65535.0f ? 65535.0f : (x)))

static float quick_median(float *v, int len) {
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - i - 1; j++) {
            if (v[j] > v[j + 1]) {
                float tmp = v[j];
                v[j] = v[j + 1];
                v[j + 1] = tmp;
            }
        }
    }
    return v[len / 2];
}

/**
 * Medianfilter für RGBA-Floatbilder auf der CPU mit OpenMP-Beschleunigung
 * @param rgba RGBA-Float-Bilddaten (in-place, 4 Kanäle, 65535.0 Skala)
 * @param width Bildbreite
 * @param height Bildhöhe
 * @param kernel_radius Radius (1 = 3x3, 2 = 5x5 ...)
 * @return 0 bei Erfolg, >0 bei Fehler
 */
int median_filter_rgbf(float *rgba, int width, int height, int kernel_radius) {
    if (!rgba || width <= 0 || height <= 0 || kernel_radius < 1 || kernel_radius > 10) return 1;

    int pixel_count = width * height;
    float *copy = (float *)malloc(pixel_count * 4 * sizeof(float));
    if (!copy) return 2;
    memcpy(copy, rgba, pixel_count * 4 * sizeof(float));

    int kernel_size = 2 * kernel_radius + 1;
    int window_area = kernel_size * kernel_size;

    #pragma omp parallel
    {
        float *r_vals = (float *)malloc(window_area * sizeof(float));
        float *g_vals = (float *)malloc(window_area * sizeof(float));
        float *b_vals = (float *)malloc(window_area * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int y = kernel_radius; y < height - kernel_radius; y++) {
            for (int x = kernel_radius; x < width - kernel_radius; x++) {
                int k = 0;
                for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
                    for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
                        int ix = x + dx;
                        int iy = y + dy;
                        int iidx = (iy * width + ix) * 4;
                        r_vals[k] = copy[iidx + 0];
                        g_vals[k] = copy[iidx + 1];
                        b_vals[k] = copy[iidx + 2];
                        k++;
                    }
                }
                int out_idx = (y * width + x) * 4;
                rgba[out_idx + 0] = quick_median(r_vals, window_area);
                rgba[out_idx + 1] = quick_median(g_vals, window_area);
                rgba[out_idx + 2] = quick_median(b_vals, window_area);
                rgba[out_idx + 3] = 65535.0f;
            }
        }

        free(r_vals);
        free(g_vals);
        free(b_vals);
    }

    free(copy);

    printf("Median filter: ok\n");

    return 0;
}

#define CLAMP(v, min, max) ((v) < (min) ? (min) : ((v) > (max) ? (max) : (v)))

/**
 * Helper: Swap two float values
 */
static inline void swapf(float *a, float *b) {
    float t = *a;
    *a = *b;
    *b = t;
}

/**
 * Helper: Median-of-three pivot selection
 */
static inline float median_of_three(float *arr, int a, int b, int c) {
    if (arr[a] < arr[b]) {
        if (arr[b] < arr[c]) return arr[b];
        else if (arr[a] < arr[c]) return arr[c];
        else return arr[a];
    } else {
        if (arr[a] < arr[c]) return arr[a];
        else if (arr[b] < arr[c]) return arr[c];
        else return arr[b];
    }
}

/**
 * Robust Quickselect to find k-th smallest element
 */
float quickselect(float *arr, int left, int right, int k) {
    while (left < right) {
        // Median-of-three pivot
        int mid = left + (right - left) / 2;
        float pivot = median_of_three(arr, left, mid, right);

        // Partition
        int i = left, j = right;
        while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
                swapf(&arr[i], &arr[j]);
                i++;
                j--;
            }
        }

        // Narrow down the search
        if (k <= j) {
            right = j;
        } else if (k >= i) {
            left = i;
        } else {
            return arr[k];
        }
    }
    return arr[left];
}

int multiscale_median_filter_rgbf1(float *rgba, int width, int height, int max_scale, float blend_factor) {
    if (!rgba || width <= 0 || height <= 0 || max_scale < 1 || blend_factor < 0.0f || blend_factor > 1.0f) {
        printf("Error: invalid parameters\n");
        return 1;
    }

    int use_quickselect = 1;
    int n_pixels = width * height;
    float *temp = (float *)malloc(sizeof(float) * 4 * n_pixels);
    if (!temp) {
        printf("Error: memory allocation failed\n");
        return 1;
    }

    for (int scale = 1; scale <= max_scale; ++scale) {
        int radius = scale;
        int ksize = 2 * radius + 1;
        int window_area = ksize * ksize;
        int half_window = window_area / 2;

        if (window_area > 225) {
            printf("Warning: kernel too large (%dx%d), skipping this scale\n", ksize, ksize);
            continue;
        }

        printf("Applying median filter: scale %d, radius %d, blend %.2f, quickselect %s\n",
               scale, radius, blend_factor, use_quickselect ? "ON" : "OFF");

        #pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float r_vals[225], g_vals[225], b_vals[225];
                int count = 0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    int yy = CLAMP(y + dy, 0, height - 1);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int xx = CLAMP(x + dx, 0, width - 1);
                        int idx = (yy * width + xx) * 4;
                        r_vals[count] = rgba[idx + 0];
                        g_vals[count] = rgba[idx + 1];
                        b_vals[count] = rgba[idx + 2];
                        count++;
                    }
                }

                float r_med, g_med, b_med;
                if (use_quickselect) {
                    r_med = quickselect(r_vals, 0, count - 1, half_window);
                    g_med = quickselect(g_vals, 0, count - 1, half_window);
                    b_med = quickselect(b_vals, 0, count - 1, half_window);
                } else {
                    // partial selection sort to median position
                    for (int i = 0; i <= half_window; ++i) {
                        int min_idx = i;
                        for (int j = i + 1; j < count; ++j)
                            if (r_vals[j] < r_vals[min_idx]) min_idx = j;
                        float tmp = r_vals[i]; r_vals[i] = r_vals[min_idx]; r_vals[min_idx] = tmp;
                    }
                    for (int i = 0; i <= half_window; ++i) {
                        int min_idx = i;
                        for (int j = i + 1; j < count; ++j)
                            if (g_vals[j] < g_vals[min_idx]) min_idx = j;
                        float tmp = g_vals[i]; g_vals[i] = g_vals[min_idx]; g_vals[min_idx] = tmp;
                    }
                    for (int i = 0; i <= half_window; ++i) {
                        int min_idx = i;
                        for (int j = i + 1; j < count; ++j)
                            if (b_vals[j] < b_vals[min_idx]) min_idx = j;
                        float tmp = b_vals[i]; b_vals[i] = b_vals[min_idx]; b_vals[min_idx] = tmp;
                    }
                    r_med = r_vals[half_window];
                    g_med = g_vals[half_window];
                    b_med = b_vals[half_window];
                }

                int out_idx = (y * width + x) * 4;
                temp[out_idx + 0] = r_med;
                temp[out_idx + 1] = g_med;
                temp[out_idx + 2] = b_med;
                temp[out_idx + 3] = rgba[out_idx + 3];
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < n_pixels; ++i) {
            int idx = i * 4;
            rgba[idx + 0] = (1.0f - blend_factor) * rgba[idx + 0] + blend_factor * temp[idx + 0];
            rgba[idx + 1] = (1.0f - blend_factor) * rgba[idx + 1] + blend_factor * temp[idx + 1];
            rgba[idx + 2] = (1.0f - blend_factor) * rgba[idx + 2] + blend_factor * temp[idx + 2];
            // alpha stays unchanged
        }
    }

    free(temp);
    printf("Multi-scale median filter: ok\n");
    return 0;
}

#endif

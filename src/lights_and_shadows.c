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
#include "lights_and_shadows.h"
#include "allsky.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef USE_CUDA

int adjust_black_point_rgbf1(float *rgbf, int width, int height,
                             double min_shift_pct, double max_shift_pct,
                             double dark_threshold) {
  if (!rgbf || width <= 0 || height <= 0)
    return 1;

  const int pixel_count = width * height;
  const int hist_bins = 2048; // fine enough for 0..1 range
  const float eps = 1e-6f;

  // --- Histogram initialization ---
  int *hist = calloc(hist_bins, sizeof(int));
  if (!hist)
    return 1;

  double min_val = 1.0, max_val = 0.0;
  double sum_L = 0.0;
  long valid_pixels = 0;

// --- Build histogram (OpenMP parallel reduction) ---
#pragma omp parallel
  {
    int local_hist[hist_bins];
    for (int i = 0; i < hist_bins; i++)
      local_hist[i] = 0;
    double local_sum = 0.0;
    double local_min = 1.0, local_max = 0.0;
    long local_valid = 0;

#pragma omp for nowait
    for (int i = 0; i < pixel_count; i++) {
      const float *px = &rgbf[i * CHANNELS];

      // Skip masked pixels (already black)
      if (px[0] <= eps && px[1] <= eps && px[2] <= eps)
        continue;

      // Compute luminance (Rec.709)
      float L = rgb_to_luma(px[0], px[1], px[2]);

      if (L < 0.0f)
        L = 0.0f;
      if (L > 1.0f)
        L = 1.0f;

      int bin = (int)(L * (hist_bins - 1));
      local_hist[bin]++;
      local_sum += L;
      if (L < local_min)
        local_min = L;
      if (L > local_max)
        local_max = L;
      local_valid++;
    }

    printf("Build histogram for local pixels\n");

    // Merge local histograms into global histogram
#pragma omp critical
    {
      for (int b = 0; b < hist_bins; b++)
        hist[b] += local_hist[b];
      sum_L += local_sum;
      if (local_min < min_val)
        min_val = local_min;
      if (local_max > max_val)
        max_val = local_max;
      valid_pixels += local_valid;
    }
  }

  if (valid_pixels == 0) {
    allsky_safe_free(hist);

      printf("Warning: No valid pixels found for black point adjustment.\n");
    return 1;
  }

  printf("Merged local histograms into global histogram\n");

  // --- Compute quantiles ---
  long target01 = (long)(valid_pixels * 0.01);
  long target99 = (long)(valid_pixels * 0.99);
  long cumulative = 0;
  float p01 = 0.0f, p99 = 1.0f;
  for (int b = 0; b < hist_bins; b++) {
    cumulative += hist[b];
    if (cumulative >= target01 && p01 == 0.0f)
      p01 = (float)b / (float)(hist_bins - 1);
    if (cumulative >= target99) {
      p99 = (float)b / (float)(hist_bins - 1);
      break;
    }
  }

  allsky_safe_free(hist);

  double avg_L = sum_L / (double)valid_pixels;


  printf("Computed quantiles for global histogram\n");

  // --- Decide shift adaptively ---
  double black_shift_pct = 0.0;

  if (avg_L > dark_threshold) {
    // bright scene: little or no adjustment
    black_shift_pct = min_shift_pct;
  } else if (p01 > 0.02 && (p99 - p01) > 0.05) {
    // underexposed or compressed shadows
    black_shift_pct = max_shift_pct;
  } else if (p01 > 0.005) {
    // mild dark compression
    black_shift_pct = (min_shift_pct + max_shift_pct) * 0.5;
  } else {
    // histogram already ok
    black_shift_pct = 0.0;
  }

  printf("[BlackPoint] valid=%ld  avgL=%.4f  p01=%.4f  p99=%.4f  "
           "shift=%.4f  (%.2f%%)\n",
           valid_pixels, avg_L, p01, p99, black_shift_pct,
           black_shift_pct * 100.0);

  if (black_shift_pct <= 0.0)
    return 0;

  // --- Apply shift ---
  const float range = (float)(max_val - min_val);
  const float black_point = (float)(min_val + black_shift_pct * range);
  const float denom = (float)(range * (1.0 - black_shift_pct));

#pragma omp parallel for
  for (int i = 0; i < pixel_count; i++) {
    float *px = &rgbf[i * CHANNELS];
    if (px[0] <= eps && px[1] <= eps && px[2] <= eps)
      continue;

    for (int c = 0; c < 3; c++) {
      float v = (px[c] - black_point) / denom;
      px[c] = clampf1(v);
    }
  }

  printf("Blackpoint adjustment: ok, %.1f%% (%.4f . %.4f)\n", 
         black_shift_pct * 100.0, min_val, black_point);

  return 0;
}

// Helper function for Median-of-three pivot selection
static inline float median_of_three_pivot(float *arr, int a, int b, int c) {
  if (arr[a] < arr[b]) {
    if (arr[b] < arr[c]) 
      return arr[b];
    else if (arr[a] < arr[c]) 
      return arr[c];
    else 
      return arr[a];
  } else {
    if (arr[a] < arr[c])
      return arr[a];
    else if (arr[b] < arr[c]) 
      return arr[c];
    else 
      return arr[b];
  }
}

// Helper function for Quickselect-based percentile calculation
static float quickselect_percentile(float *arr, int n, float percentile) {
  if (n <= 0) 
    return 0.0f;
  if (n == 1) 
    return arr[0];
  
  // Calculate index for percentile
  int k = (int)(percentile * (n - 1));
  if (k < 0)
    k = 0;
  if (k >= n) 
    k = n - 1;
  
  // For small arrays: direct sorting (more efficient)
  if (n <= 50) {
    qsort(arr, n, sizeof(float), compare_floats);
    return arr[k];
  }
  
  // Quickselect for larger arrays (O(n) instead of O(n log n))
  int left = 0, right = n - 1;
  
  while (left < right) {
    // Median-of-three pivot selection (faster than random pivot)
    int mid = left + (right - left) / 2;
    float pivot = median_of_three_pivot(arr, left, mid, right);
    
    // Partition
    int i = left, j = right;
    while (i <= j) {
      while (arr[i] < pivot)
        i++;

      while (arr[j] > pivot) 
        j--;

      if (i <= j) {
        float tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
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

int autostretch_rgbf1(float *hdr_image, int width, int height,
                          float min_val, float max_val) {
  if (!hdr_image || width <= 0 || height <= 0 || min_val < 0.0f ||
      min_val > 1.0f || max_val < 0.0f || max_val > 1.0f || max_val <= min_val)
    return 1;

  int pixel_count = width * height;
  int num_values = pixel_count * CHANNELS;
  float *all_values = (float *)allsky_safe_malloc(sizeof(float) * num_values);
  if (!all_values)
    return 1;

  // Find RGB values without clipped pixels
  int k = 0;
  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;
    for (int c = 0; c < 3; ++c) {
      float val = hdr_image[idx + c];
      if (!isfinite(val) || val < 0.0f)
        continue;
      all_values[k++] = val;
    }
  }
  // Not enough valid values
  if (k < 2) {
    printf("Autostretch: not enough valid values: k=%d\n", k);
    allsky_safe_free(all_values);
    return 1;
  }

  if (k == 2) {
    // Only 2 values: simple Min/Max stretch

    printf("Found only 2 valid values: simple Min/Max stretch\n");
    float black = all_values[0] < all_values[1] ? all_values[0] : all_values[1];
    float white = all_values[0] > all_values[1] ? all_values[0] : all_values[1];
    allsky_safe_free(all_values);
    
    // Check numerical stability
    float range = white - black;
    if (range < 1e-6f) { // Very small range
      return 1;
    }
    
    float scale = 1.0f / range;
    // Apply stretch (OpenMP parallelized)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < pixel_count; i++) {
      int idx = i * CHANNELS;
      for (int c = 0; c < 3; ++c) {
        float val = (hdr_image[idx + c] - black) * scale;
        hdr_image[idx + c] = clampf1(val);
      }
    }
    
    printf("Autostretch: ok, %.2f . %.2f (%.3f . %.3f)\n",
           min_val * 100.0f, max_val * 100.0f, black, white);
    return 0;
  }

  // Performance optimization: Quickselect instead of full sort
  // Copy for both percentiles (Quickselect modifies the array)
  float *all_values_copy = (float *)allsky_safe_malloc(sizeof(float) * k);
  if (!all_values_copy) {
    allsky_safe_free(all_values);
    printf("Autostretch: not enough memory\n");

    return 1;
  }

  memcpy(all_values_copy, all_values, sizeof(float) * k);

  // Calculate percentiles with Quickselect (OpenMP parallelized)
  // Both calls are independent and can run in parallel
  float black, white;
#pragma omp parallel sections
  {
#pragma omp section
    {
      black = quickselect_percentile(all_values, k, min_val);
    }
#pragma omp section
    {
      white = quickselect_percentile(all_values_copy, k, max_val);
    }
  }
  
  allsky_safe_free(all_values);
  allsky_safe_free(all_values_copy);

  // Check numerical stability: if white - black is too small
  float range = white - black;
  if (range < 1e-6f) { // Very small range - avoid division by almost zero
    return 1;
  }

  float scale = 1.0f / range;

  // Apply stretch (OpenMP parallelized)
#pragma omp parallel for schedule(static)
  for (int i = 0; i < pixel_count; i++) {
    int idx = i * CHANNELS;
    for (int c = 0; c < 3; ++c) {
      float val = (hdr_image[idx + c] - black) * scale;
      hdr_image[idx + c] = clampf1(val);
    }
  }

  printf("Autostretch: ok, %.2f . %.2f (%.3f . %.3f)\n", min_val * 100.0f, max_val * 100.0f, black, white);
  return 0;
}

#endif

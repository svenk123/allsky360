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
#include "image_check.h"
#include "image_loader.h"
#include "png_to_rgb8.h"
#include "stacker.h"
#include "video_encoder.h"
#ifdef USE_CUDA
#include "stacker_cuda.h"
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEFAULT_SIGMA_THRESHOLD 2.0f
/* Motion: per-channel difference (0-255) to the median background. A value
 * as low as the sigma default makes the result noise-dominated. */
#define DEFAULT_MOTION_THRESHOLD 10.0f
/* Upper bound for -w/-h: keeps width * height * 3 and the motion histogram
 * (width * height * 3 * 256 * 2 bytes) far away from integer overflow. */
#define MAX_DIMENSION 16384

/* One distinct source resolution and how many input images have it */
typedef struct {
  int width, height, count;
  int last_index; /* position of the newest image with this size */
} size_group_t;

#define MAX_SIZE_GROUPS 32

/**
 * Reads the JPEG headers of all inputs and picks the images to use.
 *
 * A config change during the night can change the image size (resolution,
 * crop). Normal timelapse: every image is used and letterboxed into the video
 * frame. Stacking needs pixel-aligned frames: only the size with the most
 * images is used (tie: the size of the newest image).
 *
 * @return Number of images in files (0 if none is usable).
 */
static int select_images(char **inputs, int n_inputs, int stacking,
                         const char **files, int *ref_width, int *ref_height) {
  size_group_t groups[MAX_SIZE_GROUPS];
  int n_groups = 0, unreadable = 0, other = 0;
  int *widths = (int *)allsky_safe_malloc(sizeof(int) * n_inputs);
  int *heights = (int *)allsky_safe_malloc(sizeof(int) * n_inputs);

  for (int i = 0; i < n_inputs; i++) {
    if (get_jpeg_dimensions(inputs[i], &widths[i], &heights[i]) != 0) {
      widths[i] = heights[i] = 0;
      unreadable++;
      continue;
    }
    int g;
    for (g = 0; g < n_groups; g++)
      if (groups[g].width == widths[i] && groups[g].height == heights[i])
        break;
    if (g == n_groups) {
      if (n_groups == MAX_SIZE_GROUPS) {
        other++;
        continue;
      }
      groups[n_groups++] = (size_group_t){widths[i], heights[i], 0, 0};
    }
    groups[g].count++;
    groups[g].last_index = i;
  }

  int best = -1;
  for (int g = 0; g < n_groups; g++)
    if (best < 0 || groups[g].count > groups[best].count ||
        (groups[g].count == groups[best].count &&
         groups[g].last_index > groups[best].last_index))
      best = g;

  if (unreadable > 0)
    fprintf(stderr, "[WARN] %d unreadable image(s) skipped.\n", unreadable);
  if (n_groups > 1 || other > 0) {
    fprintf(stderr, "[INFO] Input images have %d different sizes:", n_groups);
    for (int g = 0; g < n_groups; g++)
      fprintf(stderr, " %dx%d (%d)", groups[g].width, groups[g].height,
              groups[g].count);
    fprintf(stderr, "\n");
    if (stacking)
      fprintf(stderr,
              "[WARN] Stacking needs one size: using %dx%d, skipping %d "
              "image(s) of other sizes.\n",
              groups[best].width, groups[best].height,
              n_inputs - unreadable - groups[best].count);
    else
      fprintf(stderr, "[INFO] All sizes are used, letterboxed into the video "
                      "frame.\n");
  }

  int n = 0;
  for (int i = 0; best >= 0 && i < n_inputs; i++) {
    if (widths[i] == 0)
      continue;
    if (stacking && (widths[i] != groups[best].width ||
                     heights[i] != groups[best].height))
      continue;
    files[n++] = inputs[i];
  }

  /* Stacking: the loader double-checks the size; otherwise accept any */
  *ref_width = (stacking && best >= 0) ? groups[best].width : 0;
  *ref_height = (stacking && best >= 0) ? groups[best].height : 0;

  allsky_safe_free(widths);
  allsky_safe_free(heights);
  return n;
}

void print_usage(const char *progname) {
  fprintf(
      stderr,
      "Usage: %s -o output.mp4 -w width -h height -p fps [-v] image1.jpg ...\n",
      progname);
  fprintf(stderr, "  -m avg|max|min|sigma|diff|motion\n");
  fprintf(stderr, "  -t <threshold> sigma clipping (default: %.1f) or motion detection\n"
                  "     (difference to the median background, default: %.1f)\n",
          DEFAULT_SIGMA_THRESHOLD, DEFAULT_MOTION_THRESHOLD);
  fprintf(stderr, "  -s <filename> to save final stacked result as PNG\n");
}

void print_progress_bar(int current, int total) {
  int width = 50;
  float ratio = (float)current / total;
  int filled = (int)(ratio * width);

  printf("\r[");
  for (int i = 0; i < width; ++i) {
    printf(i < filled ? "#" : " ");
  }
  printf("] %3d%%", (int)(ratio * 100));
  fflush(stdout);
}

int main(int argc, char *argv[]) {
  const char *output_filename = NULL;
  const char *stack_output_filename = NULL;
  int width = 0, height = 0, fps = 0, verbose = 0;
  stacking_mode_t mode = STACK_NONE;
  float sigma_clip_threshold = -1.0f; /* < 0: mode-dependent default */
  int opt;

  while ((opt = getopt(argc, argv, "o:w:h:p:vs:m:t:")) != -1) {
    switch (opt) {
    case 'o':
      output_filename = optarg;
      break;
    case 'w':
      width = atoi(optarg);
      break;
    case 'h':
      height = atoi(optarg);
      break;
    case 'p':
      fps = atoi(optarg);
      break;
    case 'v':
      verbose = 1;
      break;
    case 's':
      stack_output_filename = optarg;
      break;
    case 'm':
      if (strcmp(optarg, "avg") == 0)
        mode = STACK_AVERAGE;
      else if (strcmp(optarg, "max") == 0)
        mode = STACK_MAX;
      else if (strcmp(optarg, "min") == 0)
        mode = STACK_MIN;
      else if (strcmp(optarg, "sigma") == 0)
        mode = STACK_SIGMA_CLIP;
      else if (strcmp(optarg, "diff") == 0)
        mode = STACK_DIFFERENCE;
      else if (strcmp(optarg, "motion") == 0)
        mode = STACK_MOTION;
      else {
        fprintf(stderr, "[ERROR] Unknown mode: %s\n", optarg);
        return 1;
      }
      break;
    case 't':
      sigma_clip_threshold = atof(optarg);
      break;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  int image_count = argc - optind;
  if (!output_filename || width <= 0 || height <= 0 || fps <= 0 ||
      image_count <= 0) {
    fprintf(stderr, "[ERROR] Invalid parameters.\n");
    fprintf(stderr,
            "output: %s | w: %d | h: %d | fps: %d | image_count: %d | optind: "
            "%d | argc: %d\n",
            output_filename ? output_filename : "NULL", width, height, fps,
            image_count, optind, argc);
    print_usage(argv[0]);
    return 1;
  }

  if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
    fprintf(stderr, "[ERROR] Width/height must not exceed %d.\n", MAX_DIMENSION);
    return 1;
  }

  if (sigma_clip_threshold < 0.0f)
    sigma_clip_threshold = mode == STACK_MOTION ? DEFAULT_MOTION_THRESHOLD
                                                : DEFAULT_SIGMA_THRESHOLD;

  /* Pick the images before creating the output file */
  int ref_width, ref_height;
  const char **files =
      (const char **)allsky_safe_malloc(sizeof(char *) * image_count);
  image_count = select_images(&argv[optind], image_count, mode != STACK_NONE,
                              files, &ref_width, &ref_height);
  if (image_count == 0) {
    fprintf(stderr, "[ERROR] No usable input images.\n");
    return 1;
  }

  if (init_encoder(output_filename, width, height, fps) < 0) {
    fprintf(stderr, "Encoder initialization failed.\n");
    return 1;
  }


  if (mode == STACK_MOTION) {
    size_t buffer_size = (size_t)width * height * 3;
    size_t n_pixels    = buffer_size;
    uint8_t *median_bg = (uint8_t *)allsky_safe_malloc(buffer_size);
    uint8_t *out_frame = (uint8_t *)allsky_safe_malloc(buffer_size);
    int loaded = 0;

#ifdef USE_CUDA
    // ── GPU-Pfad ────────────────────────────────────────────────────────────
    if (cuda_stacker_init(n_pixels) != 0) {
      fprintf(stderr, "CUDA stacker init failed, aborting.\n");
      return 1;
    }

    printf("Pass 1 (GPU): Building median histogram...\n");
    for (int i = 0; i < image_count; ++i) {
      uint8_t *rgb = load_and_resize_jpeg(files[i], width, height, ref_width, ref_height);
      if (!rgb) { fprintf(stderr, "[WARN] Skipping %s\n", files[i]); continue; }
      cuda_update_pixel_histogram(rgb, n_pixels);
      allsky_safe_free(rgb);
      ++loaded;
      if (verbose) print_progress_bar(loaded, image_count);
    }

    printf("\nPass 1 (GPU): Computing median from %d frames...\n", loaded);
    cuda_build_median_from_histogram(loaded, n_pixels, median_bg);

    printf("Pass 2 (GPU): Encoding motion video...\n");
    int frame_idx = 0;
    for (int i = 0; i < image_count; ++i) {
      uint8_t *rgb = load_and_resize_jpeg(files[i], width, height, ref_width, ref_height);
      if (!rgb) { fprintf(stderr, "[WARN] Skipping %s\n", files[i]); continue; }
      cuda_update_motion(rgb, median_bg, sigma_clip_threshold, n_pixels);
      cuda_export_motion(n_pixels, out_frame);
      encode_frame(out_frame);
      allsky_safe_free(rgb);
      ++frame_idx;
      if (verbose) print_progress_bar(frame_idx, loaded);
    }

    finalize_encoder();

    if (stack_output_filename) {
      cuda_export_motion(n_pixels, out_frame);
      if (save_rgb8_as_png(out_frame, width, height, 3, stack_output_filename) == 0)
        printf("\n✅ Final motion image saved as PNG: %s\n", stack_output_filename);
      else
        fprintf(stderr, "\n[ERROR] Failed to write PNG file.\n");
    }

    cuda_stacker_free();

#else
    // ── CPU-Pfad ────────────────────────────────────────────────────────────
    size_t hist_bytes = n_pixels * 256 * sizeof(uint16_t);
    printf("Pass 1 (CPU): Building median histogram (%.0f MB)...\n",
           (double)hist_bytes / (1024.0 * 1024.0));
    uint16_t *hist = (uint16_t *)calloc(n_pixels * 256, sizeof(uint16_t));
    if (!hist) {
      fprintf(stderr, "Out of memory for histogram (%.0f MB needed).\n",
              (double)hist_bytes / (1024.0 * 1024.0));
      return 1;
    }

    for (int i = 0; i < image_count; ++i) {
      uint8_t *rgb = load_and_resize_jpeg(files[i], width, height, ref_width, ref_height);
      if (!rgb) { fprintf(stderr, "[WARN] Skipping %s\n", files[i]); continue; }
      update_pixel_histogram(hist, rgb, n_pixels);
      allsky_safe_free(rgb);
      ++loaded;
      if (verbose) print_progress_bar(loaded, image_count);
    }

    printf("\nPass 1 (CPU): Computing median from %d frames...\n", loaded);
    build_median_from_histogram(hist, loaded, n_pixels, median_bg);
    free(hist);

    printf("Pass 2 (CPU): Encoding motion video...\n");
    stack_context_t stack;
    if (init_stack_context(&stack, width, height, mode, sigma_clip_threshold) != 0) {
      fprintf(stderr, "Failed to initialize stack context.\n");
      return 1;
    }
    stack.median_bg = median_bg;

    int frame_idx = 0;
    for (int i = 0; i < image_count; ++i) {
      uint8_t *rgb = load_and_resize_jpeg(files[i], width, height, ref_width, ref_height);
      if (!rgb) { fprintf(stderr, "[WARN] Skipping %s\n", files[i]); continue; }
      update_stack(&stack, rgb);
      export_stacked_rgb(&stack, out_frame);
      encode_frame(out_frame);
      allsky_safe_free(rgb);
      ++frame_idx;
      if (verbose) print_progress_bar(frame_idx, loaded);
    }

    finalize_encoder();

    if (stack_output_filename) {
      if (save_rgb8_as_png(out_frame, width, height, 3, stack_output_filename) == 0)
        printf("\n✅ Final motion image saved as PNG: %s\n", stack_output_filename);
      else
        fprintf(stderr, "\n[ERROR] Failed to write PNG file.\n");
    }

    free_stack_context(&stack);
#endif

    allsky_safe_free(median_bg);
    allsky_safe_free(out_frame);

  } else if (mode != STACK_NONE) {
    stack_context_t stack;
    if (init_stack_context(&stack, width, height, mode, sigma_clip_threshold) !=
        0) {
      fprintf(stderr, "Failed to initialize stack context.\n");
      return 1;
    }

    size_t buffer_size = (size_t)width * height * 3;
    uint8_t *out_frame = (uint8_t *)allsky_safe_malloc(buffer_size);

    int frame_idx = 0;
    for (int i = 0; i < image_count; ++i) {
      uint8_t *rgb_data =
          load_and_resize_jpeg(files[i], width, height, ref_width, ref_height);
      if (!rgb_data) {
        fprintf(stderr, "\n[WARN] Skipping image %s\n", files[i]);
        continue;
      }
      ++frame_idx;
      update_stack(&stack, rgb_data);
      export_stacked_rgb(&stack, out_frame);

      encode_frame(out_frame);
      allsky_safe_free(rgb_data);
      if (verbose)
        print_progress_bar(frame_idx, image_count);
    }

    finalize_encoder();

    if (stack_output_filename) {
      if (save_rgb8_as_png(out_frame, width, height, 3,
                           stack_output_filename) == 0) {
        printf("\n✅ Final stacked image saved as PNG: %s\n",
               stack_output_filename);
      } else {
        fprintf(stderr, "\n[ERROR] Failed to write PNG file.\n");
      }
    }

    free_stack_context(&stack);
    allsky_safe_free(out_frame);
  } else {
    uint8_t *last_frame = NULL;
    for (int i = 0; i < image_count; ++i) {
      if (verbose)
        print_progress_bar(i + 1, image_count);

      uint8_t *rgb_data =
          load_and_resize_jpeg(files[i], width, height, ref_width, ref_height);
      if (!rgb_data) {
        fprintf(stderr, "\nError loading image %s\n", files[i]);
        continue;
      }

      encode_frame(rgb_data);
      if (stack_output_filename) {
        allsky_safe_free(last_frame);
        last_frame = rgb_data;
      } else {
        allsky_safe_free(rgb_data);
      }
    }

    finalize_encoder();

    if (stack_output_filename && last_frame) {
      if (save_rgb8_as_png(last_frame, width, height, 3,
                           stack_output_filename) == 0) {
        printf("\n✅ Last frame saved as PNG: %s\n", stack_output_filename);
      } else {
        fprintf(stderr, "\n[ERROR] Failed to write PNG file.\n");
      }
      allsky_safe_free(last_frame);
    }
  }

  if (verbose) {
    print_progress_bar(image_count, image_count);
    printf("\nDone.\n");
  }

  allsky_safe_free(files);
  return 0;
}

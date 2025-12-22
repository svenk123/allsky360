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
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void print_usage(const char *progname) {
  fprintf(
      stderr,
      "Usage: %s -o output.mp4 -w width -h height -p fps [-v] image1.jpg ...\n",
      progname);
  fprintf(stderr, "  -m avg|max|min|sigma|diff|motion\n");
  fprintf(stderr, "  -t <threshold> for sigma clipping (default: 2.0)\n");
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
  float sigma_clip_threshold = 2.0f;
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

  if (init_encoder_nvenc(output_filename, width, height, fps) < 0) {
    fprintf(stderr, "Encoder initialization failed.\n");
    return 1;
  }

  int ref_width, ref_height;
  if (get_jpeg_dimensions(argv[optind], &ref_width, &ref_height) != 0) {
    fprintf(stderr, "Failed to read first image dimensions.\n");
    return 1;
  }

  if (mode != STACK_NONE) {
    stack_context_t stack;
    if (init_stack_context(&stack, width, height, mode, sigma_clip_threshold) !=
        0) {
      fprintf(stderr, "Failed to initialize stack context.\n");
      return 1;
    }

    size_t buffer_size = width * height * 3;
    uint8_t *out_frame = (uint8_t *)allsky_safe_malloc(buffer_size);

    int frame_idx = 0;
    for (int i = optind; i < argc; ++i) {
      uint8_t *rgb_data =
          load_and_resize_jpeg(argv[i], width, height, ref_width, ref_height);
      if (!rgb_data) {
        fprintf(stderr, "\n[WARN] Skipping image %s\n", argv[i]);
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
    for (int i = optind; i < argc; ++i) {
      if (verbose)
        print_progress_bar(i - optind + 1, image_count);

      uint8_t *rgb_data =
          load_and_resize_jpeg(argv[i], width, height, ref_width, ref_height);
      if (!rgb_data) {
        fprintf(stderr, "\nError loading image %s\n", argv[i]);
        continue;
      }

      encode_frame(rgb_data);
      if (rgb_data) {
        allsky_safe_free(rgb_data);
        rgb_data = NULL;
      }
    }

    finalize_encoder();
  }

  if (verbose) {
    print_progress_bar(image_count, image_count);
    printf("\nDone.\n");
  }

  return 0;
}

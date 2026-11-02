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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cairo.h>
#include "allsky_panorama.h"
#include "jpeg_to_cairo.h"
#include "png_to_cairo.h"

#define PROG_VERSION "1.0"
#define DEFAULT_JPEG_QUALITY 100

static void print_usage(const char *prog_name __attribute__((unused))) {
    printf("panorama v%s\n", PROG_VERSION);
    printf("Copyright (c) 2025 Sven Kreiensen\n");
    printf("\nConverts an allsky image (PNG or JPEG) to a panorama.\n");
    printf("\nCommand line parameters:\n");
    printf("  --infile <file>     Input image (PNG or JPEG)\n");
    printf("  --outfile <file>    Output image (JPEG)\n");
    printf("  --cx <value>        X coordinate of image center\n");
    printf("  --cy <value>        Y coordinate of image center\n");
    printf("  --theta <value>     Start angle in degrees (theta_start_deg)\n");
    printf("  --width <value>     Width of panorama in pixels\n");
    printf("  --height <value>    Height of panorama in pixels\n");
    printf("  --quality <value>   JPEG quality (0-100, default: %d)\n", DEFAULT_JPEG_QUALITY);
    printf("  --mask <file>       Optional mask image (JPEG, must match input dimensions)\n");
    printf("  --help, -h, -?      Show this help\n");
}

static int get_file_extension(const char *filename, char *ext, size_t ext_size) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) {
        return 0;
    }
    
    size_t len = strlen(dot + 1);
    if (len >= ext_size) {
        return 0;
    }
    
    strncpy(ext, dot + 1, ext_size - 1);
    ext[ext_size - 1] = '\0';
    
    // Convert to lowercase
    for (size_t i = 0; i < len; i++) {
        if (ext[i] >= 'A' && ext[i] <= 'Z') {
            ext[i] = ext[i] - 'A' + 'a';
        }
    }
    
    return 1;
}

static void apply_mask(cairo_surface_t *input, cairo_surface_t *mask) {
    int width = cairo_image_surface_get_width(input);
    int height = cairo_image_surface_get_height(input);
    int input_stride = cairo_image_surface_get_stride(input);
    int mask_stride = cairo_image_surface_get_stride(mask);
    unsigned char *input_data = cairo_image_surface_get_data(input);
    unsigned char *mask_data = cairo_image_surface_get_data(mask);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int input_idx = y * input_stride + x * 4;
            int mask_idx = y * mask_stride + x * 4;
            
            // Multiply RGB channels (normalized to 0-255)
            for (int c = 0; c < 3; c++) {
                unsigned char input_val = input_data[input_idx + c];
                unsigned char mask_val = mask_data[mask_idx + c];
                // Multiplicative blending: result = (input * mask) / 255
                input_data[input_idx + c] = (unsigned char)((input_val * mask_val) / 255);
            }
            // Keep alpha channel from input
        }
    }
    
    cairo_surface_mark_dirty(input);
}

int main(int argc, const char *argv[]) {
    const char *infile = NULL;
    const char *outfile = NULL;
    const char *mask_file = NULL;
    int cx = 0;
    int cy = 0;
    double theta = 0.0;
    int width = 0;
    int height = 0;
    int quality = DEFAULT_JPEG_QUALITY;
    
    int cx_set = 0, cy_set = 0, theta_set = 0, width_set = 0, height_set = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--infile") == 0) {
            if (i + 1 < argc) {
                infile = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --infile requires a filename\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--outfile") == 0) {
            if (i + 1 < argc) {
                outfile = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --outfile requires a filename\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--cx") == 0) {
            if (i + 1 < argc) {
                cx = atoi(argv[i + 1]);
                cx_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --cx requires a value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--cy") == 0) {
            if (i + 1 < argc) {
                cy = atoi(argv[i + 1]);
                cy_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --cy requires a value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--theta") == 0) {
            if (i + 1 < argc) {
                theta = atof(argv[i + 1]);
                theta_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --theta requires a value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--width") == 0) {
            if (i + 1 < argc) {
                width = atoi(argv[i + 1]);
                width_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --width requires a value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--height") == 0) {
            if (i + 1 < argc) {
                height = atoi(argv[i + 1]);
                height_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --height requires a value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--quality") == 0) {
            if (i + 1 < argc) {
                quality = atoi(argv[i + 1]);
                i++;
            } else {
                fprintf(stderr, "Error: --quality requires a value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--mask") == 0) {
            if (i + 1 < argc) {
                mask_file = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --mask requires a filename\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--help") == 0 || 
                   strcmp(argv[i], "-h") == 0 || 
                   strcmp(argv[i], "-?") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Unknown parameter: %s\n", argv[i]);
            fprintf(stderr, "Use --help for help\n");
            return EXIT_FAILURE;
        }
    }
    
    // Validate required parameters
    if (!infile) {
        fprintf(stderr, "Error: --infile is required\n");
        fprintf(stderr, "Use --help for help\n");
        return EXIT_FAILURE;
    }
    
    if (!outfile) {
        fprintf(stderr, "Error: --outfile is required\n");
        fprintf(stderr, "Use --help for help\n");
        return EXIT_FAILURE;
    }
    
    if (!cx_set || !cy_set || !theta_set || !width_set || !height_set) {
        fprintf(stderr, "Error: All parameters --cx, --cy, --theta, --width and --height are required\n");
        fprintf(stderr, "Use --help for help\n");
        return EXIT_FAILURE;
    }
    
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: --width and --height must be greater than 0\n");
        return EXIT_FAILURE;
    }
    
    if (quality < 0 || quality > 100) {
        fprintf(stderr, "Error: --quality must be between 0 and 100\n");
        return EXIT_FAILURE;
    }
    
    // Load input image
    cairo_surface_t *input_surface = NULL;
    char ext[16];
    
    if (!get_file_extension(infile, ext, sizeof(ext))) {
        fprintf(stderr, "Error: Cannot determine file extension: %s\n", infile);
        return EXIT_FAILURE;
    }
    
    if (strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0) {
        if (!load_jpeg_as_cairo_surface(&input_surface, infile)) {
            fprintf(stderr, "Error: Cannot load JPEG file: %s\n", infile);
            return EXIT_FAILURE;
        }
    } else if (strcmp(ext, "png") == 0) {
        if (!load_png_as_cairo_surface(&input_surface, infile)) {
            fprintf(stderr, "Error: Cannot load PNG file: %s\n", infile);
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Error: Unknown file format: %s (expected: PNG or JPEG)\n", ext);
        return EXIT_FAILURE;
    }
    
    if (!input_surface) {
        fprintf(stderr, "Error: Input image could not be loaded\n");
        return EXIT_FAILURE;
    }
    
    printf("Input image loaded: %s\n", infile);
    int input_width = cairo_image_surface_get_width(input_surface);
    int input_height = cairo_image_surface_get_height(input_surface);
    printf("Image size: %d x %d\n", input_width, input_height);
    
    // Load and apply mask if provided
    if (mask_file) {
        cairo_surface_t *mask_surface = NULL;
        
        if (!load_jpeg_as_cairo_surface(&mask_surface, mask_file)) {
            fprintf(stderr, "Error: Cannot load mask JPEG file: %s\n", mask_file);
            cairo_surface_destroy(input_surface);
            return EXIT_FAILURE;
        }
        
        if (!mask_surface) {
            fprintf(stderr, "Error: Mask image could not be loaded\n");
            cairo_surface_destroy(input_surface);
            return EXIT_FAILURE;
        }
        
        int mask_width = cairo_image_surface_get_width(mask_surface);
        int mask_height = cairo_image_surface_get_height(mask_surface);
        
        if (mask_width != input_width || mask_height != input_height) {
            fprintf(stderr, "Error: Mask dimensions (%d x %d) do not match input image dimensions (%d x %d)\n",
                    mask_width, mask_height, input_width, input_height);
            cairo_surface_destroy(input_surface);
            cairo_surface_destroy(mask_surface);
            return EXIT_FAILURE;
        }
        
        printf("Applying mask: %s\n", mask_file);
        apply_mask(input_surface, mask_surface);
        cairo_surface_destroy(mask_surface);
    }
    
    // Convert to panorama
    printf("Converting to panorama: %d x %d (center: %d, %d, theta: %.2f°)\n",
           width, height, cx, cy, theta);
    
    cairo_surface_t *panorama_surface = convert_allsky_to_panorama(
        input_surface, width, height, cx, cy, theta);
    
    if (!panorama_surface) {
        fprintf(stderr, "Error: Panorama conversion failed\n");
        cairo_surface_destroy(input_surface);
        return EXIT_FAILURE;
    }
    
    // Save panorama as JPEG
    if (!save_cairo_surface_as_jpeg(panorama_surface, outfile, quality)) {
        fprintf(stderr, "Error: Cannot save panorama: %s\n", outfile);
        cairo_surface_destroy(input_surface);
        cairo_surface_destroy(panorama_surface);
        return EXIT_FAILURE;
    }
    
    printf("Panorama successfully saved: %s\n", outfile);
    
    // Cleanup
    cairo_surface_destroy(input_surface);
    cairo_surface_destroy(panorama_surface);
    
    return EXIT_SUCCESS;
}


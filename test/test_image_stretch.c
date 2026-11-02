#include <stdio.h>
#include "jpeg_to_cairo.h"
#include "image_stretch.h"

// Include necessary functions for loading/saving images
extern cairo_surface_t *load_jpeg_as_cairo_surface(const char *filename);
extern int save_cairo_surface_as_jpeg(cairo_surface_t *surface, const char *output_filename);

int main() {
    // Load a sample JPEG image into a Cairo surface
    cairo_surface_t *surface = load_jpeg_as_cairo_surface("test_image.jpg");

    if (!surface) {
        fprintf(stderr, "Error: Failed to load test_image.jpg\n");
        return 1;
    }

    printf("Testing stretch_image_brightness()...\n");

    // Apply brightness stretching
    stretch_image_brightness(&surface, 0.5, 0.1, 0.9);

    // Save the modified image
    if (save_cairo_surface_as_jpeg(surface, "stretched_image.jpg") != 0) {
        fprintf(stderr, "Error: Failed to save stretched_image.jpg\n");
        return 1;
    }

    // Free memory
    cairo_surface_destroy(surface);

    printf("Test completed. Stretched image saved as stretched_image.jpg\n");
    return 0;
}

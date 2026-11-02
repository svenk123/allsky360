#include <stdio.h>
#include <cairo.h>
#include "fits_to_cairo.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Verwendung: %s <input.fits> <output.png>\n", argv[0]);
        return 1;
    }

    cairo_surface_t *surface = NULL;

    // 🔹 FITS laden
    if (!load_fits_as_cairo_surface(&surface, argv[1])) {
        fprintf(stderr, "Fehler beim Laden von: %s\n", argv[1]);
        return 1;
    }

    // 🔹 Als PNG speichern
    cairo_surface_write_to_png(surface, argv[2]);

    cairo_surface_destroy(surface);
    printf("FITS erfolgreich geladen und als PNG gespeichert: %s\n", argv[2]);
    return 0;
}

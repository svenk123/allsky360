#include <stdio.h>
#include <cairo.h>
#include "png_to_cairo.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Verwendung: %s <input.png> <output.png> <kompression (0-9)>\n", argv[0]);
        return 1;
    }

    cairo_surface_t *surface = NULL;

    // 🔹 PNG laden
    if (!load_png_as_cairo_surface(&surface, argv[1])) {
        fprintf(stderr, "Fehler beim Laden von: %s\n", argv[1]);
        return 1;
    }

    // 🔹 Als PNG speichern
    int compression = atoi(argv[3]);
    if (!save_cairo_surface_as_png(surface, argv[2], compression)) {
        fprintf(stderr, "Fehler beim Speichern von: %s\n", argv[2]);
        return 1;
    }

    cairo_surface_destroy(surface);
    printf("PNG erfolgreich gespeichert: %s mit Kompression %d\n", argv[2], compression);
    return 0;
}

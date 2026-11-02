#include <stdio.h>
#include <cairo.h>
#include "jpeg_to_cairo.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Verwendung: %s <input.jpg> <output.jpg> <qualität (0-100)>\n", argv[0]);
        return 1;
    }

    cairo_surface_t *surface = NULL;

    // 🔹 JPEG laden
    if (!load_jpeg_as_cairo_surface(argv[1], &surface)) {
        fprintf(stderr, "Fehler beim Laden von: %s\n", argv[1]);
        return 1;
    }

    // 🔹 Als JPEG speichern
    int quality = atoi(argv[3]);
    if (!save_cairo_surface_as_jpeg(surface, argv[2], quality)) {
        fprintf(stderr, "Fehler beim Speichern von: %s\n", argv[2]);
        return 1;
    }

    cairo_surface_destroy(surface);
    printf("JPEG erfolgreich gespeichert: %s mit Qualität %d\n", argv[2], quality);
    return 0;
}

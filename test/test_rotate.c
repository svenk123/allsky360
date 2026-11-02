#include <stdio.h>
#include <cairo.h>
#include "rotate_image.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Verwendung: %s <input.png> <output.png> <winkel>\n", argv[0]);
        return 1;
    }

    int angle = atoi(argv[3]);  // Rotationswinkel aus Argument

    // 🔹 Bild als Cairo-Surface laden
    cairo_surface_t *surface = cairo_image_surface_create_from_png(argv[1]);
    if (cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Fehler: Konnte Bild nicht laden: %s\n", argv[1]);
        return 1;
    }

    // 🔹 Bild rotieren (per Referenz)
    rotate_surface(&surface, angle);

    // 🔹 Rotiertes Bild speichern
    cairo_surface_write_to_png(surface, argv[2]);

    // 🔹 Speicher freigeben
    cairo_surface_destroy(surface);

    printf("Bild erfolgreich gedreht: %s (%d°)\n", argv[2], angle);
    return 0;
}

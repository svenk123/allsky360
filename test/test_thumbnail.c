#include <stdio.h>
#include "thumbnail.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Verwendung: %s <input.png> <output.png> <breite>\n", argv[0]);
        return 1;
    }

    int width = atoi(argv[3]);  // Thumbnail-Breite aus Argument
    return create_thumbnail(argv[1], argv[2], width);
}

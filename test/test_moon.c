#include <stdio.h>
#include <time.h>
#include "moon_calc.h"

int main() {
    time_t now = time(NULL);
    double latitude = 50.0;
    double longitude = 8.0;

    printf("Mondhöhe: %.2f°\n", moon_altitude(now, longitude, latitude));
    printf("Mondphase: %s\n", moon_phase(now));
    printf("Mondphase in Prozent: %.2f%%\n", moon_phase_percentage(now));

    return 0;
}

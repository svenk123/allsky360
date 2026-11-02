/* test_sun_calc.c - Unit Test für calculate_sun_altitude() */
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "../src/sun_calc.h"

void test_is_sun_below_6_degrees() {
    // Testfälle mit verschiedenen Breitengraden
    time_t now = time(NULL);

    printf("\nTest für Berlin (52.52° N, 13.40° E):\n");
    is_sun_below_6_degrees(now, 13.40, 52.52);

    printf("\nTest für Reykjavik (64.14° N, -21.94° E):\n");
    is_sun_below_6_degrees(now, -21.94, 64.14);

    printf("\nTest für Tromsø (69.65° N, 18.96° E):\n");
    is_sun_below_6_degrees(now, 18.96, 69.65);

    printf("\nTest für Kapstadt (-33.92° S, 18.42° E):\n");
    is_sun_below_6_degrees(now, 18.42, -33.92);

    int year = 2024, month = 6, day = 21; // Beispiel: Sommersonnenwende
    double latitude = 50.0, longitude = 8.0;

    time_t dawn, dusk;

    if (calculate_civil_twilight(year, month, day, longitude, latitude, &dawn, &dusk) == 0) {
        struct tm *dawn_tm = gmtime(&dawn);
        struct tm *dusk_tm = gmtime(&dusk);
        
        printf("Bürgerliche Dämmerung beginnt: %02d:%02d UTC\n", dawn_tm->tm_hour, dawn_tm->tm_min);
        printf("Bürgerliche Dämmerung endet: %02d:%02d UTC\n", dusk_tm->tm_hour, dusk_tm->tm_min);
    } else {
        printf("Kein Beginn/Ende der bürgerlichen Dämmerung an diesem Tag.\n");
    }
}

int main() {
    test_is_sun_below_6_degrees();
    return 0;
}

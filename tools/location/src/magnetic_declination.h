// magnetic_declination.h
#ifndef MAGNETIC_DECLINATION_H
#define MAGNETIC_DECLINATION_H

#include <time.h>

// Structure representing a single WMM coefficient
typedef struct {
    int n;
    int m;
    double gnm;
    double hnm;
    double gnm_dot;
    double hnm_dot;
} wmm_coefficient_t;

// Calculate magnetic declination for given coordinates and time
// Parameters:
//   coeffs: Pointer to WMM coefficient array
//   coeff_count: Number of coefficients in the array
//   lon: Longitude in degrees (positive east)
//   lat: Latitude in degrees (positive north)
//   timestamp: Unix time (UTC)
//   declination: Pointer to double where result will be stored (in degrees)
// Returns:
//   0 on success, >0 on failure
int get_magnetic_declination(const wmm_coefficient_t *coeffs, int coeff_count,
                             double lon, double lat, time_t timestamp,
                             double *declination);

extern const wmm_coefficient_t WMM_Coefficients[];
extern const int WMM_Coefficient_Count;

#endif // MAGNETIC_DECLINATION_H

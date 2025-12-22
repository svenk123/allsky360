// magnetic_declination.c
#include <math.h>
#include <stdio.h>
#include "magnetic_declination.h"

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define WMM_MAX_N 12

#define A 6378.137              // equatorial radius [km]
#define B 6356.7523142          // polar radius [km]
#define RE 6371.2               // reference radius for WMM [km]
#define A2 (A * A)
#define B2 (B * B)
#define A4 (A2 * A2)
#define B4 (B2 * B2)
#define C2 (A2 - B2)
#define C4 (A4 - B4)

const wmm_coefficient_t WMM_Coefficients[] = {
    {1, 0, -29351.8, 0.0, 12.0, 0.0},
    {1, 1, -1410.8, 4545.4, 9.7, -21.5},
    {2, 0, -2556.6, 0.0, -11.6, 0.0},
    {2, 1, 2951.1, -3133.6, -5.2, -27.7},
    {2, 2, 1649.3, -815.1, -8.0, -12.1},
    {3, 0, 1361.0, 0.0, -1.3, 0.0},
    {3, 1, -2404.1, -56.6, -4.2, 4.0},
    {3, 2, 1243.8, 237.5, 0.4, -0.3},
    {3, 3, 453.6, -549.5, -15.6, -4.1},
    {4, 0, 895.0, 0.0, -1.6, 0.0},
    {4, 1, 799.5, 278.6, -2.4, -1.1},
    {4, 2, 55.7, -133.9, -6.0, 4.1},
    {4, 3, -281.1, 212.0, 5.6, 1.6},
    {4, 4, 12.1, -375.6, -7.0, -4.4},
    {5, 0, -233.2, 0.0, 0.6, 0.0},
    {5, 1, 368.9, 45.4, 1.4, -0.5},
    {5, 2, 187.2, 220.2, 0.0, 2.2},
    {5, 3, -138.7, -122.9, 0.6, 0.4},
    {5, 4, -142.0, 43.0, 2.2, 1.7},
    {5, 5, 20.9, 106.1, 0.9, 1.9},
    {6, 0, 64.4, 0.0, -0.2, 0.0},
    {6, 1, 63.8, -18.4, -0.4, 0.3},
    {6, 2, 76.9, 16.8, 0.9, -1.6},
    {6, 3, -115.7, 48.8, 1.2, -0.4},
    {6, 4, -40.9, -59.8, -0.9, 0.9},
    {6, 5, 14.9, 10.9, 0.3, 0.7},
    {6, 6, -60.7, 72.7, 0.9, 0.9},
    {7, 0, 79.5, 0.0, 0.0, 0.0},
    {7, 1, -77.0, -48.9, -0.1, 0.6},
    {7, 2, -8.8, -14.4, -0.1, 0.5},
    {7, 3, 59.3, -1.0, 0.5, -0.8},
    {7, 4, 15.8, 23.4, -0.1, 0.0},
    {7, 5, 2.5, -7.4, -0.8, -1.0},
    {7, 6, -11.1, -25.1, -0.8, 0.6},
    {7, 7, 14.2, -2.3, 0.8, -0.2},
    {8, 0, 23.2, 0.0, -0.1, 0.0},
    {8, 1, 10.8, 7.1, 0.2, -0.2},
    {8, 2, -17.5, -12.6, 0.0, 0.5},
    {8, 3, 2.0, 11.4, 0.5, -0.4},
    {8, 4, -21.7, -9.7, -0.1, 0.4},
    {8, 5, 16.9, 12.7, 0.3, -0.5},
    {8, 6, 15.0, 0.7, 0.2, -0.6},
    {8, 7, -16.8, -5.2, 0.0, 0.3},
    {8, 8, 0.9, 3.9, 0.2, 0.2},
    {9, 0, 4.6, 0.0, 0.0, 0.0},
    {9, 1, 7.8, -24.8, -0.1, -0.3},
    {9, 2, 3.0, 12.2, 0.1, 0.3},
    {9, 3, -0.2, 8.3, 0.3, -0.3},
    {9, 4, -2.5, -3.3, -0.3, 0.3},
    {9, 5, -13.1, -5.2, 0.0, 0.2},
    {9, 6, 2.4, 7.2, 0.3, -0.1},
    {9, 7, 8.6, -0.6, -0.1, -0.2},
    {9, 8, -8.7, 0.8, 0.1, 0.4},
    {9, 9, -12.9, 10.0, -0.1, 0.1},
    {10, 0, -1.3, 0.0, 0.0, 0.0},
    {10, 1, -0.2, 0.4, 0.0, 0.0},
    {10, 2, -0.4, -0.4, 0.0, 0.0},
    {10, 3, -0.9, -0.2, 0.0, 0.0},
    {10, 4, 0.1, -0.6, 0.0, 0.0},
    {10, 5, 0.5, -0.1, 0.0, 0.0},
    {10, 6, -0.3, 0.2, 0.0, 0.0},
    {10, 7, -0.4, 0.3, 0.0, 0.0},
    {10, 8, 0.0, -0.2, 0.0, 0.0},
    {10, 9, 0.4, -0.3, 0.0, 0.0},
    {10, 10, -0.6, 0.1, 0.0, 0.0},
    {11, 0, 0.1, 0.0, 0.0, 0.0},
    {11, 1, 0.0, 0.0, 0.0, 0.0},
    {11, 2, 0.0, 0.0, 0.0, 0.0},
    {11, 3, 0.0, 0.0, 0.0, 0.0},
    {11, 4, 0.0, 0.0, 0.0, 0.0},
    {11, 5, 0.0, 0.0, 0.0, 0.0},
    {11, 6, 0.0, 0.0, 0.0, 0.0},
    {11, 7, 0.0, 0.0, 0.0, 0.0},
    {11, 8, 0.0, 0.0, 0.0, 0.0},
    {11, 9, 0.0, 0.0, 0.0, 0.0},
    {11, 10, 0.0, 0.0, 0.0, 0.0},
    {11, 11, 0.0, 0.0, 0.0, 0.0},
    {12, 0, 0.0, 0.0, 0.0, 0.0},
    {12, 1, 0.0, 0.0, 0.0, 0.0},
    {12, 2, 0.0, 0.0, 0.0, 0.0},
    {12, 3, 0.0, 0.0, 0.0, 0.0},
    {12, 4, 0.0, 0.0, 0.0, 0.0},
    {12, 5, 0.0, 0.0, 0.0, 0.0},
    {12, 6, 0.0, 0.0, 0.0, 0.0},
    {12, 7, 0.0, 0.0, 0.0, 0.0},
    {12, 8, 0.0, 0.0, 0.0, 0.0},
    {12, 9, 0.0, 0.0, 0.0, 0.0},
    {12, 10, 0.0, 0.0, 0.0, 0.0},
    {12, 11, 0.0, 0.0, 0.0, 0.0},
    {12, 12, 0.0, 0.0, 0.0, 0.0}
};

const int WMM_Coefficient_Count = sizeof(WMM_Coefficients) / sizeof(WMM_Coefficients[0]);

// Konvertiert Unix-Zeit in dezimales Jahr
static double unix_to_decimal_year(time_t t) {
    struct tm *ptm = gmtime(&t);
    int year = ptm->tm_year + 1900;
    int yday = ptm->tm_yday + 1;
    int days_in_year = (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) ? 366 : 365;
    return year + (double)yday / days_in_year;
}

int get_magnetic_declination(const wmm_coefficient_t *coeffs, int coeff_count,
                             double lon_deg, double lat_deg, time_t timestamp,
                             double *declination_out)
{
    if (!coeffs || coeff_count <= 0 || !declination_out)
        return 1;

    // Basisdaten vorbereiten
    double decimal_year = unix_to_decimal_year(timestamp);
    double dt = decimal_year - 2025.0;

    double rlat = lat_deg * DEG2RAD;
    double rlon = lon_deg * DEG2RAD;
    double srlat = sin(rlat), crlat = cos(rlat);
    double srlat2 = srlat * srlat;
    double crlat2 = crlat * crlat;

    // Geodätisch → sphärisch
    double q = sqrt(A2 - C2 * srlat2);
    double q2 = (A2 / B2) * (A2 / B2);
    double ct = srlat / sqrt(q2 * crlat2 + srlat2);
    double st = sqrt(1.0 - ct * ct);
    double r = sqrt((A4 - C4 * srlat2) / (q * q));
    double d = sqrt(A2 * crlat2 + B2 * srlat2);
    double ca = d / r;
    double sa = C2 * crlat * srlat / (r * d);

    double sp[13], cp[13], pp[13] = {0}, snorm[13 * 13] = {0}, dp[13][13] = {{0}};
    double k[13][13] = {{0}}, tc[13][13] = {{0}}, fn[13] = {0}, fm[13] = {0};
    sp[0] = 0.0; sp[1] = sin(rlon);
    cp[0] = 1.0; cp[1] = cos(rlon);
    for (int m = 2; m <= WMM_MAX_N; m++) {
        sp[m] = sp[1] * cp[m - 1] + cp[1] * sp[m - 1];
        cp[m] = cp[1] * cp[m - 1] - sp[1] * sp[m - 1];
    }

    // Initialisierung snorm, k, fn, fm
    snorm[0] = 1.0;
    for (int n = 1; n <= WMM_MAX_N; n++) {
        snorm[n + 0 * 13] = snorm[n - 1 + 0 * 13] * (2.0 * n - 1.0) / n;
        for (int m = 1; m <= n; m++) {
            int idx = n + m * 13;
            int idx1 = n + (m - 1) * 13;
            double factor = ((n - m + 1.0) * (m == 1 ? 2.0 : 1.0)) / (n + m);
            snorm[idx] = snorm[idx1] * sqrt(factor);
            k[m][n] = ((n - m + 1.0) * 1.0) / (n + m);
        }
        fn[n] = n + 1.0;
        fm[n] = n;
    }

    // Zeitkorrigierte Koeffizienten in tc
    for (int i = 0; i < coeff_count; i++) {
        int n = coeffs[i].n;
        int m = coeffs[i].m;
        if (m <= n && n <= WMM_MAX_N) {
            tc[m][n] = coeffs[i].gnm + coeffs[i].gnm_dot * dt;
            if (m != 0)
                tc[n][m - 1] = coeffs[i].hnm + coeffs[i].hnm_dot * dt;
        }
    }

    double aor = RE / r;
    double ar = aor;
    double br = 0.0, bt = 0.0, bp = 0.0, bpp = 0.0;

    for (int n = 1; n <= WMM_MAX_N; n++) {
        ar *= aor;
        for (int m = 0; m <= n; m++) {
            int idx = n + m * 13;
            if (n == m) {
                snorm[idx] = st * snorm[n - 1 + (m - 1) * 13];
                dp[m][n] = st * dp[m - 1][n - 1] + ct * snorm[n - 1 + (m - 1) * 13];
            } else if (n == 1 && m == 0) {
                snorm[idx] = ct * snorm[n - 1 + m * 13];
                dp[m][n] = ct * dp[m][n - 1] - st * snorm[n - 1 + m * 13];
            } else {
                if (m > n - 2) snorm[n - 2 + m * 13] = 0.0;
                if (m > n - 2) dp[m][n - 2] = 0.0;
                snorm[idx] = ct * snorm[n - 1 + m * 13] - k[m][n] * snorm[n - 2 + m * 13];
                dp[m][n] = ct * dp[m][n - 1] - st * snorm[n - 1 + m * 13] - k[m][n] * dp[m][n - 2];
            }

            double par = ar * snorm[idx];
            double temp1, temp2;

            if (m == 0) {
                temp1 = tc[m][n] * cp[m];
                temp2 = tc[m][n] * sp[m];
            } else {
                temp1 = tc[m][n] * cp[m] + tc[n][m - 1] * sp[m];
                temp2 = tc[m][n] * sp[m] - tc[n][m - 1] * cp[m];
            }

            bt -= ar * temp1 * dp[m][n];
            double fm_val = (m == 0) ? 1.0 : fm[m];
	    bp += fm_val * temp2 * par;
            br += fn[n] * temp1 * par;

            if (st == 0.0 && m == 1) {
                if (n == 1)
                    pp[n] = pp[n - 1];
                else
                    pp[n] = ct * pp[n - 1] - k[m][n] * pp[n - 2];
                bpp += fm[m] * temp2 * ar * pp[n];
            }
        }
    }

    if (st == 0.0)
        bp = bpp;
    else
        bp /= st;

    // Sphärisch → geodätisch rotieren
    double bx = -bt * ca - br * sa;
    double by = bp;

    *declination_out = atan2(by, bx) * RAD2DEG;

printf(
  "DEBUG: bt=%.2f, br=%.2f, bp=%.2f, bx=%.2f, by=%.2f, decl=%.2f°\n",
  bt, br, bp, bx, by, *declination_out
);

    return 0;
}

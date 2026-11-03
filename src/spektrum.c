/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include "allsky.h"
#include "atomic_file.h"
#include "spektrum.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Sony IMX585 quantum-efficiency curves (estimated).
 *
 * Source: community measurements and interpolation from Sony BSI CMOS data
 * for the IMX585 (similar to IMX455/IMX678 family). These are *relative*
 * spectral response values, not absolute QE percentages. They should be
 * replaced with directly measured values once available.
 *
 * Grid: 400 nm – 700 nm in 10 nm steps (31 values each).
 * -------------------------------------------------------------------------*/

/* Red channel — peak near 600 nm, sensitivity 450–700 nm */
static const float QE_R[SPEKTRUM_BINS] = {
  /* 400 */ 0.02f,
  /* 410 */ 0.03f,
  /* 420 */ 0.04f,
  /* 430 */ 0.05f,
  /* 440 */ 0.06f,
  /* 450 */ 0.07f,
  /* 460 */ 0.08f,
  /* 470 */ 0.09f,
  /* 480 */ 0.10f,
  /* 490 */ 0.10f,
  /* 500 */ 0.10f,
  /* 510 */ 0.11f,
  /* 520 */ 0.13f,
  /* 530 */ 0.16f,
  /* 540 */ 0.22f,
  /* 550 */ 0.30f,
  /* 560 */ 0.42f,
  /* 570 */ 0.55f,
  /* 580 */ 0.68f,
  /* 590 */ 0.78f,
  /* 600 */ 0.84f,
  /* 610 */ 0.88f,
  /* 620 */ 0.88f,
  /* 630 */ 0.86f,
  /* 640 */ 0.82f,
  /* 650 */ 0.74f,
  /* 660 */ 0.63f,
  /* 670 */ 0.50f,
  /* 680 */ 0.36f,
  /* 690 */ 0.23f,
  /* 700 */ 0.12f,
};

/* Green channel — peak near 530 nm, sensitivity 430–650 nm */
static const float QE_G[SPEKTRUM_BINS] = {
  /* 400 */ 0.04f,
  /* 410 */ 0.07f,
  /* 420 */ 0.12f,
  /* 430 */ 0.20f,
  /* 440 */ 0.28f,
  /* 450 */ 0.36f,
  /* 460 */ 0.44f,
  /* 470 */ 0.52f,
  /* 480 */ 0.60f,
  /* 490 */ 0.68f,
  /* 500 */ 0.74f,
  /* 510 */ 0.80f,
  /* 520 */ 0.84f,
  /* 530 */ 0.86f,
  /* 540 */ 0.84f,
  /* 550 */ 0.80f,
  /* 560 */ 0.74f,
  /* 570 */ 0.64f,
  /* 580 */ 0.52f,
  /* 590 */ 0.40f,
  /* 600 */ 0.30f,
  /* 610 */ 0.22f,
  /* 620 */ 0.15f,
  /* 630 */ 0.10f,
  /* 640 */ 0.06f,
  /* 650 */ 0.04f,
  /* 660 */ 0.02f,
  /* 670 */ 0.01f,
  /* 680 */ 0.00f,
  /* 690 */ 0.00f,
  /* 700 */ 0.00f,
};

/* Blue channel — peak near 470 nm, sensitivity 400–560 nm */
static const float QE_B[SPEKTRUM_BINS] = {
  /* 400 */ 0.30f,
  /* 410 */ 0.42f,
  /* 420 */ 0.55f,
  /* 430 */ 0.65f,
  /* 440 */ 0.72f,
  /* 450 */ 0.76f,
  /* 460 */ 0.78f,
  /* 470 */ 0.80f,
  /* 480 */ 0.78f,
  /* 490 */ 0.72f,
  /* 500 */ 0.65f,
  /* 510 */ 0.56f,
  /* 520 */ 0.46f,
  /* 530 */ 0.37f,
  /* 540 */ 0.28f,
  /* 550 */ 0.20f,
  /* 560 */ 0.13f,
  /* 570 */ 0.08f,
  /* 580 */ 0.04f,
  /* 590 */ 0.02f,
  /* 600 */ 0.01f,
  /* 610 */ 0.00f,
  /* 620 */ 0.00f,
  /* 630 */ 0.00f,
  /* 640 */ 0.00f,
  /* 650 */ 0.00f,
  /* 660 */ 0.00f,
  /* 670 */ 0.00f,
  /* 680 */ 0.00f,
  /* 690 */ 0.00f,
  /* 700 */ 0.00f,
};

/* -------------------------------------------------------------------------
 * 3×3 matrix inversion via Cramer's rule.
 * Input:  m[3][3]
 * Output: inv[3][3]
 * Returns 0 on success, 1 if the matrix is (nearly) singular.
 * -------------------------------------------------------------------------*/
static int mat3_invert(const double m[3][3], double inv[3][3]) {
  double det =
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

  if (fabs(det) < 1e-12)
    return 1;

  double inv_det = 1.0 / det;

  inv[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
  inv[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
  inv[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

  inv[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
  inv[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
  inv[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;

  inv[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
  inv[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
  inv[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

  return 0;
}

/* -------------------------------------------------------------------------
 * Public function
 * -------------------------------------------------------------------------*/
int spektrum_analyze(
    const float *rgb,
    int          width,
    int          height,
    int          cx,
    int          cy,
    int          radius,
    float        lambda,
    const char  *output_path)
{
  if (!rgb)
    return SPEKTRUM_ERR_NULL_INPUT;

  if (width <= 0 || height <= 0)
    return SPEKTRUM_ERR_INVALID_DIMS;

  /* Check that the circle overlaps the image at all */
  if (cx + radius < 0 || cx - radius >= width ||
      cy + radius < 0 || cy - radius >= height)
    return SPEKTRUM_ERR_INVALID_ROI;

  printf("[spektrum] Analyse gestartet: cx=%d cy=%d radius=%d\n",
         cx, cy, radius);

  /* ------------------------------------------------------------------
   * Step 1: average R, G, B over the circular ROI
   * ----------------------------------------------------------------*/
  double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
  long   pixel_count = 0;

  int y0 = cy - radius < 0       ? 0       : cy - radius;
  int y1 = cy + radius >= height  ? height - 1 : cy + radius;
  int x0 = cx - radius < 0       ? 0       : cx - radius;
  int x1 = cx + radius >= width   ? width - 1  : cx + radius;

  int r2 = radius * radius;

  for (int y = y0; y <= y1; ++y) {
    int dy = y - cy;
    for (int x = x0; x <= x1; ++x) {
      int dx = x - cx;
      if (dx * dx + dy * dy > r2)
        continue;

      int idx = (y * width + x) * CHANNELS;
      sum_r += rgb[idx + 0];
      sum_g += rgb[idx + 1];
      sum_b += rgb[idx + 2];
      ++pixel_count;
    }
  }

  if (pixel_count == 0)
    return SPEKTRUM_ERR_NO_PIXELS;

  double mean_r = sum_r / (double)pixel_count;
  double mean_g = sum_g / (double)pixel_count;
  double mean_b = sum_b / (double)pixel_count;

  printf("[spektrum] Messbereich: %ld Pixel verwendet\n", pixel_count);
  printf("[spektrum] Mittlere RGB-Werte: R=%.4f G=%.4f B=%.4f\n",
         mean_r, mean_g, mean_b);

  /* ------------------------------------------------------------------
   * Step 2: Wiener spectral reconstruction
   *
   * QE matrix A (3×31):
   *   A[0][j] = QE_R(λ_j)
   *   A[1][j] = QE_G(λ_j)
   *   A[2][j] = QE_B(λ_j)
   *
   * Measurement vector m = [R_mean, G_mean, B_mean]^T
   *
   * M     = A · A^T          (3×3)
   * M_reg = M + λ·I          (Tikhonov regularisation)
   * S     = A^T · M_reg⁻¹ · m  (31-element spectrum estimate)
   *
   * NOTE: With 3 channels the reconstruction has exactly 3 degrees of
   * freedom. The result is a coarse spectral fingerprint only.
   * ----------------------------------------------------------------*/

  /* Build M = A · A^T (3×3) */
  double M[3][3];
  memset(M, 0, sizeof(M));

  for (int j = 0; j < SPEKTRUM_BINS; ++j) {
    double qr = QE_R[j], qg = QE_G[j], qb = QE_B[j];
    M[0][0] += qr * qr;  M[0][1] += qr * qg;  M[0][2] += qr * qb;
    M[1][0] += qg * qr;  M[1][1] += qg * qg;  M[1][2] += qg * qb;
    M[2][0] += qb * qr;  M[2][1] += qb * qg;  M[2][2] += qb * qb;
  }

  /* Regularise: M_reg = M + λ·I */
  double lam = (double)(lambda > 0.0f ? lambda : 0.01f);
  M[0][0] += lam;
  M[1][1] += lam;
  M[2][2] += lam;

  /* Invert M_reg */
  double M_inv[3][3];
  if (mat3_invert(M, M_inv) != 0) {
    fprintf(stderr, "[spektrum] WARNUNG: Matrix ist singulaer, "
            "Regularisierung erhoehen.\n");
    return SPEKTRUM_ERR_INVALID_DIMS;
  }

  /* v = M_reg⁻¹ · m  (3-element intermediate vector) */
  double m_vec[3] = { mean_r, mean_g, mean_b };
  double v[3];
  for (int i = 0; i < 3; ++i) {
    v[i] = M_inv[i][0] * m_vec[0]
         + M_inv[i][1] * m_vec[1]
         + M_inv[i][2] * m_vec[2];
  }

  /* S[j] = A[:,j]^T · v = QE_R[j]*v[0] + QE_G[j]*v[1] + QE_B[j]*v[2] */
  float *S = (float *)allsky_safe_malloc((size_t)SPEKTRUM_BINS * sizeof(float));
  if (!S)
    return SPEKTRUM_ERR_ALLOC;

  float S_max = 0.0f;
  int   peak_idx = 0;

  for (int j = 0; j < SPEKTRUM_BINS; ++j) {
    double val = QE_R[j] * v[0] + QE_G[j] * v[1] + QE_B[j] * v[2];
    S[j] = (float)(val < 0.0 ? 0.0 : val);   /* clamp negative artefacts */
    if (S[j] > S_max) {
      S_max  = S[j];
      peak_idx = j;
    }
  }

  printf("[spektrum] Spektrum berechnet (%d Stuetzstellen, 400-700nm)\n",
         SPEKTRUM_BINS);

  /* ------------------------------------------------------------------
   * Step 3: normalise to peak = 1.0
   * ----------------------------------------------------------------*/
  if (S_max > 0.0f) {
    for (int j = 0; j < SPEKTRUM_BINS; ++j)
      S[j] /= S_max;
  }

  int peak_nm = 400 + peak_idx * 10;
  printf("[spektrum] Peak bei %dnm (normiert: 1.000)\n", peak_nm);

  /* ------------------------------------------------------------------
   * Step 4: write JSON
   * ----------------------------------------------------------------*/

  /* Build ISO-8601 UTC timestamp */
  char ts_buf[32];
  time_t now = time(NULL);
  struct tm utc_tm;
  struct tm *utc = gmtime_r(&now, &utc_tm);
  if (utc)
    strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", utc);
  else
    snprintf(ts_buf, sizeof(ts_buf), "unknown");

  /* Built incrementally, so use the open/commit form (review.md, REL-05) */
  char tmp_path[PATH_MAX];
  FILE *fp = atomic_file_open(output_path, tmp_path, sizeof(tmp_path));
  if (!fp) {
    fprintf(stderr, "[spektrum] FEHLER: Datei '%s' konnte nicht "
            "geoeffnet werden.\n", output_path);
    allsky_safe_free(S);
    return SPEKTRUM_ERR_WRITE;
  }

  fprintf(fp, "{\n");
  fprintf(fp, "    \"timestamp\": \"%s\",\n", ts_buf);
  fprintf(fp, "    \"cx\": %d,\n", cx);
  fprintf(fp, "    \"cy\": %d,\n", cy);
  fprintf(fp, "    \"radius\": %d,\n", radius);
  fprintf(fp, "    \"pixels_used\": %ld,\n", pixel_count);
  fprintf(fp, "    \"mean_rgb\": [%.6f, %.6f, %.6f],\n",
          mean_r, mean_g, mean_b);

  /* wavelengths array */
  fprintf(fp, "    \"wavelengths_nm\": [");
  for (int j = 0; j < SPEKTRUM_BINS; ++j) {
    fprintf(fp, "%d", 400 + j * 10);
    if (j < SPEKTRUM_BINS - 1)
      fprintf(fp, ", ");
  }
  fprintf(fp, "],\n");

  /* normalised spectrum array */
  fprintf(fp, "    \"spectrum\": [");
  for (int j = 0; j < SPEKTRUM_BINS; ++j) {
    fprintf(fp, "%.4f", S[j]);
    if (j < SPEKTRUM_BINS - 1)
      fprintf(fp, ", ");
  }
  fprintf(fp, "],\n");

  fprintf(fp, "    \"spectrum_normalized\": true\n");
  fprintf(fp, "}\n");

  if (atomic_file_commit(fp, tmp_path, output_path) != 0) {
    fprintf(stderr, "[spektrum] FEHLER: '%s' konnte nicht geschrieben werden.\n",
            output_path);
    allsky_safe_free(S);
    return SPEKTRUM_ERR_WRITE;
  }

  allsky_safe_free(S);

  printf("[spektrum] Ergebnis geschrieben: %s\n", output_path);
  return SPEKTRUM_OK;
}

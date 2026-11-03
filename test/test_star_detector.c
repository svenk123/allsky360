/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
/*
 * Test of the raw star detector (src/star_detector.c).
 *
 *   test_star_detector
 *       synthetic RGGB frame with known stars, hot pixels, a line and a large
 *       blob; checks detection rate, positions and rejections.
 *
 *   test_star_detector FILE.raw CX CY RADIUS [PSF_SIGMA DETECT_SIGMA] [OUT.json]
 *       runs the detector on a real INDIGO raw frame (e.g. a raw dump).
 */
#include "allsky.h"
#include "star_detector.h"

#include <indigo/indigo_bus.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Deterministic Gaussian random numbers (xorshift + Box-Muller) */
static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static double urand(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return ((rng >> 11) + 0.5) / 9007199254740992.0;
}
static double grand(void) {
  return sqrt(-2.0 * log(urand())) * cos(2.0 * M_PI * urand());
}

/* Integrated Gaussian over a pixel (x-0.5 .. x+0.5) */
static double pix_gauss(double x, double c, double s) {
  return 0.5 * (erf((x + 0.5 - c) / (s * M_SQRT2)) -
                erf((x - 0.5 - c) / (s * M_SQRT2)));
}

typedef struct {
  double x, y, flux;
} truth_t;

static int run_file(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    perror(argv[1]);
    return 1;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *raw = malloc((size_t)n + 1);
  if (!raw || fread(raw, 1, (size_t)n, f) != (size_t)n) {
    fprintf(stderr, "read error\n");
    fclose(f);
    return 1;
  }
  fclose(f);
  raw[n] = 0;

  /* STAR_TEST_INVERT=1: negative image (65535 - v, no clipping). Stars
   * become holes, the noise keeps its statistics: everything found then is
   * a false detection (estimate of the false-positive rate). */
  const indigo_raw_header *h = (const indigo_raw_header *)raw;
  if (getenv("STAR_TEST_INVERT") && h->signature == INDIGO_RAW_MONO16) {
    uint16_t *px = (uint16_t *)(raw + sizeof(indigo_raw_header));
    size_t np = (size_t)h->width * h->height;
    for (size_t i = 0; i < np; i++)
      px[i] = (uint16_t)(65535 - px[i]);
    printf("inverted (negative image)\n");
  }

  star_detect_params_t p = {.psf_sigma = 1.0, .detect_sigma = 5.0,
                            .max_fwhm = 8.0, .max_elongation = 0.5,
                            .max_stars = 5000};
  if (argc > 5)
    p.psf_sigma = atof(argv[5]);
  if (argc > 6)
    p.detect_sigma = atof(argv[6]);

  star_t *stars = NULL;
  int ns = 0;
  star_detect_stats_t st;
  int rc = find_stars_raw(raw, (size_t)n, 0, 0, 0, atoi(argv[2]),
                          atoi(argv[3]), atoi(argv[4]), &p, &stars, &ns, &st);
  if (rc == 0) {
    double fw = 0, sn = 0;
    for (int i = 0; i < ns; i++) {
      fw += stars[i].fwhm;
      sn += stars[i].snr;
    }
    printf("stars: %d, mean FWHM %.2f px, mean SNR %.1f\n", ns,
           ns ? fw / ns : 0.0, ns ? sn / ns : 0.0);
    for (int i = 0; i < ns && i < 5; i++)
      printf("  #%d x=%.1f y=%.1f flux=%.0f snr=%.1f fwhm=%.2f e=%.2f\n", i,
             stars[i].x, stars[i].y, stars[i].flux, stars[i].snr,
             stars[i].fwhm, stars[i].elongation);
    if (argc > 7)
      stars_to_json(stars, ns, &st, argv[7]);
  }
  allsky_safe_free(stars);
  free(raw);
  return rc;
}

int main(int argc, char **argv) {
  if (argc >= 5)
    return run_file(argc, argv);

  const int W = 1200, H = 1000;
  const double bg = 1200.0, read_noise = 6.0, gain_e = 1.0; /* ADU */
  const double psf = 1.0;                                   /* raw px */
  const int NSTARS = 150, NHOT = 60;

  const char meta[] = "SIMPLE=T;BAYERPAT='RGGB';XBAYROFF=0;YBAYROFF=0;";
  size_t data = sizeof(indigo_raw_header) + (size_t)W * H * 2;
  size_t size = data + sizeof(meta);
  unsigned char *raw = calloc(1, size + 1);
  indigo_raw_header *hdr = (indigo_raw_header *)raw;
  hdr->signature = INDIGO_RAW_MONO16;
  hdr->width = W;
  hdr->height = H;
  memcpy(raw + data, meta, sizeof(meta));
  uint16_t *pix = (uint16_t *)(raw + sizeof(indigo_raw_header));

  /* Sky with a gradient (light dome) */
  double *img = malloc(sizeof(double) * W * H);
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      img[y * W + x] = bg * (1.0 + 0.6 * x / W + 0.2 * y / H);

  /* Stars: fluxes log-uniform 150 .. 60000 ADU */
  truth_t truth[NSTARS];
  for (int i = 0; i < NSTARS; i++) {
    truth[i].x = 80 + urand() * (W - 160);
    truth[i].y = 80 + urand() * (H - 160);
    truth[i].flux = 150.0 * pow(400.0, urand());
    for (int y = (int)truth[i].y - 6; y <= (int)truth[i].y + 6; y++)
      for (int x = (int)truth[i].x - 6; x <= (int)truth[i].x + 6; x++)
        img[y * W + x] += truth[i].flux * pix_gauss(x, truth[i].x, psf) *
                          pix_gauss(y, truth[i].y, psf);
  }
  /* False sources: a straight line (wire / edge) and a large blob (lamp,
   * cloud lit from below) */
  for (int x = 300; x < 700; x++)
    for (int dy = -1; dy <= 1; dy++)
      img[(150 + dy) * W + x] += 400.0 * (dy == 0 ? 1.0 : 0.4);
  for (int y = 550; y < 750; y++)
    for (int x = 750; x < 950; x++)
      img[y * W + x] += 3000.0 * exp(-((x - 850.0) * (x - 850.0) +
                                       (y - 650.0) * (y - 650.0)) /
                                     (2.0 * 12.0 * 12.0));

  /* Noise: shot noise + read noise */
  for (int i = 0; i < W * H; i++) {
    double v = img[i] + grand() * sqrt(img[i] / gain_e + read_noise * read_noise);
    pix[i] = (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
  }
  /* Hot pixels: single pixels, way above the sky */
  int hotx[NHOT], hoty[NHOT];
  for (int i = 0; i < NHOT; i++) {
    hotx[i] = 20 + (int)(urand() * (W - 40));
    hoty[i] = 20 + (int)(urand() * (H - 40));
    pix[hoty[i] * W + hotx[i]] = (uint16_t)(bg * 1.6 + 2000 + urand() * 20000);
  }
  free(img);

  star_detect_params_t p = {.psf_sigma = psf, .detect_sigma = 5.0,
                            .max_fwhm = 8.0, .max_elongation = 0.5,
                            .max_stars = 1000};
  star_t *stars = NULL;
  int ns = 0;
  star_detect_stats_t st;
  /* SQM circle inside the frame, like on the camera */
  const double CX = 600, CY = 500, R = 440;
  if (find_stars_raw(raw, size, 0, 0, 0, (int)CX, (int)CY, (int)R, &p, &stars,
                     &ns, &st) != 0) {
    fprintf(stderr, "FAIL: detector returned an error\n");
    return 1;
  }

  /* Match detections to the truth (2 px). Only stars inside the circle
   * count; "bright" = expected aperture SNR >= 10. */
  int matched = 0, matched_bright = 0, bright = 0, inside = 0, false_pos = 0,
      on_hot = 0, on_line = 0, on_blob = 0;
  double max_err_bright = 0, sum_err2 = 0;
  int *used = calloc(NSTARS, sizeof(int));
  double *snr_exp = calloc(NSTARS, sizeof(double));
  for (int i = 0; i < NSTARS; i++) {
    double sky = bg * (1.0 + 0.6 * truth[i].x / W + 0.2 * truth[i].y / H);
    /* aperture of ~20 raw pixels */
    snr_exp[i] = truth[i].flux / sqrt(truth[i].flux + 20.0 * (sky + 36.0));
    if (hypot(truth[i].x - CX, truth[i].y - CY) > R)
      continue;
    inside++;
    if (snr_exp[i] >= 10.0)
      bright++;
  }
  for (int s = 0; s < ns; s++) {
    int best = -1;
    double bd = 2.0;
    for (int i = 0; i < NSTARS; i++) {
      double d = hypot(stars[s].x - truth[i].x, stars[s].y - truth[i].y);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    if (best >= 0 && !used[best]) {
      used[best] = 1;
      matched++;
      sum_err2 += bd * bd;
      if (snr_exp[best] >= 10.0) {
        matched_bright++;
        if (bd > max_err_bright)
          max_err_bright = bd;
      }
      if (getenv("STAR_TEST_VERBOSE") && bd > 0.75)
        printf("  offset %.2f px: snr %.1f (expected %.1f) fwhm %.2f\n", bd,
               stars[s].snr, snr_exp[best], stars[s].fwhm);
      continue;
    }
    false_pos++;
    if (getenv("STAR_TEST_VERBOSE"))
      printf("  false: x=%.1f y=%.1f snr=%.1f fwhm=%.2f e=%.2f\n", stars[s].x,
             stars[s].y, stars[s].snr, stars[s].fwhm, stars[s].elongation);
    for (int h = 0; h < NHOT; h++)
      if (hypot(stars[s].x - hotx[h], stars[s].y - hoty[h]) < 2.0)
        on_hot++;
    if (stars[s].y > 145 && stars[s].y < 155 && stars[s].x > 290 &&
        stars[s].x < 710)
      on_line++;
    if (hypot(stars[s].x - 850, stars[s].y - 650) < 40)
      on_blob++;
  }
  free(used);
  free(snr_exp);
  double rms_err = matched ? sqrt(sum_err2 / matched) : 0;

  printf("synthetic: %d stars in the circle (%d with SNR >= 10), %d detected\n",
         inside, bright, ns);
  printf("  matched %d (bright %d/%d), position error rms %.2f px, bright max "
         "%.2f px\n",
         matched, matched_bright, bright, rms_err, max_err_bright);
  printf("  false %d (hot pixels %d, line %d, blob %d), hot pixels corrected "
         "%d/%d\n",
         false_pos, on_hot, on_line, on_blob, st.hot_pixels, NHOT);

  int fail = 0;
  if (matched_bright < bright * 0.95) {
    fprintf(stderr, "FAIL: less than 95%% of the bright stars found\n");
    fail = 1;
  }
  if (on_hot > 0 || on_line > 0 || on_blob > 0) {
    fprintf(stderr, "FAIL: hot pixels, line or blob detected as stars\n");
    fail = 1;
  }
  if (false_pos > 3) {
    fprintf(stderr, "FAIL: too many false detections\n");
    fail = 1;
  }
  if (max_err_bright > 0.75) {
    fprintf(stderr, "FAIL: position error too large\n");
    fail = 1;
  }
  allsky_safe_free(stars);
  free(raw);
  printf(fail ? "FAILED\n" : "OK\n");
  return fail;
}

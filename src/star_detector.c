/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include "star_detector.h"
#include "allsky.h"
#include "atomic_file.h"
#include "indigo_raw_to_rgbf.h"

#include <indigo/indigo_bus.h>
#include <json-c/json.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Background mesh tile size in (binned) pixels */
#define BG_TILE 32
/* Minimum fraction of a tile that must lie inside the region to be used */
#define BG_TILE_MIN_FILL 0.25f
/* Hot pixel: this many raw noise sigma above its neighbours' median ... */
#define HOT_SIGMA 6.0f
/* ... while no neighbour gets more than this fraction of its excess */
#define HOT_NEIGHBOUR_FRACTION 0.12f
/* Moments and flux use pixels above this many noise sigma */
#define MOMENT_SIGMA 2.0f
/* Extended source: ring median above this fraction of the peak */
#define RING_FRACTION 0.15f
/* Upper limit for the MAD noise samples (taken with a stride) */
#define MAX_NOISE_SAMPLES 400000

/* ── small helpers ──────────────────────────────────────────────────────── */

static void swapf(float *a, float *b) {
  float t = *a;
  *a = *b;
  *b = t;
}

/* k-th smallest element (0-based), reorders the array (Hoare quickselect) */
static float select_kth(float *v, int n, int k) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    float pivot = v[(lo + hi) / 2];
    int i = lo, j = hi;
    while (i <= j) {
      while (v[i] < pivot)
        i++;
      while (v[j] > pivot)
        j--;
      if (i <= j) {
        swapf(&v[i], &v[j]);
        i++;
        j--;
      }
    }
    if (k <= j)
      hi = j;
    else if (k >= i)
      lo = i;
    else
      return v[k];
  }
  return v[k];
}

static float median_inplace(float *v, int n) {
  if (n <= 0)
    return 0.0f;
  return select_kth(v, n, n / 2);
}

/* Median and 1.4826*MAD of v[0..n) after `iter` rounds of 3-sigma clipping.
 * Reorders and overwrites v. */
static void clipped_stats(float *v, int n, int iter, float *med_out,
                          float *sigma_out) {
  float med = 0.0f, sigma = 0.0f;
  float *tmp = allsky_safe_malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
  for (int it = 0; it <= iter && n > 0; it++) {
    med = median_inplace(v, n);
    for (int i = 0; i < n; i++)
      tmp[i] = fabsf(v[i] - med);
    sigma = 1.4826f * median_inplace(tmp, n);
    if (it == iter || sigma <= 0.0f)
      break;
    int m = 0;
    for (int i = 0; i < n; i++)
      if (fabsf(v[i] - med) <= 3.0f * sigma)
        v[m++] = v[i];
    if (m == n || m < 8)
      break;
    n = m;
  }
  allsky_safe_free(tmp);
  *med_out = med;
  *sigma_out = sigma;
}

static inline int in_circle_f(float x, float y, float cx, float cy, float r) {
  float dx = x - cx, dy = y - cy;
  return dx * dx + dy * dy <= r * r;
}

/* Bounded search for a FITS keyword after the pixel data */
static int raw_has_bayer_pattern(const unsigned char *raw, size_t raw_size,
                                 size_t data_end) {
  static const char key[] = "BAYERPAT=";
  const size_t klen = sizeof(key) - 1;
  if (data_end >= raw_size)
    return 0;
  for (size_t i = data_end; i + klen <= raw_size; i++)
    if (memcmp(raw + i, key, klen) == 0)
      return 1;
  return 0;
}

/* Value of raw pixel (x, y) in ADU (luminance for RGB formats) */
static inline float raw_pixel(const unsigned char *data, uint32_t type,
                              int width, int x, int y) {
  size_t i = (size_t)y * width + x;
  switch (type) {
  case INDIGO_RAW_MONO8:
    return data[i];
  case INDIGO_RAW_MONO16:
    return ((const uint16_t *)data)[i];
  case INDIGO_RAW_RGB24: {
    const unsigned char *p = data + i * 3;
    return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
  }
  case INDIGO_RAW_RGBA32: {
    const unsigned char *p = data + i * 4;
    return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
  }
  case INDIGO_RAW_ABGR32: {
    const unsigned char *p = data + i * 4;
    return 0.2126f * p[3] + 0.7152f * p[2] + 0.0722f * p[1];
  }
  case INDIGO_RAW_RGB48: {
    const uint16_t *p = (const uint16_t *)data + i * 3;
    return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
  }
  default:
    return 0.0f;
  }
}

/* Robust per-pixel noise from differences of neighbours of the same colour
 * (step 2 for Bayer, 1 otherwise): insensitive to gradients and stars. */
static float noise_from_differences(const float *img, int w, int h, int step,
                                    const unsigned char *mask) {
  long total = (long)w * h;
  int stride = (int)(total / MAX_NOISE_SAMPLES) + 1;
  float *s = allsky_safe_malloc((size_t)(total / stride + 1) * sizeof(float));
  int n = 0;
  for (long i = 0; i < total; i += stride) {
    int x = (int)(i % w);
    if (x + step >= w || !mask[i] || !mask[i + step])
      continue;
    s[n++] = img[i] - img[i + step];
  }
  float med, sigma;
  clipped_stats(s, n, 2, &med, &sigma);
  allsky_safe_free(s);
  return sigma / sqrtf(2.0f);
}

/* ── background mesh ────────────────────────────────────────────────────── */

/* Fills bg[] and rms[] (per pixel) from a tile mesh. The noise comes from
 * neighbour differences (pixel noise, insensitive to gradients) or, with
 * noise_from_values, from the clipped spread of the values (for a filtered
 * image, whose pixels are correlated). Returns 0 on success. */
static int background_mesh(const float *img, const unsigned char *mask, int w,
                           int h, int noise_from_values, float *bg,
                           float *rms) {
  int gw = (w + BG_TILE - 1) / BG_TILE;
  int gh = (h + BG_TILE - 1) / BG_TILE;
  int ng = gw * gh;
  float *tb = allsky_safe_malloc((size_t)ng * sizeof(float));
  float *tr = allsky_safe_malloc((size_t)ng * sizeof(float));
  unsigned char *valid = allsky_safe_malloc((size_t)ng);
  float *buf = allsky_safe_malloc((size_t)BG_TILE * BG_TILE * sizeof(float));
  float *dif = allsky_safe_malloc((size_t)BG_TILE * BG_TILE * sizeof(float));
  int nvalid = 0;

  for (int gy = 0; gy < gh; gy++) {
    for (int gx = 0; gx < gw; gx++) {
      int x0 = gx * BG_TILE, y0 = gy * BG_TILE;
      int x1 = x0 + BG_TILE < w ? x0 + BG_TILE : w;
      int y1 = y0 + BG_TILE < h ? y0 + BG_TILE : h;
      int n = 0, nd = 0;
      for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) {
          size_t i = (size_t)y * w + x;
          if (!mask[i])
            continue;
          buf[n++] = img[i];
          /* Noise from horizontal neighbour differences: unlike the spread
           * of the values it does not grow with a gradient in the tile */
          if (x + 1 < w && mask[i + 1])
            dif[nd++] = img[i] - img[i + 1];
        }
      int g = gy * gw + gx;
      valid[g] = (n >= (int)(BG_TILE_MIN_FILL * BG_TILE * BG_TILE) ||
                  (n > 16 && n >= (int)(0.5f * (x1 - x0) * (y1 - y0)))) &&
                 nd > 8;
      if (valid[g]) {
        float med, sig, dmed, dsig;
        clipped_stats(buf, n, 3, &med, &sig);
        clipped_stats(dif, nd, 3, &dmed, &dsig);
        tb[g] = med;
        tr[g] = noise_from_values ? sig : dsig / sqrtf(2.0f);
        nvalid++;
      }
    }
  }
  allsky_safe_free(buf);
  allsky_safe_free(dif);

  if (nvalid == 0) {
    allsky_safe_free(tb);
    allsky_safe_free(tr);
    allsky_safe_free(valid);
    return 1;
  }

  /* Fill invalid tiles from valid neighbours, growing outwards */
  for (int changed = 1; changed;) {
    changed = 0;
    unsigned char *nv = allsky_safe_malloc_copy((size_t)ng, valid);
    for (int gy = 0; gy < gh; gy++)
      for (int gx = 0; gx < gw; gx++) {
        int g = gy * gw + gx;
        if (valid[g])
          continue;
        float sb = 0, sr = 0;
        int c = 0;
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++) {
            int nx = gx + dx, ny = gy + dy;
            if (nx < 0 || ny < 0 || nx >= gw || ny >= gh || !valid[ny * gw + nx])
              continue;
            sb += tb[ny * gw + nx];
            sr += tr[ny * gw + nx];
            c++;
          }
        if (c) {
          tb[g] = sb / c;
          tr[g] = sr / c;
          nv[g] = 1;
          changed = 1;
        }
      }
    memcpy(valid, nv, (size_t)ng);
    allsky_safe_free(nv);
  }
  allsky_safe_free(valid);

  /* 3x3 median filter of the mesh: removes tiles spoiled by bright stars */
  float *fb = allsky_safe_malloc((size_t)ng * sizeof(float));
  float *fr = allsky_safe_malloc((size_t)ng * sizeof(float));
  for (int gy = 0; gy < gh; gy++)
    for (int gx = 0; gx < gw; gx++) {
      float vb[9], vr[9];
      int c = 0;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          int nx = gx + dx, ny = gy + dy;
          if (nx < 0 || ny < 0 || nx >= gw || ny >= gh)
            continue;
          vb[c] = tb[ny * gw + nx];
          vr[c] = tr[ny * gw + nx];
          c++;
        }
      fb[gy * gw + gx] = median_inplace(vb, c);
      fr[gy * gw + gx] = median_inplace(vr, c);
    }
  allsky_safe_free(tb);
  allsky_safe_free(tr);

  /* Bilinear interpolation between tile centres */
#pragma omp parallel for
  for (int y = 0; y < h; y++) {
    float fy = (y + 0.5f) / BG_TILE - 0.5f;
    int y0 = (int)floorf(fy);
    float ty = fy - y0;
    int ya = y0 < 0 ? 0 : (y0 >= gh ? gh - 1 : y0);
    int yb = y0 + 1 < 0 ? 0 : (y0 + 1 >= gh ? gh - 1 : y0 + 1);
    for (int x = 0; x < w; x++) {
      float fx = (x + 0.5f) / BG_TILE - 0.5f;
      int x0 = (int)floorf(fx);
      float tx = fx - x0;
      int xa = x0 < 0 ? 0 : (x0 >= gw ? gw - 1 : x0);
      int xb = x0 + 1 < 0 ? 0 : (x0 + 1 >= gw ? gw - 1 : x0 + 1);
      float b00 = fb[ya * gw + xa], b10 = fb[ya * gw + xb];
      float b01 = fb[yb * gw + xa], b11 = fb[yb * gw + xb];
      float r00 = fr[ya * gw + xa], r10 = fr[ya * gw + xb];
      float r01 = fr[yb * gw + xa], r11 = fr[yb * gw + xb];
      bg[y * w + x] = (b00 * (1 - tx) + b10 * tx) * (1 - ty) +
                      (b01 * (1 - tx) + b11 * tx) * ty;
      rms[y * w + x] = (r00 * (1 - tx) + r10 * tx) * (1 - ty) +
                       (r01 * (1 - tx) + r11 * tx) * ty;
    }
  }
  allsky_safe_free(fb);
  allsky_safe_free(fr);
  return 0;
}

/* Separable convolution with a normalised 1D kernel (edges clamped) */
static void convolve_separable(const float *in, float *out, int w, int h,
                               const float *k, int kr) {
  float *tmp = allsky_safe_malloc((size_t)w * h * sizeof(float));
#pragma omp parallel for
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      float s = 0.0f;
      for (int i = -kr; i <= kr; i++) {
        int xx = x + i < 0 ? 0 : (x + i >= w ? w - 1 : x + i);
        s += k[i + kr] * in[y * w + xx];
      }
      tmp[y * w + x] = s;
    }
#pragma omp parallel for
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      float s = 0.0f;
      for (int i = -kr; i <= kr; i++) {
        int yy = y + i < 0 ? 0 : (y + i >= h ? h - 1 : y + i);
        s += k[i + kr] * tmp[yy * w + x];
      }
      out[y * w + x] = s;
    }
  allsky_safe_free(tmp);
}

static int compare_flux_desc(const void *a, const void *b) {
  float fa = ((const star_t *)a)->flux, fb = ((const star_t *)b)->flux;
  return (fa < fb) - (fa > fb);
}

/* ── detector ───────────────────────────────────────────────────────────── */

int find_stars_raw(const unsigned char *raw_data, size_t raw_size,
                   int is_monochrome, int offset_x, int offset_y, int cx,
                   int cy, int radius, const star_detect_params_t *params,
                   star_t **stars_out, int *num_stars_out,
                   star_detect_stats_t *stats_out) {
  if (!raw_data || !params || !stars_out || !num_stars_out || radius <= 0)
    return 1;
  *stars_out = NULL;
  *num_stars_out = 0;

  star_detect_stats_t st;
  memset(&st, 0, sizeof(st));

  size_t expected = indigo_raw_expected_size(raw_data);
  if (expected == 0 || raw_size < expected) {
    fprintf(stderr, "Star detector: invalid or incomplete raw frame\n");
    return 2;
  }
  const indigo_raw_header *hdr = (const indigo_raw_header *)raw_data;
  const uint32_t type = hdr->signature;
  const int rw_full = (int)hdr->width, rh_full = (int)hdr->height;
  const unsigned char *pixels = raw_data + sizeof(indigo_raw_header);

  const int is_bayer = !is_monochrome &&
                       (type == INDIGO_RAW_MONO8 || type == INDIGO_RAW_MONO16) &&
                       raw_has_bayer_pattern(raw_data, raw_size, expected);
  const int bin = is_bayer ? 2 : 1;
  st.binning = bin;

  /* Region in raw coordinates, a margin for the filter kernel around it */
  const float rcx = (float)cx + offset_x, rcy = (float)cy + offset_y;
  /* Half a background tile around the circle, so that the background mesh
   * does not have to extrapolate at the edge of the circle */
  const int margin = BG_TILE * bin / 2 + 4 * bin +
                     (int)ceilf(3.0f * (float)params->psf_sigma);
  int x0 = (int)floorf(rcx - radius) - margin;
  int y0 = (int)floorf(rcy - radius) - margin;
  int x1 = (int)ceilf(rcx + radius) + margin;
  int y1 = (int)ceilf(rcy + radius) + margin;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > rw_full) x1 = rw_full;
  if (y1 > rh_full) y1 = rh_full;
  /* Keep the Bayer cells intact */
  x0 -= x0 % bin;
  y0 -= y0 % bin;
  int rw = (x1 - x0) / bin * bin, rh = (y1 - y0) / bin * bin;
  if (rw < 8 * bin || rh < 8 * bin) {
    fprintf(stderr, "Star detector: region (%d,%d r=%d) outside the image\n",
            cx, cy, radius);
    return 3;
  }

  /* 1. Copy the region and correct isolated hot pixels */
  size_t rn = (size_t)rw * rh;
  float *r = allsky_safe_malloc(rn * sizeof(float));
  unsigned char *rmask = allsky_safe_malloc(rn);
  for (int y = 0; y < rh; y++)
    for (int x = 0; x < rw; x++) {
      r[(size_t)y * rw + x] = raw_pixel(pixels, type, rw_full, x0 + x, y0 + y);
      rmask[(size_t)y * rw + x] =
          in_circle_f(x0 + x + 0.5f, y0 + y + 0.5f, rcx, rcy, (float)radius);
    }
  float sigma_raw = noise_from_differences(r, rw, rh, bin, rmask);

  float *rc = allsky_safe_malloc_copy(rn * sizeof(float), r);
  int hot = 0;
  if (sigma_raw > 0.0f) {
#pragma omp parallel for reduction(+ : hot)
    for (int y = 1; y < rh - 1; y++)
      for (int x = 1; x < rw - 1; x++) {
        float v = r[(size_t)y * rw + x];
        float nb[8];
        int c = 0;
        float nmax = -1e30f;
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy)
              continue;
            float u = r[(size_t)(y + dy) * rw + x + dx];
            nb[c++] = u;
            if (u > nmax)
              nmax = u;
          }
        float base = median_inplace(nb, 8);
        float excess = v - base;
        if (excess > HOT_SIGMA * sigma_raw &&
            nmax - base < HOT_NEIGHBOUR_FRACTION * excess) {
          rc[(size_t)y * rw + x] = base;
          hot++;
        }
      }
  }
  st.hot_pixels = hot;
  allsky_safe_free(r);
  allsky_safe_free(rmask);

  /* 2. Bin Bayer cells (sum of R, G, G, B) */
  const int bw = rw / bin, bh = rh / bin;
  const size_t bn = (size_t)bw * bh;
  float *b = allsky_safe_malloc(bn * sizeof(float));
  unsigned char *mask = allsky_safe_malloc(bn);
  /* Region centre and radius in binned coordinates */
  const float bcx = (rcx - x0) / bin, bcy = (rcy - y0) / bin;
  const float bradius = (float)radius / bin;
  for (int y = 0; y < bh; y++)
    for (int x = 0; x < bw; x++) {
      float s = 0.0f;
      for (int j = 0; j < bin; j++)
        for (int i = 0; i < bin; i++)
          s += rc[(size_t)(y * bin + j) * rw + x * bin + i];
      b[(size_t)y * bw + x] = s;
      mask[(size_t)y * bw + x] =
          in_circle_f(x + 0.5f, y + 0.5f, bcx, bcy, bradius);
    }
  allsky_safe_free(rc);

  /* 3. Background and noise */
  float *bg = allsky_safe_malloc(bn * sizeof(float));
  float *rms = allsky_safe_malloc(bn * sizeof(float));
  unsigned char *bgmask = allsky_safe_malloc(bn);
  memset(bgmask, 1, bn); /* the whole box, including the margin */
  int bg_rc = background_mesh(b, bgmask, bw, bh, 0, bg, rms);
  if (bg_rc != 0) {
    allsky_safe_free(bgmask);
    fprintf(stderr, "Star detector: no background in the region\n");
    allsky_safe_free(b);
    allsky_safe_free(mask);
    allsky_safe_free(bg);
    allsky_safe_free(rms);
    return 4;
  }
  {
    float *v = allsky_safe_malloc(bn * sizeof(float));
    int n = 0;
    for (size_t i = 0; i < bn; i++)
      if (mask[i])
        v[n++] = bg[i];
    st.background = median_inplace(v, n) / (float)(bin * bin);
    st.noise = noise_from_differences(b, bw, bh, 1, mask) / (float)bin;
    allsky_safe_free(v);
  }

  float *d = allsky_safe_malloc(bn * sizeof(float));
  for (size_t i = 0; i < bn; i++)
    d[i] = b[i] - bg[i];
  allsky_safe_free(b);
  allsky_safe_free(bg);

  /* 4. Matched filter */
  float ks = (float)params->psf_sigma / bin;
  if (ks < 0.5f)
    ks = 0.5f;
  int kr = (int)ceilf(3.0f * ks);
  if (kr < 1)
    kr = 1;
  float *k = allsky_safe_malloc((size_t)(2 * kr + 1) * sizeof(float));
  float ksum = 0.0f, ksq = 0.0f;
  for (int i = -kr; i <= kr; i++) {
    k[i + kr] = expf(-(float)(i * i) / (2.0f * ks * ks));
    ksum += k[i + kr];
  }
  for (int i = 0; i <= 2 * kr; i++) {
    k[i] /= ksum;
    ksq += k[i] * k[i];
  }
  /* Noise of the filtered image = pixel noise * sqrt(sum of 2D kernel²) */
  const float filter_noise = ksq;
  float *f = allsky_safe_malloc(bn * sizeof(float));
  convolve_separable(d, f, bw, bh, k, kr);
  allsky_safe_free(k);

  /* Noise of the filtered image, measured: the sky is not white noise.
   * Structure at the size of a star (cloud texture, Milky Way, moonlit
   * haze, pattern noise) passes the matched filter and would otherwise be
   * detected. The prediction from the pixel noise is the lower limit. */
  float *fbg = allsky_safe_malloc(bn * sizeof(float));
  float *frms = allsky_safe_malloc(bn * sizeof(float));
  if (background_mesh(f, bgmask, bw, bh, 1, fbg, frms) != 0) {
    memset(fbg, 0, bn * sizeof(float));
    for (size_t i = 0; i < bn; i++)
      frms[i] = rms[i] * filter_noise;
  }
  allsky_safe_free(bgmask);

  /* 5. + 6. Peaks, measurement, rejection */
  int cap = 1024, count = 0;
  star_t *stars = allsky_safe_malloc((size_t)cap * sizeof(star_t));
  const float thr = (float)params->detect_sigma;
  const int ap = kr > 2 ? kr : 2; /* measurement radius (binned pixels) */
  /* Pixel variance of a uniformly lit pixel: subtracted from the moments */
  const float pixvar = 1.0f / 12.0f;

  for (int y = ap; y < bh - ap; y++) {
    for (int x = ap; x < bw - ap; x++) {
      size_t i = (size_t)y * bw + x;
      if (!mask[i])
        continue;
      float noise = rms[i];
      float fnoise = frms[i] > noise * filter_noise ? frms[i]
                                                    : noise * filter_noise;
      if (noise <= 0.0f || f[i] - fbg[i] <= thr * fnoise)
        continue;
      /* Local maximum in 5x5; ties go to the first pixel in raster order */
      int is_max = 1;
      for (int dy = -2; dy <= 2 && is_max; dy++)
        for (int dx = -2; dx <= 2 && is_max; dx++) {
          if (!dx && !dy)
            continue;
          int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= bw || ny >= bh)
            continue;
          float u = f[(size_t)ny * bw + nx];
          if (u > f[i] || (u == f[i] && (dy < 0 || (dy == 0 && dx < 0))))
            is_max = 0;
        }
      if (!is_max)
        continue;
      st.candidates++;

      /* Moments of the significant pixels, flux of the aperture */
      double sw = 0, sx = 0, sy = 0, flux = 0;
      int npix = 0, nap = 0;
      float peak = 0.0f;
      for (int dy = -ap; dy <= ap; dy++)
        for (int dx = -ap; dx <= ap; dx++) {
          if (dx * dx + dy * dy > ap * ap)
            continue;
          float v = d[(size_t)(y + dy) * bw + x + dx];
          flux += v;
          nap++;
          if (v > peak)
            peak = v;
          if (v > MOMENT_SIGMA * noise) {
            sw += v;
            sx += v * dx;
            sy += v * dy;
            npix++;
          }
        }
      if (sw <= 0.0 || flux <= 0.0)
        continue;

      /* Extended source (lamp, lit cloud, glow): a star has fallen to the
       * background two pixels outside the aperture, an extended source has
       * not. The median ignores a neighbouring star in part of the ring. */
      {
        float ring[128];
        int nr = 0;
        const int r1 = ap + 1, r2 = ap + 3;
        for (int dy = -r2; dy <= r2; dy++)
          for (int dx = -r2; dx <= r2; dx++) {
            int rr = dx * dx + dy * dy;
            int nx = x + dx, ny = y + dy;
            if (rr < r1 * r1 || rr > r2 * r2 || nx < 0 || ny < 0 ||
                nx >= bw || ny >= bh || nr >= 128)
              continue;
            ring[nr++] = d[(size_t)ny * bw + nx];
          }
        if (nr > 8 && median_inplace(ring, nr) >
                          RING_FRACTION * peak + MOMENT_SIGMA * noise) {
          st.rejected_size++;
          continue;
        }
      }
      double mx = sx / sw, my = sy / sw;
      double mxx = 0, myy = 0, mxy = 0;
      for (int dy = -ap; dy <= ap; dy++)
        for (int dx = -ap; dx <= ap; dx++) {
          if (dx * dx + dy * dy > ap * ap)
            continue;
          float v = d[(size_t)(y + dy) * bw + x + dx];
          if (v > MOMENT_SIGMA * noise) {
            mxx += v * (dx - mx) * (dx - mx);
            myy += v * (dy - my) * (dy - my);
            mxy += v * (dx - mx) * (dy - my);
          }
        }
      mxx = mxx / sw + pixvar;
      myy = myy / sw + pixvar;
      mxy /= sw;
      double var = 0.5 * (mxx + myy);
      float fwhm = (float)(2.3548 * sqrt(var) * bin);
      double diff = sqrt((mxx - myy) * (mxx - myy) + 4.0 * mxy * mxy);
      float elong = (float)(diff / (mxx + myy));

      if (fwhm > params->max_fwhm) {
        st.rejected_size++;
        continue;
      }
      /* Too few pixels for a meaningful shape: only the size test applies */
      if (npix >= 4 && elong > params->max_elongation) {
        st.rejected_shape++;
        continue;
      }

      if (count == cap) {
        cap *= 2;
        star_t *tmp = allsky_safe_realloc(stars, (size_t)cap * sizeof(star_t));
        if (!tmp)
          break;
        stars = tmp;
      }
      /* Binned pixel (x+mx) covers raw pixels x0+bin*(x+mx) ... +bin-1 */
      float raw_x = x0 + (float)((x + mx) * bin) + 0.5f * (bin - 1);
      float raw_y = y0 + (float)((y + my) * bin) + 0.5f * (bin - 1);
      stars[count++] = (star_t){
          .x = raw_x - offset_x,
          .y = raw_y - offset_y,
          .flux = (float)flux,
          .peak = peak,
          .snr = (float)(flux / (noise * sqrt((double)nap))),
          .fwhm = fwhm,
          .elongation = elong,
      };
    }
  }
  allsky_safe_free(d);
  allsky_safe_free(f);
  allsky_safe_free(fbg);
  allsky_safe_free(frms);
  allsky_safe_free(rms);
  allsky_safe_free(mask);

  /* Brightest first; the limit drops the faintest, not a part of the sky */
  qsort(stars, (size_t)count, sizeof(star_t), compare_flux_desc);
  if (params->max_stars > 0 && count > params->max_stars) {
    st.truncated = count - params->max_stars;
    count = params->max_stars;
  }
  if (count == 0) {
    allsky_safe_free(stars);
    stars = NULL;
  }

  *stars_out = stars;
  *num_stars_out = count;
  if (stats_out)
    *stats_out = st;

  printf("Star detector: %d stars (candidates %d, too large %d, elongated %d, "
         "hot pixels %d, dropped %d), background %.1f ADU, noise %.2f ADU, "
         "binning %dx%d, star sigma %.2f px, threshold %.1f sigma, "
         "region %d,%d r=%d\n",
         count, st.candidates, st.rejected_size, st.rejected_shape,
         st.hot_pixels, st.truncated, st.background, st.noise, bin, bin,
         params->psf_sigma, params->detect_sigma, cx, cy, radius);
  return 0;
}

/* ── output ─────────────────────────────────────────────────────────────── */

/* Number with a fixed number of decimals in the JSON text (a plain double
 * would be written as 1518.5899999999999) */
static struct json_object *json_float(double v, int decimals) {
  char buf[48];
  snprintf(buf, sizeof(buf), "%.*f", decimals, v);
  return json_object_new_double_s(strtod(buf, NULL), buf);
}

int stars_to_json(const star_t *stars, int num_stars,
                  const star_detect_stats_t *stats, const char *filename) {
  if ((!stars && num_stars > 0) || num_stars < 0 || !filename)
    return 1;

  struct json_object *root = json_object_new_object();
  struct json_object *arr = json_object_new_array();
  if (!root || !arr) {
    json_object_put(root);
    json_object_put(arr);
    return 2;
  }

  for (int i = 0; i < num_stars; i++) {
    struct json_object *o = json_object_new_object();
    json_object_object_add(o, "x", json_float(stars[i].x, 2));
    json_object_object_add(o, "y", json_float(stars[i].y, 2));
    json_object_object_add(o, "flux", json_float(stars[i].flux, 1));
    json_object_object_add(o, "peak", json_float(stars[i].peak, 1));
    json_object_object_add(o, "snr", json_float(stars[i].snr, 1));
    json_object_object_add(o, "fwhm", json_float(stars[i].fwhm, 2));
    json_object_object_add(o, "elongation", json_float(stars[i].elongation, 2));
    json_object_array_add(arr, o);
  }
  json_object_object_add(root, "stars", arr);

  if (stats) {
    json_object_object_add(root, "background", json_float(stats->background, 2));
    json_object_object_add(root, "noise", json_float(stats->noise, 3));
    json_object_object_add(root, "binning", json_object_new_int(stats->binning));
    json_object_object_add(root, "hot_pixels", json_object_new_int(stats->hot_pixels));
    json_object_object_add(root, "candidates", json_object_new_int(stats->candidates));
    json_object_object_add(root, "rejected_size", json_object_new_int(stats->rejected_size));
    json_object_object_add(root, "rejected_shape", json_object_new_int(stats->rejected_shape));
    json_object_object_add(root, "truncated", json_object_new_int(stats->truncated));
  }

  /* Written atomically - the archive page reads these files (REL-05) */
  const char *json_str =
      json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
  int rc = write_text_file_atomic(filename, json_str) != 0 ? 4 : 0;
  json_object_put(root);
  if (rc == 0)
    printf("Written %d stars to file: %s\n", num_stars, filename);
  return rc;
}

/* ── SQM from the star count ────────────────────────────────────────────── */

int estimate_sqm_from_star_count(int num_stars, float a, float b,
                                 float *sqm_out) {
  if (num_stars < 1 || !sqm_out)
    return 1;

  *sqm_out = a + b * log10f((float)num_stars);
  return 0;
}

int estimate_sqm_corrected(int num_stars, float exposure_s, float gain_db,
                           float a, float b, float *sqm_out) {
  if (num_stars < 1 || exposure_s <= 0.0f || !sqm_out)
    return 1;

  float correction_factor =
      powf(exposure_s, 0.5f) * powf(2.0f, gain_db / 12.0f);
  if (correction_factor <= 0.0f)
    return 2;

  float corrected_count = (float)num_stars / correction_factor;
  *sqm_out = a + b * log10f(corrected_count);
  return 0;
}

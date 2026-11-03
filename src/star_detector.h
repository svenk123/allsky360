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
#ifndef STAR_DETECTOR_H
#define STAR_DETECTOR_H

#include <stddef.h>

/*
 * Star detector working on the linear INDIGO raw frame (before debayering
 * and any processing), restricted to a circular region of interest.
 *
 * Pipeline (classical source extraction, cf. SExtractor/SEP, DAOFIND):
 *   1. Cosmetic correction: isolated hot pixels are replaced by the median of
 *      their neighbours (a star always spreads light into its neighbours).
 *   2. Bayer frames are binned 2x2 (one R, two G, one B per superpixel):
 *      no debayer artefacts, better signal-to-noise ratio.
 *   3. Background and noise mesh (tiles, sigma-clipped median / MAD),
 *      median-filtered and bilinearly interpolated, then subtracted.
 *   4. Matched filter with a Gaussian of the star size.
 *   5. Detection: local maxima above detect_sigma times the local noise of
 *      the filtered image (a significance threshold, independent of sky
 *      brightness, exposure and gain).
 *   6. Measurement and rejection: centroid, FWHM and elongation from second
 *      moments, aperture flux and SNR; too large or elongated sources
 *      (clouds, trees, lights, edges) are rejected.
 */

typedef struct {
  float x, y;       /* centroid in processing-image coordinates (after crop) */
  float flux;       /* background-subtracted aperture flux [raw ADU] */
  float peak;       /* peak above background of the (binned) pixel [raw ADU] */
  float snr;        /* aperture signal-to-noise ratio */
  float fwhm;       /* full width at half maximum [image pixels] */
  float elongation; /* 0 = round ... 1 = line */
} star_t;

typedef struct {
  double psf_sigma;      /* Gaussian sigma of a star [image pixels] */
  double detect_sigma;   /* detection threshold in noise sigma (filtered) */
  double max_fwhm;       /* reject sources wider than this [image pixels] */
  double max_elongation; /* reject sources more elongated than this */
  int max_stars;         /* keep at most this many (brightest first) */
} star_detect_params_t;

typedef struct {
  float background;  /* median sky background per raw pixel [ADU] */
  float noise;       /* background noise per raw pixel [ADU] */
  int binning;       /* 2 for Bayer frames, 1 for mono/RGB */
  int hot_pixels;    /* isolated hot pixels corrected */
  int candidates;    /* peaks above the detection threshold */
  int rejected_size; /* rejected: FWHM > max_fwhm */
  int rejected_shape;/* rejected: elongation > max_elongation */
  int truncated;     /* stars dropped because of max_stars */
} star_detect_stats_t;

/**
 * Detect stars in an INDIGO raw frame.
 *
 * @param raw_data: INDIGO raw frame (header, pixel data, FITS keywords).
 * @param raw_size: size of raw_data in bytes.
 * @param is_monochrome: 1 = ignore a Bayer pattern (mono camera).
 * @param offset_x, offset_y: crop offset; image = raw - offset.
 * @param cx, cy, radius: circular region in processing-image coordinates.
 * @param params: detection parameters.
 * @param stars_out: allocated array of stars, brightest first (free with
 *                   allsky_safe_free()); NULL when none were found.
 * @param num_stars_out: number of stars.
 * @param stats_out: optional statistics (may be NULL).
 * @return: 0 on success, >0 on error.
 */
int find_stars_raw(const unsigned char *raw_data, size_t raw_size,
                   int is_monochrome, int offset_x, int offset_y, int cx,
                   int cy, int radius, const star_detect_params_t *params,
                   star_t **stars_out, int *num_stars_out,
                   star_detect_stats_t *stats_out);

/**
 * Write stars (and the detection statistics) to a JSON file (atomic).
 *
 * Format: {"stars":[{"x":..,"y":..,"flux":..,"snr":..,"fwhm":..}, ...],
 *          "background":.., "noise":.., ...}
 *
 * @return: 0 on success, >0 on error.
 */
int stars_to_json(const star_t *stars, int num_stars,
                  const star_detect_stats_t *stats, const char *filename);

/**
 * Estimate sky quality (SQM) from number of detected stars using a logarithmic model.
 *
 * @param num_stars: number of detected stars in the defined ROI.
 * @param a: calibration parameter (e.g. 23.0) – maximum SQM for dark sky.
 * @param b: calibration slope (e.g. 1.5) – sensitivity of the star count to SQM.
 * @param sqm_out: pointer to float where the estimated SQM value will be stored.
 * @return: 0 on success, >0 on error (e.g. invalid input).
 */
int estimate_sqm_from_star_count(int num_stars, float a, float b, float *sqm_out);

/**
 * Estimate sky quality (SQM) from star count, corrected for exposure and gain.
 *
 * @param num_stars: number of detected stars in the ROI.
 * @param exposure_s: exposure time in seconds (e.g. 1.5 for 1500 ms).
 * @param gain_db: camera gain in dB (e.g. 0–30).
 * @param a: SQM model parameter (e.g. 23.0).
 * @param b: SQM slope parameter (e.g. 1.5).
 * @param sqm_out: pointer to float where the SQM result will be stored.
 * @return: 0 on success, >0 on invalid input or division by zero.
 */
int estimate_sqm_corrected(int num_stars, float exposure_s, float gain_db,
                           float a, float b, float *sqm_out);

#endif // STAR_DETECTOR_H

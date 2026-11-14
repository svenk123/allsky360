# Image Processing Pipeline

This document describes **what** happens to every image between the camera and
the JPEG file, **in which order**, and **which `config.ini` keys** control each
step. How the exposures are fused into one HDR image is described in
[HDR-FUSION.md](HDR-FUSION.md).

All processing works on float RGBA images in **linear light, 0.0 – 1.0**. On a
CUDA build the steps after the fusion run on the GPU in place on the same
(zero-copy) buffer; the CPU build runs the same steps with OpenMP.

---

## 1. Overview

```
 per exposure (HDR frame 0 … n-1)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ raw frame ─► crop ─► debayer ─► brightness / noise measurement           │
 │           └─► star detection for SQM (frame 0, raw data)                 │
 │           ─► multiscale median filter (night)                            │
 └──────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
 HDR fusion ─► white balance ─► SCNR (night) ─► HDRMT (optional) ─► normalize
                     │
                     ▼
 autostretch ─► black point ─► ACDNR ─► wavelet sharpen ─► clarity ─► dehaze
                     │
                     ▼
 gamma (fixed or auto) ─► Reinhard (optional) ─► tone curve (optional)
                     │
                     ▼
 saturation ─► SQM / focus / next exposure ─► text overlay ─► JPEG, thumbnail,
                                                              panorama
```

| # | Step | Enabled by | Default |
|---|------|-----------|---------|
| 1 | Crop | `crop_image` | 0 |
| 2 | Debayer | `camera_debayer` (nni, bilinear, vng) | vng |
| 3 | Multiscale median filter | `nighttime_multiscale_median_filter` | 1 (night only) |
| 4 | HDR fusion | `hdr` | 1 |
| 5 | White balance | `daytime_/nighttime_white_balance_red/blue` | – |
| 6 | SCNR (green reduction) | `scnr_filter` | 1 (night only) |
| 7 | HDR multiscale transform | `hdrmt` | 0 (experimental) |
| 8 | Autostretch | `autostretch` | 1 |
| 9 | Black point | `daytime_/nighttime_move_blackpoint` > 0 | night 0.001 |
| 10 | ACDNR noise reduction | `acdnr_filter` | 1 |
| 11 | Wavelet sharpening | `wavelet_sharpen` | 1 |
| 12 | Clarity (local contrast) | `clarity_filter` | 0 |
| 13 | Dehaze | `dehaze_amount` > 0 | 0.0 |
| 14 | Gamma | `gamma` / `auto_gamma` | 0.9 / 0 |
| 15 | Reinhard tone mapping | `reinhard_tonemap` | 0 |
| 16 | Tone curve (S-curve) | `tone_curve` | 0 |
| 17 | Saturation | `saturation` ≠ 1.0 | 2.0 |
| 18 | JPEG export | `jpeg_quality`, `thumbnail`, `panorama` | – |

Monochrome cameras (`camera_monochrome=1`) skip debayering, white balance,
SCNR and saturation.

**Tip:** to see the effect of each step, send the running camera the signal
`SIGUSR2`:

```bash
sudo systemctl kill -s USR2 allsky360-camera
```

The next image is then also saved after each stage (debug images numbered
801 – 810) into the image directory. This happens once; send the signal again
for another series.

---

## 2. Before the fusion (per exposure)

### 2.1 Crop

`crop_image=1` cuts the region `crop_width` × `crop_height` at
`crop_x_offset`/`crop_y_offset` out of the raw frame before anything else
happens. Use it to drop unused sensor area around the fisheye circle. The
camera stops with an error if the region lies outside the sensor.

### 2.2 Debayer

Color cameras deliver a Bayer mosaic. `camera_debayer` selects the algorithm:
`nni` (fastest), `bilinear`, or `vng` (best quality, slowest). See
[CAMERA_SETTINGS.md](CAMERA_SETTINGS.md).

### 2.3 Measurements

The median brightness inside `measure_brightness_area` (fraction of
`image_horizon_radius`) drives the auto-exposure. At night the background
noise is measured inside `nighttime_measure_noise_radius`; the fusion uses it
to down-weight noisy short exposures.

The star detector for the **SQM** estimate runs on the *raw* data of HDR
frame 0 (the longest exposure), before any filter can change the stars.

### 2.4 Multiscale median filter (night)

Removes fine noise and hot pixels at high gain.

| Key | Meaning |
|---|---|
| `nighttime_multiscale_median_filter` | 1 = on |
| `nighttime_multiscale_median_filter_max_scale` | 1 – 3, size of the structures removed |
| `nighttime_multiscale_median_filter_amount` | 0.0 – 1.0, blend with the original |

`max_scale=1` with a low amount leaves the stars sharp; `max_scale=3` with
amount 1.0 makes the image visibly soft.

---

## 3. Fusion and color

### 3.1 HDR fusion

The exposures are merged by a multi-scale (Laplacian pyramid) exposure
fusion, frame by frame to keep the memory peak low. If no frame is
over-exposed, only frame 0 is used. With `hdr=0` the single exposure goes
straight on. Details and all `hdr_*` keys: [HDR-FUSION.md](HDR-FUSION.md).

### 3.2 White balance

The HDR frames are fused **raw**; the white balance is applied once to the
fused linear image. It multiplies red and blue with fixed factors (green is
always 1.0), separately for day and night:

| Key | Example |
|---|---|
| `daytime_white_balance_red`, `daytime_white_balance_blue` | depends on the camera |
| `nighttime_white_balance_red`, `nighttime_white_balance_blue` | 1.0 / 2.0 |

See [DAYTIME_NIGHTTIME_SETTINGS.md](DAYTIME_NIGHTTIME_SETTINGS.md).

### 3.3 SCNR – green / light-pollution reduction (night)

Subtractive Chromatic Noise Reduction removes the green cast typical for
sodium and LED light pollution. It needs white-balanced data, so with HDR it
runs once after the fusion.

| Key | Meaning |
|---|---|
| `scnr_filter` | 1 = on (night only) |
| `scnr_filter_amount` | 0.0 – 1.0 |
| `scnr_filter_protection` | `none`, `average_neutral`, `maximum_neutral` |

`average_neutral` is the gentle choice; `maximum_neutral` removes more green
and can turn green aurora grey – lower the amount on aurora nights.

### 3.4 HDR multiscale transform (experimental)

`hdrmt=1` compresses large-scale brightness differences (e.g. a bright
horizon) while keeping small-scale detail. The strength is fixed in the code,
with separate presets for day and night.

### 3.5 Normalization

The fused image is brought back into 0.0 – 1.0. Values above
`hdr_highlight_knee` are compressed softly instead of being clipped, so the
overall brightness does not depend on the number of HDR frames.

---

## 4. Tonal range and detail

### 4.1 Autostretch

Stretches the histogram so that the `autostretch_min_val` percentile becomes
black and the `autostretch_max_val` percentile becomes white.

| Key | Default |
|---|---|
| `autostretch` | 1 |
| `autostretch_min_val` | 0.01 |
| `autostretch_max_val` | 0.99 |

### 4.2 Black point

Pulls the sky background down to a darker level. Active when
`daytime_move_blackpoint` / `nighttime_move_blackpoint` is greater than 0; the
`*_min_shift_pct`, `*_max_shift_pct` and `*_dark_threshold` keys limit how far
it moves. Autostretch usually does most of this job; keep the black point
small when both are on.

### 4.3 ACDNR – adaptive noise reduction

Smooths flat, low-contrast areas (sky background) strongly and leaves
structures (stars, edges) almost untouched. Luminance and color noise are
treated separately. It runs **before** sharpening, so the sharpener does not
amplify the noise.

| Key | Default | Meaning |
|---|---|---|
| `acdnr_filter` | 1 | on/off |
| `acdnr_filter_lum_stddev` | 1.5 | blur radius for luminance |
| `acdnr_filter_lum_amount` | 0.15 | strength for luminance (0 – 1) |
| `acdnr_filter_lum_iterations` | 2 | passes |
| `acdnr_filter_lum_kernel_size` | 3 | kernel size (odd) |
| `acdnr_filter_chrom_stddev` | 2.0 | blur radius for color |
| `acdnr_filter_chrom_amount` | 0.4 | strength for color (0 – 1) |
| `acdnr_filter_chrom_iterations` | 2 | passes |
| `acdnr_filter_chrom_kernel_size` | 5 | kernel size (odd) |

Color noise can be smoothed much more strongly than luminance noise without
the image looking soft.

### 4.4 Wavelet sharpening

Splits the image into small, medium and large detail layers and amplifies
each one separately.

| Key | Default | Meaning |
|---|---|---|
| `wavelet_sharpen` | 1 | on/off |
| `wavelet_sharpen_gain_small` | 2.0 | fine detail (stars) |
| `wavelet_sharpen_gain_medium` | 0.0 | medium structures |
| `wavelet_sharpen_gain_large` | 0.0 | large structures (clouds) |

A gain of 0.0 leaves a layer unchanged.

### 4.5 Clarity – local contrast

Raises the contrast of mid-sized structures (clouds, Milky Way) without
changing the overall brightness.

| Key | Default | Meaning |
|---|---|---|
| `clarity_filter` | 0 | on/off |
| `clarity_filter_strength` | 0.35 | strength |
| `clarity_filter_radius` | 24 | size of the structures in pixels |
| `clarity_filter_midtone_width` | 0.25 | how far beyond the mid-tones it acts |
| `clarity_filter_preserve_highlights` | 1 | protect bright areas |
| `clarity_filter_mask_mode` | 1 | use a mid-tone mask |

Too much clarity causes halos around the horizon and the moon.

### 4.6 Dehaze

Removes haze and milky light pollution by estimating how much of each pixel
is scattered light.

| Key | Default | Meaning |
|---|---|---|
| `dehaze_amount` | 0.0 | strength, 0.0 = off |
| `dehaze_estimate` | 0.01 | fraction of darkest pixels used to estimate the haze |

---

## 5. Brightness curve and color

### 5.1 Gamma

Applies `output = input ^ gamma`. Values **below 1.0 brighten** the shadows
and mid-tones, values above 1.0 darken them.

| Key | Default | Meaning |
|---|---|---|
| `gamma` | 0.9 | fixed gamma, used when `auto_gamma=0` |
| `auto_gamma` | 0 | 1 = compute gamma from the mean brightness of the scene |
| `auto_gamma_target` | 0.18 | target mean brightness (middle grey) |
| `auto_gamma_min`, `auto_gamma_max` | 0.15, 1.2 | limits for the computed gamma |

The gamma used for each image is printed in the log (`Gamma: … (auto=…)`).

### 5.2 Reinhard tone mapping (optional)

`reinhard_tonemap=1` compresses bright areas (moon, street lights) with the
global Reinhard operator after gamma. `reinhard_white_point=0` detects the
white point automatically; a value > 0 maps this brightness to full white.

### 5.3 Tone curve (optional)

`tone_curve=1` applies a parametric S-curve to the luminance:

| Key | Range | Effect |
|---|---|---|
| `tone_curve_shadow_lift` | 0.0 – 0.5 | lifts shadows and lower mid-tones, black stays black (0.10 subtle … 0.30 strong) |
| `tone_curve_midtone_contrast` | 0.0 – 0.5 | S-curve around 0.5, more punch (0.15 subtle … 0.40 strong) |

### 5.4 Saturation

`saturation` multiplies the color intensity: 1.0 = unchanged, > 1.0 more
color, < 1.0 less. All-sky cameras usually need 1.5 – 2.5 because the fisheye
optics and the light pollution wash out the colors. Too much saturation makes
the sky background blotchy.

---

## 6. Output

After the image processing the camera

1. estimates the **SQM** value (only at astronomical night, see the `sqm_*`
   keys) and measures the **focus** inside `focus_radius`,
2. computes exposure and gain for the next image,
3. prints the date and time (station time zone, `location_timezone`) and the
   other text overlays into the image,
4. writes the JPEG (`jpeg_quality`, nvJPEG on the GPU or libjpeg on the CPU),
   the thumbnail (`thumbnail`, `thumbnail_width`) and the horizon panorama
   (`panorama`, `panorama_*`) into `images/YYYYMMDD/` (UTC day),
5. stores the image metadata in the daily database and in the image JSON,
6. starts the `postprocessing_script` (e.g. the AI sky classification), if
   configured.

---

## 7. Tuning tips

* Change one key at a time and compare the result over a whole night – the
  look changes a lot between moonless nights, moonlight and twilight.
* Noise first, sharpness second: if stars look "crunchy", lower
  `wavelet_sharpen_gain_small` before raising the noise reduction.
* Background too bright or grey: raise `autostretch_min_val` slightly or use a
  small black point shift; do not lower `gamma` further.
* Green or orange sky: correct the white balance first, then use SCNR.
* Keep clarity and dehaze moderate – small changes have large visual effects.
* The config is re-read at the start of every capture cycle, so changes take
  effect with the next image; no restart is needed.

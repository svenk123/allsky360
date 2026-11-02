# HDR Post-Processing Pipeline – Detailed Documentation

This document explains, in clear and practical terms,  
**what** each image-processing function in the HDR pipeline does  
and **why** it is executed in this specific order.

The pipeline operates on **float-based RGB(A) images** in the **0.0 – 1.0** range,  
initially in **linear light**, and then prepares them for display or export.

---

## 1. Why We Need Post-Processing After HDR Merge

After `hdr_merge_rgbf()` you have an **HDR image**, but:

- It often looks **flat or dull**.  
- Bright and dark regions are physically correct but **not visually balanced**.  
- Colors may appear **washed-out or tinted**.  
- Noise from the source images is **still fully visible**.  
- And the image is **not yet in a display-ready color space** (sRGB).

In short:

The HDR merge gives you **all the raw information**, but post-processing turns it into a **visually usable and natural-looking image**.

The following steps are required to:
1. Map the huge HDR range into visible contrast (tone-mapping & stretching),
2. Balance colors and brightness (white balance),
3. Suppress noise while preserving real detail,
4. Enhance structure and contrast (clarity / dehaze),
5. Convert to the display color space and compress for output.

---

## 2. Overview of the Pipeline

Executed in this order:

```C
hdr_merge_rgbf();             // Combine exposures (create HDR)
tone_map_rgbf();              // Compress dynamic range
autostretch_rgbf();           // Adjust histogram / visual exposure
white_balance_rgbf();         // Color and white balance
acdnr_rgbf_cuda();            // Noise reduction
wavelet_sharpen_rgbf_cuda();  // Multi-scale sharpening
clarity_rgbf();               // Local (mid-scale) contrast
dehaze_rgbf();                // Global haze removal
gamma_apply_rgbf();           // Linear → sRGB gamma
saturation_rgbf();            // Adjust color intensity
nvjpeg_encode();              // Export to JPEG
```

Below each function is explained in plain English.

---

## 3. Function Descriptions

### 3.1 hdr_merge_rgbf() – Combine Multiple Exposures

Purpose
Merges several frames with different exposure times (e.g. 15 s + 60 s) into one image.

How it works
* Bright regions (moon, lights, clouds) come from short exposures to avoid clipping.
* Dark regions (stars, faint sky) come from long exposures to keep signal and detail.
* Result: a single float HDR image containing information from all inputs.

Why it’s needed
A single exposure can’t cover the full brightness range without losing data.
HDR merge preserves detail in both shadows and highlights.

Output look
Physically rich but visually flat — further adjustments are required.

---

## 3.2 tone_map_rgbf() – Compress HDR Dynamic Range

Purpose
Maps the physical HDR range into the visible 0.0 – 1.0 range.

How it works
Applies a nonlinear curve (e.g. Reinhard or logarithmic)
to brighten shadows and tame highlights.

Why it’s needed
Without tone mapping:
* Shadows appear black,
* Highlights are blown out.

Tone mapping produces a usable “base” image for further correction.

Result
Visibly balanced, yet often still low in contrast.

---

### 3.3 autostretch_rgbf(min_val, max_val) – Automatic Histogram Stretch

Purpose
Expands the tonal range so the darkest and brightest pixels
span the full display range.

How it works
* Analyzes the luminance histogram.
* Finds percentiles min_val / max_val (e.g. 1 % and 99 %).
* Maps that interval to 0 – 1 to maximize visible contrast.

Why it’s needed
Tone mapping compresses dynamic range but doesn’t fully utilize it.
Auto-stretch adjusts exposure and black/white points automatically.

Typical values
min_val = 0.01 – 0.03, max_val = 0.97 – 0.99.

Result
The image looks brighter, more contrasty, and properly exposed.
It usually replaces the need for move_blackpoint_rgbf().

---

### 3.4 white_balance_rgbf() – Color and White Balance

PThis function performs three tasks at once:

✔ 1. Background Neutralization

It analyzes the central region of the frame
(a configurable sub-frame percentage, e.g. the central 20–50%)
and ensures that the background color becomes neutral (gray).
This removes green/yellow/blue casts from:
	•	light pollution
	•	sensor bias
	•	uneven sky glow
	•	lens flare or gradients

✔ 2. Automatic White Balance

It computes the mean brightness of R, G, and B channels
and scales them so that neutral areas become truly neutral.
This aligns the channels for correct color representation.

✔ 3. Highlight Limiting (3-sigma Clipping)

The function limits extreme outliers using a
3× standard-deviation (“3 sigma”) rule:

max_val = target + 3 * max(stddev_r, stddev_g, stddev_b);

This effectively suppresses:
	•	hot pixels
	•	bright noise spikes
	•	over-exposed flare regions

without damaging normal image content.

3. How the Function Selects the Data Region

The parameter:

subframe_percent

defines how much of the central image area is used for calibration.

Example:
	•	subframe_percent = 30 → use central 30% × 30% area
	•	subframe_percent = 50 → use central half of the image

This avoids including:
	•	bright buildings
	•	the moon
	•	horizon glow
	•	strong stars
	•	camera artefacts

The center of the frame usually gives the best representation
of the true background color of the sky.

---

### 3.5 acdnr_rgbf_cuda() – Adaptive Contrast-Driven Noise Reduction

Purpose
Reduces noise while preserving edges and fine structures.

How it works
* Measures local contrast or variance.
* Smooths low-contrast regions strongly (sky, background).
* Leaves detailed regions mostly untouched.

Why it’s needed
HDR merging and stretching amplify sensor noise.
ACDNR suppresses background noise before sharpening.

Why here
Denoise must run before sharpening; otherwise,
you’d re-blur the enhanced details.

Result
A smoother, cleaner image without destroying structure.

---

### 3.6 wavelet_sharpen_rgbf_cuda() – Multi-Scale Detail Sharpening

Purpose
Enhances real details on several spatial scales
without amplifying noise or halos.

How it works
* Separates the image into small, medium, and large detail layers.
* Boosts each layer by user-defined strength.
* Recombines them into one sharpened image.

Why it’s needed
After denoising, fine structures appear soft.
Wavelet sharpening restores crispness in a controlled way.

Typical parameters
gain_small = 0.6, gain_medium = 0.3, gain_large = 0.1.

Result
Sharper stars, textures, and edges — natural, not over-sharpened.

---

### 3.7 clarity_rgbf() – Local Contrast (Mid-Scale)

Purpose
Adds depth and texture by increasing contrast in mid-sized structures.

How it works
Enhances differences between neighboring areas
without changing overall brightness.

Why it’s needed
Tone-mapped images often appear flat.
Clarity emphasizes shapes and gives a three-dimensional feel.

Order
Runs after sharpening, because sharpening handles tiny details
and clarity works on larger ones.

Result
Improved depth and presence; use moderately to avoid halos.

---

### 3.8 dehaze_rgbf() – Atmospheric Haze Removal

Purpose
Removes haze, fog, or light pollution for clearer, deeper skies.

How it works
Estimates a transmission map from brightness and color,
then restores contrast and color in washed-out regions.

Why it’s needed
All-sky and landscape images often suffer from atmospheric scatter
that makes them appear milky.

Order
Executed after clarity, since dehaze acts globally
while clarity operates locally.

Result
Cleaner sky, stronger contrast; excessive use may cause unnatural color.

---

### 3.9 gamma_apply_rgbf() – Convert Linear to sRGB Gamma

Purpose
Transforms the image from linear light to the nonlinear sRGB space
expected by monitors and JPEG viewers.

How it works
Applies a gamma curve (pow(x, 1/2.2) or similar)
to match human light perception.

Why it’s needed
Without gamma correction:
* Shadows look too dark,
* Midtones look wrong,
* Overall brightness feels off.

Order
Gamma must be applied after all linear operations, because filters like sharpening and denoise rely on linear math.

Result
A perceptually correct, display-ready image.

---

### 3.10 saturation_rgbf() – Adjust Color Intensity

Purpose
Increases or decreases the vividness of colors.

How it works
Converts RGB to a color space (HSV/HSI) or separates luminance, then multiplies the saturation component by a factor.

Why after gamma
Color perception depends on display brightness. Doing this in linear space causes wrong hues or noise amplification.

Result
More vibrant or more subdued colors, depending on the factor. Enhances auroras and sunsets — but avoid oversaturation.

---

### 3.11 nvjpeg_encode() – Final JPEG Export

Purpose
Converts the processed float image to 8-bit RGB and saves it as JPEG.

How it works
* Scales 0.0 – 1.0 floats to 0–255 bytes.
* Uses NVIDIA’s NVJPEG encoder on the GPU.
* Outputs a compact, ready-to-view file.

Why last
After encoding, color and tone information are fixed;
no further processing should be done.

Result
A display-ready, compressed image for web, archive, or dashboard use.

---

## 4. Full Processing Chain Summary

1. Create HDR – Combine all exposures into one high-range image.
2. Normalize Dynamic Range – Tone-map and stretch brightness for visibility.
3. Balance Color & Reduce Noise – White balance and denoise for clarity.
4. Enhance Details & Depth – Sharpen, add clarity, remove haze.
5. Prepare for Output – Apply gamma, adjust saturation, export to JPEG.

Example Implementation

hdr_merge_rgbf();             // Merge multiple exposures
tone_map_rgbf();              // Compress HDR to visible range
autostretch_rgbf(0.01,0.99);  // Optimize brightness & contrast
white_balance_rgbf();         // Neutralize color channels
acdnr_rgbf_cuda();            // Noise reduction
wavelet_sharpen_rgbf_cuda();  // Fine-detail sharpening
clarity_rgbf();               // Mid-scale contrast enhancement
dehaze_rgbf();                // Global haze correction
gamma_apply_rgbf();           // Linear → sRGB conversion
saturation_rgbf();            // Adjust color intensity
nvjpeg_encode();              // Export as JPEG

---

## 5. Implementation Notes

* All operations before gamma_apply_rgbf() must work in linear space.
* Use persistent GPU memory between CUDA steps to avoid transfer overhead.
* Log luminance statistics (avgL, p01, p99) after tone mapping or stretch for adaptive parameter tuning.
* Keep sharpening and dehaze moderate — small parameter changes can have large visual effects.

---

## Summary

This pipeline transforms raw HDR merges into natural, display-ready images:

* Tone & Exposure -> balanced
* Color -> neutral
* Noise -> reduced
* Details -> enhanced
* Output -> visually clean, ready for publication

It provides a reproducible, structured workflow suitable for astrophotography, allsky cameras, or any imaging system producing high-dynamic-range data.

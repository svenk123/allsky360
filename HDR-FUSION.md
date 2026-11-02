# 🌌 HDR Fusion Pipeline – Capture & Processing Documentation

This document explains how the **HDR fusion process** works in the Allsky image pipeline,  
how it is configured via `config.ini`, and what each HDR-related parameter does in practice.

The goal is to keep the explanation understandable even for non-experts,  
without going into formulas.

---

## 1. Enabling HDR in `config.ini`

HDR fusion can be turned on in `config.ini` using:

```ini
hdr = 1
```

* hdr = 0 HDR disabled, only a single exposure is used.
* hdr = 1 HDR enabled, the system can capture and fuse up to 5 frames with different exposure times.

The HDR pipeline then works in two phases:

1. Capture phase- decide how many frames to take and with which exposure times.
2. Fusion phase - combine those frames into a single HDR image.

---

## 2. Exposure Bracketing: How HDR Frames Are Captured

### 2.1 Maximum Number of HDR Frames

* The system captures up to 5 images per HDR cycle:
* one longest exposure (reference),
* and up to four additional shorter exposures if bright regions are detected as overexposed.

Not every scene needs all 5 frames.
If the scene is not extremely bright, fewer frames are enough.

---

## 2.2 Daytime vs Nighttime Exposure Factor

Two parameters control how aggressively shorter exposures are chosen:

```ini
daytime_hdr_exposure_factor   = 5.0
nighttime_hdr_exposure_factor = 10.0
```

Conceptual meaning:
* These factors define how much exposure span the HDR stack should cover.
* The shortest exposure will be roughly the longest exposure divided by this factor, depending on how much clipping is detected.

Daytime example (daytime_hdr_exposure_factor = 5.0):
* The sky is bright, sun or strong reflections may be visible.
* The system uses a factor around 5 to make sure:
 * long exposures still see shadows and clouds,
 * short exposures prevent the sun and bright clouds from completely blowing out.

Nighttime example (nighttime_hdr_exposure_factor = 10.0):
* At night, the dynamic range can be enormous:
 * very faint stars and nebulosity,
 * but also bright lamps, the moon, or illuminated clouds.
* A larger factor (10.0) allows the system to:
 * keep very long exposures for faint structures,
 * while still adding much shorter exposures for bright objects.

You can think of these factors as:

*How much difference in brightness do I want my HDR stack to cover between darkest details and brightest highlights?*

---

## 3. Overexposure Check Per Frame

During the capture loop, each newly captured frame is checked for overexposure using:

```C
hdr_check_overexposure_rgbf1()
```

This function decides whether:
* the current frame is too bright (too many clipped pixels), and
* whether another, shorter exposure should be captured.

### 3.1 Overexposure Threshold Parameters

Two config.ini parameters control this behavior:

```ini
hdr_clipping_threshold = 0.98
hdr_min_clipped_pixels = 0.00001
````

**hdr_clipping_threshold = 0.98**
This defines which pixel brightness is considered *almost clipped*.
In practice:
* A pixel with luminance near 1.0 (e.g. ≥ 0.98) is treated as *at the limit*.
* Values above this threshold are candidates for being too bright or already clipped.

**hdr_min_clipped_pixels = 0.00001**
This is the minimum fraction of pixels that may exceed the clipping threshold before the frame is considered overexposed.

Example:
* 0.00001 corresponds to 0.001 % of all pixels.
* On a 4-megapixel image (~4,000,000 pixels), this is about 40 pixels.
* If more than ~40 pixels are above the clipping threshold, the frame is classified as overexposed.

Practical logic:
* If very few pixels are clipped (e.g. a tiny star or a single lamp), that is usually acceptable.
* If many pixels blow out (e.g. bright clouds or large reflections), the system reacts by capturing a shorter exposure.

---

## 4. Reference Frame and Exposure Statistics

### 4.1 Reference Frame (Longest Exposure)

Among all captured HDR frames. The frame with the longest exposure time is treated as the reference image.
This reference usually contains the best signal-to-noise ratio (especially in darker regions), is used as a baseline for brightness and color during fusion.

Shorter exposures are primarily there to rescue bright areas that would be overexposed in the longest frame.

---

## 4.2 Measuring Per-Frame Mean Brightness

For each image and each color channel, the pipeline computes mean/median brightness using:

```C
hdr_compute_mean_rgbf1()
```

What this function does:
For each HDR input frame:
* computes luminance or per-channel statistics (R, G, B),
* measures how bright the frame is on average.

These values are later used in the fusion step to:
* normalize frames relative to each other,
* avoid color shifts between exposures,
* account for differences in overall brightness.

Why it matters:
Without per-frame brightness statistics, combining multiple exposures can:
* cause unnatural color casts,
* over- or underweight certain frames,
* lead to biased HDR results.

---

## 5. Multi-Scale HDR Fusion with Laplacian Pyramids

The core of the HDR process is:

```C
hdr_multi_scale_fusion_laplacian_rgbf1()
```

This function performs a multi-scale Laplacian fusion of up to 5 frames.

### 5.1 Intuitive Idea

Instead of simply averaging frames or blending them with a single mask, the image is broken down into several detail layers:
* very fine details (stars, small edges),
* medium structures (cloud shapes, buildings, horizon structures),
* large structures (overall sky gradients, bright/dark regions).

For each layer, the fusion decides which exposure contributes what:
* In bright areas, the shorter exposures are preferred (to avoid clipping).
* In dark and mid-tone areas, the longer exposures are preferred (better signal, less noise).
* The result is reconstructed from all layers into one final HDR image.

You can think of it as:

**For every detail size, pick the best information from the most suitable exposure and blend it smoothly.**

---

### 5.2 Key Parameters of hdr_multi_scale_fusion_laplacian_rgbf1()

These are controlled in config.ini:

```ini
hdr_y_max_expected          = 1.0
hdr_contrast_weight_strength = 0.5
hdr_pyramid_levels_override = 6
```

#### 5.2.1 hdr_y_max_expected = 1.0

This parameter defines how **bright is bright** in the HDR inputs, so the fusion can handle highlights consistently. It tells the algorithm what maximum luminance range to expect in the input images.
It is used to normalize weights and avoid out-of-range assumptions.

Typical usage:
1.0 means:
* The input frames are expected to be roughly normalized to the 0.0–1.0 range after preliminary scaling/normalization.
* If your inputs can be temporarily higher than 1.0, you might set this to a higher value (advanced use).

---

#### 5.2.2 hdr_contrast_weight_strength = 0.5

This parameter tells the algorithm how much to **trust** contrast when deciding which exposure contributes the details. It controls how strongly local contrast influences the weighting of each exposure.

Behavior:

Low value (e.g. 0.0):
* Fusion relies mainly on exposure-based criteria (e.g. highlight protection, basic brightness).

High value (e.g. 1.0):
* Regions with higher local contrast (sharp details, structure) are strongly favored from the best-exposed frame.

0.5 is a balanced setting. Both exposure and local contrast influence the decision. High-contrast regions (details) are chosen from the most suitable exposure, but without completely ignoring the exposure logic.

---

#### 5.2.4 hdr_pyramid_levels_override = 6

This parameter controls how many *layers of detail* the HDR fusion uses to mix the exposures.
It sets the number of pyramid levels used in the multi-scale Laplacian fusion.

What it means:
Each pyramid level represents a different detail size:
* lower levels result in very fine detail,
* higher levels result in larger structures and gradients.

6 levels means:
* a good coverage from fine to fairly large structures,
* smoother transitions between exposures across all scales.

Effects:
More levels:
* better handling of large gradients and complex scenes,
* smoother, more natural blending,
* slightly higher computational cost.
Fewer levels:
* faster, but less flexible,
* transitions may become harsher or less **HDR-like**.

---

## 6. Optional HDR Tone Mapping (hdrmt=1)

After HDR fusion, you can optionally apply a high dynamic range tone-mapping step.

In config.ini:
```ini
hdrmt = 1
```

* hdrmt = 0 no additional HDR tone mapping after fusion.
* hdrmt = 1 apply an extra tone-mapping pass on the fused HDR result.

What it does:
* Further compresses or reshapes the dynamic range, specifically tailored for high dynamic range content.

Useful when:
* the scene still has very bright and very dark areas after fusion,
* you want a more **HDR-look** or better visibility in shadows.

Effect:
* Can improve overall balance and visibility,
* But should be tuned carefully to avoid an unnatural “overcooked HDR” look.

---

## 7. Normalizing the Result: hdr_normalize_range_rgbf1()

After optional HDR tone mapping, the pipeline calls:

```C
hdr_normalize_range_rgbf1()
```

Purpose:
* Ensures that the final HDR-fused image is normalized to a clean range, usually 0.0–1.0.
* Removes any remaining offsets or scale drift from the fusion and tone mapping.

Why this is important:
All later post-processing steps (noise reduction, sharpening, clarity, dehaze, gamma, saturation)
assume a well-defined, normalized input range.
It also prevents unwanted clipping or washed-out images later.

---

## 8. HDR Fusion in the Overall Pipeline

In the full processing chain, the HDR steps look like this:

// HDR capture and fusion
// (controlled by hdr=1 and related parameters in config.ini)
capture_hdr_frames();                     // capture up to 5 exposures
hdr_merge_rgbf();                         // multi-frame HDR fusion (Laplacian)
if (hdrmt == 1) {
    hdr_tone_mapping_rgbf1();             // optional HDR-specific tone mapping
}
hdr_normalize_range_rgbf1();              // normalize to a clean 0..1 range

// Then continue with the standard post-processing pipeline:
tone_map_rgbf();                          // global tone mapping (if used in your design)
autostretch_rgbf(0.01, 0.99);             // visual stretching
white_balance_rgbf();
acdnr_rgbf_cuda();
wavelet_sharpen_rgbf_cuda();
clarity_rgbf();
dehaze_rgbf();
gamma_apply_rgbf();
saturation_rgbf();
nvjpeg_encode();

(The exact placement of tone_map_rgbf() vs. hdrmt depends on how you structure your pipeline;
the key idea is: HDR fusion → HDR tone mapping (optional) → normalization → main post-processing.)

---

## Summary
* HDR is enabled with hdr = 1 and can use up to 5 exposures.
* Exposure factors (daytime_hdr_exposure_factor, nighttime_hdr_exposure_factor) control how much brightness range is covered.
* Overexposure detection (hdr_check_overexposure_rgbf1(), hdr_clipping_threshold, hdr_min_clipped_pixels) decides when to add shorter exposures.
* Per-frame brightness statistics (hdr_compute_mean_rgbf1()) help to keep fusion balanced and color-stable.
* The core fusion happens in hdr_multi_scale_fusion_laplacian_rgbf1() with parameters controlling luminance range, color handling, contrast weighting, and the number of scales.
* Optional HDR tone mapping (hdrmt = 1) refines the final dynamic range.
* hdr_normalize_range_rgbf1() prepares the output for the standard post-processing pipeline.

This gives you a flexible, robust HDR fusion system that is both technically sound
and understandable for non-experts looking into the code or documentation.

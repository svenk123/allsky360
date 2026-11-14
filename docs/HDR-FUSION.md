# HDR Capture and Fusion

An all-sky image has to show faint stars and the bright moon, a dark
landscape and bright clouds near the sun at the same time. A single exposure
cannot do that. With HDR enabled, Allsky360 takes up to five exposures of
different length and fuses them into one image.

This document explains how the exposures are chosen, how they are fused, and
what the `hdr_*` keys in `config.ini` do. What happens to the fused image
afterwards is described in [POSTPROCESSING.md](POSTPROCESSING.md).

---

## 1. Switching HDR on and off

```ini
hdr=1
```

| Value | Effect |
|---|---|
| `0` | Only one exposure per image. Fastest, lowest memory use. |
| `1` | Up to 5 exposures per image (default). Shorter exposures are only taken when the first one is over-exposed, so in a dark night HDR costs nothing. |

---

## 2. How the exposures are chosen

1. **Frame 0** uses the exposure time and gain of the auto-exposure. It is the
   longest exposure and the reference for brightness and color.
2. After each frame the camera checks whether the frame is **over-exposed**
   (see section 3). If not, the series ends here.
3. If it is, the next frame is shorter by `*_hdr_exposure_factor` **EV
   steps** (one EV = half the exposure time):

   ```
   t(i) = t(0) / 2^(i × factor)
   ```

4. The series ends after 5 frames, as soon as a frame is not over-exposed, or
   when the next exposure would be shorter than `camera_min_exposure`. In the
   last case one final frame is taken at the camera minimum if that still
   adds dynamic range.

All frames use the same gain.

| Key | Default | Meaning |
|---|---|---|
| `daytime_hdr_exposure_factor` | 2.0 | EV steps between two frames by day |
| `nighttime_hdr_exposure_factor` | 2.0 | EV steps between two frames by night |

Example with frame 0 at 60 s:

| Factor | Frame 0 | Frame 1 | Frame 2 | Frame 3 | Frame 4 |
|---|---|---|---|---|---|
| 1.0 | 60 s | 30 s | 15 s | 7.5 s | 3.75 s |
| 2.0 | 60 s | 15 s | 3.75 s | 0.94 s | 0.23 s |
| 3.0 | 60 s | 7.5 s | 0.94 s | 0.12 s | 0.015 s |

A larger factor covers a larger brightness range with the same number of
frames, but the fusion has to bridge larger steps. 1.0 – 2.0 gives the best
results. If the log shows `HDR ladder stops before frame …`, the steps hit the
camera minimum; lower the factor to use more frames.

---

## 3. Over-exposure check

| Key | Default | Meaning |
|---|---|---|
| `hdr_clipping_threshold` | 0.99998 | Brightness (0 – 1) from which a pixel counts as clipped. Must be just below the sensor's maximum value; depends on the bit depth. |
| `hdr_min_clipped_pixels` | 0.00001 | Fraction of all pixels that may be clipped before the frame counts as over-exposed. 0.00001 = 0.001 %, about 50 pixels of a 5-megapixel image. |

A few clipped pixels (a bright star, a single lamp) do not trigger another
frame. Large clipped areas (moon, lit clouds, the sun) do.

---

## 4. The fusion

The frames are fused by **multi-scale exposure fusion**: every frame is split
into detail layers (Laplacian pyramid) – fine detail such as stars, medium
structures such as cloud edges, and large brightness gradients. For every
pixel and every layer a weight decides how much each frame contributes:

* **well exposed** pixels (near the brightness of the reference frame) get a
  high weight,
* **clipped** pixels get a low weight,
* pixels with **high local contrast** (real detail) get a higher weight,
* at night, pixels of short exposures that are only **noise** get a lower
  weight.

The layers are blended with these weights and put back together. Because the
blending happens per detail size, there are no hard edges between areas taken
from different frames. The frames are added one after another, so the memory
need does not grow with the number of frames.

| Key | Default | Meaning |
|---|---|---|
| `hdr_weight_sigma` | 0.8 | Width of the "well exposed" weight curve. Smaller = each pixel comes more strictly from the frame where it is best exposed; larger = softer blending. |
| `hdr_weight_clip_factor` | 0.90 | How strongly near-clipped pixels are suppressed (0 – 1). `0` = determined automatically from the shortest frame. |
| `hdr_contrast_weight_strength` | 0.6 | Influence of local contrast on the weights (0 = none, 1 = strong). |
| `hdr_noise_fusion_strength` | 1.0 | Down-weighting of short exposures where they only contain noise (0 = off, 1 = full). Uses the noise measured at night (`nighttime_measure_noise_radius`); has no effect by day. |
| `hdr_y_max_expected` | 1.0 | Brightness scale for the weights. `0` = detect from the image. A value > 0 is raised automatically with the number of frames (×1.2 for 2 frames, ×1.4 for 3, …). |
| `hdr_pyramid_levels_override` | 0 | Number of pyramid levels, 1 – 6; `0` = automatic from the image size. More levels give smoother large-scale transitions. |

If frame 0 is not over-exposed, the fusion is skipped and frame 0 is used
directly.

---

## 5. After the fusion

1. **White balance** – the frames are fused raw; the day or night white
   balance is applied once to the fused image.
2. **SCNR** (night) – green reduction, see [POSTPROCESSING.md](POSTPROCESSING.md).
3. **HDR multiscale transform** (`hdrmt=1`, experimental) – compresses large
   brightness differences such as a bright horizon while keeping fine detail.
   The strength is fixed in the code, with separate presets for day and night.
4. **Normalization** – the fused image keeps the brightness of frame 0; only
   values above the knee are compressed softly into the range up to white:

| Key | Default | Meaning |
|---|---|---|
| `hdr_highlight_knee` | 0.8 | Start of the highlight shoulder (0 – 0.99). Higher = brighter highlights, less room for detail in them. `0` = old behaviour: divide by the image maximum (the image gets darker with every additional frame). |

Then the normal image processing continues with autostretch, noise reduction
and so on.

---

## 6. Checking the result

The page **Camera → HDR** of the web interface shows the exposures of the
latest image, the fusion parameters, the measured noise and – if enabled –
the weight maps.

| Key | Default | Meaning |
|---|---|---|
| `hdr_save_exposures` | 1 | Save a preview of every exposure (`latest_image_t0.jpg` … `t4.jpg`) for the HDR page. |
| `hdr_weight_maps` | 0 | Save the weight map of every frame as an image (`hdr_weight_map_img0.jpg` …). Shows which areas come from which frame. |
| `hdr_weight_stats` | 1 | Print weight statistics of every frame to the log. |

The camera log (`journalctl -u allsky360-camera -f`) shows for every image the exposure of every frame, whether
it was over-exposed and how many frames were fused
(`Can use N of 5 HDR frames`).

---

## 7. Tips

* **Moon or sun burnt out:** raise the exposure factor so the shortest frame
  becomes shorter, or lower `hdr_clipping_threshold` slightly (e.g. 0.995) so
  shorter frames are taken earlier.
* **Halos or visible borders around bright areas:** raise `hdr_weight_sigma`
  or lower `hdr_contrast_weight_strength`.
* **Noisy bright areas at night:** keep `hdr_noise_fusion_strength` at 1.0 and
  make sure the noise measurement area covers dark sky.
* **Highlights too grey or flat:** raise `hdr_highlight_knee` towards 0.9.
* **Weak computer or little memory:** a larger exposure factor reaches the
  camera minimum with fewer frames; `hdr=0` switches HDR off completely.

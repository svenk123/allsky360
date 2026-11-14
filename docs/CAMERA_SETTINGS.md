# Camera Settings

These parameters control your camera's behavior and are important for the auto-exposure algorithm. Most values are sensor-dependent, so check your camera's documentation if you're unsure.

## Exposure Times

### camera_max_exposure

The maximum exposure time in seconds that the auto-exposure algorithm is allowed to use. The default is 60 seconds, which is usually sufficient for most all-sky captures.

If you're at a very dark site, you can increase this value to capture fainter stars. However, you'll also increase the risk of motion blur from wind or vibrations. 60 seconds is usually a good compromise.

### camera_min_exposure

The minimum exposure time in seconds for the auto-exposure algorithm. On many modern sensors, the minimum is 32 microseconds (0.000032 seconds).

If you set this value too high, the algorithm won't be able to react quickly enough in very bright conditions (like at sunrise). The default value of 0.000032 seconds (32µs) works well for most cameras.

Set it to the real minimum of your camera (see its data sheet or the INDIGO control panel). The HDR exposure series stops at this value: a shorter request would be clamped silently by the driver and only produce a duplicate frame. The example configuration uses 0.0007 s for the Player One Uranus-C.

## Gain

### camera_min_gain and camera_max_gain

These two parameters define the allowed gain range. Gain is specified as an integer and varies significantly between camera types.

**camera_min_gain** is usually 0, which corresponds to the sensor's native sensitivity. Some cameras allow negative values (similar to ISO 50 or ISO 25 on DSLRs), but that's rare.

**camera_max_gain** varies a lot between cameras. Typical values are between 200 and 400, though some cameras go much higher. Your camera's documentation is the best source here – or just try it out. If the auto-exposure algorithm reaches the maximum gain but the image is still too dark, you can increase the value.

Important: High gain values mean more noise. The algorithm tries to compensate, but there's a limit. On my camera, I've capped the maximum at 300 because beyond that the noise just becomes too dominant.

## Monochrome Cameras vs. Color Cameras

### camera_monochrome

If your camera has a monochrome sensor (no color filter array), set this parameter to `1`. This tells the system to skip all color-related operations like saturation adjustment, color calibration, and white balance – operations that don't make sense for grayscale images.

The default is `0`, which assumes a color sensor. Most astronomy cameras these days are color cameras, so you'll only need to change this if you're using a dedicated monochrome camera.

Monochrome sensors are often used for narrowband imaging or when you want maximum sensitivity, since they don't have the color filter array that blocks some light. But for general all-sky photography, color cameras work fine and give you more flexibility.

### camera_debayer

Most astronomy cameras use a Bayer filter pattern on the sensor, where each pixel only captures one color (red, green, or blue). The debayer algorithm converts this single-color-per-pixel data into a full-color image by interpolating the missing color values.

You can choose from three algorithms:

**`nni`** (Nearest Neighbor Interpolation) – This is the simplest and fastest method. It just copies the nearest pixel's color value. It's quick, but you'll notice some color artifacts, especially around stars and fine details. I'd only use this if processing speed is critical and image quality isn't.

**`bilinear`** – This uses bilinear interpolation to estimate missing color values by averaging neighboring pixels. It's a good middle ground: better quality than NNI without being too slow. The results are smoother, but you might still see some color fringing on bright stars.

**`vng`** (Variable Number of Gradients) – This is the default and usually the best choice. It analyzes gradients in the image to make smarter decisions about color interpolation. The results are noticeably better, with less color artifacts and sharper details. It's a bit slower than the other methods, but for all-sky images the quality difference is worth it.

The default is `vng`, which works well for most cases. If you're processing a lot of images and speed becomes an issue, you can try `bilinear` as a compromise. I'd avoid `nni` unless you really need the speed.

## Binning

### camera_binning2x2

This parameter enables or disables 2x2 binning, if your camera supports it. Binning combines adjacent pixels on the sensor – with 2x2 binning, four pixels (a 2x2 block) are combined into one.

The main benefit is increased sensitivity: you get roughly 4x the light per "pixel" (since you're combining 4 pixels), which means you can use shorter exposure times or capture fainter objects. The trade-off is lower resolution – your image will be half the width and height.

Set this to `1` to enable 2x2 binning, or `0` to disable it (full resolution). The default is `0`.

Whether binning makes sense depends on what you're doing. For all-sky cameras, full resolution is usually better since you want to capture as much detail as possible. But if you're having trouble getting enough light or need faster processing, binning can help. Some cameras also have less read noise when binned, which can improve image quality in very dark conditions.

Not all cameras support binning, so if your camera doesn't, this setting won't do anything. Check your camera's documentation to see if it's supported.

## Practical Tips

When setting up a new camera, start with the default values. Most cameras work fine with them. Only adjust if you notice the auto-exposure algorithm hitting limits (like always staying at maximum exposure during night captures).

For the gain values, you should really check your camera's documentation – they're too specific to guess.

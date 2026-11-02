# Camera Settings

These parameters control your camera's behavior and are important for the auto-exposure algorithm. Most values are sensor-dependent, so check your camera's documentation if you're unsure.

## Exposure Times

### camera_max_exposure

The maximum exposure time in seconds that the auto-exposure algorithm is allowed to use. The default is 60 seconds, which is usually sufficient for most all-sky captures.

If you're at a very dark site, you can increase this value to capture fainter stars. However, you'll also increase the risk of motion blur from wind or vibrations. 60 seconds is usually a good compromise.

### camera_min_exposure

The minimum exposure time in seconds for the auto-exposure algorithm. On many modern sensors, the minimum is 32 microseconds (0.000032 seconds).

If you set this value too high, the algorithm won't be able to react quickly enough in very bright conditions (like at sunrise). The default value of 0.000032 seconds (32µs) works well for most cameras.

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

### camera_r_factor, camera_g_factor, camera_b_factor

These three parameters are the weighting factors for red, green, and blue pixels when converting RAW data to a color image. They account for the different sensitivity of various pixel types on the sensor.

If you don't specify these parameters, the Rec.709 standard values are used:

- `camera_r_factor = 0.2126`
- `camera_g_factor = 0.7152`
- `camera_b_factor = 0.0722`

For most cases, this is perfectly fine. Rec.709 is an established standard and works well with most modern sensors.

If you notice the colors aren't quite right – too much green or too little red, for example – you can experiment here. The three values should add up to 1.0, otherwise the image will be too bright or too dark.

For example: On some sensors, the green channel is a bit more sensitive. You could reduce `camera_g_factor` slightly (e.g., to 0.70) and distribute the difference to the other two channels.

Here are some example values for common sensors:

| Sensor | R Factor | G Factor | B Factor | Notes |
| --- | --- | --- | --- | --- |
| Rec.709 (Recommended) | 0.2126 | 0.7152 | 0.0722 | Default choice. Best for general use, DSLR and unknown sensors. |
| Sony IMX571 (ASI2600/QHY268) | 0.2944 | 0.5021 | 0.2035 | 26MP APS-C BSI (STARVIS). Excellent balance for broadband. |
| Sony IMX533 (ASI533/Saturn-C/Saturn-M) | 0.2910 | 0.5072 | 0.2018 | 9MP 1" Square BSI (STARVIS). Popular square format. |
| Sony IMX455 (ASI6200/QHY600) | 0.2987 | 0.5001 | 0.2013 | 61MP Full Frame BSI (STARVIS). Full frame reference sensor. |
| Sony IMX294 (ASI294) | 0.3068 | 0.5008 | 0.1925 | 11.7MP 4/3" BSI. High sensitivity 4/3 format. |
| Sony IMX183 (ASI183) | 0.2967 | 0.4983 | 0.2050 | 20MP 1" BSI. High resolution 1-inch sensor. |
| Sony IMX178 (ASI178) | 0.2346 | 0.5206 | 0.2448 | 6.4MP 1/1.8" BSI. High resolution entry-level sensor. |
| Sony IMX224 (ASI224) | 0.3402 | 0.4765 | 0.1833 | 1.27MP 1/3" BSI. Classic planetary sensor. High Red response. |
| Sony IMX585 (ASI585) - STARVIS 2 | 0.3431 | 0.4822 | 0.1747 | 8.3MP 1/1.2" BSI (STARVIS 2). NIR optimized. Excellent for H-Alpha/Narrowband. |
| Sony IMX662 (ASI662) - STARVIS 2 | 0.3430 | 0.4821 | 0.1749 | 2.1MP 1/2.8" BSI (STARVIS 2). Planetary/Guiding. High Red/NIR sensitivity. |
| Sony IMX678/715 - STARVIS 2 | 0.3426 | 0.4825 | 0.1750 | BSI (STARVIS 2). High resolution planetary/security sensors. |
| Panasonic MN34230 (ASI1600/QHY163) | 0.2650 | 0.5250 | 0.2100 | 4/3" CMOS. Classic Mono/OSC sensor. |

If your camera uses one of these sensors, you can use the corresponding values. For sensors not listed here, Rec.709 is usually a safe starting point.


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

# Daytime and Nighttime Settings

These parameters control how the camera behaves during daytime and nighttime captures. The system automatically switches between day and night modes based on the sun's position, and each mode has its own set of exposure, gain, and processing parameters.

## Daytime Settings

### daytime_capture

Enable or disable image capture during daytime hours. Set this to `1` to capture images during the day, or `0` to only capture at night.

If you're primarily interested in nighttime astronomy, you might want to disable daytime captures to save storage space and processing time. However, daytime captures can be useful for monitoring sky conditions, cloud cover, and creating time-lapse videos that show the full day/night cycle.

The default is `1` (enabled). Even if you're not interested in daytime images, having them can help with troubleshooting and understanding your system's behavior.

### daytime_startexposure

The initial exposure time in seconds when entering daytime mode or when the system starts during daytime hours.

For daytime captures, you'll typically want very short exposures – the sky is extremely bright compared to nighttime. The default value of 0.0045 seconds (4.5 milliseconds) is a good starting point for most cameras and locations.

If your images are consistently overexposed during daytime, try reducing this value (e.g., 0.002 or 0.003 seconds). If they're too dark, increase it slightly, but be careful – daytime exposures longer than about 0.01 seconds will likely be overexposed unless you're in very cloudy conditions.

The auto-exposure algorithm will adjust from this starting point, but having a reasonable initial value helps it converge faster.

### daytime_startgain

The initial gain value when entering daytime mode. For daytime captures, you'll almost always want this set to `0` (the camera's native sensitivity).

Daytime conditions provide plenty of light, so there's no need to increase gain. Higher gain values would just add noise without any benefit. The auto-exposure algorithm will primarily adjust exposure time during daytime, not gain.

### daytime_meanbrightness

The target mean brightness level for the auto-exposure algorithm during daytime, expressed as a fraction between 0 and 1.

The default is 0.20 (20% of maximum brightness), which works well for most daytime conditions. This gives you a good balance – bright enough to see details in clouds and sky, but not so bright that highlights are completely blown out.

If your daytime images look too dark, try increasing this to 0.25 or 0.30. If they're too bright and losing detail in clouds, try reducing it to 0.15. The algorithm will continuously adjust exposure to maintain this target brightness.

Keep in mind that daytime conditions can vary dramatically – a bright sunny day vs. heavy cloud cover. The algorithm will adapt, but the mean brightness target determines the general exposure level it's aiming for.

### daytime_white_balance_blue

The blue channel multiplier for white balance during daytime captures. This adjusts the color temperature to make daytime images look more natural.

The default is 1.4, which typically produces a slightly warmer, more natural-looking sky. Higher values (e.g., 1.6) will make the image warmer/more yellow, while lower values (e.g., 1.2) will make it cooler/more blue.

Daytime white balance is tricky because the color temperature of daylight changes throughout the day – warmer at sunrise/sunset, cooler at midday. The value here is a compromise that works reasonably well across most daytime conditions.

If your daytime images have a strong color cast, you can adjust this along with `daytime_white_balance_red` to fine-tune the colors.

### daytime_white_balance_red

The red channel multiplier for white balance during daytime. The default is 1.0, which means no adjustment to the red channel.

Together with `daytime_white_balance_blue`, this controls the overall color balance. The green channel is typically left at 1.0 (no adjustment) since it's used as the reference.

If you find your daytime images are too red, you can reduce this value (e.g., 0.9). If they're too cyan (lacking red), increase it (e.g., 1.1). But in most cases, adjusting the blue channel is sufficient.

### daytime_move_blackpoint

Enable or disable black point adjustment during daytime processing. Set to `1` to enable, `0` to disable.

When enabled, the system automatically adjusts the black point (the darkest value in the image) to improve contrast and make the image look better. This is particularly useful for daytime images where you want good contrast in clouds and sky.

The default is 0.01, which enables a subtle adjustment. The actual amount of adjustment is controlled by the min/max shift parameters below.

If you're using `autostretch` instead (which is often preferred), you can disable this by setting it to `0`.

### daytime_move_blackpoint_min_shift_pct

The minimum percentage by which the black point can be shifted when adjusting contrast. The default is 0.005 (0.5%).

This sets a lower bound on how much the black point adjustment can change the image. Very small values mean the adjustment will be subtle, while larger values allow more aggressive contrast enhancement.

In practice, you rarely need to change this. The default works well for most cases.

### daytime_move_blackpoint_max_shift_pct

The maximum percentage by which the black point can be shifted. The default is 0.02 (2%).

This prevents the black point adjustment from being too aggressive and crushing shadows. If the algorithm wants to shift the black point by more than this amount, it will be capped at this maximum.

For daytime images, 2% is usually plenty. Going higher might make clouds look too contrasty or create harsh transitions.

### daytime_move_blackpoint_dark_threshold

A threshold value that determines when black point adjustment should be applied. The default is 0.2.

If the image is darker than this threshold (mean brightness below 0.2), the black point adjustment will be more conservative. This prevents the algorithm from over-adjusting already-dark images.

You typically don't need to change this unless you're working in very specific lighting conditions.

### daytime_hdr_exposure_factor

The exposure factor used for HDR (High Dynamic Range) processing during daytime. The default is 2.0.

HDR processing captures multiple exposures and combines them to capture a wider range of brightness levels. This is particularly useful for daytime images where you have both bright sky and darker foreground elements.

The exposure factor determines how much longer the second exposure should be compared to the base exposure. A factor of 2.0 means the second exposure is twice as long, which captures more detail in shadows while the first exposure captures highlights.

Valid values are typically 1.0 (no HDR, just use the base exposure) or 2.0 (double the exposure for the second frame). Higher values like 3.0 or 4.0 are possible but usually create too much difference between exposures, making fusion difficult.

If HDR is disabled (`hdr=0`), this parameter has no effect.

## Nighttime Settings

### nighttime_sunangle

The sun's altitude in degrees below the horizon at which the system switches from daytime to nighttime mode. The default is -6 degrees.

This is the "astronomical twilight" threshold. When the sun is 6 degrees or more below the horizon, the system considers it nighttime and switches to nighttime capture settings.

You can adjust this to change when the day/night transition occurs:

- **-6 degrees** (default): Astronomical twilight. The sky is quite dark, suitable for most astronomical imaging.

- **-12 degrees**: Very dark conditions. The system will wait longer before switching to night mode, ensuring the sky is fully dark.

- **-3 degrees**: Earlier transition. The system switches to night mode sooner, which might be useful if you want to start capturing earlier, but the sky will still have some twilight glow.

The -6 degree default is a good compromise for most users. It's dark enough for good nighttime images while not waiting unnecessarily long after sunset.

### nighttime_startexposure

The initial exposure time in seconds when entering nighttime mode or when the system starts during nighttime hours.

The default is 10.0 seconds, which is a reasonable starting point for most all-sky cameras. This gives the auto-exposure algorithm a good baseline to work from.

If you're at a very dark site, you might want to start with a longer exposure (e.g., 15 or 20 seconds) to capture fainter stars. At light-polluted sites, shorter exposures (e.g., 5 seconds) might be better to avoid overexposing the sky background.

The algorithm will adjust from this starting point, but having it close to what you expect helps it converge faster.

### nighttime_startgain

The initial gain value when entering nighttime mode. The default is `0` (native sensitivity).

Unlike daytime, nighttime conditions often benefit from increased gain to capture fainter stars and details. However, starting at 0 lets the algorithm decide whether to increase gain based on the actual conditions.

If you know your site is very dark and you want to prioritize sensitivity, you could start with a higher gain (e.g., 50 or 100). But in most cases, letting the algorithm adjust is better.

### nighttime_meanbrightness

The target mean brightness level for auto-exposure during nighttime, expressed as a fraction between 0 and 1.

The default is 0.15 (15% of maximum brightness), which is lower than the daytime target. This makes sense – you want nighttime images to be darker to preserve star visibility and avoid washing out the sky with light pollution.

If your nighttime images are too bright (stars getting washed out), try reducing this to 0.10 or 0.12. If they're too dark and you're losing faint stars, try increasing it to 0.18 or 0.20, but be careful – too bright and you'll start seeing light pollution dominate.

The optimal value depends heavily on your site's light pollution level. At dark sites, you can go lower (0.10-0.12). At light-polluted sites, you might need to go higher (0.18-0.20) to see anything, but you'll trade off some star visibility.

### nighttime_white_balance_blue

The blue channel multiplier for white balance during nighttime captures. The default is 2.0.

Nighttime images typically need more blue adjustment than daytime because artificial lighting (streetlights, etc.) tends to be warm/yellow. The higher blue multiplier helps balance this out and make stars look more natural.

Higher values (e.g., 2.5) will make the image cooler/more blue, which can help counteract warm light pollution. Lower values (e.g., 1.5) will make it warmer.

The 2.0 default is a good starting point, but you might need to adjust based on your local light pollution characteristics. Sodium vapor streetlights (common in many areas) are very warm, so you might need even higher blue values.

### nighttime_white_balance_red

The red channel multiplier for white balance during nighttime. The default is 1.0 (no adjustment).

Like daytime, the red channel is typically left at 1.0 unless you have specific color issues. Most light pollution adjustments can be handled with the blue channel.

### nighttime_move_blackpoint

Enable or disable black point adjustment during nighttime processing. The default is 0.001 (enabled, but more subtle than daytime).

Nighttime black point adjustment is typically more conservative than daytime because you want to preserve the dark sky background and not crush faint stars.

The default 0.001 enables very subtle adjustment. If you're using `autostretch` (which is often preferred for nighttime), you can disable this by setting it to `0`.

### nighttime_move_blackpoint_min_shift_pct

The minimum percentage shift for nighttime black point adjustment. The default is 0.005 (0.5%), same as daytime.

This works the same way as the daytime version, but the overall adjustment is typically smaller due to the lower base value in `nighttime_move_blackpoint`.

### nighttime_move_blackpoint_max_shift_pct

The maximum percentage shift for nighttime black point adjustment. The default is 0.02 (2%), same as daytime.

Again, this works like the daytime version but is applied more conservatively overall.

### nighttime_move_blackpoint_dark_threshold

The dark threshold for nighttime black point adjustment. The default is 0.2, same as daytime.

This parameter works the same way as the daytime version.

### nighttime_hdr_exposure_factor

The exposure factor for HDR processing during nighttime. The default is 2.0, same as daytime.

HDR can be useful for nighttime images when you have both bright stars/planets and faint deep-sky objects or aurora. The 2.0 factor works well for most cases.

If HDR is disabled (`hdr=0`), this parameter has no effect.

### nighttime_measure_noise

Enable or disable noise measurement during nighttime captures. Set to `1` to enable, `0` to disable. The default is `1` (enabled).

When enabled, the system measures the noise level in a specific region of the image. This information is used by noise reduction algorithms and can help the system make better decisions about exposure and gain.

Noise measurement is particularly useful at high gain settings where noise becomes significant. The measurement helps the system understand how much noise reduction is needed.

### nighttime_measure_noise_radius

The radius in pixels of the region used for noise measurement. The default is 250 pixels.

This defines a circular area (centered on the image) where noise is measured. The region should be relatively uniform – typically a patch of sky without bright stars or other features.

A radius of 250 pixels works well for most camera resolutions. For very high-resolution cameras, you might want to increase this (e.g., 300 or 400). For lower-resolution cameras, you could decrease it (e.g., 150 or 200).

The important thing is that the measurement region contains a representative sample of the sky background, not bright stars or other features that would skew the noise measurement.

## Practical Tips

The day/night transition is automatic based on sun position, so you don't need to manually switch modes. However, the `nighttime_sunangle` parameter lets you fine-tune when this happens.

For most users, the default values work well. The main things you might want to adjust:

- **nighttime_meanbrightness**: This is probably the most important parameter to tune. It directly affects how bright your nighttime images are and how many stars you can see vs. how much light pollution shows up.

- **White balance values**: If your images have a strong color cast (too yellow from light pollution or too blue), adjust the white balance multipliers.

- **Start exposure times**: If the auto-exposure algorithm takes too long to converge when switching modes, adjust the start exposure values to be closer to what you expect.

Remember that these are starting points – the auto-exposure algorithm will adjust from these values based on actual conditions. But having good starting values helps the system work more efficiently.


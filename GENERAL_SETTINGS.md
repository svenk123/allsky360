# General Settings

These parameters define your observing location and basic capture behavior. They're used throughout the system, including the main camera daemon and the meteo and aurora tools.

## Observing Location

### location_latitude

Your observing site's latitude in decimal degrees. This is used for calculating sun and moon positions, determining day/night transitions, and fetching location-specific weather and aurora data.

The format is flexible: you can specify it as a decimal number with an optional hemisphere suffix. For example:

- `51.2349N` (northern hemisphere)
- `51.2349` (treated as positive, same as N)
- `-34.5678` or `34.5678S` (southern hemisphere)

If you use the N/S suffix, the parser automatically handles the sign. Without a suffix, positive values are assumed. For southern latitudes, you can either use a negative number or the S suffix.

**Important**: This value is also used by the meteo tool to fetch weather data from Open-Meteo and by the aurora tool to query NOAA aurora forecasts. Make sure it's accurate – even a few degrees off can affect the quality of these external data sources.

### location_longitude

Your observing site's longitude in decimal degrees. Like latitude, this is critical for astronomical calculations and external data services.

The format works the same way:

- `7.4479E` (eastern hemisphere)
- `7.4479` (treated as positive, same as E)
- `-122.4194` or `122.4194W` (western hemisphere)

For most locations, you'll want at least 4-5 decimal places for reasonable accuracy. That gives you precision down to about 10 meters, which is more than enough for all-sky imaging.

Both the meteo and aurora tools use this value along with latitude to fetch location-specific data. The aurora tool, for instance, uses your coordinates to find the nearest point in NOAA's aurora forecast grid and calculate local aurora activity levels.

### location_altitude

Your observing site's altitude above sea level in meters. This is used for atmospheric calculations and can affect how the system interprets sky brightness measurements.

Most locations are between 0 and 2000 meters. If you're at a high-altitude site (above 2000m), you might notice slightly different behavior in the auto-exposure algorithm, since atmospheric extinction is reduced.

This is a simple integer value – just enter the altitude in meters. You can find your altitude using GPS, topographic maps, or online tools like Google Earth.

### location_city

A descriptive name for your location. This is purely informational and appears in metadata and logs. It doesn't affect any calculations, but it's useful for keeping track of multiple installations or when reviewing logs later.

You can use any format you like – city name, city/country, or even a custom identifier. The value should be in quotes if it contains spaces or special characters.

Example: `"Musterstadt/Germany"` or `"My Observatory"`

### location_owner

Your name or the name of the person/organization operating the camera. Like location_city, this is metadata only and doesn't affect functionality.

Use quotes if the name contains spaces.

## Capture Interval

### capture_interval

The time between image captures in seconds. This is the base interval that the camera daemon uses to schedule captures.

The default is 120 seconds (2 minutes), which works well for most all-sky setups. You get a good balance between capturing enough data and not overwhelming your storage or processing pipeline.

For daytime captures, 120 seconds is usually fine – the sky doesn't change dramatically in two minutes. At night, you might want to consider your goals:

- **120 seconds (2 min)**: Good general-purpose interval. Captures most meteor events and aurora activity while keeping file sizes manageable.

- **60 seconds (1 min)**: Better for capturing fast-moving events like meteors or rapidly changing aurora. You'll generate more files, but you're less likely to miss brief events.

- **30 seconds**: Very high cadence. Only use this if you're specifically interested in fast events or doing detailed time-lapse work. Storage requirements increase significantly.

- **300 seconds (5 min)**: Lower cadence, useful if storage is limited or you're primarily interested in long-term sky conditions rather than transient events.

Keep in mind that shorter intervals mean more images to process and store. A 2-minute interval generates 720 images per day. At 30 seconds, that jumps to 2,880 images. Make sure your storage and processing pipeline can handle the load.

The interval applies to both daytime and nighttime captures. The system will take an image every `capture_interval` seconds, regardless of whether it's day or night (assuming daytime_capture is enabled).

## Practical Tips

When setting up a new installation, double-check your coordinates. Even small errors can cause issues:

- Incorrect sun/moon position calculations
- Wrong day/night transition times
- Inaccurate weather or aurora data

You can verify your coordinates using GPS, Google Maps (right-click and select coordinates), or online coordinate converters. Most smartphones have GPS apps that will give you accurate coordinates.

For the capture interval, start with the default 120 seconds and adjust based on your needs. If you find you're missing interesting events, try 60 seconds. If storage becomes an issue, you can always increase it later.


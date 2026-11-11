/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
/*
 * Shared by allsky360-keogram and allsky360-startrails: the keys of
 * config.ini they need, the time range options and the list of the allsky
 * images of a range (images/YYYYMMDD/image-YYYYMMDDHHMMSS.jpg, UTC).
 */
#ifndef TOOLS_IMAGE_SET_H
#define TOOLS_IMAGE_SET_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    char image_directory[PATH_MAX];
    double zenith_x, zenith_y;   /* image_zenith_x/y (pixels) */
    double radius;               /* image_horizon_radius (pixels) */
    double north_deg;            /* image_north_angle (degrees clockwise from up) */
    double longitude;            /* location_longitude */
    int have_longitude;
} tools_config_t;

typedef struct {
    time_t ts;
    char path[PATH_MAX];
    void *data;   /* per-image result of the tool, NULL when unreadable */
} image_entry_t;

/** Defaults as in config.ini.default. */
void tools_config_defaults(tools_config_t *cfg);

/** Read the keys above from a key=value config.ini; -1 when unreadable. */
int tools_config_load(const char *path, tools_config_t *cfg);

/** Unix seconds, YYYYMMDD (00:00 UTC), "YYYY-MM-DD HH:MM[:SS]" or
 * YYYY-MM-DDTHH:MM[:SS] (UTC); 0 on success. */
int parse_time_utc(const char *s, time_t *out);

/** "YYYY-MM-DD HH:MM:SS UTC" */
void format_time_utc(time_t t, char *buf, size_t len);

/**
 * Time range from the --night/--start/--end options (any may be NULL):
 * night = the night after the evening of YYYYMMDD, from solar noon to the
 * next solar noon at the longitude (12:00 UTC without one); start alone =
 * 24 hours; nothing = the last 24 hours. Prints the error and returns -1.
 */
int resolve_time_range(const char *night, const char *start_s, const char *end_s,
                       const tools_config_t *cfg, time_t *start, time_t *end);

/**
 * All image-YYYYMMDDHHMMSS.jpg in [start, end) of the UTC day directories,
 * sorted by time; night_only: only images whose JSON has night_mode 1.
 * Returns NULL with *count 0 when there are none (or on out of memory).
 */
image_entry_t *collect_images(const char *dir, time_t start, time_t end, int night_only,
                              int verbose, size_t *count);

/** Median distance between consecutive images (the capture interval), s. */
double median_interval(const image_entry_t *img, size_t n);

/** Write a json-c object to path.tmp and rename it; 0 on success. */
struct json_object;
int write_json_atomic(const char *path, struct json_object *obj);

#endif /* TOOLS_IMAGE_SET_H */

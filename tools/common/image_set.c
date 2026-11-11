/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
#define _GNU_SOURCE   /* timegm (default), strptime (XPG) */
#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <json-c/json.h>

#include "image_set.h"

#define DAY_S 86400

void tools_config_defaults(tools_config_t *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    snprintf(cfg->image_directory, sizeof(cfg->image_directory), "/home/allskyuser/images");
    cfg->zenith_x = 1925;
    cfg->zenith_y = 1185;
    cfg->radius = 1090;
}

static char *trim(char *s)
{
    while (isspace((unsigned char)*s))
        s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1]))
        *--e = '\0';
    /* optional quotes around the value */
    if (e - s >= 2 && s[0] == '"' && e[-1] == '"') {
        e[-1] = '\0';
        s++;
    }
    return s;
}

int tools_config_load(const char *path, tools_config_t *cfg)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return -1;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        char *s = trim(line);
        if (*s == '#' || *s == ';' || *s == '\0')
            continue;
        char *eq = strchr(s, '=');
        if (!eq)
            continue;
        *eq = '\0';
        char *key = trim(s), *val = trim(eq + 1);
        if (!strcmp(key, "image_directory"))
            snprintf(cfg->image_directory, sizeof(cfg->image_directory), "%s", val);
        else if (!strcmp(key, "image_zenith_x"))
            cfg->zenith_x = atof(val);
        else if (!strcmp(key, "image_zenith_y"))
            cfg->zenith_y = atof(val);
        else if (!strcmp(key, "image_horizon_radius"))
            cfg->radius = atof(val);
        else if (!strcmp(key, "image_north_angle"))
            cfg->north_deg = atof(val);
        else if (!strcmp(key, "location_longitude")) {
            cfg->longitude = atof(val);
            cfg->have_longitude = 1;
        }
    }
    fclose(fp);
    return 0;
}

int parse_time_utc(const char *s, time_t *out)
{
    struct tm tm;
    size_t n = strlen(s);
    int digits = n > 0;

    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)s[i]))
            digits = 0;
    if (digits && n == 8) {                 /* YYYYMMDD: 00:00 UTC */
        memset(&tm, 0, sizeof(tm));
        if (!strptime(s, "%Y%m%d", &tm))
            return -1;
        *out = timegm(&tm);
        return 0;
    }
    if (digits && n > 8) {                  /* Unix seconds */
        *out = (time_t)strtoll(s, NULL, 10);
        return 0;
    }
    static const char *formats[] = { "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                                     "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M" };
    for (size_t f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
        memset(&tm, 0, sizeof(tm));
        const char *end = strptime(s, formats[f], &tm);
        if (end && (*end == '\0' || (*end == 'Z' && end[1] == '\0'))) {
            *out = timegm(&tm);
            return 0;
        }
    }
    return -1;
}

void format_time_utc(time_t t, char *buf, size_t len)
{
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(buf, len, "%Y-%m-%d %H:%M:%S UTC", &tm);
}

int resolve_time_range(const char *night, const char *start_s, const char *end_s,
                       const tools_config_t *cfg, time_t *start, time_t *end)
{
    if (night) {
        time_t day;
        if (strlen(night) != 8 || parse_time_utc(night, &day) != 0) {
            fprintf(stderr, "--night expects YYYYMMDD\n");
            return -1;
        }
        /* solar noon in UTC: 12:00 minus the longitude (15° per hour) */
        double offset = cfg->have_longitude ? cfg->longitude / 15.0 * 3600.0 : 0.0;
        *start = day + 12 * 3600 - (time_t)lround(offset);
        *start -= *start % 60;   /* full minute */
        *end = *start + DAY_S;
    } else if (start_s) {
        if (parse_time_utc(start_s, start) != 0) {
            fprintf(stderr, "Cannot parse --start %s\n", start_s);
            return -1;
        }
        *end = *start + DAY_S;
    } else {
        *end = time(NULL);
        *start = *end - DAY_S;
    }
    if (end_s && parse_time_utc(end_s, end) != 0) {
        fprintf(stderr, "Cannot parse --end %s\n", end_s);
        return -1;
    }
    if (*end <= *start) {
        fprintf(stderr, "The end lies before the start\n");
        return -1;
    }
    return 0;
}

static int cmp_image(const void *a, const void *b)
{
    time_t ta = ((const image_entry_t *)a)->ts, tb = ((const image_entry_t *)b)->ts;
    return (ta > tb) - (ta < tb);
}

/* night_mode of the image's JSON sidecar: 1, 0, or -1 when unknown */
static int image_night_mode(const char *jpg_path)
{
    char json_path[PATH_MAX];
    size_t n = strlen(jpg_path);
    if (n < 4 || n + 1 >= sizeof(json_path))
        return -1;
    memcpy(json_path, jpg_path, n - 4);
    strcpy(json_path + n - 4, ".json");
    json_object *root = json_object_from_file(json_path), *v;
    if (!root)
        return -1;
    int mode = json_object_object_get_ex(root, "night_mode", &v) ? json_object_get_int(v) : -1;
    json_object_put(root);
    return mode;
}

image_entry_t *collect_images(const char *dir, time_t start, time_t end, int night_only,
                              int verbose, size_t *count)
{
    image_entry_t *list = NULL;
    size_t n = 0, cap = 0;

    *count = 0;
    for (time_t day = start - start % DAY_S; day < end; day += DAY_S) {
        struct tm tm;
        char daydir[PATH_MAX], ymd[16];
        gmtime_r(&day, &tm);
        strftime(ymd, sizeof(ymd), "%Y%m%d", &tm);
        if (snprintf(daydir, sizeof(daydir), "%s/%s", dir, ymd) >= (int)sizeof(daydir))
            continue;
        DIR *d = opendir(daydir);
        if (!d) {
            if (verbose)
                fprintf(stderr, "No directory %s\n", daydir);
            continue;
        }
        struct dirent *e;
        while ((e = readdir(d))) {
            char stamp[15];
            struct tm it;
            /* image-YYYYMMDDHHMMSS.jpg exactly (not panorama-, not thumbnails) */
            if (strlen(e->d_name) != 6 + 14 + 4 || strncmp(e->d_name, "image-", 6) ||
                strcmp(e->d_name + 20, ".jpg"))
                continue;
            memcpy(stamp, e->d_name + 6, 14);
            stamp[14] = '\0';
            memset(&it, 0, sizeof(it));
            if (!strptime(stamp, "%Y%m%d%H%M%S", &it))
                continue;
            time_t ts = timegm(&it);
            if (ts < start || ts >= end)
                continue;
            if (n == cap) {
                cap = cap ? cap * 2 : 512;
                image_entry_t *tmp = realloc(list, cap * sizeof(*list));
                if (!tmp) {
                    closedir(d);
                    free(list);
                    return NULL;
                }
                list = tmp;
            }
            list[n].ts = ts;
            list[n].data = NULL;
            if (snprintf(list[n].path, sizeof(list[n].path), "%s/%s", daydir, e->d_name) >= (int)sizeof(list[n].path))
                continue;
            if (night_only && image_night_mode(list[n].path) != 1)
                continue;
            n++;
        }
        closedir(d);
    }
    if (n == 0) {
        free(list);
        return NULL;
    }
    qsort(list, n, sizeof(*list), cmp_image);
    *count = n;
    return list;
}

static int cmp_time(const void *a, const void *b)
{
    time_t ta = *(const time_t *)a, tb = *(const time_t *)b;
    return (ta > tb) - (ta < tb);
}

double median_interval(const image_entry_t *img, size_t n)
{
    if (n < 2)
        return 60.0;
    time_t *d = malloc((n - 1) * sizeof(*d));
    if (!d)
        return 60.0;
    for (size_t i = 1; i < n; i++)
        d[i - 1] = img[i].ts - img[i - 1].ts;
    qsort(d, n - 1, sizeof(*d), cmp_time);
    double m = (double)d[(n - 1) / 2];
    free(d);
    return m > 0 ? m : 60.0;
}

int write_json_atomic(const char *path, json_object *obj)
{
    char tmp[PATH_MAX + 8];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp))
        return -1;
    if (json_object_to_file_ext(tmp, obj, JSON_C_TO_STRING_PLAIN) != 0 || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

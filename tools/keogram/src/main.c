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
 * allsky360-keogram: keogram of the allsky images of a time range.
 *
 * Every column of the keogram is the north-south meridian through the zenith
 * of one image (north at the top, south at the bottom). The columns are
 * placed by time: the x axis runs linearly from --start to --end, so gaps in
 * the capture stay visible as black columns. Image files and days are UTC
 * (images/YYYYMMDD/image-YYYYMMDDHHMMSS.jpg), like everything stored.
 */
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <json-c/json.h>

#include "image_set.h"
#include "jpeg_io.h"
#include "keogram.h"

#define DEFAULT_CONFIG "/home/allskyuser/config.ini"
#define DEFAULT_HEIGHT 720
#define MAX_WIDTH 8192

static void usage(const char *prog)
{
    printf("Usage: %s [options]\n"
           "Keogram (north-south meridian of every image over time) of the allsky images.\n\n"
           "Time range (UTC; TIME: Unix seconds, YYYYMMDD, \"YYYY-MM-DD HH:MM[:SS]\" or YYYY-MM-DDTHH:MM):\n"
           "  -n, --night DATE        the night after the evening of DATE (YYYYMMDD): from solar\n"
           "                          noon to the next solar noon at location_longitude\n"
           "  -s, --start TIME        start of the range\n"
           "  -e, --end TIME          end of the range (default: start + 24 h)\n"
           "                          without -n/-s: the last 24 hours\n"
           "      --trim              shorten the range to the first and last image used\n\n"
           "Options:\n"
           "  -c, --config FILE       config.ini (default %s): image_directory,\n"
           "                          image_zenith_x/y, image_horizon_radius, image_north_angle,\n"
           "                          location_longitude\n"
           "  -i, --images DIR        image directory (overrides image_directory)\n"
           "  -o, --output FILE       keogram JPEG (default keogram.jpg)\n"
           "  -j, --json FILE         also write a JSON description (time range, size, images)\n"
           "  -W, --width N           width in pixels (default: one column per capture interval)\n"
           "  -H, --height N          height in pixels, horizon to horizon (default %d)\n"
           "  -w, --slice-width N     pixels averaged across the meridian (default 5)\n"
           "      --night-only        only images with night_mode 1 in their JSON\n"
           "      --zenith X,Y        zenith in image pixels (overrides image_zenith_x/y)\n"
           "      --radius R          horizon radius in pixels (overrides image_horizon_radius)\n"
           "      --north DEG         north angle, clockwise from up (overrides image_north_angle)\n"
           "  -q, --quality N         JPEG quality (default 90)\n"
           "  -v, --verbose           progress output\n"
           "  -h, --help              this help\n\n"
           "Exit codes: 0 written, 1 usage/config error, 2 no images in the range, 3 write error\n",
           prog, DEFAULT_CONFIG, DEFAULT_HEIGHT);
}

int main(int argc, char **argv)
{
    const char *config_path = DEFAULT_CONFIG, *images_dir = NULL;
    const char *output = "keogram.jpg", *json_out = NULL;
    const char *night = NULL, *start_s = NULL, *end_s = NULL;
    int width = 0, height = DEFAULT_HEIGHT, slice_width = 5, quality = 90;
    int night_only = 0, trim = 0, verbose = 0;
    double zx = NAN, zy = NAN, radius = NAN, north = NAN;

    enum { OPT_NIGHT_ONLY = 1000, OPT_ZENITH, OPT_RADIUS, OPT_NORTH, OPT_TRIM };
    static const struct option long_opts[] = {
        { "night", required_argument, 0, 'n' },
        { "start", required_argument, 0, 's' },
        { "end", required_argument, 0, 'e' },
        { "trim", no_argument, 0, OPT_TRIM },
        { "config", required_argument, 0, 'c' },
        { "images", required_argument, 0, 'i' },
        { "output", required_argument, 0, 'o' },
        { "json", required_argument, 0, 'j' },
        { "width", required_argument, 0, 'W' },
        { "height", required_argument, 0, 'H' },
        { "slice-width", required_argument, 0, 'w' },
        { "night-only", no_argument, 0, OPT_NIGHT_ONLY },
        { "zenith", required_argument, 0, OPT_ZENITH },
        { "radius", required_argument, 0, OPT_RADIUS },
        { "north", required_argument, 0, OPT_NORTH },
        { "quality", required_argument, 0, 'q' },
        { "verbose", no_argument, 0, 'v' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 }
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "n:s:e:c:i:o:j:W:H:w:q:vh", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'n': night = optarg; break;
        case 's': start_s = optarg; break;
        case 'e': end_s = optarg; break;
        case 'c': config_path = optarg; break;
        case 'i': images_dir = optarg; break;
        case 'o': output = optarg; break;
        case 'j': json_out = optarg; break;
        case 'W': width = atoi(optarg); break;
        case 'H': height = atoi(optarg); break;
        case 'w': slice_width = atoi(optarg); break;
        case 'q': quality = atoi(optarg); break;
        case 'v': verbose = 1; break;
        case OPT_TRIM: trim = 1; break;
        case OPT_NIGHT_ONLY: night_only = 1; break;
        case OPT_ZENITH:
            if (sscanf(optarg, "%lf,%lf", &zx, &zy) != 2) {
                fprintf(stderr, "--zenith expects X,Y\n");
                return 1;
            }
            break;
        case OPT_RADIUS: radius = atof(optarg); break;
        case OPT_NORTH: north = atof(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 1;
        }
    }
    if (optind < argc) {
        fprintf(stderr, "Unexpected argument: %s\n", argv[optind]);
        return 1;
    }

    tools_config_t cfg;
    tools_config_defaults(&cfg);
    if (tools_config_load(config_path, &cfg) != 0) {
        /* all geometry given on the command line: the config is optional */
        if (isnan(zx) || isnan(radius) || !images_dir) {
            fprintf(stderr, "Cannot read %s: %s\n", config_path, strerror(errno));
            return 1;
        }
    }
    if (images_dir)
        snprintf(cfg.image_directory, sizeof(cfg.image_directory), "%s", images_dir);
    keogram_geometry_t geo = { cfg.zenith_x, cfg.zenith_y, cfg.radius, cfg.north_deg };
    if (!isnan(zx)) { geo.zenith_x = zx; geo.zenith_y = zy; }
    if (!isnan(radius)) geo.radius = radius;
    if (!isnan(north)) geo.north_deg = north;

    if (height < 16 || height > 8192 || width < 0 || width > MAX_WIDTH || geo.radius <= 0 ||
        quality < 1 || quality > 100 || slice_width < 1) {
        fprintf(stderr, "Invalid size, radius, quality or slice width\n");
        return 1;
    }

    time_t start, end;
    if (resolve_time_range(night, start_s, end_s, &cfg, &start, &end) != 0)
        return 1;

    char s1[40], s2[40];
    format_time_utc(start, s1, sizeof(s1));
    format_time_utc(end, s2, sizeof(s2));
    if (verbose)
        printf("Range %s .. %s, images from %s\n", s1, s2, cfg.image_directory);

    size_t n = 0;
    image_entry_t *img = collect_images(cfg.image_directory, start, end, night_only, verbose, &n);
    if (!img) {
        fprintf(stderr, "No %simages between %s and %s in %s\n", night_only ? "night (night_mode 1) " : "",
                s1, s2, cfg.image_directory);
        return 2;
    }

    /* Decode at the smallest size that still gives height rows over the
     * horizon diameter: JPEG DCT scaling by 1/2, 1/4 or 1/8 */
    int scale = 1;
    while (scale < 8 && 2.0 * geo.radius / (scale * 2) >= height)
        scale *= 2;

    const double interval = median_interval(img, n);
    /* an image fills the columns up to this distance from its time; larger
     * gaps stay black */
    const double max_gap = 1.5 * interval;
    const int half_width = slice_width / 2;

    /* The geometry in config.ini fits one image size: that of the newest
     * image (the current camera settings). Images of another size (crop
     * switched on or off within the range) would give wrong columns. */
    int img_w = 0, img_h = 0;
    if (jpeg_read_size(img[n - 1].path, &img_w, &img_h) != 0) {
        fprintf(stderr, "Cannot read the size of %s\n", img[n - 1].path);
        free(img);
        return 2;
    }
    /* libjpeg rounds the scaled size up */
    const int exp_w = (img_w + scale - 1) / scale, exp_h = (img_h + scale - 1) / scale;
    if (geo.zenith_x - geo.radius < 0 || geo.zenith_x + geo.radius > img_w ||
        geo.zenith_y - geo.radius < 0 || geo.zenith_y + geo.radius > img_h)
        fprintf(stderr, "Note: horizon circle (zenith %.0f,%.0f, radius %.0f) reaches outside the "
                        "%dx%d image; those parts stay black\n",
                geo.zenith_x, geo.zenith_y, geo.radius, img_w, img_h);

    /* One meridian slice per image, decoded in parallel */
    size_t failed = 0, other_size = 0, done = 0;
    #pragma omp parallel for schedule(dynamic, 4) reduction(+:failed, other_size)
    for (size_t i = 0; i < n; i++) {
        int w = 0, h = 0;
        uint8_t *rgb = jpeg_read_rgb_scaled(img[i].path, scale, &w, &h);
        if (rgb && (w != exp_w || h != exp_h)) {
            other_size++;
            if (verbose)
                fprintf(stderr, "Skipped %s (%dx%d, not %dx%d)\n", img[i].path, w * scale, h * scale, img_w, img_h);
            free(rgb);
            continue;
        }
        uint8_t *slice = rgb ? malloc((size_t)height * 3) : NULL;
        if (slice) {
            keogram_slice(rgb, w, h, (double)scale, &geo, height, half_width, slice);
            img[i].data = slice;
        } else {
            failed++;
            fprintf(stderr, "Skipped %s (cannot be read)\n", img[i].path);
        }
        free(rgb);
        if (verbose) {
            size_t k;
            #pragma omp atomic capture
            k = ++done;
            if (k % 50 == 0 || k == n)
                fprintf(stderr, "\r%zu/%zu images", k, n);
        }
    }
    if (verbose)
        fprintf(stderr, "\n");
    if (other_size)
        fprintf(stderr, "Skipped %zu images with another size than %dx%d (the geometry fits only one size)\n",
                other_size, img_w, img_h);
    const size_t used = n - failed - other_size;
    if (used == 0) {
        fprintf(stderr, "No usable images\n");
        free(img);
        return 2;
    }

    /* --trim: the range of the images used, half an interval on each side */
    if (trim) {
        size_t first = 0, last = n - 1;
        while (!img[first].data) first++;
        while (!img[last].data) last--;
        start = img[first].ts - (time_t)(interval / 2);
        end = img[last].ts + (time_t)(interval / 2) + 1;
        format_time_utc(start, s1, sizeof(s1));
        format_time_utc(end, s2, sizeof(s2));
    }
    if (width == 0) {
        width = (int)lround((double)(end - start) / interval);
        if (width < 100) width = 100;
        if (width > MAX_WIDTH) width = MAX_WIDTH;
    }
    if (verbose)
        printf("%zu images, interval %.0f s, decoded at 1/%d, keogram %d x %d\n",
               used, interval, scale, width, height);

    /* Columns by time: the nearest image within max_gap */
    uint8_t *out = calloc((size_t)width * height * 3, 1);
    if (!out) {
        fprintf(stderr, "Out of memory\n");
        return 3;
    }
    const double col_s = (double)(end - start) / width;
    size_t j = 0, filled = 0;
    for (int x = 0; x < width; x++) {
        double t = (double)start + (x + 0.5) * col_s;
        while (j + 1 < n && fabs((double)img[j + 1].ts - t) <= fabs((double)img[j].ts - t))
            j++;
        /* nearest readable image around j */
        const image_entry_t *best = NULL;
        double best_d = max_gap;
        for (size_t k = j > 2 ? j - 2 : 0; k < n && k <= j + 2; k++) {
            double dt = fabs((double)img[k].ts - t);
            if (img[k].data && dt <= best_d) {
                best = &img[k];
                best_d = dt;
            }
        }
        if (!best)
            continue;
        filled++;
        const uint8_t *slice = best->data;
        for (int y = 0; y < height; y++)
            memcpy(out + ((size_t)y * width + x) * 3, slice + (size_t)y * 3, 3);
    }

    int rc = jpeg_write_rgb(output, out, width, height, quality);
    if (rc != 0) {
        fprintf(stderr, "Cannot write %s (%d)\n", output, rc);
        return 3;
    }

    if (json_out) {
        json_object *jo = json_object_new_object();
        json_object_object_add(jo, "source", json_object_new_string("allsky360-keogram"));
        json_object_object_add(jo, "generated", json_object_new_int64((int64_t)time(NULL)));
        json_object_object_add(jo, "start", json_object_new_int64((int64_t)start));
        json_object_object_add(jo, "end", json_object_new_int64((int64_t)end));
        json_object_object_add(jo, "width", json_object_new_int(width));
        json_object_object_add(jo, "height", json_object_new_int(height));
        json_object_object_add(jo, "seconds_per_column", json_object_new_double(col_s));
        json_object_object_add(jo, "images", json_object_new_int((int)used));
        json_object_object_add(jo, "images_unreadable", json_object_new_int((int)failed));
        json_object_object_add(jo, "images_other_size", json_object_new_int((int)other_size));
        json_object_object_add(jo, "image_width", json_object_new_int(img_w));
        json_object_object_add(jo, "image_height", json_object_new_int(img_h));
        json_object_object_add(jo, "first_image", json_object_new_int64((int64_t)img[0].ts));
        json_object_object_add(jo, "last_image", json_object_new_int64((int64_t)img[n - 1].ts));
        json_object_object_add(jo, "interval", json_object_new_double(interval));
        json_object_object_add(jo, "night_only", json_object_new_boolean(night_only));
        /* row 0 = northern horizon, last row = southern horizon */
        json_object_object_add(jo, "top", json_object_new_string("N"));
        json_object_object_add(jo, "bottom", json_object_new_string("S"));
        if (write_json_atomic(json_out, jo) != 0) {
            fprintf(stderr, "Cannot write %s\n", json_out);
            rc = 3;
        }
        json_object_put(jo);
    }

    printf("%s: %d x %d, %zu images (%zu unreadable, %zu other size), %s .. %s, %.0f%% of the columns filled\n",
           output, width, height, used, failed, other_size, s1, s2, 100.0 * filled / width);

    for (size_t i = 0; i < n; i++)
        free(img[i].data);
    free(img);
    free(out);
    return rc ? 3 : 0;
}

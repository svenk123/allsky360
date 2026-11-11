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
 * allsky360-startrails: star trails of the allsky images of a time range.
 *
 * Every output pixel is the brightest value of that pixel over all images
 * ("lighten" stacking), so the stars draw their arcs around the celestial
 * pole. Only night images (night_mode 1 in their JSON) by default;
 * --max-brightness also leaves out images that are brighter on average.
 */
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <json-c/json.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "image_set.h"
#include "jpeg_io.h"

#define DEFAULT_CONFIG "/home/allskyuser/config.ini"
/* The camera exposes night images to a target brightness, so their mean is
 * no measure of how dark the sky was: no brightness limit by default, the
 * night images (night_mode 1) are the filter */
#define DEFAULT_MAX_BRIGHTNESS 1.0
#define DEFAULT_SCALE 2

static void usage(const char *prog)
{
    printf("Usage: %s [options]\n"
           "Star trails (brightest value of every pixel over all images) of the allsky images.\n\n"
           "Time range (UTC; TIME: Unix seconds, YYYYMMDD, \"YYYY-MM-DD HH:MM[:SS]\" or YYYY-MM-DDTHH:MM):\n"
           "  -n, --night DATE          the night after the evening of DATE (YYYYMMDD): from solar\n"
           "                            noon to the next solar noon at location_longitude\n"
           "  -s, --start TIME          start of the range\n"
           "  -e, --end TIME            end of the range (default: start + 24 h)\n"
           "                            without -n/-s: the last 24 hours\n\n"
           "Options:\n"
           "  -c, --config FILE         config.ini (default %s): image_directory,\n"
           "                            location_longitude\n"
           "  -i, --images DIR          image directory (overrides image_directory)\n"
           "  -o, --output FILE         star trails JPEG (default startrails.jpg)\n"
           "  -j, --json FILE           also write a JSON description (time range, images)\n"
           "  -b, --max-brightness F    leave out images brighter than F on average, 0..1\n"
           "                            (default %.2f = keep all; -v shows the brightness)\n"
           "  -S, --scale N             output at 1/N of the image size: 1, 2, 4 or 8 (default %d)\n"
           "      --all                 also day images (default: only night_mode 1)\n"
           "  -q, --quality N           JPEG quality (default 92)\n"
           "  -v, --verbose             progress output and the brightness of every image\n"
           "  -h, --help                this help\n\n"
           "Exit codes: 0 written, 1 usage/config error, 2 no (usable) images in the range, 3 write error\n",
           prog, DEFAULT_CONFIG, DEFAULT_MAX_BRIGHTNESS, DEFAULT_SCALE);
}

/* Mean luminance 0..1 of an RGB image (every 4th pixel is enough) */
static double mean_luminance(const uint8_t *rgb, size_t pixels)
{
    double sum = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < pixels; i += 4, n++) {
        const uint8_t *p = rgb + i * 3;
        sum += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    }
    return n ? sum / n / 255.0 : 0.0;
}

int main(int argc, char **argv)
{
    const char *config_path = DEFAULT_CONFIG, *images_dir = NULL;
    const char *output = "startrails.jpg", *json_out = NULL;
    const char *night = NULL, *start_s = NULL, *end_s = NULL;
    double max_brightness = DEFAULT_MAX_BRIGHTNESS;
    int scale = DEFAULT_SCALE, quality = 92, all = 0, verbose = 0;

    enum { OPT_ALL = 1000 };
    static const struct option long_opts[] = {
        { "night", required_argument, 0, 'n' },
        { "start", required_argument, 0, 's' },
        { "end", required_argument, 0, 'e' },
        { "config", required_argument, 0, 'c' },
        { "images", required_argument, 0, 'i' },
        { "output", required_argument, 0, 'o' },
        { "json", required_argument, 0, 'j' },
        { "max-brightness", required_argument, 0, 'b' },
        { "scale", required_argument, 0, 'S' },
        { "all", no_argument, 0, OPT_ALL },
        { "quality", required_argument, 0, 'q' },
        { "verbose", no_argument, 0, 'v' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 }
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "n:s:e:c:i:o:j:b:S:q:vh", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'n': night = optarg; break;
        case 's': start_s = optarg; break;
        case 'e': end_s = optarg; break;
        case 'c': config_path = optarg; break;
        case 'i': images_dir = optarg; break;
        case 'o': output = optarg; break;
        case 'j': json_out = optarg; break;
        case 'b': max_brightness = atof(optarg); break;
        case 'S': scale = atoi(optarg); break;
        case 'q': quality = atoi(optarg); break;
        case 'v': verbose = 1; break;
        case OPT_ALL: all = 1; break;
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 1;
        }
    }
    if (optind < argc) {
        fprintf(stderr, "Unexpected argument: %s\n", argv[optind]);
        return 1;
    }
    if ((scale != 1 && scale != 2 && scale != 4 && scale != 8) || quality < 1 || quality > 100 ||
        max_brightness <= 0.0 || max_brightness > 1.0) {
        fprintf(stderr, "Invalid scale (1, 2, 4, 8), quality (1..100) or max brightness (0..1]\n");
        return 1;
    }

    tools_config_t cfg;
    tools_config_defaults(&cfg);
    if (tools_config_load(config_path, &cfg) != 0 && !images_dir) {
        fprintf(stderr, "Cannot read %s: %s\n", config_path, strerror(errno));
        return 1;
    }
    if (images_dir)
        snprintf(cfg.image_directory, sizeof(cfg.image_directory), "%s", images_dir);

    time_t start, end;
    if (resolve_time_range(night, start_s, end_s, &cfg, &start, &end) != 0)
        return 1;
    char s1[40], s2[40];
    format_time_utc(start, s1, sizeof(s1));
    format_time_utc(end, s2, sizeof(s2));
    if (verbose)
        printf("Range %s .. %s, images from %s\n", s1, s2, cfg.image_directory);

    size_t n = 0;
    image_entry_t *img = collect_images(cfg.image_directory, start, end, !all, verbose, &n);
    if (!img) {
        fprintf(stderr, "No %simages between %s and %s in %s\n", all ? "" : "night (night_mode 1) ",
                s1, s2, cfg.image_directory);
        return 2;
    }

    /* All images must have the size of the newest one (crop switched on or
     * off within the range gives other sizes that do not line up) */
    int img_w = 0, img_h = 0;
    if (jpeg_read_size(img[n - 1].path, &img_w, &img_h) != 0) {
        fprintf(stderr, "Cannot read the size of %s\n", img[n - 1].path);
        free(img);
        return 2;
    }
    const int out_w = (img_w + scale - 1) / scale, out_h = (img_h + scale - 1) / scale;
    const size_t pixels = (size_t)out_w * out_h;

    /* One maximum buffer per thread, merged at the end */
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    uint8_t **acc = calloc((size_t)threads, sizeof(*acc));
    if (!acc) {
        fprintf(stderr, "Out of memory\n");
        return 3;
    }
    for (int t = 0; t < threads; t++) {
        acc[t] = calloc(pixels * 3, 1);
        if (!acc[t]) {
            fprintf(stderr, "Out of memory (%d buffers of %dx%d)\n", threads, out_w, out_h);
            return 3;
        }
    }
    if (verbose)
        printf("%zu images, output %d x %d (1/%d), %d threads\n", n, out_w, out_h, scale, threads);

    /* data marks an image as used (pointer to a static flag) */
    static char used_flag;
    size_t failed = 0, other_size = 0, bright = 0, done = 0;
    #pragma omp parallel for schedule(dynamic, 2) reduction(+:failed, other_size, bright)
    for (size_t i = 0; i < n; i++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        int w = 0, h = 0;
        uint8_t *rgb = jpeg_read_rgb_scaled(img[i].path, scale, &w, &h);
        if (!rgb) {
            failed++;
            fprintf(stderr, "Skipped %s (cannot be read)\n", img[i].path);
        } else if (w != out_w || h != out_h) {
            other_size++;
            if (verbose)
                fprintf(stderr, "Skipped %s (%dx%d, not %dx%d)\n", img[i].path, w * scale, h * scale, img_w, img_h);
        } else {
            double lum = mean_luminance(rgb, pixels);
            if (verbose)
                fprintf(stderr, "%s brightness %.3f\n", img[i].path, lum);
            if (lum > max_brightness) {
                bright++;
                if (verbose)
                    fprintf(stderr, "Skipped %s (brightness %.3f > %.3f)\n", img[i].path, lum, max_brightness);
            } else {
                uint8_t *a = acc[tid];
                for (size_t k = 0; k < pixels * 3; k++)
                    if (rgb[k] > a[k])
                        a[k] = rgb[k];
                img[i].data = &used_flag;
            }
        }
        free(rgb);
        if (verbose) {
            size_t k;
            #pragma omp atomic capture
            k = ++done;
            if (k % 50 == 0 || k == n)
                fprintf(stderr, "%zu/%zu images\n", k, n);
        }
    }

    const size_t used = n - failed - other_size - bright;
    if (other_size)
        fprintf(stderr, "Skipped %zu images with another size than %dx%d\n", other_size, img_w, img_h);
    if (bright)
        fprintf(stderr, "Skipped %zu images brighter than %.2f on average\n", bright, max_brightness);
    if (used == 0) {
        fprintf(stderr, "No usable images (all unreadable, of another size or too bright)\n");
        for (int t = 0; t < threads; t++)
            free(acc[t]);
        free(acc);
        free(img);
        return 2;
    }

    /* merge the thread buffers into the first */
    for (int t = 1; t < threads; t++) {
        for (size_t k = 0; k < pixels * 3; k++)
            if (acc[t][k] > acc[0][k])
                acc[0][k] = acc[t][k];
        free(acc[t]);
    }

    int rc = jpeg_write_rgb(output, acc[0], out_w, out_h, quality);
    if (rc != 0) {
        fprintf(stderr, "Cannot write %s (%d)\n", output, rc);
        rc = 3;
    }

    time_t first = 0, last = 0;
    for (size_t i = 0; i < n; i++) {
        if (!img[i].data)
            continue;
        if (!first)
            first = img[i].ts;
        last = img[i].ts;
    }

    if (rc == 0 && json_out) {
        json_object *jo = json_object_new_object();
        json_object_object_add(jo, "source", json_object_new_string("allsky360-startrails"));
        json_object_object_add(jo, "generated", json_object_new_int64((int64_t)time(NULL)));
        json_object_object_add(jo, "start", json_object_new_int64((int64_t)start));
        json_object_object_add(jo, "end", json_object_new_int64((int64_t)end));
        json_object_object_add(jo, "first_image", json_object_new_int64((int64_t)first));
        json_object_object_add(jo, "last_image", json_object_new_int64((int64_t)last));
        json_object_object_add(jo, "width", json_object_new_int(out_w));
        json_object_object_add(jo, "height", json_object_new_int(out_h));
        json_object_object_add(jo, "scale", json_object_new_int(scale));
        json_object_object_add(jo, "images", json_object_new_int((int)used));
        json_object_object_add(jo, "images_too_bright", json_object_new_int((int)bright));
        json_object_object_add(jo, "images_unreadable", json_object_new_int((int)failed));
        json_object_object_add(jo, "images_other_size", json_object_new_int((int)other_size));
        json_object_object_add(jo, "max_brightness", json_object_new_double(max_brightness));
        json_object_object_add(jo, "night_only", json_object_new_boolean(!all));
        if (write_json_atomic(json_out, jo) != 0) {
            fprintf(stderr, "Cannot write %s\n", json_out);
            rc = 3;
        }
        json_object_put(jo);
    }

    if (rc == 0) {
        char f1[40], f2[40];
        format_time_utc(first, f1, sizeof(f1));
        format_time_utc(last, f2, sizeof(f2));
        printf("%s: %d x %d, %zu images (%zu too bright, %zu unreadable, %zu other size), %s .. %s\n",
               output, out_w, out_h, used, bright, failed, other_size, f1, f2);
    }

    free(acc[0]);
    free(acc);
    free(img);
    return rc;
}

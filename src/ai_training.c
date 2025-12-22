/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <png.h>

#include <json-c/json.h>

#include "ai_training.h"

int save_patches_with_metadata(const float *rgba, int width, int height, int patch_size,
                               const int (*positions)[2], int n_patches,
                               const char *out_dir, const char *prefix,
                               struct json_object *meta_data)
{
    char filepath[512];
    static int p = 0;
    for (p = 0; p < n_patches; ++p)
    {
        int x0 = positions[p][0];
        int y0 = positions[p][1];

        // Validate patch position
        if (x0 < 0 || y0 < 0 || x0 + patch_size > width || y0 + patch_size > height)
        {
            fprintf(stderr, "Invalid patch position: (%d, %d)\n", x0, y0);
            return 1;
        }

        // Create PNG filename
        snprintf(filepath, sizeof(filepath), "%s/%s_patch%03d.png", out_dir, prefix, p);
        FILE *fp = fopen(filepath, "wb");
        if (!fp)
        {
            fprintf(stderr, "Failed to open file: %s\n", filepath);
            return 1;
        }

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!png_ptr || !info_ptr)
        {
            fclose(fp);
            fprintf(stderr, "PNG struct creation failed\n");
            return 1;
        }
        if (setjmp(png_jmpbuf(png_ptr)))
        {
            fclose(fp);
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fprintf(stderr, "PNG write error\n");
            return 1;
        }
        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, patch_size, patch_size,
                     16, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);

        // Write rows
        png_bytep row = (png_bytep)malloc(6 * patch_size); // 3 channels x 2 bytes
        for (int y = 0; y < patch_size; ++y)
        {
            for (int x = 0; x < patch_size; ++x)
            {
                int idx = ((y0 + y) * width + (x0 + x)) * 4;
                for (int c = 0; c < 3; ++c)
                {
                    float val = rgba[idx + c];
                    uint16_t out_val = (uint16_t)(val < 0.0f ? 0 : val > 1.0f ? 65535
                                                                              : val * 65535.0f + 0.5f);
                    row[x * 6 + c * 2 + 0] = (out_val >> 8) & 0xFF;
                    row[x * 6 + c * 2 + 1] = out_val & 0xFF;
                }
            }
            png_write_row(png_ptr, row);
        }
        free(row);
        png_write_end(png_ptr, NULL);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);

        // Create JSON metadata file
        snprintf(filepath, sizeof(filepath), "%s/%s_patch%03d.json", out_dir, prefix, p);
        FILE *json_fp = fopen(filepath, "w");
        if (!json_fp)
        {
            fprintf(stderr, "Failed to open JSON file: %s\n", filepath);
            return 1;
        }
        const char *json_str = json_object_to_json_string_ext(meta_data, JSON_C_TO_STRING_PRETTY);
        fprintf(json_fp, "%s\n", json_str);
        fclose(json_fp);
    }

    return 0;
}

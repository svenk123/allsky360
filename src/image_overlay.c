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
#include "image_overlay.h"
#include "allsky.h"
#include "jpeg_to_cairo.h"
#include <cairo/cairo.h>
#include <freetype/freetype.h>
#include <freetype/ftglyph.h>
#include <freetype/ftoutln.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int is_monochrome_image(cairo_surface_t *surface) {
  if (!surface)
    return 0;

  // Check if surface is grayscale
  if (cairo_image_surface_get_format(surface) == CAIRO_FORMAT_A8) {
    return 1;
  }

  // Check if rgb values are equal
  if (cairo_image_surface_get_format(surface) == CAIRO_FORMAT_RGB24) {
    unsigned char *data = cairo_image_surface_get_data(surface);
    int width = cairo_image_surface_get_width(surface);
    int height = cairo_image_surface_get_height(surface);
    int stride = cairo_image_surface_get_stride(surface);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        unsigned char *pixel = data + y * stride + x * CHANNELS;
        unsigned char b = pixel[0];
        unsigned char g = pixel[1];
        unsigned char r = pixel[2];

        if (r != g || g != b) {
          return 0; // Not a monochrome surface
        }
      }
    }
    return 1; // All pixels are equal
  }

  return 0; // Can not detect
}

#define FONT_SIZE 24
#define TEXT_COLOR 1.0, 0.0, 0.0 // Red
#define HISTOGRAM_BINS 256       // Number of bins for histogram
#define TEXT_RED 1.0
#define TEXT_GREEN 0.0
#define TEXT_BLUE 0.0
#define TEXT_ALPHA 1.0

static void draw_text(cairo_t *cr, const char *text, int x, int y) {
  // Set anti-aliasing and hinting for better text rendering
  cairo_font_options_t *font_options = cairo_font_options_create();
  cairo_font_options_set_antialias(font_options, CAIRO_ANTIALIAS_NONE);
  cairo_font_options_set_hint_style(font_options, CAIRO_HINT_STYLE_FULL);
  cairo_font_options_set_hint_metrics(font_options, CAIRO_HINT_METRICS_ON);
  cairo_set_font_options(cr, font_options);
  cairo_font_options_destroy(font_options);

  // Set text color to pure red
  cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);

  // Set font
  cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
                         CAIRO_FONT_WEIGHT_NORMAL);
  cairo_set_font_size(cr, 24);

  // Set blend mode (important for overlays)
  cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

  // Set text color again (because rectangle may have overridden it)
  cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);

  // Move and draw text
  cairo_move_to(cr, x, y);
  cairo_show_text(cr, text);

  // Ensure rendering updates
  cairo_stroke(cr);
}

int overlay_datetime_on_image(cairo_surface_t **surface, time_t timestamp) {
  if (!surface || !(*surface)) {
    return 1;
  }

  cairo_t *cr = cairo_create(*surface);

  // Format timestamp into datetime string
  char datetime[50];
  struct tm *tm_info = localtime(&timestamp);
  snprintf(datetime, sizeof(datetime), "%04d-%02d-%02d %02d:%02d:%02d",
           tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
           tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);
  draw_text(cr, datetime, 10, 30);

  cairo_destroy(cr);

  return 0;
}

int draw_text_red_freetype_rgbf1(float *rgbf, int width, int height,
                                 const char *text, int pixel_height, int xpos,
                                 int ypos, const char *font_path) {
  if (!rgbf || width <= 0 || height <= 0 || !text || pixel_height <= 0 ||
      !font_path)
    return 1;

  if (xpos < 0 || xpos >= width)
    return 2;
  if (ypos < 0 || ypos >= height)
    return 3;

  FT_Library ft;
  if (FT_Init_FreeType(&ft)) {
    return 4;
  }

  // Use a standard system font (you can parametrize this)
  FT_Face face;
  if (FT_New_Face(ft, font_path, 0, &face)) {
    FT_Done_FreeType(ft);
    return 5;
  }

  FT_Set_Pixel_Sizes(face, 0, pixel_height);

  // First pass: measure text width and max height
  int pen_x = 0;
  int max_h = 0;

  for (const char *p = text; *p; p++) {
    if (FT_Load_Char(face, *p, FT_LOAD_RENDER))
      continue;

    // Use advance.x instead of bitmap.width for correct spacing
    pen_x +=
        (face->glyph->advance.x >> 6); // advance is in 26.6 fixed point format
    if ((int)face->glyph->bitmap.rows > max_h)
      max_h = (int)face->glyph->bitmap.rows;
  }

  int text_w = pen_x;
  int text_h = max_h;

  if (text_w <= 0 || text_h <= 0) {
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    return 6;
  }

  // Allocate monochrome buffer
  unsigned char *mono = (unsigned char *)allsky_safe_malloc(text_w * text_h);
  if (!mono) {
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    return 7;
  }
  memset(mono, 0, text_w * text_h);

  // Second pass: render characters into mono[]
  pen_x = 0;
  for (const char *p = text; *p; p++) {
    if (FT_Load_Char(face, *p, FT_LOAD_RENDER))
      continue;

    FT_Bitmap *bm = &face->glyph->bitmap;
    int gw = bm->width;
    int gh = bm->rows;
    int left = face->glyph->bitmap_left; // Horizontal offset
    int pitch = bm->pitch;

    // Use bitmap_left for correct horizontal positioning
    int x_offset = pen_x + left;

    // Handle negative pitch (bottom-to-top bitmap)
    if (pitch < 0) {
      // Bitmap is stored bottom-to-top, start from the last row
      unsigned char *bitmap_base = bm->buffer + (gh - 1) * (-pitch);

      for (int y = 0; y < gh; y++) {
        int src_y = y;
        int dst_y =
            y + (face->glyph->bitmap_top > 0 ? 0 : -face->glyph->bitmap_top);

        // Calculate source address with negative pitch
        unsigned char *src_row = bitmap_base - y * (-pitch);

        if (dst_y >= 0 && dst_y < text_h && x_offset >= 0 &&
            x_offset + gw <= text_w) {
          memcpy(&mono[(dst_y * text_w) + x_offset], src_row, gw);
        }
      }
    } else {
      // Normal top-to-bottom bitmap

      for (int y = 0; y < gh; y++) {
        int src_y = y;
        int dst_y =
            y + (face->glyph->bitmap_top > 0 ? 0 : -face->glyph->bitmap_top);

        if (dst_y >= 0 && dst_y < text_h && x_offset >= 0 &&
            x_offset + gw <= text_w) {
          memcpy(&mono[(dst_y * text_w) + x_offset], &bm->buffer[src_y * gw],
                 gw);
        }
      }
    }
    // Use advance.x for correct character spacing
    pen_x += (face->glyph->advance.x >> 6);
  }

  // Now blend into RGB float image
  // Only red channel is written: R = mono * 1.0f, G = 0, B = 0

  // Protect bounds (clamp if text goes near edges)
  int max_x = xpos + text_w;
  int max_y = ypos + text_h;
  if (max_x > width)
    max_x = width;
  if (max_y > height)
    max_y = height;

#pragma omp parallel for
  for (int y = ypos; y < max_y; y++) {
    int ty = y - ypos;

    for (int x = xpos; x < max_x; x++) {
      int tx = x - xpos;

      unsigned char v = mono[ty * text_w + tx];
      if (v == 0)
        continue;

      float val = (float)v / 255.0f * 1.0f;

      size_t idx = ((size_t)y * width + x) * CHANNELS;

      rgbf[idx + 0] = val;  // R
      rgbf[idx + 1] = 0.0f; // G
      rgbf[idx + 2] = 0.0f; // B
    }
  }

  allsky_safe_free(mono);
  FT_Done_Face(face);
  FT_Done_FreeType(ft);

  printf("Text overlayed at x=%d, y=%d: %s\n", xpos, ypos, text);

  return 0;
}

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
#ifndef JPEG_ERROR_H
#define JPEG_ERROR_H

#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>

/*
 * libjpeg error handler that returns to the caller instead of exit().
 *
 * The default handler (jpeg_std_error) calls exit() on every fatal error,
 * e.g. a write error on a full disk or a truncated input file, which would
 * terminate the whole daemon. Usage:
 *
 *   struct allsky_jpeg_err_mgr jerr;
 *   cinfo.err = jpeg_std_error(&jerr.pub);
 *   jerr.pub.error_exit = allsky_jpeg_error_exit;
 *   if (setjmp(jerr.setjmp_buf)) {
 *     jpeg_destroy_compress(&cinfo);  // and close/remove the file
 *     return error;
 *   }
 *
 * Locals changed after setjmp() and used in the error branch must be
 * volatile.
 */
struct allsky_jpeg_err_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buf;
};

static inline void allsky_jpeg_error_exit(j_common_ptr cinfo) {
  struct allsky_jpeg_err_mgr *err = (struct allsky_jpeg_err_mgr *)cinfo->err;
  (*cinfo->err->output_message)(cinfo);
  longjmp(err->setjmp_buf, 1);
}

#endif // JPEG_ERROR_H

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
#ifndef ALLSKY_H
#define ALLSKY_H

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PROG_VERSION "2.2"

#define DEBAYER_ALG_NNI 0
#define DEBAYER_ALG_BILINEAR 1
#define DEBAYER_ALG_VNG 2

/* RGBA */
#define CHANNELS 4

#define ALLSKY_DEBUG_DISABLED 0
#define ALLSKY_DEBUG_NORMAL 1
#define ALLSKY_DEBUG_MEMORY 2
#define ALLSKY_DEBUG_TIMING 3

/* Macro for timing functions.
 *
 * Uses "if (1) { ... } else ((void)0)" rather than the usual "do { ... }
 * while (0)": inside a do-while, a 'continue' in the wrapped block would bind
 * to the macro's own loop and merely fall out of the macro, silently carrying
 * on with the code that follows instead of skipping the enclosing iteration
 * (see review.md, BUG-03). With the if/else form, 'continue' and 'break' bind
 * to the loop at the call site, which is what every caller expects. The
 * trailing 'else ((void)0)' keeps the macro a single statement and protects
 * against a dangling else. */
#define TIMED_BLOCK(func_name, block)                                          \
  if (1) {                                                                     \
    if (allsky_debug >= ALLSKY_DEBUG_TIMING)                                   \
      function_start = clock();                                                \
    block if (allsky_debug >= ALLSKY_DEBUG_TIMING) {                           \
      function_end = clock();                                                  \
      printf("_______________________________%s() processing time: %.2f ms\n", \
             func_name,                                                        \
             1000.0 * (function_end - function_start) / CLOCKS_PER_SEC);       \
    }                                                                          \
  } else                                                                       \
    ((void)0)

/***** Safe memory allocation functions **********/

static inline void *allsky_safe_malloc(size_t size) {
  void *pointer = malloc(size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_malloc: out of memory (%zu bytes)\n", size);
    abort();
  }
  memset(pointer, 0, size);
  return pointer;
}

static inline void *allsky_safe_malloc_copy(size_t size, void *from) {
  void *pointer = malloc(size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_malloc_copy: out of memory (%zu bytes)\n", size);
    abort();
  }
  memcpy(pointer, from, size);
  return pointer;
}

static inline void *allsky_safe_realloc(void *pointer, size_t size) {
  pointer = realloc(pointer, size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_realloc: out of memory (%zu bytes)\n", size);
    abort();
  }
  return pointer;
}

static inline void *allsky_safe_realloc_copy(void *pointer, size_t size,
                                             void *from) {
  pointer = realloc(pointer, size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_realloc_copy: out of memory (%zu bytes)\n", size);
    abort();
  }
  memcpy(pointer, from, size);
  return pointer;
}

static inline void *allsky_safe_calloc(size_t nmemb, size_t size) {
  void *pointer = calloc(nmemb, size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_calloc: out of memory (%zu x %zu bytes)\n", nmemb, size);
    abort();
  }
  return pointer;
}

static inline void allsky_safe_free(void *pointer) {
  if (pointer)
    free(pointer);
}

/***** Safe string functions **********/

static inline char *allsky_safe_strncpy(char *dst, const char *src,
                                        size_t size) {
  if (size > 0) {
    size_t len = strnlen(src, size - 1);
    memcpy(dst, src, len);
    dst[len] = '\0';
  }

  return dst;
}

static inline char *allsky_safe_strdup(const char *s) {
  if (!s) {
    fprintf(stderr, "allsky_safe_strdup: NULL argument\n");
    abort();
  }

  char *copy = strdup(s);
  if (!copy) {
    fprintf(stderr, "allsky_safe_strdup: out of memory (%zu bytes)\n",
            strlen(s) + 1);
    abort();
  }

  return copy;
}

static inline void allsky_safe_strcpy(char *dest, const char *src,
                                      size_t dest_size) {
  if (!dest || !src || dest_size == 0)
    return;

  strncpy(dest, src, dest_size - 1);
  dest[dest_size - 1] = '\0';
}

static inline void allsky_safe_strcat(char *dest, const char *src,
                                      size_t dest_size) {
  if (!dest || !src || dest_size == 0)
    return;

  size_t dest_len = strnlen(dest, dest_size);

  if (dest_len < dest_size - 1) {
    allsky_safe_strcpy(dest + dest_len, src, dest_size - dest_len);
  }
}

static inline int allsky_safe_snprintf(char *buf, size_t buf_size,
                                       const char *fmt, ...) {
  if (!buf || buf_size == 0 || !fmt) {
    return 2;
  }

  va_list args;
  va_start(args, fmt);
  int len = vsnprintf(buf, buf_size, fmt, args);
  va_end(args);

  if (len < 0) {
    return 2; // snprintf error
  }
  if ((size_t)len >= buf_size) {
    buf[buf_size - 1] = '\0';
    return 1; // String truncated
  }

  return 0;
}

/****** Safe file operations **************/

/* These report and return instead of asserting. An assert() turns an ordinary
 * runtime condition - a full filesystem, a read-only directory - into SIGABRT,
 * and compiling with -DNDEBUG would flip the very same code into ignoring the
 * error completely (see review.md, REL-04). Callers must check the result.
 *
 * For publishing a file where readers must never observe a partial write, use
 * src/atomic_file.h instead. */

static inline FILE *allsky_safe_fopen(const char *filename, const char *mode) {
  if (!filename || !mode)
    return NULL;

  FILE *file = fopen(filename, mode);
  if (!file)
    fprintf(stderr, "allsky_safe_fopen: cannot open %s (mode %s): %s\n",
            filename, mode, strerror(errno));

  return file;
}

static inline size_t allsky_safe_fread(void *ptr, size_t size, size_t count,
                                       FILE *stream) {
  if (!ptr || !stream)
    return 0;

  size_t result = fread(ptr, size, count, stream);
  if (result != count)
    fprintf(stderr, "allsky_safe_fread: read %zu of %zu items (%s)\n", result,
            count, ferror(stream) ? strerror(errno) : "end of file");

  return result;
}

static inline size_t allsky_safe_fwrite(const void *ptr, size_t size,
                                        size_t count, FILE *stream) {
  if (!ptr || !stream)
    return 0;

  size_t result = fwrite(ptr, size, count, stream);
  if (result != count)
    fprintf(stderr, "allsky_safe_fwrite: wrote %zu of %zu items: %s\n", result,
            count, strerror(errno));

  return result;
}

/* Returns 0 on success, EOF on error - like fclose() itself. */
static inline int allsky_safe_fclose(FILE *stream) {
  if (!stream)
    return 0;

  int status = fclose(stream);
  if (status != 0)
    fprintf(stderr, "allsky_safe_fclose: close failed: %s\n", strerror(errno));

  return status;
}

static inline int allsky_safe_mkdir(const char *path, mode_t mode) {
  if (!path) {
    fprintf(stderr, "%s: Invalid path argument.\n", __func__);
    return 1;
  }

  if (mkdir(path, mode) == -1) {
    if (errno == EEXIST) {
      // No error here
      return 0;
    }

    switch (errno) {
    case EACCES:
      fprintf(stderr, "%s: Permission denied: %s\n", __func__, path);
      break;
    case ENOENT:
      fprintf(stderr, "%s: A component of the path does not exist: %s\n",
              __func__, path);
      break;
    case ENOTDIR:
      fprintf(stderr, "%s: A component of the path is not a directory: %s\n",
              __func__, path);
      break;
    case EROFS:
      fprintf(stderr, "%s: Read-only file system: %s\n", __func__, path);
      break;
    case ENOSPC:
      fprintf(stderr, "%s: No space left on device: %s\n", __func__, path);
      break;
    case ENAMETOOLONG:
      fprintf(stderr, "%s: Path name too long: %s\n", __func__, path);
      break;
    case ELOOP:
      fprintf(stderr, "%s: Too many symbolic links: %s\n", __func__, path);
      break;
    case EPERM:
      fprintf(stderr, "%s: Operation not permitted: %s\n", __func__, path);
      break;
    case EFAULT:
      fprintf(stderr, "%s: Bad address: %s\n", __func__, path);
      break;
    default:
      fprintf(stderr, "%s: mkdir failed for %s: ", __func__, path);
      perror(NULL);
      break;
    }
    return 1;
  }
  return 0;
}

static inline int allsky_safe_rename(const char *old_path,
                                     const char *new_path) {
  if (!old_path || !new_path) {
    fprintf(stderr, "%s: Invalid arguments.\n", __func__);
    return 1;
  }

  if (rename(old_path, new_path) == -1) {
    switch (errno) {
    case EACCES:
      fprintf(stderr, "%s: Permission denied (%s -> %s)\n", __func__, old_path,
              new_path);
      break;
    case EISDIR:
      fprintf(stderr, "%s: Target is a directory (%s -> %s)\n", __func__,
              old_path, new_path);
      break;
    case ENOENT:
      fprintf(stderr,
              "%s: Source or target directory does not exist (%s -> %s)\n",
              __func__, old_path, new_path);
      break;
    case ENOTDIR:
      fprintf(stderr, "%s: Component of path is not a directory (%s -> %s)\n",
              __func__, old_path, new_path);
      break;
    case EROFS:
      fprintf(stderr, "%s: Read-only file system (%s -> %s)\n", __func__,
              old_path, new_path);
      break;
    case EXDEV:
      fprintf(stderr,
              "%s: Source and target are on different filesystems (%s -> %s)\n",
              __func__, old_path, new_path);
      break;
    case ENOSPC:
      fprintf(stderr, "%s: No space left on device (%s -> %s)\n", __func__,
              old_path, new_path);
      break;
    case EEXIST:
      fprintf(stderr,
              "%s: Target file exists and cannot be overwritten (%s -> %s)\n",
              __func__, old_path, new_path);
      break;
    default:
      perror("allsky_safe_rename");
      break;
    }
    return 1;
  }

  return 0;
}

static inline int allsky_check_directory(const char *path) {
  struct stat st;

  // Check existence
  if (stat(path, &st) != 0)
    return 1;

  // Is a directory?
  if (!S_ISDIR(st.st_mode))
    return 2;

  // Writeable
  if (access(path, W_OK) != 0)
    return 3;

  return 0;
}

/***** Utility functions **********/

static inline float clampf1(float x) {
  return (x < 0.0f) ? 0.0f : (x > 1.0f ? 1.0f : x);
}

/* Compute Rec.709 linear luminance from RGB. */
static inline float rgb_to_luminance_rec709(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static inline float rgb_to_luminance(float r, float g, float b, float r_factor, float g_factor, float b_factor) {
  return r_factor * r + g_factor * g + b_factor * b;
}

static inline int compare_floats(const void *a, const void *b) {
  float fa = *(const float *)a;
  float fb = *(const float *)b;
  return (fa > fb) - (fa < fb);
}


#endif // ALLSKY_H

/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "atomic_file.h"

FILE *atomic_file_open(const char *path, char *tmp_path, size_t tmp_path_size) {
  if (!path || path[0] == '\0' || !tmp_path || tmp_path_size == 0)
    return NULL;

  /* The temporary file must live in the same directory as the destination:
   * rename() is only atomic within one filesystem. */
  int len = snprintf(tmp_path, tmp_path_size, "%s.tmp", path);
  if (len < 0 || (size_t)len >= tmp_path_size) {
    fprintf(stderr, "atomic_file: path too long: %s\n", path);
    return NULL;
  }

  FILE *file = fopen(tmp_path, "w");
  if (!file)
    fprintf(stderr, "atomic_file: cannot create %s: %s\n", tmp_path,
            strerror(errno));

  return file;
}

int atomic_file_commit(FILE *file, const char *tmp_path, const char *path) {
  if (!file || !tmp_path || !path)
    return 1;

  /* ferror() catches write errors that fprintf() silently swallowed, e.g. a
   * full filesystem. */
  if (ferror(file)) {
    fprintf(stderr, "atomic_file: write error on %s\n", tmp_path);
    fclose(file);
    unlink(tmp_path);
    return 1;
  }

  if (fflush(file) != 0 || fsync(fileno(file)) != 0) {
    fprintf(stderr, "atomic_file: cannot flush %s: %s\n", tmp_path,
            strerror(errno));
    fclose(file);
    unlink(tmp_path);
    return 1;
  }

  if (fclose(file) != 0) {
    fprintf(stderr, "atomic_file: cannot close %s: %s\n", tmp_path,
            strerror(errno));
    unlink(tmp_path);
    return 1;
  }

  if (rename(tmp_path, path) != 0) {
    fprintf(stderr, "atomic_file: cannot rename %s to %s: %s\n", tmp_path, path,
            strerror(errno));
    unlink(tmp_path);
    return 1;
  }

  return 0;
}

void atomic_file_abort(FILE *file, const char *tmp_path) {
  if (file)
    fclose(file);
  if (tmp_path)
    unlink(tmp_path);
}

int write_text_file_atomic(const char *path, const char *text) {
  if (!path || !text)
    return 1;

  char tmp_path[4096];
  FILE *file = atomic_file_open(path, tmp_path, sizeof(tmp_path));
  if (!file)
    return 1;

  size_t len = strlen(text);
  if (len > 0 && fwrite(text, 1, len, file) != len) {
    fprintf(stderr, "atomic_file: short write on %s\n", tmp_path);
    atomic_file_abort(file, tmp_path);
    return 1;
  }

  if (len == 0 || text[len - 1] != '\n')
    fputc('\n', file);

  return atomic_file_commit(file, tmp_path, path);
}

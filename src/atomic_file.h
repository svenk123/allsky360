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
#ifndef ATOMIC_FILE_H
#define ATOMIC_FILE_H

#include <stdio.h>
#include <stddef.h>

/*
 * Publishing a file in a single step.
 *
 * The web API and the dashboard poll files such as latest_image.json while the
 * daemon rewrites them. Writing straight to the destination lets a reader see a
 * truncated file, and a power loss leaves one behind permanently
 * (see review.md, REL-05). These helpers write to a temporary file next to the
 * destination and rename() it into place, which is atomic within a filesystem:
 * a reader sees either the old or the new content, never a mixture.
 */

/**
 * Open a temporary file next to `path` for writing.
 * Use this for content that is produced incrementally.
 *
 * @param path: final destination
 * @param tmp_path: [out] buffer receiving the temporary path, needed for commit
 * @param tmp_path_size: size of that buffer
 * @return open stream, or NULL on error
 */
FILE *atomic_file_open(const char *path, char *tmp_path, size_t tmp_path_size);

/**
 * Flush, fsync, close and rename the temporary file onto `path`.
 * The stream is closed in every case, also on failure.
 *
 * @return 0 on success, non-zero on error (the temporary file is removed then)
 */
int atomic_file_commit(FILE *file, const char *tmp_path, const char *path);

/**
 * Give up: close the stream and remove the temporary file. `path` stays as it
 * was.
 */
void atomic_file_abort(FILE *file, const char *tmp_path);

/**
 * Write a complete text in one step. A trailing newline is appended when the
 * text does not already end with one.
 *
 * @return 0 on success, non-zero on error (`path` stays untouched)
 */
int write_text_file_atomic(const char *path, const char *text);

#endif // ATOMIC_FILE_H

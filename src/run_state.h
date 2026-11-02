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
#ifndef RUN_STATE_H
#define RUN_STATE_H

#include <stddef.h>

/* Name of the state file inside the directory passed to the functions below */
#define RUN_STATE_FILENAME "allsky360-state"

/**
 * Read one value from the persistent run state.
 *
 * The state survives a daemon restart and a reboot, which is what makes
 * once-per-day work such as the end-of-night processing restart-safe: without
 * it a crash across the dawn transition silently skipped a whole night
 * (see review.md, REL-06).
 *
 * @param directory: directory holding the state file
 * @param key: key to look up
 * @param value: [out] buffer for the value, always NUL-terminated
 * @param value_size: size of the buffer
 * @return 0 when the key was found, 1 otherwise (value is set to "")
 */
int run_state_get(const char *directory, const char *key, char *value,
                  size_t value_size);

/**
 * Store one value in the persistent run state.
 *
 * The file is rewritten atomically (temporary file, fsync, rename), so a power
 * loss during the write leaves either the old or the new content, never a
 * truncated file.
 *
 * @param directory: directory holding the state file
 * @param key: key to store
 * @param value: value to store (must not contain a newline)
 * @return 0 on success, non-zero on error
 */
int run_state_set(const char *directory, const char *key, const char *value);

#endif // RUN_STATE_H

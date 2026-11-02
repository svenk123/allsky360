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
#ifndef RETENTION_H
#define RETENTION_H

#include "config.h"

/**
 * Enforce the archive retention policy.
 *
 * The archive is organised per UTC day (YYYYMMDD) and a day is always removed
 * as a whole, across all four managed classes:
 *
 *   <image_directory>/YYYYMMDD/           full images, thumbnails, panoramas,
 *                                         per-image JSON, debug images
 *   <video_directory>/<mode>_YYYYMMDD.mp4 timelapse videos
 *   <video_directory>/<mode>_YYYYMMDD.png timelapse still frames
 *   <database_directory>/database_YYYYMMDD.db[-wal|-shm]   daily database
 *
 * Two limits are applied, whichever bites first:
 *   - age:  days older than retention_max_days are removed
 *   - size: while the managed data exceeds retention_max_size_gb, the oldest
 *           remaining day is removed
 *
 * The newest retention_keep_days days are never touched, regardless of the
 * limits, so a misconfigured value cannot delete the night in progress.
 *
 * @param config: loaded configuration
 * @param today_str: current UTC day as "YYYYMMDD"
 * @return 0 on success (also when nothing had to be deleted), 1 on error
 */
int run_retention(const config_t *config, const char *today_str);

#endif // RETENTION_H

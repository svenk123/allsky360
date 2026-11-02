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
#ifndef SD_NOTIFY_H
#define SD_NOTIFY_H

/*
 * Minimal implementation of the systemd notify protocol (see sd_notify(3)),
 * so the daemon does not need libsystemd. Without systemd (NOTIFY_SOCKET not
 * set, e.g. 'make camera-run') all functions are no-ops.
 */

/**
 * Send a raw state string such as "READY=1" or "STATUS=..." to systemd.
 * @return 0 if sent or not running under systemd, -1 on error
 */
int allsky_sd_notify(const char *state);

/**
 * Tell systemd that the service is alive (WATCHDOG=1).
 * Rate-limited to one message per second, so it can be called from loops.
 */
void allsky_watchdog_ping(void);

#endif /* SD_NOTIFY_H */

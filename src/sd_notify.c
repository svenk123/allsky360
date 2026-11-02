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
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "sd_notify.h"

int allsky_sd_notify(const char *state) {
  const char *path = getenv("NOTIFY_SOCKET");
  if (!path || !*path || !state)
    return 0;

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;

  size_t path_len = strlen(path);
  if (path_len >= sizeof(addr.sun_path))
    return -1;
  memcpy(addr.sun_path, path, path_len);

  /* '@' marks a socket in the abstract namespace */
  if (addr.sun_path[0] == '@')
    addr.sun_path[0] = '\0';

  int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (fd < 0)
    return -1;

  socklen_t addr_len =
      (socklen_t)(offsetof(struct sockaddr_un, sun_path) + path_len);
  ssize_t sent = sendto(fd, state, strlen(state), MSG_NOSIGNAL,
                        (const struct sockaddr *)&addr, addr_len);
  close(fd);

  return sent < 0 ? -1 : 0;
}

void allsky_watchdog_ping(void) {
  static time_t last_ping = 0;
  struct timespec now;

  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return;
  if (last_ping != 0 && now.tv_sec - last_ping < 1)
    return;

  last_ping = now.tv_sec;
  allsky_sd_notify("WATCHDOG=1");
}

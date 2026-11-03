/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
#include "timezone.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define ZONEINFO_DIR "/usr/share/zoneinfo/"

static char applied[128] = "\x01"; /* never a valid name: first call applies */
static char effective[128] = "UTC";

/* IANA name of the system time zone (/etc/timezone or /etc/localtime) */
static void system_zone_name(char *out, size_t size) {
  snprintf(out, size, "UTC");

  FILE *f = fopen("/etc/timezone", "r");
  if (f) {
    char line[128];
    if (fgets(line, sizeof(line), f)) {
      line[strcspn(line, " \t\r\n")] = '\0';
      if (line[0]) {
        snprintf(out, size, "%s", line);
        fclose(f);
        return;
      }
    }
    fclose(f);
  }

  char target[PATH_MAX];
  ssize_t n = readlink("/etc/localtime", target, sizeof(target) - 1);
  if (n > 0) {
    target[n] = '\0';
    const char *p = strstr(target, "zoneinfo/");
    if (p)
      snprintf(out, size, "%s", p + strlen("zoneinfo/"));
  }
}

/* A name from the time zone database: no absolute path, no "..", a file */
static int valid_zone_name(const char *name) {
  if (!name[0] || name[0] == '/' || strstr(name, "..") ||
      strlen(name) > 100)
    return 0;
  for (const char *p = name; *p; p++)
    if (!(( *p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
          (*p >= '0' && *p <= '9') || *p == '/' || *p == '_' || *p == '-' ||
          *p == '+'))
      return 0;
  char path[256];
  snprintf(path, sizeof(path), ZONEINFO_DIR "%s", name);
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

int tz_apply(const char *iana) {
  const char *name = iana ? iana : "";
  if (strcmp(name, applied) == 0)
    return 0;
  snprintf(applied, sizeof(applied), "%s", name);

  int rc = 0;
  if (name[0] && valid_zone_name(name)) {
    setenv("TZ", name, 1);
    snprintf(effective, sizeof(effective), "%s", name);
  } else {
    if (name[0]) {
      fprintf(stderr,
              "WARNING: location_timezone '%s' is not a known time zone "
              "(see timedatectl list-timezones), using the system time zone\n",
              name);
      rc = 1;
    }
    unsetenv("TZ");
    system_zone_name(effective, sizeof(effective));
  }
  tzset();

  char now[64];
  tz_format_local(time(NULL), "%Y-%m-%d %H:%M:%S %Z", now, sizeof(now));
  printf("Time zone for display: %s (local time now %s, UTC offset %+ld s); "
         "storage is UTC\n",
         effective, now, tz_offset_seconds(time(NULL)));
  return rc;
}

const char *tz_name(void) { return effective; }

long tz_offset_seconds(time_t t) {
  struct tm tm_local;
  if (localtime_r(&t, &tm_local) == NULL)
    return 0;
  return (long)tm_local.tm_gmtoff;
}

size_t tz_format_local(time_t t, const char *fmt, char *buf, size_t size) {
  struct tm tm_local;
  if (!buf || size == 0)
    return 0;
  buf[0] = '\0';
  if (localtime_r(&t, &tm_local) == NULL)
    return 0;
  return strftime(buf, size, fmt, &tm_local);
}

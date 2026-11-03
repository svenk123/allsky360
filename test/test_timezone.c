/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
/*
 * Test of src/timezone.c: display offsets across the summer/winter time
 * changes, and that the UTC day used for storage does not depend on the
 * display time zone.
 */
#include "timezone.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static int failures = 0;

static void expect_offset(time_t t, long want, const char *what) {
  long got = tz_offset_seconds(t);
  char buf[64];
  tz_format_local(t, "%Y-%m-%d %H:%M:%S %Z", buf, sizeof(buf));
  int ok = got == want;
  printf("%s %-34s %s  offset %+ld (want %+ld)\n", ok ? "ok  " : "FAIL", what,
         buf, got, want);
  failures += !ok;
}

static void expect_str(const char *got, const char *want, const char *what) {
  int ok = strcmp(got, want) == 0;
  printf("%s %-34s '%s'%s%s\n", ok ? "ok  " : "FAIL", what, got,
         ok ? "" : " want ", ok ? "" : want);
  failures += !ok;
}

/* The UTC day as the camera builds images/YYYYMMDD */
static void utc_day(time_t t, char *out, size_t n) {
  struct tm tm;
  gmtime_r(&t, &tm);
  snprintf(out, n, "%04d%02d%02d", (tm.tm_year + 1900) % 10000, (tm.tm_mon + 1) % 100,
           tm.tm_mday % 100);
}

int main(void) {
  /* 2026: CEST starts Sun 29 March 01:00 UTC, ends Sun 25 October 01:00 UTC */
  const time_t spring = 1774746000; /* 2026-03-29 01:00:00 UTC */
  const time_t autumn = 1792890000; /* 2026-10-25 01:00:00 UTC */

  tz_apply("Europe/Berlin");
  expect_str(tz_name(), "Europe/Berlin", "zone in use");
  expect_offset(spring - 1, 3600, "spring, 1 s before the change");
  expect_offset(spring, 7200, "spring, at the change");
  expect_offset(autumn - 1, 7200, "autumn, 1 s before the change");
  expect_offset(autumn, 3600, "autumn, at the change");
  /* Autumn: 02:30 local happens twice; the UTC instants stay distinct */
  char a[64], b[64];
  tz_format_local(autumn - 1800, "%H:%M %Z", a, sizeof(a));
  tz_format_local(autumn + 1800, "%H:%M %Z", b, sizeof(b));
  expect_str(a, "02:30 CEST", "autumn, first 02:30");
  expect_str(b, "02:30 CET", "autumn, second 02:30");

  /* Storage: the UTC day does not depend on the display zone */
  const time_t late = 1793746800; /* 2026-11-03 23:00:00 UTC = 00:00 CET */
  char d1[16], d2[16];
  utc_day(late, d1, sizeof(d1));
  tz_apply("Pacific/Auckland");
  utc_day(late, d2, sizeof(d2));
  expect_str(d1, "20261103", "UTC day, Europe/Berlin");
  expect_str(d2, "20261103", "UTC day, Pacific/Auckland");
  expect_offset(late, 46800, "Auckland (NZDT) offset");

  /* Invalid names fall back to the system zone */
  int rc = tz_apply("Mars/Olympus_Mons");
  printf("%s %-34s rc=%d, zone in use %s\n", rc == 1 ? "ok  " : "FAIL",
         "invalid name rejected", rc, tz_name());
  failures += rc != 1;
  rc = tz_apply("../../etc/passwd");
  printf("%s %-34s rc=%d\n", rc == 1 ? "ok  " : "FAIL", "path name rejected",
         rc);
  failures += rc != 1;

  printf(failures ? "FAILED (%d)\n" : "test_timezone: all tests passed\n",
         failures);
  return failures != 0;
}

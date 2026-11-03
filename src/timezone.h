/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
#ifndef ALLSKY_TIMEZONE_H
#define ALLSKY_TIMEZONE_H

#include <stddef.h>
#include <time.h>

/*
 * Time handling in allsky360:
 *   - Everything that is stored is UTC: Unix timestamps in the databases,
 *     the day of images/YYYYMMDD and database_YYYYMMDD.db, file names.
 *     Always use gmtime_r() for these, never localtime().
 *   - The station time zone (config: location_timezone, an IANA name such
 *     as Europe/Berlin) is only used for display: the time printed into the
 *     image and the "timezone" written for the web pages. The offset of a
 *     moment (summer/winter time) comes from the time zone database for that
 *     moment, so a DST change or a changed system time zone never touches
 *     stored data.
 */

/**
 * Use the IANA time zone `iana` for localtime_r() in this process (sets TZ).
 * An empty name uses the system time zone. An unknown name is reported and
 * the system time zone is used. Calling it again with the same name does
 * nothing, so it can run after every configuration load.
 *
 * @return: 0 on success, 1 if the name was invalid (system zone in use).
 */
int tz_apply(const char *iana);

/** IANA name of the time zone in use (e.g. "Europe/Berlin", "Etc/UTC"). */
const char *tz_name(void);

/** UTC offset in seconds of the station time zone at time t (DST aware). */
long tz_offset_seconds(time_t t);

/** strftime() of t in the station time zone (%Z gives CET/CEST/UTC). */
size_t tz_format_local(time_t t, const char *fmt, char *buf, size_t size);

#endif

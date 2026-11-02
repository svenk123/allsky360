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
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "allsky.h"
#include "retention.h"

/* An archive of one day per entry; 4096 days is ~11 years. */
#define RETENTION_MAX_DAYS 4096

/* Video basenames written by scripts/endofnight.sh */
static const char *const video_prefixes[] = {"normal", "max", "motion"};
#define VIDEO_PREFIX_COUNT                                                     \
  (sizeof(video_prefixes) / sizeof(video_prefixes[0]))

typedef struct {
  char day[9]; /* YYYYMMDD */
  unsigned long long bytes;
} retention_day_t;

/*** Helpers ****************************************************************/

/* A day key must be exactly eight digits. Everything that deletes uses this
 * as its gate, so nothing outside the YYYYMMDD naming scheme can be touched. */
static int is_day_key(const char *name) {
  if (!name || strlen(name) != 8)
    return 0;

  for (int i = 0; i < 8; i++)
    if (name[i] < '0' || name[i] > '9')
      return 0;

  return 1;
}

/* Reject paths that are obviously unsafe to walk or delete below. */
static int is_usable_directory(const char *path) {
  if (!path || path[0] == '\0')
    return 0;
  if (strcmp(path, "/") == 0)
    return 0;
  if (strstr(path, "..") != NULL)
    return 0;

  struct stat st;
  if (lstat(path, &st) != 0)
    return 0;

  return S_ISDIR(st.st_mode) ? 1 : 0;
}

/* UTC midnight of a YYYYMMDD key. Returns (time_t)-1 on a malformed key. */
static time_t day_to_time(const char *day) {
  if (!is_day_key(day))
    return (time_t)-1;

  struct tm tm_day;
  memset(&tm_day, 0, sizeof(tm_day));
  tm_day.tm_year = (day[0] - '0') * 1000 + (day[1] - '0') * 100 +
                   (day[2] - '0') * 10 + (day[3] - '0') - 1900;
  tm_day.tm_mon = (day[4] - '0') * 10 + (day[5] - '0') - 1;
  tm_day.tm_mday = (day[6] - '0') * 10 + (day[7] - '0');
  tm_day.tm_hour = 12; /* midday avoids any DST/rounding edge */

  return timegm(&tm_day);
}

static int compare_days(const void *a, const void *b) {
  return strcmp(((const retention_day_t *)a)->day,
                ((const retention_day_t *)b)->day);
}

/* Add a day to the set if it is not present yet. */
static void add_day(retention_day_t *days, int *count, const char *day) {
  if (!is_day_key(day) || *count >= RETENTION_MAX_DAYS)
    return;

  for (int i = 0; i < *count; i++)
    if (strcmp(days[i].day, day) == 0)
      return;

  allsky_safe_strncpy(days[*count].day, day, sizeof(days[*count].day));
  days[*count].bytes = 0;
  (*count)++;
}

/* Size of a directory tree. Symlinks are counted by their own size and never
 * followed, so a link can never pull foreign data into the accounting. */
static unsigned long long tree_size(const char *path, int depth) {
  if (depth > 8)
    return 0;

  DIR *dir = opendir(path);
  if (!dir)
    return 0;

  unsigned long long total = 0;
  struct dirent *entry;

  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;

    char child[PATH_MAX];
    if (allsky_safe_snprintf(child, sizeof(child), "%s/%s", path,
                             entry->d_name))
      continue; /* truncated - skip rather than touch the wrong path */

    struct stat st;
    if (lstat(child, &st) != 0)
      continue;

    if (S_ISDIR(st.st_mode))
      total += tree_size(child, depth + 1);
    else
      total += (unsigned long long)st.st_size;
  }

  closedir(dir);
  return total;
}

/* Human readable size into a caller-supplied buffer, so several values can be
 * formatted in one printf(). */
static const char *human_bytes(unsigned long long bytes, char *buf,
                               size_t buf_size) {
  const double k = 1024.0;

  if (bytes >= 1024ULL * 1024ULL * 1024ULL)
    snprintf(buf, buf_size, "%.2f GiB", (double)bytes / (k * k * k));
  else if (bytes >= 1024ULL * 1024ULL)
    snprintf(buf, buf_size, "%.1f MiB", (double)bytes / (k * k));
  else
    snprintf(buf, buf_size, "%llu KiB", bytes / 1024ULL);

  return buf;
}

static unsigned long long file_size(const char *path) {
  struct stat st;
  if (lstat(path, &st) != 0)
    return 0;
  return S_ISREG(st.st_mode) ? (unsigned long long)st.st_size : 0;
}

/* Recursively delete a directory tree. Symlinks are unlinked, never followed. */
static int remove_tree(const char *path, int depth, int dry_run) {
  if (depth > 8) {
    fprintf(stderr, "retention: refusing to recurse deeper than 8 levels: %s\n",
            path);
    return 1;
  }

  DIR *dir = opendir(path);
  if (!dir) {
    fprintf(stderr, "retention: cannot open %s: %s\n", path, strerror(errno));
    return 1;
  }

  int failed = 0;
  struct dirent *entry;

  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;

    char child[PATH_MAX];
    if (allsky_safe_snprintf(child, sizeof(child), "%s/%s", path,
                             entry->d_name)) {
      fprintf(stderr, "retention: path too long below %s - skipped\n", path);
      failed = 1;
      continue;
    }

    struct stat st;
    if (lstat(child, &st) != 0)
      continue;

    if (S_ISDIR(st.st_mode)) {
      if (remove_tree(child, depth + 1, dry_run))
        failed = 1;
    } else if (!dry_run && unlink(child) != 0) {
      fprintf(stderr, "retention: cannot delete %s: %s\n", child,
              strerror(errno));
      failed = 1;
    }
  }

  closedir(dir);

  if (!dry_run && !failed && rmdir(path) != 0) {
    fprintf(stderr, "retention: cannot remove directory %s: %s\n", path,
            strerror(errno));
    failed = 1;
  }

  return failed;
}

/*** Per-day accounting *****************************************************/

/* Build the list of file paths that belong to one day outside the image
 * directory (databases and videos). Returns the number of paths filled in. */
static int day_side_files(const config_t *config, const char *day,
                          char paths[][PATH_MAX], int max_paths) {
  int n = 0;
  static const char *const db_suffix[] = {"", "-wal", "-shm"};
  static const char *const video_ext[] = {"mp4", "png"};

  for (size_t s = 0; s < sizeof(db_suffix) / sizeof(db_suffix[0]); s++) {
    if (n >= max_paths)
      break;
    if (allsky_safe_snprintf(paths[n], PATH_MAX, "%s/database_%s.db%s",
                             config->database_directory, day, db_suffix[s]) == 0)
      n++;
  }

  for (size_t p = 0; p < VIDEO_PREFIX_COUNT; p++) {
    for (size_t e = 0; e < sizeof(video_ext) / sizeof(video_ext[0]); e++) {
      if (n >= max_paths)
        break;
      if (allsky_safe_snprintf(paths[n], PATH_MAX, "%s/%s_%s.%s",
                               config->video_directory, video_prefixes[p], day,
                               video_ext[e]) == 0)
        n++;
    }
  }

  return n;
}

#define MAX_SIDE_FILES 16

static unsigned long long day_bytes(const config_t *config, const char *day) {
  unsigned long long total = 0;

  char image_day[PATH_MAX];
  if (allsky_safe_snprintf(image_day, sizeof(image_day), "%s/%s",
                           config->image_directory, day) == 0)
    total += tree_size(image_day, 0);

  /* INDIGO raw dumps use the same per-day layout and are ~16 MB per frame */
  char raw_day[PATH_MAX];
  if (is_usable_directory(config->raw_dump_directory) &&
      allsky_safe_snprintf(raw_day, sizeof(raw_day), "%s/%s",
                           config->raw_dump_directory, day) == 0)
    total += tree_size(raw_day, 0);

  char paths[MAX_SIDE_FILES][PATH_MAX];
  int n = day_side_files(config, day, paths, MAX_SIDE_FILES);
  for (int i = 0; i < n; i++)
    total += file_size(paths[i]);

  return total;
}

static int delete_day(const config_t *config, const char *day, int dry_run) {
  if (!is_day_key(day)) {
    fprintf(stderr, "retention: refusing to delete malformed day '%s'\n", day);
    return 1;
  }

  int failed = 0;

  char image_day[PATH_MAX];
  if (allsky_safe_snprintf(image_day, sizeof(image_day), "%s/%s",
                           config->image_directory, day) == 0) {
    struct stat st;
    if (lstat(image_day, &st) == 0 && S_ISDIR(st.st_mode)) {
      if (remove_tree(image_day, 0, dry_run))
        failed = 1;
    }
  }

  char raw_day[PATH_MAX];
  if (is_usable_directory(config->raw_dump_directory) &&
      allsky_safe_snprintf(raw_day, sizeof(raw_day), "%s/%s",
                           config->raw_dump_directory, day) == 0) {
    struct stat st;
    if (lstat(raw_day, &st) == 0 && S_ISDIR(st.st_mode)) {
      if (remove_tree(raw_day, 0, dry_run))
        failed = 1;
    }
  }

  char paths[MAX_SIDE_FILES][PATH_MAX];
  int n = day_side_files(config, day, paths, MAX_SIDE_FILES);
  for (int i = 0; i < n; i++) {
    struct stat st;
    if (lstat(paths[i], &st) != 0)
      continue;
    if (!dry_run && unlink(paths[i]) != 0) {
      fprintf(stderr, "retention: cannot delete %s: %s\n", paths[i],
              strerror(errno));
      failed = 1;
    }
  }

  return failed;
}

/*** Entry point ************************************************************/

int run_retention(const config_t *config, const char *today_str) {
  if (!config || !is_day_key(today_str))
    return 1;

  if (!config->retention)
    return 0;

  if (!is_usable_directory(config->image_directory)) {
    fprintf(stderr, "retention: image_directory '%s' unusable - skipped\n",
            config->image_directory);
    return 1;
  }

  const int dry_run = config->retention_dry_run ? 1 : 0;

  int keep_days = config->retention_keep_days;
  if (keep_days < 2)
    keep_days = 2; /* today and the night that may still span midnight */

  int max_days = config->retention_max_days;
  if (max_days < keep_days)
    max_days = keep_days;

  double max_gb = config->retention_max_size_gb;
  if (max_gb < 0.0)
    max_gb = 0.0;
  const unsigned long long max_bytes =
      (unsigned long long)(max_gb * 1024.0 * 1024.0 * 1024.0);

  /* 1) Collect every day present in the image directory. Videos and databases
   *    of a day without images are picked up in step 2. */
  retention_day_t *days =
      allsky_safe_calloc(RETENTION_MAX_DAYS, sizeof(retention_day_t));
  int count = 0;

  DIR *dir = opendir(config->image_directory);
  if (!dir) {
    fprintf(stderr, "retention: cannot read %s: %s\n", config->image_directory,
            strerror(errno));
    allsky_safe_free(days);
    return 1;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL)
    if (is_day_key(entry->d_name))
      add_day(days, &count, entry->d_name);
  closedir(dir);

  /* 2a) Days that only exist as a raw dump */
  if (is_usable_directory(config->raw_dump_directory)) {
    DIR *rawdir = opendir(config->raw_dump_directory);
    if (rawdir) {
      while ((entry = readdir(rawdir)) != NULL)
        if (is_day_key(entry->d_name))
          add_day(days, &count, entry->d_name);
      closedir(rawdir);
    }
  }

  /* 2b) Days that only have a database left (image directory already gone) */
  if (is_usable_directory(config->database_directory)) {
    DIR *dbdir = opendir(config->database_directory);
    if (dbdir) {
      while ((entry = readdir(dbdir)) != NULL) {
        char day[9];
        /* database_YYYYMMDD.db... */
        if (strncmp(entry->d_name, "database_", 9) == 0 &&
            strlen(entry->d_name) >= 17) {
          memcpy(day, entry->d_name + 9, 8);
          day[8] = '\0';
          add_day(days, &count, day);
        }
      }
      closedir(dbdir);
    }
  }

  if (count == 0) {
    allsky_safe_free(days);
    return 0;
  }

  qsort(days, (size_t)count, sizeof(retention_day_t), compare_days);

  /* 3) Size per day and total */
  unsigned long long total = 0;
  for (int i = 0; i < count; i++) {
    days[i].bytes = day_bytes(config, days[i].day);
    total += days[i].bytes;
  }

  char buf_a[32], buf_b[32];
  printf("retention: %d day(s) in the archive, %s used, limits: "
         "%d days / %.2f GiB, keeping the newest %d day(s)%s\n",
         count, human_bytes(total, buf_a, sizeof(buf_a)), max_days, max_gb,
         keep_days, dry_run ? " [DRY RUN]" : "");

  const time_t today_time = day_to_time(today_str);
  int deletable = count - keep_days; /* index bound: never touch the newest */
  int removed = 0;
  unsigned long long freed = 0;

  /* 4) Age limit */
  for (int i = 0; i < deletable; i++) {
    time_t day_time = day_to_time(days[i].day);
    if (day_time == (time_t)-1 || today_time == (time_t)-1)
      continue;

    long age_days = (long)((today_time - day_time) / 86400);
    if (age_days <= max_days)
      break; /* sorted ascending - everything after this is younger */

    printf("retention: removing %s (%ld days old, %s)%s\n", days[i].day,
           age_days, human_bytes(days[i].bytes, buf_a, sizeof(buf_a)),
           dry_run ? " [DRY RUN]" : "");

    if (delete_day(config, days[i].day, dry_run) == 0) {
      total -= days[i].bytes;
      freed += days[i].bytes;
      days[i].bytes = 0;
      removed++;
    }
  }

  /* 5) Size limit - keep removing the oldest remaining day */
  for (int i = 0; i < deletable && total > max_bytes; i++) {
    if (days[i].bytes == 0)
      continue; /* already gone in step 4 */

    printf("retention: removing %s (%s) to stay below the size limit "
           "(%s used, limit %.2f GiB)%s\n",
           days[i].day, human_bytes(days[i].bytes, buf_a, sizeof(buf_a)),
           human_bytes(total, buf_b, sizeof(buf_b)), max_gb,
           dry_run ? " [DRY RUN]" : "");

    if (delete_day(config, days[i].day, dry_run) == 0) {
      total -= days[i].bytes;
      freed += days[i].bytes;
      days[i].bytes = 0;
      removed++;
    } else {
      break; /* do not spin on a directory we cannot delete */
    }
  }

  if (removed)
    printf("retention: %d day(s) removed, %s freed, %s remaining%s\n", removed,
           human_bytes(freed, buf_a, sizeof(buf_a)),
           human_bytes(total, buf_b, sizeof(buf_b)),
           dry_run ? " [DRY RUN - nothing was deleted]" : "");
  else
    printf("retention: nothing to do\n");

  allsky_safe_free(days);
  return 0;
}

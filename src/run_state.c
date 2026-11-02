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
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "allsky.h"
#include "atomic_file.h"
#include "run_state.h"

#define RUN_STATE_MAX_KEYS 32
#define RUN_STATE_LINE 512

typedef struct {
  char key[128];
  char value[256];
} run_state_entry_t;

static int state_path(const char *directory, char *out, size_t out_size) {
  if (!directory || directory[0] == '\0' || !out)
    return 1;

  return allsky_safe_snprintf(out, out_size, "%s/%s", directory,
                              RUN_STATE_FILENAME) != 0;
}

/* Read the whole file. Unknown or malformed lines are skipped, so a damaged
 * state file degrades to "value not found" instead of breaking the daemon. */
static int state_load(const char *directory, run_state_entry_t *entries,
                      int max_entries) {
  char path[PATH_MAX];
  if (state_path(directory, path, sizeof(path)))
    return 0;

  FILE *file = fopen(path, "r");
  if (!file)
    return 0; /* no state yet - not an error */

  int count = 0;
  char line[RUN_STATE_LINE];

  while (count < max_entries && fgets(line, sizeof(line), file)) {
    char *newline = strchr(line, '\n');
    if (newline)
      *newline = '\0';

    if (line[0] == '\0' || line[0] == '#')
      continue;

    char *sep = strchr(line, '=');
    if (!sep || sep == line)
      continue;

    *sep = '\0';
    allsky_safe_strncpy(entries[count].key, line, sizeof(entries[count].key));
    allsky_safe_strncpy(entries[count].value, sep + 1,
                        sizeof(entries[count].value));
    count++;
  }

  fclose(file);
  return count;
}

int run_state_get(const char *directory, const char *key, char *value,
                  size_t value_size) {
  if (!key || !value || value_size == 0)
    return 1;

  value[0] = '\0';

  run_state_entry_t entries[RUN_STATE_MAX_KEYS];
  int count = state_load(directory, entries, RUN_STATE_MAX_KEYS);

  for (int i = 0; i < count; i++) {
    if (strcmp(entries[i].key, key) == 0) {
      allsky_safe_strncpy(value, entries[i].value, value_size);
      return 0;
    }
  }

  return 1;
}

int run_state_set(const char *directory, const char *key, const char *value) {
  if (!key || key[0] == '\0' || !value)
    return 1;
  if (strchr(key, '=') || strchr(key, '\n') || strchr(value, '\n')) {
    fprintf(stderr, "run_state: refusing to store '%s' - illegal character\n",
            key);
    return 1;
  }

  run_state_entry_t entries[RUN_STATE_MAX_KEYS];
  int count = state_load(directory, entries, RUN_STATE_MAX_KEYS);

  int found = 0;
  for (int i = 0; i < count; i++) {
    if (strcmp(entries[i].key, key) == 0) {
      allsky_safe_strncpy(entries[i].value, value, sizeof(entries[i].value));
      found = 1;
      break;
    }
  }

  if (!found) {
    if (count >= RUN_STATE_MAX_KEYS) {
      fprintf(stderr, "run_state: state file is full, '%s' not stored\n", key);
      return 1;
    }
    allsky_safe_strncpy(entries[count].key, key, sizeof(entries[count].key));
    allsky_safe_strncpy(entries[count].value, value,
                        sizeof(entries[count].value));
    count++;
  }

  char path[PATH_MAX];
  if (state_path(directory, path, sizeof(path)))
    return 1;

  /* Written through the shared atomic helper: temporary file, fsync, rename. */
  char tmp_path[PATH_MAX];
  FILE *file = atomic_file_open(path, tmp_path, sizeof(tmp_path));
  if (!file)
    return 1;

  fprintf(file, "# allsky360 run state - written by the camera daemon\n");
  for (int i = 0; i < count; i++)
    fprintf(file, "%s=%s\n", entries[i].key, entries[i].value);

  return atomic_file_commit(file, tmp_path, path);
}

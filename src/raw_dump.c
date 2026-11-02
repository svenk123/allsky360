#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "allsky.h"
#include "raw_dump.h"

int save_indigo_raw_dump(const char *raw_dump_directory, const char *today_str,
                         const char *now_str, int frame_index,
                         const unsigned char *raw_buffer,
                         size_t raw_buffer_size, char *saved_path,
                         size_t saved_path_size) {
  if (!raw_dump_directory || !today_str || !now_str || !raw_buffer ||
      raw_buffer_size == 0 || frame_index < 0) {
    fprintf(stderr, "%s: invalid argument\n", __func__);
    return 1;
  }

  if (allsky_safe_mkdir(raw_dump_directory, 0777))
    return 1;

  char daily_directory[PATH_MAX + 1];
  if (allsky_safe_snprintf(daily_directory, sizeof(daily_directory), "%s/%s",
                           raw_dump_directory, today_str)) {
    fprintf(stderr, "WARNING: String %s truncated\n", daily_directory);
    return 1;
  }

  if (allsky_safe_mkdir(daily_directory, 0777))
    return 1;

  char filepath[PATH_MAX + 1];
  if (allsky_safe_snprintf(filepath, sizeof(filepath),
                           "%s/raw-%s-%s_T%d.indigo", daily_directory,
                           today_str, now_str, frame_index)) {
    fprintf(stderr, "WARNING: String %s truncated\n", filepath);
    return 1;
  }

  FILE *file = fopen(filepath, "wb");
  if (!file) {
    fprintf(stderr, "%s: failed to open %s: %s\n", __func__, filepath,
            strerror(errno));
    return 1;
  }

  size_t bytes_written = fwrite(raw_buffer, 1, raw_buffer_size, file);
  if (bytes_written != raw_buffer_size) {
    fprintf(stderr, "%s: failed to write complete raw dump %s (%zu of %zu bytes)\n",
            __func__, filepath, bytes_written, raw_buffer_size);
    fclose(file);
    return 1;
  }

  if (fclose(file) != 0) {
    fprintf(stderr, "%s: failed to close %s: %s\n", __func__, filepath,
            strerror(errno));
    return 1;
  }

  if (saved_path && saved_path_size > 0) {
    if (allsky_safe_snprintf(saved_path, saved_path_size, "%s", filepath)) {
      fprintf(stderr, "WARNING: String %s truncated\n", saved_path);
      return 1;
    }
  }

  return 0;
}

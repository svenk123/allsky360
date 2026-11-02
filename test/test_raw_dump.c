#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/raw_dump.h"

static void test_save_indigo_raw_dump_writes_expected_file(void) {
  char temp_template[] = "/tmp/allsky360-raw-dump-XXXXXX";
  char *temp_dir = mkdtemp(temp_template);
  assert(temp_dir != NULL);

  const unsigned char raw_data[] = {0x49, 0x4e, 0x44, 0x49, 0x47, 0x4f};
  char saved_path[PATH_MAX + 1];

  int result = save_indigo_raw_dump(temp_dir, "20260426", "175401", 2,
                                    raw_data, sizeof(raw_data), saved_path,
                                    sizeof(saved_path));
  assert(result == 0);
  assert(strstr(saved_path, "/20260426/raw-20260426-175401_T2.indigo") != NULL);

  FILE *file = fopen(saved_path, "rb");
  assert(file != NULL);

  unsigned char buffer[sizeof(raw_data)] = {0};
  size_t bytes_read = fread(buffer, 1, sizeof(buffer), file);
  assert(bytes_read == sizeof(raw_data));
  assert(memcmp(buffer, raw_data, sizeof(raw_data)) == 0);
  assert(fclose(file) == 0);

  assert(remove(saved_path) == 0);

  char daily_dir[PATH_MAX + 1];
  snprintf(daily_dir, sizeof(daily_dir), "%s/20260426", temp_dir);
  assert(rmdir(daily_dir) == 0);
  assert(rmdir(temp_dir) == 0);
}

int main(void) {
  test_save_indigo_raw_dump_writes_expected_file();
  return 0;
}

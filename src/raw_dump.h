#ifndef RAW_DUMP_H
#define RAW_DUMP_H

#include <stddef.h>

int save_indigo_raw_dump(const char *raw_dump_directory, const char *today_str,
                         const char *now_str, int frame_index,
                         const unsigned char *raw_buffer,
                         size_t raw_buffer_size, char *saved_path,
                         size_t saved_path_size);

#endif /* RAW_DUMP_H */

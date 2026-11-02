#ifndef AI_FARMING_JSON_OUTPUT_H
#define AI_FARMING_JSON_OUTPUT_H

#include <stdio.h>
#include <stddef.h>
#include <time.h>

/* Escapes src for safe embedding as a JSON string value into buf (quotes,
 * backslashes, control chars). Always NUL-terminates; truncates rather than
 * overflowing if buf is too small. */
static inline void json_escape(const char *src, char *buf, size_t bufsize) {
    size_t o = 0;
    if (bufsize == 0) return;
    for (size_t i = 0; src && src[i] != '\0'; i++) {
        unsigned char c = (unsigned char)src[i];
        char esc[8];
        size_t elen;
        switch (c) {
            case '"':  esc[0] = '\\'; esc[1] = '"';  elen = 2; break;
            case '\\': esc[0] = '\\'; esc[1] = '\\'; elen = 2; break;
            case '\n': esc[0] = '\\'; esc[1] = 'n';  elen = 2; break;
            case '\r': esc[0] = '\\'; esc[1] = 'r';  elen = 2; break;
            case '\t': esc[0] = '\\'; esc[1] = 't';  elen = 2; break;
            default:
                if (c < 0x20) {
                    elen = (size_t)snprintf(esc, sizeof(esc), "\\u%04x", c);
                } else {
                    esc[0] = (char)c;
                    elen = 1;
                }
        }
        if (o + elen >= bufsize - 1) break;
        for (size_t k = 0; k < elen; k++) buf[o++] = esc[k];
    }
    buf[o] = '\0';
}

#define JSON_SUCCESS_INT(tool, key, value) \
    printf("{\"success\": true, \"tool\": \"%s\", \"result\": {\"%s\": %ld}, \"timestamp\": %ld}\n", \
           (tool), (key), (long)(value), (long)time(NULL))

#define JSON_SUCCESS_FLOAT(tool, key, value) \
    printf("{\"success\": true, \"tool\": \"%s\", \"result\": {\"%s\": %.2f}, \"timestamp\": %ld}\n", \
           (tool), (key), (double)(value), (long)time(NULL))

#define JSON_SUCCESS_ACTION(tool, action, detail) \
    do { \
        char _jo_esc[256]; \
        json_escape((detail), _jo_esc, sizeof(_jo_esc)); \
        printf("{\"success\": true, \"tool\": \"%s\", \"result\": {\"action\": \"%s\", \"detail\": \"%s\"}, \"timestamp\": %ld}\n", \
               (tool), (action), _jo_esc, (long)time(NULL)); \
    } while (0)

/* For results with more than one field: build the JSON object body yourself
 * (e.g. via snprintf) and pass it as result_json, e.g. "{\"a\": 1, \"b\": 2.5}". */
#define JSON_SUCCESS_RESULT(tool, result_json) \
    printf("{\"success\": true, \"tool\": \"%s\", \"result\": %s, \"timestamp\": %ld}\n", \
           (tool), (result_json), (long)time(NULL))

#define JSON_ERROR(tool, msg, code, retry_safe) \
    do { \
        char _jo_eesc[512]; \
        json_escape((msg), _jo_eesc, sizeof(_jo_eesc)); \
        printf("{\"success\": false, \"tool\": \"%s\", \"error\": \"%s\", \"error_code\": \"%s\", \"metadata\": {\"retry_safe\": %s}}\n", \
               (tool), _jo_eesc, (code), (retry_safe) ? "true" : "false"); \
    } while (0)

#endif

/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license 
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#ifndef CONFIG_H
#define CONFIG_H

#include <limits.h>

typedef struct {
    double latitude;
    double longitude;
    int altitude;
    char database_directory[NAME_MAX+1];

} config_t;

int load_config(const char *filename, config_t *config);

int show_config(config_t *config);

/**
 * @brief Converts the given config_t structure into a JSON file.
 * 
 * @param filename The path to the output JSON file.
 * @param config Pointer to the configuration structure to serialize.
 * @return int Returns 0 on success, >0 on error.
 */
int config_to_json(const char *filename, const config_t *config);

#endif /* CONFIG_H */

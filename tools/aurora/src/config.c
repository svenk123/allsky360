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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <json-c/json.h>

#include "config.h"

int load_config(const char *filename, config_t *config) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return 1;
    }

    // Set default values
    config->latitude=50.1109f;
    config->longitude=8.6821f;
    config->altitude=100;
    snprintf(config->database_directory, sizeof(config->database_directory), "/opt/allsky360/database");

    char line[256], key[50], value[100];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || strlen(line) < 3) 
	    continue;
        
        // Remove spaces
        char *trim_line = strtok(line, "\n");
        
        if (sscanf(trim_line, "%49[^=] = \"%99[^\"]\"", key, value) == 2 ||
            sscanf(trim_line, "%49[^=] = %99s", key, value) == 2) {

            if (strcmp(key, "latitude") == 0)
                config->latitude = atof(value);
            else if (strcmp(key, "longitude") == 0)
                config->longitude = atof(value);
            else if (strcmp(key, "altitude") == 0)
                config->altitude = atoi(value);
            else if (strcmp(key, "database_directory") == 0)
                snprintf(config->database_directory, sizeof(config->database_directory), "%.*s", (int)sizeof(config->database_directory)-1, value);

        }
    }

    fclose(file);

    return 0;
}

int show_config(config_t *config) {
    // Debug
    printf("\nLoaded configuration:\n");

    printf("latitude: %.6f\n", config->latitude);
    printf("longitude: %.6f\n", config->longitude);
    printf("altitude: %dm\n", config->altitude);
    printf("database_directory: %s\n", config->database_directory);

    return 0;
}

int config_to_json(const char *filename, const config_t *config) {
    if (!filename || !config) return 1;

    FILE *file = fopen(filename, "w");
    if (!file) return 2;

    struct json_object *jroot = json_object_new_object();

    // Generated input from generate_config_to_json.py
json_object_object_add(jroot, "latitude", json_object_new_double(config->latitude));
json_object_object_add(jroot, "longitude", json_object_new_double(config->longitude));
json_object_object_add(jroot, "altitude", json_object_new_int(config->altitude));

    // Schreibe JSON in Datei
    const char *json_str = json_object_to_json_string_ext(jroot, JSON_C_TO_STRING_SPACED);
    //const char *json_str = json_object_to_json_string_ext(jroot, JSON_C_TO_STRING_PRETTY);
    if (fprintf(file, "%s\n", json_str) < 0) {
        fclose(file);
        json_object_put(jroot); // Speicher freigeben
        return 3;
    }

    fclose(file);
    json_object_put(jroot); // Speicher freigeben
    return 0;
}

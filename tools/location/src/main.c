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
#include <time.h>
#include <unistd.h>
#include <json-c/json.h>
#include "magnetic_declination.h"
#include "config.h"

#define PROG_VERSION "1.0"

int debug = 0;

int main(int argc, const char *argv[]) {
    const char *config_path = NULL;
    double longitude = 0.0;
    double latitude = 0.0;
    double altitude = 0.0;
    int longitude_set = 0, latitude_set = 0, altitude_set = 0;
    const char *outfile = NULL;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0)) {
            if (i + 1 < argc) {
                config_path = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --config requires a file path\n");
                exit(EXIT_FAILURE);
            }
        } else if ((strcmp(argv[i], "--outfile") == 0 || strcmp(argv[i], "-o") == 0)) {
            if (i + 1 < argc) {
                outfile = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --outfile requires a file path\n");
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--longitude") == 0) {
            if (i + 1 < argc) {
                longitude = atof(argv[i + 1]);
                longitude_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --longitude requires a value\n");
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--latitude") == 0) {
            if (i + 1 < argc) {
                latitude = atof(argv[i + 1]);
                latitude_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --latitude requires a value\n");
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--altitude") == 0) {
            if (i + 1 < argc) {
                altitude = atof(argv[i + 1]);
                altitude_set = 1;
                i++;
            } else {
                fprintf(stderr, "Error: --altitude requires a value\n");
                exit(EXIT_FAILURE);
            }
        } else if ((strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-v") == 0)) {
	    debug=1;
        } else if ((strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-?") == 0)) {
            printf("allsky360-location v%s\n", PROG_VERSION);
            printf("Copyright (c) 2025 Sven Kreiensen\n");
            printf("\nCommand line parameters:\n");
            printf("--config file       Configuration file\n");
	    printf("--outfile|-o file   Write output to file\n");
            printf("--longitude value   Longitude in decimal degrees\n");
            printf("--latitude value    Latitude in decimal degrees\n");
            printf("--altitude value    Altitude in meters\n");
	    printf("--debug|-v          Enable debug output\n");
            printf("--help|-h|-?        Command help\n");
            return EXIT_SUCCESS;
        }
    }

    if (!config_path) {
        fprintf(stderr, "ERROR: No configuration file. Call: %s --config <path>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Load configuration
    config_t config;
    if (load_config(config_path, &config) != 0) {
        fprintf(stderr, "ERROR: Failed to load configuration file %s.\n", config_path);
        return EXIT_FAILURE;
    }

    if (!longitude_set) longitude = config.longitude;
    if (!latitude_set) latitude = config.latitude;
    if (!altitude_set) altitude = config.altitude;

    time_t now = time(NULL);
    double decl;

    int err = get_magnetic_declination(WMM_Coefficients, WMM_Coefficient_Count,
                                       longitude, latitude, now, &decl);
    if (err == 0) {
	printf("magnetic declination: %.2f°\n", decl);
    } else {
        fprintf(stderr, "Fehler bei Berechnung!\n");
        return EXIT_FAILURE;
    }

    json_object *jout = json_object_new_object();

    json_object_object_add(jout, "longitude", json_object_new_double(longitude));
    json_object_object_add(jout, "latitude", json_object_new_double(latitude));
    json_object_object_add(jout, "altitude", json_object_new_double(altitude));
    json_object_object_add(jout, "magnetic_declination", json_object_new_double(decl));
    json_object_object_add(jout, "city", json_object_new_string(config.city));

    const char *json_str = json_object_to_json_string_ext(jout, JSON_C_TO_STRING_PRETTY);
    if (outfile) {
	FILE *fp = fopen(outfile, "w");

	if (!fp) {
	    fprintf(stderr, "Failed to write JSON to file: %s\n", outfile);
	} else {
	    fprintf(fp, "%s\n", json_str);
	    fclose(fp);

	    if (debug) {
    		printf("JSON written to file: %s\n", outfile);
	    }
	}
    }
    printf("%s\n", json_str);

    json_object_put(jout);

    return EXIT_SUCCESS;
}

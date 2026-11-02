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
#include <ctype.h>
#include <errno.h>
#include "coordinate_parser.h"

int parse_coordinate(const char *input, double *out_value) {
    if (!input || !out_value) 
	return 1;

    char *endptr;
    double value = strtod(input, &endptr);
    if (endptr == input) 
	return 1;  // No valid number

    // Check for optional hemisphere suffix
    while (isspace((unsigned char)*endptr)) 
	endptr++; // skip spaces

    char hemi = toupper((unsigned char)*endptr);

    switch (hemi) {
        case 'N':  // north = +
        case 'E':  // east = +
        case '\0': // no suffix = treat as is
            *out_value = value;
            return 0;
        case 'S':  // south = -
        case 'W':  // west = -
            *out_value = -value;
            return 0;
        default:
            return 1;  // invalid suffix
    }
}

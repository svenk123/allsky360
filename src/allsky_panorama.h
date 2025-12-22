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
#ifndef ALLSKY_PANORAMA_H
#define ALLSKY_PANORAMA_H

#include <cairo.h>

/**
 *  Convert allsky image to panorama image
 * @param fisheye_surface: pointer to the fisheye image surface
 * @param panorama_width: width of the panorama image
 * @param panorama_height: height of the panorama image
 * @param cx: center x of the panorama image
 * @param cy: center y of the panorama image
 * @param theta_start_deg: start angle of the panorama image
 * @return: pointer to the panorama image surface
 */
cairo_surface_t *convert_allsky_to_panorama(cairo_surface_t *fisheye_surface, int panorama_width, int panorama_height, int cx, int cy, double theta_start_deg);

#endif // ALLSKY_PANORAMA_H

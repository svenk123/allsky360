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
#ifndef ROTATE_IMAGE_H
#define ROTATE_IMAGE_H

#include <cairo.h>

/**
 * Rotates a Cairo surface by 90°, 180°, or 270° (by reference).
 *
 * @param surface: pointer to the Cairo surface pointer (will be updated with the rotated surface).
 * @param angle: rotation angle in degrees (must be 90, 180, or 270).
 */
int rotate_surface(cairo_surface_t **surface, int angle);

#endif // ROTATE_IMAGE_H

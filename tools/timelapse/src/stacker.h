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
#ifndef STACKER_H
#define STACKER_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    STACK_NONE,
    STACK_AVERAGE,
    STACK_MAX,
    STACK_MIN,
    STACK_MEDIAN,  // noch nicht implementiert
    STACK_SIGMA_CLIP,
    STACK_DIFFERENCE,
    STACK_MOTION
} stacking_mode_t;

typedef struct {
    float *accum;     // Akkumulierte Daten (float RGB)
    float *accum2;    // Für Sigma-Clipping: quadrierte Werte
    float *background;
    size_t count;     // Anzahl gestackter Bilder
    int width;
    int height;
    float motion_threshold;
    stacking_mode_t mode;
    float sigma_clip_threshold;  // z.B. 2.0
} stack_context_t;

int init_stack_context(stack_context_t *ctx, int width, int height, stacking_mode_t mode, float sigma_clip_threshold);

void update_stack(stack_context_t *ctx, const uint8_t *rgb_data);

void export_stacked_rgb(const stack_context_t *ctx, uint8_t *rgb_out);

void free_stack_context(stack_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // STACKER_H

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
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "stacker.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int init_stack_context(stack_context_t *ctx, int width, int height, stacking_mode_t mode, float sigma_clip_threshold) {
    size_t size = width * height * 3;
    ctx->width = width;
    ctx->height = height;
    ctx->count = 0;
    ctx->mode = mode;
    ctx->sigma_clip_threshold = sigma_clip_threshold;
    ctx->motion_threshold = sigma_clip_threshold; //10.0f;
    ctx->accum = (float *)calloc(size, sizeof(float));
    ctx->accum2 = (float *)calloc(size, sizeof(float));
    ctx->background = (float *)calloc(size, sizeof(float));

    if (!ctx->accum)
	return 1;

    if (mode == STACK_SIGMA_CLIP || mode == STACK_MOTION) {
        if (!ctx->accum2) 
	    return 1;
    }

    if (mode == STACK_DIFFERENCE) {
        if (!ctx->background) 
	    return 1;
    }

    return 0;
}

void update_stack(stack_context_t *ctx, const uint8_t *rgb_data) {
    size_t size = ctx->width * ctx->height * 3;
    ctx->count++;

    // Hintergrundinitialisierung für min-stack (optional)
    if (ctx->mode == STACK_SIGMA_CLIP || ctx->mode == STACK_MOTION) {
        if (ctx->count == 1) {
#pragma omp parallel for
            for (size_t i = 0; i < size; ++i)
                ctx->background[i] = (float)rgb_data[i];
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                float val = (float)rgb_data[i];
                if (val < ctx->background[i])
                    ctx->background[i] = val;
            }
        }
    }

    if (ctx->mode == STACK_MOTION) {
#pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            float val = (float)rgb_data[i];
            float bg  = ctx->background[i];
            float diff = fabsf(val - bg);

            if (diff >= ctx->motion_threshold) {
                ctx->accum[i] += diff;
            }
        }

        return;
    }

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float val = (float)rgb_data[i];

        switch (ctx->mode) {
            case STACK_AVERAGE:
                ctx->accum[i] += val;
                break;

            case STACK_MAX:
                if (ctx->count == 1 || val > ctx->accum[i])
                    ctx->accum[i] = val;
                break;

            case STACK_MIN:
                if (ctx->count == 1 || val < ctx->accum[i])
                    ctx->accum[i] = val;
                break;

            case STACK_SIGMA_CLIP:
                ctx->accum[i]  += val;
                ctx->accum2[i] += val * val;
                break;

            case STACK_DIFFERENCE:
                if (ctx->count == 1) {
                    ctx->background[i] = val;
                    ctx->accum[i] = 0.0f;
                } else {
                    float diff = fabsf(val - ctx->background[i]);
                    ctx->accum[i] = diff;
                }
                break;

            default:
                break;
        }
    }
}

void export_stacked_rgb(const stack_context_t *ctx, uint8_t *rgb_out) {
    size_t size = ctx->width * ctx->height * 3;
    float norm = (ctx->mode == STACK_AVERAGE && ctx->count > 0) ? (1.0f / ctx->count) : 1.0f;

    switch (ctx->mode) {
        case STACK_SIGMA_CLIP:
#pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                float sum = ctx->accum[i];
                float sum2 = ctx->accum2[i];
                float mean = sum / ctx->count;
                float variance = (sum2 / ctx->count) - (mean * mean);
                float sigma = sqrtf(fmaxf(variance, 0.0f));
                float threshold = ctx->sigma_clip_threshold * sigma;
                float clipped = fminf(fmaxf(mean, mean - threshold), mean + threshold);
                rgb_out[i] = (uint8_t)fminf(255.0f, fmaxf(0.0f, clipped + 0.5f));
            }
            break;
        case STACK_MOTION:
            {
                // Finde den maximalen Wert für dynamische Skalierung
                float max_val = 0.0f;
#pragma omp parallel for reduction(max:max_val)
                for (size_t i = 0; i < size; ++i) {
                    if (ctx->accum[i] > max_val)
                        max_val = ctx->accum[i];
                }
                
                // Skaliere alle Werte auf 0-255 Bereich
                if (max_val > 0.0f) {
                    float scale = 255.0f / max_val;
#pragma omp parallel for
                    for (size_t i = 0; i < size; ++i) {
                        float val = ctx->accum[i] * scale;
                        if (val > 255.0f)
                            val = 255.0f;
                        if (val < 0.0f)
                            val = 0.0f;
                        rgb_out[i] = (uint8_t)(val + 0.5f);
                    }
                } else {
                    // Alle Werte sind 0
#pragma omp parallel for
                    for (size_t i = 0; i < size; ++i) {
                        rgb_out[i] = 0;
                    }
                }
            }
            break;
        case STACK_DIFFERENCE:
        case STACK_MIN:
        case STACK_MAX:
        case STACK_AVERAGE:
        default:
#pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                float val = ctx->accum[i] * norm;
                if (val > 255.0f)
		    val = 255.0f;
                if (val < 0.0f)
		    val = 0.0f;
                rgb_out[i] = (uint8_t)(val + 0.5f);
            }

            break;
    }
}

void free_stack_context(stack_context_t *ctx) {
    free(ctx->accum);
    free(ctx->accum2);
    free(ctx->background);

    ctx->accum = NULL;
    ctx->accum2 = NULL;
    ctx->background = NULL;

    ctx->count = 0;
}

#ifdef __cplusplus
}
#endif

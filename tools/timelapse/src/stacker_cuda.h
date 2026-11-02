/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 *
 * CUDA-beschleunigter Motion-Stacker (Median-Hintergrund + Max-Akkumulator)
 *
 *****************************************************************************/
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Allokiert GPU-Puffer fuer width*height*3 Pixel.
// Gibt 0 bei Erfolg, -1 bei CUDA-Fehler zurueck.
int  cuda_stacker_init(size_t n_pixels);

// Gibt alle GPU-Puffer frei.
void cuda_stacker_free(void);

// Pass 1: Histogramm eines Frames (CPU-Speicher) auf der GPU aktualisieren.
void cuda_update_pixel_histogram(const uint8_t *rgb_host, size_t n_pixels);

// Nach Pass 1: Median aus dem Histogramm berechnen, Ergebnis nach Host kopieren.
void cuda_build_median_from_histogram(int n_frames, size_t n_pixels,
                                      uint8_t *median_out_host);

// Pass 2: Motion-Akkumulator mit einem Frame (CPU) aktualisieren.
void cuda_update_motion(const uint8_t *rgb_host, const uint8_t *median_bg_host,
                        float threshold, size_t n_pixels);

// Akkumulator dynamisch skalieren und als uint8 RGB nach Host kopieren.
void cuda_export_motion(size_t n_pixels, uint8_t *out_host);

#ifdef __cplusplus
}
#endif

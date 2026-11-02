# Bugfix src/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 20 bugs identified in the `src/` code review, covering Critical/High/Medium/Low severity issues across memory safety, race conditions, arithmetic overflows, signal handling, and algorithmic correctness.

**Architecture:** Fixes are applied file-by-file. No new abstractions are introduced. Each fix is minimal and targeted. The CUDA dual-implementation pattern (CPU `.c` + GPU `.cu` + wrapper `.c`) is preserved throughout.

**Tech Stack:** C (GNU11), CUDA, OpenMP, libjpeg, SQLite3, libnova

---

## Files Modified

| File | Bugs Fixed |
|------|-----------|
| `include/allsky.h` | #3 — assert as OOM handler (Critical) |
| `src/jpeg_functions.c` | #1 — no setjmp libjpeg handler (Critical) |
| `src/sun_calc.c` | #2 JD offset, #17 latitude optimization (Critical/Medium) |
| `src/dehaze_filter.c` | #4 OpenMP race + int overflow (Critical/Medium) |
| `src/main.c` | #5 signal handler, #6 shared globals, #8 argv bounds, #9 time_t, #11 post-processing path (High/Medium) |
| `src/debayer_bilinear.c` | #7 — wrong green divisor (High) |
| `src/acdnr_filter.c` | #8 — mask never applied to box_blur (High) |
| `src/daily_database.c` | #9 — sqlite3_close on open failure (High) |
| `src/hdr_merge.c` | #10 — integer overflow in pixel_count (High) |
| `src/lights_and_shadows.c` | #15 — division by zero (Medium) |
| `src/debayer_vng.c` | #16 — uninitialized border pixels (Medium) |
| `src/white_balance_cuda.cu` | #14 cudaDeviceSynchronize unchecked, #18 cudaMemset unchecked (Low) |
| `src/config.c` | #20 — dead write wrong sizeof (Low) |

---

## Task 1: Fix `allsky_safe_malloc/realloc/calloc` — assert as OOM handler

**Files:**
- Modify: `include/allsky.h:63–103`

**Problem:** All `allsky_safe_*` functions use `assert(pointer != NULL)`. With `-DNDEBUG` (release builds), asserts are stripped — the function silently returns `NULL`, making every caller's NULL-check dead code and causing NULL dereferences.

**Fix:** Replace `assert` with an explicit `if (!pointer) { fprintf(stderr, ...); abort(); }` in all four functions. Behavior in debug builds is identical; release builds now also abort with a clear message instead of silently returning NULL.

- [ ] **Edit `include/allsky.h` — replace assert with explicit abort:**

```c
static inline void *allsky_safe_malloc(size_t size) {
  void *pointer = malloc(size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_malloc: out of memory (%zu bytes)\n", size);
    abort();
  }
  memset(pointer, 0, size);
  return pointer;
}

static inline void *allsky_safe_malloc_copy(size_t size, void *from) {
  void *pointer = malloc(size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_malloc_copy: out of memory (%zu bytes)\n", size);
    abort();
  }
  memcpy(pointer, from, size);
  return pointer;
}

static inline void *allsky_safe_realloc(void *pointer, size_t size) {
  pointer = realloc(pointer, size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_realloc: out of memory (%zu bytes)\n", size);
    abort();
  }
  return pointer;
}

static inline void *allsky_safe_realloc_copy(void *pointer, size_t size,
                                             void *from) {
  pointer = realloc(pointer, size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_realloc_copy: out of memory (%zu bytes)\n", size);
    abort();
  }
  memcpy(pointer, from, size);
  return pointer;
}

static inline void *allsky_safe_calloc(size_t nmemb, size_t size) {
  void *pointer = calloc(nmemb, size);
  if (!pointer) {
    fprintf(stderr, "allsky_safe_calloc: out of memory (%zu x %zu bytes)\n", nmemb, size);
    abort();
  }
  return pointer;
}
```

- [ ] **Remove `#include <assert.h>`** from `include/allsky.h` if no other assert remains in the file (verify with `grep -n assert include/allsky.h`).

- [ ] **Build to verify:** `make 2>&1 | grep -E "error:|warning:" | head -30`

---

## Task 2: Fix libjpeg fatal error handling — add setjmp/longjmp

**Files:**
- Modify: `src/jpeg_functions.c`

**Problem:** `jpeg_std_error`'s default `error_exit` calls `exit(1)`, terminating the entire daemon on any JPEG error (I/O failure, OOM during Huffman encoding, etc.).

**Fix:** Install a custom `error_exit` that uses `longjmp` to return control to the caller. Apply to both `save_jpeg_rgbf1` and `save_jpeg_rgbf16`.

- [ ] **Add error manager struct and helper — at top of `src/jpeg_functions.c` after the includes:**

```c
/* Custom libjpeg error handler that uses longjmp instead of exit() */
struct allsky_jpeg_err_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buf;
};

static void allsky_jpeg_error_exit(j_common_ptr cinfo) {
  struct allsky_jpeg_err_mgr *err = (struct allsky_jpeg_err_mgr *)cinfo->err;
  (*cinfo->err->output_message)(cinfo); /* print error to stderr */
  longjmp(err->setjmp_buf, 1);
}
```

- [ ] **Rewrite JPEG encoding block in `save_jpeg_rgbf1` (lines 77–108):**

Replace the block from `struct jpeg_compress_struct cinfo;` to `allsky_safe_free(rgb8);` with:

```c
  struct jpeg_compress_struct cinfo;
  struct allsky_jpeg_err_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = allsky_jpeg_error_exit;

  if (setjmp(jerr.setjmp_buf)) {
    /* libjpeg error: clean up and return error */
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    allsky_safe_free(rgb8);
    return 5;
  }

  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = new_w;
  cinfo.image_height = new_h;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, compression_ratio, TRUE);
  jpeg_start_compress(&cinfo, TRUE);

  JSAMPROW row_pointer[1];
  int row_stride = new_w * 3;

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &rgb8[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);
  allsky_safe_free(rgb8);

  printf("Save JPEG: ok, filename: %s\n", filename);
  return 0;
```

- [ ] **Apply same change to `save_jpeg_rgbf16`** (same pattern, `#ifndef USE_CUDA` block).

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 3: Fix sun_calc.c — JD offset bug + wrong latitude optimization

**Files:**
- Modify: `src/sun_calc.c`

**Problem A (Critical):** `ln_get_solar_rst_horizon` returns absolute Julian Days in `rst.rise`/`rst.set`. Adding `jd` produces a date ~2.4 million days in the future.

**Problem B (Medium):** Early `return 0` for `|latitude| > 54°` / `> 60°` is only valid at summer solstice. In winter these latitudes have valid astronomical nights.

- [ ] **Fix JD offset in `calculate_civil_twilight` — remove `jd +` from both `ln_get_date` calls:**

```c
  /* Begin of civil twilight (dawn) */
  ln_get_date(rst.rise, &date);   /* was: jd + rst.rise */
  ...
  /* End of civil twilight (dusk) */
  ln_get_date(rst.set, &date);    /* was: jd + rst.set */
```

- [ ] **Check for the same pattern elsewhere in sun_calc.c:**

```bash
grep -n "jd + rst\." src/sun_calc.c
```

Fix any additional occurrences identically.

- [ ] **Remove latitude short-circuit from `is_sun_below_18_degrees` (around line 32):**

Delete these lines:
```c
    /* Optimization: if latitude > 54° or < -54°, sun may never reach -18° */
    if (latitude > 54.0 || latitude < -54.0) {
        printf("Latitude |%.2f| > 54° → Sun may never go below -18°.\n", latitude);
        return 0;
    }
```

- [ ] **Remove latitude short-circuit from `is_sun_below_6_degrees` (around line 71):**

Delete these lines:
```c
    /* Latitude optimization: If latitude > 60° or < -60° => sun can never go below -6° */
    if (latitude > 60.0 || latitude < -60.0) {
        printf("Latitude is outside the critical range ...\n");
        return 0;
    }
```

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 4: Fix dehaze_filter.c — OpenMP race condition + integer overflow

**Files:**
- Modify: `src/dehaze_filter.c`

**Problem A (Critical):** `reduction(+:count)` creates thread-private copies of `count`. The subsequent `#pragma omp atomic capture` operates on the thread-private copy, not shared memory — multiple threads get the same `local_idx` and overwrite the same `valid_luma` slot.

**Problem B (Medium):** `int pixel_count = width * height` — int overflow for large images.

- [ ] **Read the full `estimate_haze_level` function to confirm line numbers:**

```bash
grep -n "reduction\|atomic\|pixel_count\|int pixel_count\|size_t pixel_count" src/dehaze_filter.c
```

- [ ] **Fix integer overflow — change `int pixel_count` to `size_t`:**

```c
/* Before: */
int pixel_count = width * height;
/* After: */
size_t pixel_count = (size_t)width * height;
```

Also fix the loop variable to `size_t` where needed:
```c
for (size_t i = 0; i < pixel_count; ++i) {
    size_t idx = i * 4;
```

- [ ] **Fix OpenMP race — remove `reduction(+:count)` clause, keep only `atomic capture`:**

```c
/* Before: */
#pragma omp parallel for reduction(+ : count)
  for (int i = 0; i < pixel_count; ++i) {
    ...
    if (!(r == 0.0f && g == 0.0f && b == 0.0f)) {
      int local_idx;
#pragma omp atomic capture
      local_idx = count++;
      valid_luma[local_idx] = luma[i];
    }
  }

/* After: */
#pragma omp parallel for
  for (size_t i = 0; i < pixel_count; ++i) {
    size_t idx = i * 4;
    float r = rgba_image[idx + 0];
    float g = rgba_image[idx + 1];
    float b = rgba_image[idx + 2];

    if (!(r == 0.0f && g == 0.0f && b == 0.0f)) {
      int local_idx;
#pragma omp atomic capture
      local_idx = count++;
      valid_luma[local_idx] = luma[i];
    }
  }
```

- [ ] **Fix the same `int pixel_count` pattern in the caller function `perceptual_dehaze_rgbf1_multiscale_full`:**

```bash
grep -n "int pixel_count\|int.*width \* height" src/dehaze_filter.c
```

Change all `int pixel_count = width * height` to `size_t pixel_count = (size_t)width * height` in this file.

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 5: Fix main.c — signal handler async-safety

**Files:**
- Modify: `src/main.c`

**Problem:** `cleanup_and_exit` calls `printf`, `free`, `exit`, and INDIGO functions — all non-async-signal-safe. `debug_pipeline_images` is not `volatile sig_atomic_t`.

- [ ] **Find the `keep_running` and `debug_pipeline_images` declarations:**

```bash
grep -n "keep_running\|debug_pipeline_images" src/main.c | head -10
```

- [ ] **Change `keep_running` and `debug_pipeline_images` to `volatile sig_atomic_t`:**

```c
/* Before (wherever they are declared, typically near top of file): */
static int keep_running = 1;
static int debug_pipeline_images = 0;

/* After: */
static volatile sig_atomic_t keep_running = 1;
static volatile sig_atomic_t debug_pipeline_images = 0;
```

- [ ] **Rewrite `cleanup_and_exit` to only set the flag:**

```c
void cleanup_and_exit(int signum) {
  (void)signum;
  keep_running = 0;
}
```

- [ ] **Move actual cleanup to the main loop exit path.** Find where the main `while (keep_running)` loop ends and add cleanup after it:

```bash
grep -n "while (keep_running)\|cleanup_resources\|indigo_stop" src/main.c | head -20
```

Ensure `cleanup_resources()` is called after the loop exits (not in the signal handler).

- [ ] **Rewrite `debug_pipeline_handler` to only set the flag:**

```c
void debug_pipeline_handler(int signum) {
  (void)signum;
  debug_pipeline_images = 1;
}
```

- [ ] **Build:** `make 2>&1 | grep -E "error:|warning:.*sig_atomic" | head -20`

---

## Task 6: Fix main.c — shared global `count` needs memory visibility

**Files:**
- Modify: `src/main.c`

**Problem:** `static int count = 1` is written by the INDIGO callback thread (`--count`) and polled by the main loop (`while (count > 0)`) without any synchronisation. On ARM (Jetson Orin) writes may not be visible without a barrier.

- [ ] **Check current type of `count`:**

```bash
grep -n "static int count\|static.*count" src/main.c | head -5
```

- [ ] **Change `count` to `volatile int`** (sufficient for single-writer single-reader spin; `_Atomic` would be stronger but requires C11):

```c
/* Before: */
static int count = 1;

/* After: */
static volatile int count = 1;
```

Note: `exposure_sec` and `gain` are doubles written from main and read from callbacks. Add `volatile` to these too:

```c
/* Before: */
double exposure_sec = 0.0;
double gain = 0;

/* After: */
volatile double exposure_sec = 0.0;
volatile double gain = 0;
```

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 7: Fix main.c — argv bounds, time_t truncation, post-processing path

**Files:**
- Modify: `src/main.c`

**Three independent fixes in main.c.**

- [ ] **Fix argv[i+1] bounds check for `-e`/`-g` (around line 716):**

```c
/* Before: */
    } else if ((strcmp(argv[i], "--start-exposure") == 0 ||
                strcmp(argv[i], "-e") == 0)) {
      exposure_sec = atof(argv[i + 1]);
      i++;
    } else if ((strcmp(argv[i], "--start-gain") == 0 ||
                strcmp(argv[i], "-g") == 0)) {
      gain = atof(argv[i + 1]);
      i++;

/* After: */
    } else if ((strcmp(argv[i], "--start-exposure") == 0 ||
                strcmp(argv[i], "-e") == 0)) {
      if (i + 1 < argc) {
        exposure_sec = atof(argv[i + 1]);
        i++;
      } else {
        fprintf(stderr, "Error: -e/--start-exposure requires a value\n");
        exit(1);
      }
    } else if ((strcmp(argv[i], "--start-gain") == 0 ||
                strcmp(argv[i], "-g") == 0)) {
      if (i + 1 < argc) {
        gain = atof(argv[i + 1]);
        i++;
      } else {
        fprintf(stderr, "Error: -g/--start-gain requires a value\n");
        exit(1);
      }
```

- [ ] **Fix `time_t` truncation (around line 899):**

```c
/* Before: */
  int time_1 = time(NULL) - config.capture_interval - 1;

/* After: */
  time_t time_1 = time(NULL) - config.capture_interval - 1;
```

- [ ] **Fix post-processing called with pre-rename path. Find the rename + postprocessing sequence:**

```bash
grep -n "allsky_safe_rename\|run_postprocessing\|archive_image_filepath\|latest_image_filepath" src/main.c | tail -20
```

- [ ] **Pass `archive_image_filepath` to `run_postprocessing`:**

```c
/* Before: */
      if (run_postprocessing(today_str, now_str, is_night, latest_image_filepath, config.image_directory) != 0) {

/* After: */
      if (run_postprocessing(today_str, now_str, is_night, archive_image_filepath, config.image_directory) != 0) {
```

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 8: Fix debayer_bilinear.c — wrong green channel divisor

**Files:**
- Modify: `src/debayer_bilinear.c`

**Problem:** The `case 0x00` (R pixel in RGGB) uses `/ 3.0` instead of `/ 2.0` for the green interpolation, making every R-position pixel 33% too green-deficient.

- [ ] **Find all occurrences of `/ 3.0` in the file:**

```bash
grep -n "/ 3\.0\|/ 3\.0f" src/debayer_bilinear.c
```

- [ ] **Fix both occurrences (rgb16 variant around line 56, rgbf variant around line 136):**

```c
/* Before: */
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 3.0
                                     : raw[idx];

/* After: */
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0
                                     : raw[idx];
```

And for the rgbf variant:
```c
/* Before: */
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 3.0f
                                     : (float)raw[idx];

/* After: */
        g = (x > 0 && x < width - 1) ? (raw[idx - 1] + raw[idx + 1]) / 2.0f
                                     : (float)raw[idx];
```

- [ ] **Verify no other `/ 3` remains:** `grep -n "/ 3" src/debayer_bilinear.c`

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 9: Fix acdnr_filter.c — computed mask never applied to box_blur

**Files:**
- Modify: `src/acdnr_filter.c`

**Problem:** The Laplace-based detail-protection mask is computed, then immediately freed without influencing the blur. Result: the blur is applied unconditionally to the entire image, destroying star detail.

**Fix:** Add an optional `const float *mask` parameter to `box_blur`. Where `mask != NULL`, blend using `mask[i] * amount` as the effective strength (0 = edge, no blur; 1 = smooth, full blur). The chrominance calls pass `NULL` (unchanged behaviour).

- [ ] **Modify `box_blur` signature and implementation:**

```c
static void box_blur(float *channel, int width, int height, int radius,
                     float amount, const float *mask) {
  float *temp = (float *)allsky_safe_malloc(width * height * sizeof(float));
  if (!temp)
    return;
  memcpy(temp, channel, width * height * sizeof(float));

#pragma omp parallel for collapse(2)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = 0.0f;
      int count = 0;

      for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
          int ix = x + dx;
          int iy = y + dy;
          if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
            sum += temp[iy * width + ix];
            count++;
          }
        }
      }
      float blurred = sum / count;
      float effective_amount = (mask != NULL) ? mask[y * width + x] * amount : amount;
      channel[y * width + x] =
          (1.0f - effective_amount) * temp[y * width + x] + effective_amount * blurred;
    }
  }

  allsky_safe_free(temp);
}
```

- [ ] **Update luminance blur call to pass the mask (around line 130):**

```c
/* Before: */
    box_blur(Y, width, height, structure_size_l, amount_l);
    allsky_safe_free(mask);

/* After: */
    box_blur(Y, width, height, structure_size_l, amount_l, mask);
    allsky_safe_free(mask);
```

- [ ] **Update chrominance blur calls to pass NULL (lines 137–138):**

```c
/* Before: */
    box_blur(U, width, height, structure_size_c, amount_c);
    box_blur(V, width, height, structure_size_c, amount_c);

/* After: */
    box_blur(U, width, height, structure_size_c, amount_c, NULL);
    box_blur(V, width, height, structure_size_c, amount_c, NULL);
```

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 10: Fix daily_database.c — sqlite3_close missing on open failure

**Files:**
- Modify: `src/daily_database.c`

**Problem:** SQLite docs require calling `sqlite3_close(db)` even when `sqlite3_open` returns an error (db may be non-NULL). Two functions have this bug.

- [ ] **Fix `create_daily_database` open-failure path:**

```bash
grep -n "sqlite3_open\|return 1" src/daily_database.c | head -10
```

```c
/* Before: */
    int rc = sqlite3_open(db_filename, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

/* After: */
    int rc = sqlite3_open(db_filename, &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 1;
    }
```

- [ ] **Apply the same fix to `insert_measurement`** (same pattern, second `sqlite3_open` call in the file).

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 11: Fix hdr_merge.c — integer overflow in pixel_count

**Files:**
- Modify: `src/hdr_merge.c`

**Problem:** `int pixel_count = width * height` (in three separate functions) overflows for images larger than ~46341×46341. The main HDR merge function uses `const int pixel_count` and `int idx = i * CHANNELS` in OpenMP loops.

- [ ] **Find all integer pixel_count declarations:**

```bash
grep -n "int pixel_count\|const int pixel_count" src/hdr_merge.c
```

- [ ] **Fix each occurrence — change `int pixel_count = width * height` to `size_t pixel_count = (size_t)width * height`.** There are approximately 4 locations (lines ~147, 173, 404, 1212). Fix all of them.

```c
/* Before: */
  int pixel_count = width * height;
  /* or */
  const int pixel_count = width * height;

/* After: */
  size_t pixel_count = (size_t)width * height;
```

- [ ] **Fix all loop variables that use `pixel_count` as bound.** After changing `pixel_count` to `size_t`, loop variables `int i` used as `for (int i = 0; i < pixel_count; ++i)` must also become `size_t`:

```bash
grep -n "for (int i = 0; i < pixel_count" src/hdr_merge.c
```

For each: `for (size_t i = 0; i < pixel_count; ++i)`

- [ ] **Fix `int idx = i * CHANNELS` where `i` is now `size_t`:**

```bash
grep -n "int idx = i \* CHANNELS" src/hdr_merge.c
```

```c
/* Before: */
    int idx = i * CHANNELS;
/* After: */
    size_t idx = i * (size_t)CHANNELS;
```

- [ ] **Build:** `make 2>&1 | grep -E "error:|warning:.*conversion" | head -30`

---

## Task 12: Fix lights_and_shadows.c — division by zero when range == 0

**Files:**
- Modify: `src/lights_and_shadows.c`

**Problem:** When all valid pixels have identical luminance (`max_val == min_val`), `range = 0` and `denom = 0`. The apply loop then divides every pixel value by zero → NaN/Inf.

- [ ] **Locate the `range` and `denom` computation:**

```bash
grep -n "const float range\|const float denom\|black_shift_pct" src/lights_and_shadows.c | head -10
```

- [ ] **Add early return before the apply loop when `denom` is degenerate:**

```c
  const float range = (float)(max_val - min_val);
  const float black_point = (float)(min_val + black_shift_pct * range);
  const float denom = (float)(range * (1.0 - black_shift_pct));

  /* Guard: nothing to stretch when all pixels have the same luminance */
  if (denom < 1e-7f)
    return 0;

  /* Apply shift */
  ...
```

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 13: Fix debayer_vng.c — uninitialized border pixels

**Files:**
- Modify: `src/debayer_vng.c`

**Problem:** Both VNG functions iterate only over the interior (skipping `half_kernel` border pixels). The output buffer is allocated via `allsky_safe_malloc` (not calloc). The border ring contains uninitialized heap data returned to the caller.

- [ ] **Find both VNG function start points and their output buffer allocations:**

```bash
grep -n "allsky_safe_malloc\|allsky_safe_calloc\|for (int y = half_kernel" src/debayer_vng.c
```

- [ ] **Zero the output buffer for `debayer_vng_rgb16` immediately after allocation:**

```c
/* Add directly after the malloc of 'rgb': */
memset(rgb, 0, (size_t)width * height * CHANNELS * sizeof(unsigned short));
```

- [ ] **Zero the output buffer for `debayer_vng_rgbf` immediately after allocation:**

```c
/* Add directly after the malloc of 'rgbf': */
memset(rgbf, 0, (size_t)width * height * CHANNELS * sizeof(float));
```

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 14: Fix white_balance_cuda.cu — unchecked CUDA calls

**Files:**
- Modify: `src/white_balance_cuda.cu`

**Problem A:** `cudaDeviceSynchronize()` return value discarded (line ~386) — GPU kernel faults are silently swallowed.

**Problem B:** Seven `cudaMemset` calls (lines ~194–200) ignore return values — failed memsets leave accumulators uninitialised.

- [ ] **Find the unchecked `cudaDeviceSynchronize()` call:**

```bash
grep -n "cudaDeviceSynchronize" src/white_balance_cuda.cu
```

- [ ] **Add error check (use the existing `err` variable and the error-handling pattern already used in the same function):**

```c
/* Before: */
  cudaDeviceSynchronize();

/* After: */
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_sum_r);
    cudaFree(d_sum_g);
    cudaFree(d_sum_b);
    cudaFree(d_sum_r2);
    cudaFree(d_sum_g2);
    cudaFree(d_sum_b2);
    cudaFree(d_count);
    return 1;
  }
```

- [ ] **Wrap the 7 `cudaMemset` calls with error checks.** Use the cleanup-and-return pattern already established in the function. Example for the block:

```c
  err = cudaMemset(d_sum_r, 0, sizeof(double));
  if (err != cudaSuccess) goto cuda_cleanup_stats;
  err = cudaMemset(d_sum_g, 0, sizeof(double));
  if (err != cudaSuccess) goto cuda_cleanup_stats;
  err = cudaMemset(d_sum_b, 0, sizeof(double));
  if (err != cudaSuccess) goto cuda_cleanup_stats;
  err = cudaMemset(d_sum_r2, 0, sizeof(double));
  if (err != cudaSuccess) goto cuda_cleanup_stats;
  err = cudaMemset(d_sum_g2, 0, sizeof(double));
  if (err != cudaSuccess) goto cuda_cleanup_stats;
  err = cudaMemset(d_sum_b2, 0, sizeof(double));
  if (err != cudaSuccess) goto cuda_cleanup_stats;
  err = cudaMemset(d_count, 0, sizeof(int));
  if (err != cudaSuccess) goto cuda_cleanup_stats;
  goto cuda_memset_ok;

cuda_cleanup_stats:
  printf("cudaMemset failed: %s\n", cudaGetErrorString(err));
  cudaFree(d_sum_r); cudaFree(d_sum_g); cudaFree(d_sum_b);
  cudaFree(d_sum_r2); cudaFree(d_sum_g2); cudaFree(d_sum_b2);
  cudaFree(d_count);
  return 1;
cuda_memset_ok:;
```

- [ ] **Build with CUDA:** `make USE_CUDA=1 2>&1 | grep -E "error:" | head -20` (skip if no CUDA toolkit available; the CPU build should still pass)

---

## Task 15: Fix config.c — dead write with wrong sizeof argument

**Files:**
- Modify: `src/config.c`

**Problem:** `snprintf(config->indigoserver, sizeof(config->indigoservername), "remote")` uses the size of a *different* field (`indigoservername`) for the buffer `indigoserver`. Additionally, the very next line overwrites `indigoserver` with `"127.0.0.1"`, making line 211 a dead write.

- [ ] **Remove the dead write (line 211):**

```bash
grep -n 'snprintf.*indigoserver.*"remote"' src/config.c
```

Delete the line:
```c
  snprintf(config->indigoserver, sizeof(config->indigoservername), "remote");
```

The following line `snprintf(config->indigoserver, sizeof(config->indigoserver), "127.0.0.1");` is correct and sufficient.

- [ ] **Build:** `make 2>&1 | grep -E "error:" | head -20`

---

## Task 16: Final build and test verification

- [ ] **Full CPU build:**

```bash
make clean && make 2>&1 | tail -20
```

Expected: `bin/allsky360-camera` produced, no errors.

- [ ] **Run tests:**

```bash
make tests
bin/test_sun_calc
```

- [ ] **Check for remaining `assert(` in memory helpers:**

```bash
grep -n "assert(" include/allsky.h
```

Expected: no `assert` calls remain.

- [ ] **Check for remaining `/ 3\.0` in debayer files:**

```bash
grep -rn "/ 3\.0" src/debayer_*.c
```

Expected: no results.

- [ ] **Check sun_calc for remaining `jd + rst`:**

```bash
grep -n "jd + rst\." src/sun_calc.c
```

Expected: no results.

- [ ] **Commit all fixes:**

```bash
git add include/allsky.h \
        src/jpeg_functions.c \
        src/sun_calc.c \
        src/dehaze_filter.c \
        src/main.c \
        src/debayer_bilinear.c \
        src/acdnr_filter.c \
        src/daily_database.c \
        src/hdr_merge.c \
        src/lights_and_shadows.c \
        src/debayer_vng.c \
        src/white_balance_cuda.cu \
        src/config.c
git commit -m "fix: address 20 bugs from src/ code review

- allsky.h: replace assert with explicit abort in safe-alloc helpers
- jpeg_functions: add setjmp/longjmp libjpeg error handler
- sun_calc: fix rst.rise/rst.set absolute JD (remove jd+ offset); remove wrong latitude short-circuit
- dehaze_filter: fix OpenMP reduction+atomic conflict; size_t pixel_count
- main: async-signal-safe handlers; volatile count/globals; argv bounds check; time_t; correct post-process path
- debayer_bilinear: fix green channel divisor 3->2
- acdnr_filter: pass computed mask to box_blur
- daily_database: sqlite3_close on open failure
- hdr_merge: size_t pixel_count to prevent int overflow
- lights_and_shadows: guard denom==0 before divide
- debayer_vng: memset output buffer to zero border pixels
- white_balance_cuda: check cudaDeviceSynchronize and cudaMemset return values
- config: remove dead write with wrong sizeof"
```

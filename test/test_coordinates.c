/* Tests for parse_lat_lon() (review.md, BUG-10) */
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../src/coordinates.h"

static void expect(const char *value, int is_latitude, double expected) {
  double result = 999.0;
  int rc = parse_lat_lon(value, is_latitude, &result);
  if (rc != 0 || fabs(result - expected) > 1e-9) {
    fprintf(stderr, "FAIL: '%s' -> rc=%d result=%f, expected %f\n", value, rc,
            result, expected);
    assert(0);
  }
}

static void expect_invalid(const char *value, int is_latitude) {
  double result = 42.0;
  int rc = parse_lat_lon(value, is_latitude, &result);
  if (rc == 0 || result != 42.0) {
    fprintf(stderr, "FAIL: '%s' accepted as %f\n", value, result);
    assert(0);
  }
}

int main(void) {
  /* Acceptance criteria from review.md */
  expect("33.9249S", 1, -33.9249);   /* Cape Town */
  expect("74.0060W", 0, -74.0060);   /* New York */
  expect("-33.9249", 1, -33.9249);   /* signed values unchanged */
  expect("-18.4241", 0, -18.4241);

  /* Northern/eastern suffixes and signs */
  expect("51.2349N", 1, 51.2349);
  expect("7.4479E", 0, 7.4479);
  expect("51.2349", 1, 51.2349);
  expect("+7.4479", 0, 7.4479);
  expect("33.9249 s", 1, -33.9249);  /* blank and lower case */
  expect("  12.5w  ", 0, -12.5);
  expect("90", 1, 90.0);
  expect("-180", 0, -180.0);

  /* Invalid input must be rejected, never silently mirrored */
  expect_invalid("", 1);
  expect_invalid("abc", 1);
  expect_invalid("-33.9249S", 1);    /* sign and suffix */
  expect_invalid("51.2349E", 1);     /* longitude suffix on a latitude */
  expect_invalid("7.4479N", 0);      /* latitude suffix on a longitude */
  expect_invalid("51.2349NN", 1);
  expect_invalid("51.2349 N x", 1);
  expect_invalid("90.1", 1);         /* out of range */
  expect_invalid("180.5W", 0);
  expect_invalid("nan", 1);

  printf("test_coordinates: all tests passed\n");
  return 0;
}

/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sensors_config.h"

void sensors_config_defaults(sensors_config_t *c) {
  memset(c, 0, sizeof(*c));
  snprintf(c->database_directory, sizeof(c->database_directory),
           "/home/allskyuser/database");
  snprintf(c->device, sizeof(c->device), "/dev/ttyTHS1");
  c->baudrate = 4800;
  snprintf(c->gpio_chip, sizeof(c->gpio_chip), "/dev/gpiochip0");
  c->gpio_line = 112; /* Jetson PR.04 / UART1_RTS */
  c->timeout_ms = 1200;
  c->retries = 3;
  c->function = 0x03;

  c->weather_address = 0;
  c->weather_register = 0x01F4;
  c->wind_speed_address = 0;
  c->wind_speed_register = 1;
  c->wind_speed_scale = 1.0;
  c->wind_direction_address = 0;
  c->wind_direction_register = 1;
  c->wind_calm_speed = 0.3; /* Beaufort 0, calm: below 0.3 m/s */
  c->cabinet_address = 0;
  c->cabinet_register = 0;
}

/* Integer with range check; accepts decimal and 0x hex */
static void set_int(const char *key, const char *value, long min, long max,
                    int *target) {
  char *end = NULL;
  errno = 0;
  long v = strtol(value, &end, 0);
  if (end == value || *end != '\0' || errno != 0 || v < min || v > max) {
    fprintf(stderr, "WARNING: config.ini: invalid %s '%s' (allowed %ld..%ld), using %d\n",
            key, value, min, max, *target);
    return;
  }
  *target = (int)v;
}

static void set_double(const char *key, const char *value, double min,
                       double max, double *target) {
  char *end = NULL;
  errno = 0;
  double v = strtod(value, &end);
  if (end == value || *end != '\0' || errno != 0 || v < min || v > max) {
    fprintf(stderr, "WARNING: config.ini: invalid %s '%s', using %g\n", key,
            value, *target);
    return;
  }
  *target = v;
}

static void set_string(const char *key, const char *value, char *target,
                       size_t size) {
  if (strlen(value) >= size) {
    fprintf(stderr, "WARNING: config.ini: %s is longer than %zu characters, ignored\n",
            key, size - 1);
    return;
  }
  snprintf(target, size, "%s", value);
}

/* Trim blanks (and a trailing CR) on both ends, in place */
static char *trim(char *s) {
  while (*s == ' ' || *s == '\t')
    s++;
  size_t len = strlen(s);
  while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t' ||
                     s[len - 1] == '\r' || s[len - 1] == '\n'))
    s[--len] = '\0';
  return s;
}

int sensors_config_load(const char *filename, sensors_config_t *c) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "ERROR: cannot open %s: %s\n", filename, strerror(errno));
    return -1;
  }

  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    if (!strchr(line, '\n') && !feof(f)) {
      int ch;
      while ((ch = fgetc(f)) != EOF && ch != '\n')
        ;
      continue; /* over-long line, not one of ours */
    }
    char *p = trim(line);
    if (*p == '\0' || *p == '#')
      continue;
    char *eq = strchr(p, '=');
    if (!eq)
      continue;
    *eq = '\0';
    char *key = trim(p);
    char *value = trim(eq + 1);
    size_t vlen = strlen(value);
    if (vlen >= 2 && value[0] == '"' && value[vlen - 1] == '"') {
      value[vlen - 1] = '\0';
      value++;
    }

    if (strcmp(key, "database_directory") == 0)
      set_string(key, value, c->database_directory, sizeof(c->database_directory));
    else if (strncmp(key, "modbus_", 7) != 0)
      continue;
    else if (strcmp(key, "modbus_device") == 0)
      set_string(key, value, c->device, sizeof(c->device));
    else if (strcmp(key, "modbus_baudrate") == 0)
      set_int(key, value, 1200, 115200, &c->baudrate);
    else if (strcmp(key, "modbus_gpio_chip") == 0)
      set_string(key, value, c->gpio_chip, sizeof(c->gpio_chip));
    else if (strcmp(key, "modbus_gpio_line") == 0)
      set_int(key, value, -1, 1023, &c->gpio_line);
    else if (strcmp(key, "modbus_timeout_ms") == 0)
      set_int(key, value, 50, 10000, &c->timeout_ms);
    else if (strcmp(key, "modbus_retries") == 0)
      set_int(key, value, 1, 10, &c->retries);
    else if (strcmp(key, "modbus_function") == 0)
      set_int(key, value, 3, 4, &c->function);
    else if (strcmp(key, "modbus_weather_address") == 0)
      set_int(key, value, 0, 247, &c->weather_address);
    else if (strcmp(key, "modbus_weather_register") == 0)
      set_int(key, value, 0, 0xFFF0, &c->weather_register);
    else if (strcmp(key, "modbus_weather_wind_direction_offset") == 0)
      set_int(key, value, 0, 359, &c->weather_wind_direction_offset);
    else if (strcmp(key, "modbus_wind_speed_address") == 0)
      set_int(key, value, 0, 247, &c->wind_speed_address);
    else if (strcmp(key, "modbus_wind_speed_register") == 0)
      set_int(key, value, 0, 0xFFFF, &c->wind_speed_register);
    else if (strcmp(key, "modbus_wind_speed_scale") == 0)
      set_double(key, value, 0.001, 10000.0, &c->wind_speed_scale);
    else if (strcmp(key, "modbus_wind_direction_address") == 0)
      set_int(key, value, 0, 247, &c->wind_direction_address);
    else if (strcmp(key, "modbus_wind_direction_register") == 0)
      set_int(key, value, 0, 0xFFFF, &c->wind_direction_register);
    else if (strcmp(key, "modbus_wind_direction_offset") == 0)
      set_int(key, value, 0, 359, &c->wind_direction_offset);
    else if (strcmp(key, "modbus_wind_calm_speed") == 0)
      set_double(key, value, 0.0, 10.0, &c->wind_calm_speed);
    else if (strcmp(key, "modbus_cabinet_address") == 0)
      set_int(key, value, 0, 247, &c->cabinet_address);
    else if (strcmp(key, "modbus_cabinet_register") == 0)
      set_int(key, value, 0, 0xFFFE, &c->cabinet_register);
    else
      fprintf(stderr, "WARNING: config.ini: unknown key '%s' (ignored)\n", key);
  }
  fclose(f);
  return 0;
}

void sensors_config_print(const sensors_config_t *c) {
  printf("database_directory: %s\n", c->database_directory);
  printf("bus: %s %d baud, DE GPIO %s line %d, timeout %d ms, %d attempts, function 0x%02X\n",
         c->device, c->baudrate, c->gpio_chip, c->gpio_line, c->timeout_ms,
         c->retries, c->function);
  printf("weather sensor:  address %d, register 0x%04X, wind offset %d\n",
         c->weather_address, c->weather_register, c->weather_wind_direction_offset);
  printf("anemometer:      address %d, register %d, scale %g\n",
         c->wind_speed_address, c->wind_speed_register, c->wind_speed_scale);
  printf("wind vane:       address %d, register %d, offset %d, held below %g m/s\n",
         c->wind_direction_address, c->wind_direction_register, c->wind_direction_offset,
         c->wind_calm_speed);
  printf("cabinet sensor:  address %d, register %d\n", c->cabinet_address,
         c->cabinet_register);
}

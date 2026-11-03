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
#include <fcntl.h>
#include <gpiod.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "modbus_rtu.h"

/* Silence before every request. Modbus RTU needs at least 3.5 character
 * times between frames; the sensors on this bus additionally drop requests
 * that follow an answer too quickly. */
#define INTER_FRAME_US 100000
/* Pause between two attempts of the same request */
#define RETRY_DELAY_US 200000
/* How long to wait for another poller to release the bus */
#define LOCK_WAIT_MS 15000

uint16_t modbus_crc16(const uint8_t *data, int len) {
  uint16_t crc = 0xFFFF;
  for (int i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++)
      crc = (crc & 1) ? (uint16_t)((crc >> 1) ^ 0xA001) : (uint16_t)(crc >> 1);
  }
  return crc;
}

static speed_t baud_constant(int baudrate) {
  switch (baudrate) {
  case 1200: return B1200;
  case 2400: return B2400;
  case 4800: return B4800;
  case 9600: return B9600;
  case 19200: return B19200;
  case 38400: return B38400;
  case 57600: return B57600;
  case 115200: return B115200;
  default: return 0;
  }
}

/* Wait for the exclusive lock, other allsky360 pollers hold it only briefly */
static int lock_port(int fd, const char *device) {
  for (int waited = 0; waited < LOCK_WAIT_MS; waited += 100) {
    if (flock(fd, LOCK_EX | LOCK_NB) == 0)
      return 0;
    if (errno != EWOULDBLOCK) {
      fprintf(stderr, "ERROR: cannot lock %s: %s\n", device, strerror(errno));
      return -1;
    }
    usleep(100000);
  }
  fprintf(stderr, "ERROR: %s is busy (locked by another process)\n", device);
  return -1;
}

int modbus_open(modbus_bus_t *bus, const char *device, int baudrate,
                const char *gpio_chip, int gpio_line, int timeout_ms,
                int retries) {
  memset(bus, 0, sizeof(*bus));
  bus->fd = -1;
  bus->timeout_ms = timeout_ms > 0 ? timeout_ms : 1200;
  bus->retries = retries > 0 ? retries : 3;

  speed_t speed = baud_constant(baudrate);
  if (!speed) {
    fprintf(stderr, "ERROR: unsupported baud rate %d\n", baudrate);
    return -1;
  }

  int fd = open(device, O_RDWR | O_NOCTTY | O_CLOEXEC);
  if (fd < 0) {
    fprintf(stderr, "ERROR: cannot open %s: %s\n", device, strerror(errno));
    return -1;
  }
  if (lock_port(fd, device) != 0) {
    close(fd);
    return -1;
  }

  struct termios tty;
  if (tcgetattr(fd, &tty) != 0) {
    fprintf(stderr, "ERROR: tcgetattr %s: %s\n", device, strerror(errno));
    close(fd);
    return -1;
  }
  cfmakeraw(&tty);
  cfsetispeed(&tty, speed);
  cfsetospeed(&tty, speed);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~(PARENB | CSTOPB | CSIZE | CRTSCTS); /* 8N1, no flow control */
  tty.c_cflag |= CS8;
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;
  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    fprintf(stderr, "ERROR: tcsetattr %s: %s\n", device, strerror(errno));
    close(fd);
    return -1;
  }
  tcflush(fd, TCIOFLUSH);
  bus->fd = fd;

  if (gpio_line >= 0 && gpio_chip && *gpio_chip) {
    struct gpiod_chip *chip = gpiod_chip_open(gpio_chip);
    if (!chip) {
      fprintf(stderr, "ERROR: cannot open GPIO chip %s: %s\n", gpio_chip,
              strerror(errno));
      modbus_close(bus);
      return -1;
    }
    struct gpiod_line *line = gpiod_chip_get_line(chip, (unsigned)gpio_line);
    if (!line || gpiod_line_request_output(line, "allsky360-sensors", 0) < 0) {
      fprintf(stderr, "ERROR: cannot use GPIO line %d of %s as output: %s\n",
              gpio_line, gpio_chip, strerror(errno));
      gpiod_chip_close(chip);
      modbus_close(bus);
      return -1;
    }
    bus->gpio_chip = chip;
    bus->gpio_line = line;
  }
  return 0;
}

void modbus_close(modbus_bus_t *bus) {
  if (bus->gpio_line)
    gpiod_line_release((struct gpiod_line *)bus->gpio_line);
  if (bus->gpio_chip)
    gpiod_chip_close((struct gpiod_chip *)bus->gpio_chip);
  if (bus->fd >= 0)
    close(bus->fd); /* also releases the flock */
  bus->gpio_line = NULL;
  bus->gpio_chip = NULL;
  bus->fd = -1;
}

/* Read up to `len` bytes; stops at the timeout. Returns the bytes read. */
static int read_with_timeout(int fd, uint8_t *buf, int len, int timeout_ms) {
  int total = 0;
  while (total < len) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    struct timeval tv = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    int rv = select(fd + 1, &rfds, NULL, NULL, &tv);
    if (rv < 0 && errno == EINTR)
      continue;
    if (rv <= 0)
      break;
    ssize_t n = read(fd, buf + total, (size_t)(len - total));
    if (n <= 0)
      break;
    total += (int)n;
  }
  return total;
}

static void set_direction(modbus_bus_t *bus, int transmit) {
  if (bus->gpio_line)
    gpiod_line_set_value((struct gpiod_line *)bus->gpio_line, transmit);
}

int modbus_read_registers(modbus_bus_t *bus, uint8_t address, uint8_t function,
                          uint16_t reg, uint16_t count, uint16_t *values) {
  if (bus->fd < 0 || address < 1 || address > 247 || count < 1 ||
      count > MODBUS_MAX_REGISTERS || (function != 0x03 && function != 0x04))
    return MODBUS_ERR_ARGS;

  uint8_t req[8] = {address,          function,         (uint8_t)(reg >> 8),
                    (uint8_t)reg,     (uint8_t)(count >> 8), (uint8_t)count};
  uint16_t crc = modbus_crc16(req, 6);
  req[6] = (uint8_t)(crc & 0xFF);
  req[7] = (uint8_t)(crc >> 8);

  const int expected = 3 + count * 2 + 2;
  uint8_t resp[3 + MODBUS_MAX_REGISTERS * 2 + 2];

  for (int attempt = 1; attempt <= bus->retries; attempt++) {
    usleep(attempt == 1 ? INTER_FRAME_US : RETRY_DELAY_US);
    tcflush(bus->fd, TCIOFLUSH);

    set_direction(bus, 1);
    ssize_t written = write(bus->fd, req, sizeof(req));
    tcdrain(bus->fd); /* until the last bit has left the UART */
    set_direction(bus, 0);
    if (written != (ssize_t)sizeof(req)) {
      fprintf(stderr, "WARNING: write to the bus failed: %s\n", strerror(errno));
      continue;
    }

    int n = read_with_timeout(bus->fd, resp, expected, bus->timeout_ms);

    /* Exception answer: address, function|0x80, code, CRC (5 bytes) */
    if (n >= 5 && resp[0] == address && resp[1] == (function | 0x80) &&
        modbus_crc16(resp, 3) == (uint16_t)(resp[3] | (resp[4] << 8))) {
      if (bus->verbose)
        fprintf(stderr, "Modbus exception 0x%02X from address %u, register 0x%04X\n",
                resp[2], address, reg);
      return MODBUS_ERR_EXCEPTION;
    }

    if (n == expected && resp[0] == address && resp[1] == function &&
        resp[2] == count * 2 &&
        modbus_crc16(resp, expected - 2) ==
            (uint16_t)(resp[expected - 2] | (resp[expected - 1] << 8))) {
      for (int i = 0; i < count; i++)
        values[i] = (uint16_t)((resp[3 + i * 2] << 8) | resp[4 + i * 2]);
      return MODBUS_OK;
    }

    if (bus->verbose) {
      fprintf(stderr, "address %u register 0x%04X: attempt %d/%d failed (%d of %d bytes:",
              address, reg, attempt, bus->retries, n, expected);
      for (int i = 0; i < n; i++)
        fprintf(stderr, " %02X", resp[i]);
      fprintf(stderr, ")\n");
    }
  }
  return MODBUS_ERR_TIMEOUT;
}

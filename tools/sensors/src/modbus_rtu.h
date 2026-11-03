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
#ifndef MODBUS_RTU_H
#define MODBUS_RTU_H

#include <stdint.h>

/*
 * Minimal Modbus RTU master over a serial RS485 line.
 *
 * The RS485 transceiver direction (DE/RE) can be switched by a GPIO line
 * (libgpiod v1), as on the Jetson carrier board used here: high while
 * sending, low while receiving. Transceivers with automatic direction
 * control need no GPIO (gpio_line < 0).
 */

#define MODBUS_MAX_REGISTERS 32

typedef struct {
  int fd;
  void *gpio_chip; /* struct gpiod_chip * */
  void *gpio_line; /* struct gpiod_line *, NULL = no direction switching */
  int timeout_ms;  /* wait for an answer (the sensors need up to ~1 s) */
  int retries;     /* attempts per request */
  int verbose;
} modbus_bus_t;

/* Result of modbus_read_registers() */
#define MODBUS_OK 0
#define MODBUS_ERR_TIMEOUT -1   /* no or incomplete/corrupt answer */
#define MODBUS_ERR_EXCEPTION -2 /* slave answered with an exception code */
#define MODBUS_ERR_ARGS -3

/**
 * Open the serial port (8N1, raw) and the optional direction GPIO.
 * Also takes an exclusive lock on the port, so two pollers never talk on the
 * bus at the same time.
 * @return 0 on success, -1 on error (message on stderr)
 */
int modbus_open(modbus_bus_t *bus, const char *device, int baudrate,
                const char *gpio_chip, int gpio_line, int timeout_ms,
                int retries);

void modbus_close(modbus_bus_t *bus);

/**
 * Read `count` 16-bit registers starting at `reg` with function code
 * `function` (0x03 holding or 0x04 input registers).
 * @return MODBUS_OK, MODBUS_ERR_TIMEOUT, MODBUS_ERR_EXCEPTION or
 *         MODBUS_ERR_ARGS
 */
int modbus_read_registers(modbus_bus_t *bus, uint8_t address, uint8_t function,
                          uint16_t reg, uint16_t count, uint16_t *values);

/* CRC-16/MODBUS, exposed for the unit test */
uint16_t modbus_crc16(const uint8_t *data, int len);

#endif /* MODBUS_RTU_H */

/* wind_speed.c
 * Liest einen RS485-Windgeschwindigkeitssensor (Modbus-RTU) aus und gibt die
 * Windgeschwindigkeit in m/s, km/h, Knoten und als Beaufort-Stufe aus.
 *
 * Kommunikation wie weather-sensor / wind-direction: /dev/ttyTHS1, 4800 Baud
 * 8N1, RTS/DE ueber GPIO (libgpiod, /dev/gpiochip0). Die Windgeschwindigkeit
 * steht im Holding-Register 1. Modbus-Register sind immer 16 Bit breit.
 *
 * Default ist --scale 1, d.h. der Rohwert wird unveraendert als m/s
 * interpretiert (Vergleichsmessung mit dem Wetterstations-Sensor legt nahe,
 * dass dieser Sensor in ganzen m/s liefert; noch nicht mit bekannter
 * Windgeschwindigkeit verifiziert). Sensoren mit Rohwert = m/s * 10 (Aufloesung
 * 0,1 m/s) lassen sich per --scale 10 abfragen.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <getopt.h>
#include <sys/select.h>
#include <gpiod.h>
#include "json_output.h"

#define TOOL_NAME       "wind-speed"
#define SERIAL_DEV      "/dev/ttyTHS1"
#define GPIO_CHIP       "/dev/gpiochip0"
#define GPIO_LINE_OFFS  112          /* PR.04, UART1_RTS_PR4 (Pin 207) */
#define BAUD            B4800
#define DEFAULT_ADDR    1
#define FUNC_READ_HOLD  0x03
#define REG_WIND_SPEED  1            /* Windgeschwindigkeit */
#define TIMEOUT_MS      1200
#define RETRIES         3
#define RESP_LEN        7            /* addr, func, bytecount, 2 Daten, 2 CRC */

static bool g_json = false;

/* Einheitliche Fehlerausgabe: JSON (stdout) oder Klartext (stderr), Exit 1 */
static int fail(const char *msg, const char *code, bool retry_safe) {
    if (g_json) JSON_ERROR(TOOL_NAME, msg, code, retry_safe);
    else fprintf(stderr, "Fehler (%s): %s\n", code, msg);
    return 1;
}

static uint16_t crc16_modbus(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            if (crc & 1) crc = (crc >> 1) ^ 0xA001;
            else crc >>= 1;
        }
    }
    return crc;
}

static bool check_crc(const uint8_t *resp, size_t len) {
    uint16_t recv = resp[len - 2] | (resp[len - 1] << 8);
    return recv == crc16_modbus(resp, len - 2);
}

static int serial_open(const char *dev) {
    int fd = open(dev, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Fehler beim Oeffnen von %s: %s", dev, strerror(errno));
        if (errno == ENOENT || errno == ENODEV)
            fail(msg, "HARDWARE_NOT_FOUND", false);
        else if (errno == EACCES || errno == EPERM)
            fail(msg, "PERMISSION_DENIED", false);
        else
            fail(msg, "UNKNOWN_ERROR", false);
        return -1;
    }

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "tcgetattr fehlgeschlagen: %s", strerror(errno));
        fail(msg, "HARDWARE_NOT_FOUND", false);
        close(fd);
        return -1;
    }
    cfmakeraw(&tty);
    cfsetispeed(&tty, BAUD);
    cfsetospeed(&tty, BAUD);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | CSTOPB | CSIZE | CRTSCTS);
    tty.c_cflag |= CS8;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "tcsetattr fehlgeschlagen: %s", strerror(errno));
        fail(msg, "HARDWARE_NOT_FOUND", false);
        close(fd);
        return -1;
    }
    tcflush(fd, TCIOFLUSH);
    return fd;
}

static ssize_t serial_read_timeout(int fd, uint8_t *buf, size_t expected_len, int timeout_ms) {
    size_t total = 0;
    while (total < expected_len) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        if (select(fd + 1, &rfds, NULL, NULL, &tv) <= 0) break;
        ssize_t n = read(fd, buf + total, expected_len - total);
        if (n <= 0) break;
        total += (size_t)n;
    }
    return (ssize_t)total;
}

static struct gpiod_line *gpio_setup(struct gpiod_chip **chip_out) {
    struct gpiod_chip *chip = gpiod_chip_open(GPIO_CHIP);
    if (!chip) {
        char msg[160];
        snprintf(msg, sizeof(msg), "GPIO-Chip %s konnte nicht geoeffnet werden: %s",
                 GPIO_CHIP, strerror(errno));
        fail(msg, (errno == EACCES || errno == EPERM) ? "PERMISSION_DENIED" : "HARDWARE_NOT_FOUND", false);
        return NULL;
    }
    struct gpiod_line *line = gpiod_chip_get_line(chip, GPIO_LINE_OFFS);
    if (!line || gpiod_line_request_output(line, TOOL_NAME, 0) < 0) {
        char msg[160];
        snprintf(msg, sizeof(msg), "GPIO-Line %d nicht verfuegbar: %s", GPIO_LINE_OFFS, strerror(errno));
        fail(msg, (errno == EACCES || errno == EPERM) ? "PERMISSION_DENIED" : "HARDWARE_NOT_FOUND", false);
        gpiod_chip_close(chip);
        return NULL;
    }
    *chip_out = chip;
    return line;
}

static bool read_register(int fd, struct gpiod_line *rts, uint8_t addr, uint16_t reg,
                          uint16_t *value) {
    uint8_t req[8] = { addr, FUNC_READ_HOLD, reg >> 8, reg & 0xFF, 0, 1 };
    uint16_t crc = crc16_modbus(req, 6);
    req[6] = crc & 0xFF;
    req[7] = crc >> 8;
    uint8_t resp[RESP_LEN];

    for (int attempt = 0; attempt < RETRIES; attempt++) {
        tcflush(fd, TCIFLUSH);
        gpiod_line_set_value(rts, 1);
        ssize_t w = write(fd, req, sizeof(req));
        (void)w;
        tcdrain(fd);
        gpiod_line_set_value(rts, 0);

        ssize_t n = serial_read_timeout(fd, resp, RESP_LEN, TIMEOUT_MS);
        if (n == RESP_LEN && check_crc(resp, RESP_LEN) &&
            resp[0] == addr && resp[1] == FUNC_READ_HOLD && resp[2] == 2) {
            *value = (uint16_t)((resp[3] << 8) | resp[4]);
            return true;
        }
        if (attempt + 1 < RETRIES) usleep(50000);
    }
    return false;
}

#define DEFAULT_SCALE   1.0          /* Rohwert / SCALE = m/s */
#define MAX_PLAUSIBLE   60.0         /* m/s, darueber: Sensor defekt/falsche Skalierung */

/* Obergrenzen der Beaufort-Stufen 0-11 in m/s, darueber 12 */
static const double BEAUFORT_LIMIT[12] = {
    0.3, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6
};

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Verwendung: %s [-a ADRESSE] [-s TEILER] [-j] [-h]\n"
        "  -a, --address ADRESSE  Modbus-Slave-Adresse 1-247 (Default: %d)\n"
        "  -s, --scale TEILER     Rohwert wird durch TEILER geteilt = m/s\n"
        "                         (Default: 1, d.h. Rohwert in m/s;\n"
        "                         10 = Rohwert in 0,1 m/s)\n"
        "  -j, --json             Ausgabe als JSON statt Klartext\n"
        "  -h, --help             Diese Hilfe anzeigen\n"
        "\n"
        "Exit-Code 0 bei Erfolg, 1 bei Fehler.\n",
        prog, DEFAULT_ADDR);
}

static bool parse_int(const char *s, long lo, long hi, long *out) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v < lo || v > hi) return false;
    *out = v;
    return true;
}

static bool parse_double(const char *s, double lo, double hi, double *out) {
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0' || v < lo || v > hi) return false;
    *out = v;
    return true;
}

int main(int argc, char *argv[]) {
    long address = DEFAULT_ADDR;
    double scale = DEFAULT_SCALE;

    /* -j vorab suchen, damit auch Argumentfehler im gewuenschten Format kommen */
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-j") || !strcmp(argv[i], "--json")) g_json = true;

    static struct option long_opts[] = {
        {"address", required_argument, 0, 'a'},
        {"scale",   required_argument, 0, 's'},
        {"json",    no_argument,       0, 'j'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    char msg[192];
    while ((opt = getopt_long(argc, argv, "a:s:jh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'a':
                if (!parse_int(optarg, 1, 247, &address)) {
                    snprintf(msg, sizeof(msg), "Ungueltige Adresse: %s (erlaubt: 1-247)", optarg);
                    return fail(msg, "INVALID_ARGS", false);
                }
                break;
            case 's':
                if (!parse_double(optarg, 0.001, 100000.0, &scale)) {
                    snprintf(msg, sizeof(msg), "Ungueltiger Teiler: %s (erlaubt: 0.001-100000)", optarg);
                    return fail(msg, "INVALID_ARGS", false);
                }
                break;
            case 'j':
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return fail("Unbekannte Option", "INVALID_ARGS", false);
        }
    }

    int fd = serial_open(SERIAL_DEV);
    if (fd < 0) return 1;

    struct gpiod_chip *chip = NULL;
    struct gpiod_line *rts = gpio_setup(&chip);
    if (!rts) {
        close(fd);
        return 1;
    }

    uint16_t raw = 0;
    bool ok = read_register(fd, rts, (uint8_t)address, REG_WIND_SPEED, &raw);

    gpiod_line_release(rts);
    gpiod_chip_close(chip);
    close(fd);

    if (!ok) {
        snprintf(msg, sizeof(msg),
                 "Keine gueltige Antwort erhalten (Timeout nach mehreren Versuchen), Adresse %ld, Register 0x%04X",
                 address, REG_WIND_SPEED);
        return fail(msg, "SENSOR_TIMEOUT", true);
    }

    double ms = raw / scale;
    if (ms > MAX_PLAUSIBLE) {
        snprintf(msg, sizeof(msg),
                 "Windgeschwindigkeit ausserhalb des plausiblen Bereichs: %.1f m/s (Rohwert %u, Teiler %.4g, erwartet 0-%.0f)",
                 ms, raw, scale, MAX_PLAUSIBLE);
        return fail(msg, "SENSOR_OUT_OF_RANGE", false);
    }

    double kmh = ms * 3.6;
    double knots = ms / 0.514444;
    int bft = 12;
    for (int i = 0; i < 12; i++) {
        if (ms <= BEAUFORT_LIMIT[i]) { bft = i; break; }
    }

    if (g_json) {
        char result[256];
        snprintf(result, sizeof(result),
                 "{\"address\": %ld, \"raw\": %u, \"scale\": %.4g, \"wind_speed_ms\": %.2f, "
                 "\"wind_speed_kmh\": %.2f, \"wind_speed_knots\": %.2f, \"beaufort\": %d}",
                 address, raw, scale, ms, kmh, knots, bft);
        JSON_SUCCESS_RESULT(TOOL_NAME, result);
    } else {
        printf("Adresse: %ld\n", address);
        printf("Rohwert: %u (Teiler %.4g)\n", raw, scale);
        printf("Windgeschwindigkeit: %.2f m/s = %.2f km/h = %.2f kn (Beaufort %d)\n",
               ms, kmh, knots, bft);
    }
    return 0;
}

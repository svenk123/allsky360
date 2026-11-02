/* modbus_scan.c
 * Sucht Modbus-RTU-Geraete am RS485-Bus: fragt einen Adressbereich nacheinander
 * mit einem Read-Request (Default: Holding-Register 1) ab. Jedes Geraet, das
 * mit einer CRC-gueltigen Antwort reagiert -- auch mit einer Modbus-Exception
 * (z.B. "Register nicht vorhanden") -- gilt als gefunden.
 *
 * Bus wie bei den anderen Tools: /dev/ttyTHS1, 8N1, RTS/DE ueber GPIO
 * (libgpiod, /dev/gpiochip0, Line 112). Die Baudrate ist einstellbar.
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

#define TOOL_NAME        "modbus-scan"
#define SERIAL_DEV       "/dev/ttyTHS1"
#define GPIO_CHIP        "/dev/gpiochip0"
#define GPIO_LINE_OFFS   112          /* PR.04, UART1_RTS_PR4 (Pin 207) */
#define DEFAULT_BAUD     4800
#define DEFAULT_START    1
#define DEFAULT_END      247
#define DEFAULT_REGISTER 1
#define DEFAULT_FUNC     0x03
#define DEFAULT_TIMEOUT  1200
#define DEFAULT_RETRIES  2
#define MAX_RESP_LEN     8

static bool g_json = false;

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
    if (len < 5) return false;
    uint16_t recv = resp[len - 2] | (resp[len - 1] << 8);
    return recv == crc16_modbus(resp, len - 2);
}

/* Baudrate -> termios-Konstante; 0 wenn nicht unterstuetzt */
static speed_t baud_to_speed(long baud) {
    switch (baud) {
        case 1200:   return B1200;
        case 2400:   return B2400;
        case 4800:   return B4800;
        case 9600:   return B9600;
        case 19200:  return B19200;
        case 38400:  return B38400;
        case 57600:  return B57600;
        case 115200: return B115200;
        default:     return 0;
    }
}

static int serial_open(const char *dev, speed_t speed) {
    int fd = open(dev, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Fehler beim Oeffnen von %s: %s", dev, strerror(errno));
        if (errno == ENOENT || errno == ENODEV) fail(msg, "HARDWARE_NOT_FOUND", false);
        else if (errno == EACCES || errno == EPERM) fail(msg, "PERMISSION_DENIED", false);
        else fail(msg, "UNKNOWN_ERROR", false);
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
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);
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

/* Liest, bis eine vollstaendige Antwort (Normalantwort oder 5-Byte-Exception)
 * da ist, max_len erreicht ist oder timeout_ms lang nichts mehr kommt. */
static ssize_t serial_read_response(int fd, uint8_t *buf, size_t max_len,
                                    uint8_t func, int timeout_ms) {
    size_t total = 0;
    while (total < max_len) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        if (select(fd + 1, &rfds, NULL, NULL, &tv) <= 0) break;
        ssize_t n = read(fd, buf + total, max_len - total);
        if (n <= 0) break;
        total += (size_t)n;
        if (total >= 5 && buf[1] == (uint8_t)(func | 0x80)) break;
        if (total >= 3 && buf[1] == func && total >= (size_t)(3 + buf[2] + 2)) break;
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

/* Liefert Antwortlaenge (CRC-gueltig, passende Adresse) oder 0 */
static size_t probe(int fd, struct gpiod_line *rts, uint8_t addr, uint8_t func,
                    uint16_t reg, int timeout_ms, int retries, uint8_t *resp) {
    uint8_t req[8] = { addr, func, reg >> 8, reg & 0xFF, 0, 1 };
    uint16_t crc = crc16_modbus(req, 6);
    req[6] = crc & 0xFF;
    req[7] = crc >> 8;

    for (int attempt = 0; attempt < retries; attempt++) {
        tcflush(fd, TCIFLUSH);
        gpiod_line_set_value(rts, 1);
        ssize_t w = write(fd, req, sizeof(req));
        (void)w;
        tcdrain(fd);
        gpiod_line_set_value(rts, 0);

        ssize_t n = serial_read_response(fd, resp, MAX_RESP_LEN, func, timeout_ms);
        if (n >= 5 && check_crc(resp, (size_t)n) && resp[0] == addr &&
            (resp[1] == func || resp[1] == (uint8_t)(func | 0x80)))
            return (size_t)n;
        if (attempt + 1 < retries) usleep(30000);
    }
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Verwendung: %s [-b BAUD] [-s START] [-e ENDE] [-r REGISTER] [-f FUNKTION]\n"
        "                  [-t TIMEOUT_MS] [-R VERSUCHE] [-j] [-h]\n"
        "  -b, --baud BAUD        Baudrate: 1200, 2400, 4800, 9600, 19200, 38400,\n"
        "                         57600, 115200 (Default: %d), Format 8N1\n"
        "  -s, --start ADRESSE    Erste Adresse, 1-247 (Default: %d)\n"
        "  -e, --end ADRESSE      Letzte Adresse, 1-247 (Default: %d)\n"
        "  -r, --register REG     Abgefragtes Register, dezimal oder 0x-hex\n"
        "                         (Default: %d)\n"
        "  -f, --function FUNC    0x03 Holding- oder 0x04 Input-Register (Default: 0x%02X)\n"
        "  -t, --timeout MS       Timeout pro Anfrage in ms (Default: %d)\n"
        "  -R, --retries N        Versuche pro Adresse (Default: %d)\n"
        "  -j, --json             Ausgabe als JSON statt Klartext\n"
        "  -h, --help             Diese Hilfe anzeigen\n"
        "\n"
        "Exit-Code: 0 = mindestens ein Geraet gefunden, 2 = keines gefunden,\n"
        "1 = Fehler. Ohne Antwort kostet jede Adresse TIMEOUT x VERSUCHE; der\n"
        "Default von 1200 ms passt zu Sensoren mit ~1 s Antwortzeit, schnellere\n"
        "Geraete lassen sich mit z.B. -t 300 deutlich zuegiger scannen.\n",
        prog, DEFAULT_BAUD, DEFAULT_START, DEFAULT_END, DEFAULT_REGISTER,
        DEFAULT_FUNC, DEFAULT_TIMEOUT, DEFAULT_RETRIES);
}

/* Zahl dezimal oder mit 0x-Praefix, strikt */
static bool parse_num(const char *s, long lo, long hi, long *out) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 0);
    if (errno || end == s || *end != '\0' || v < lo || v > hi) return false;
    *out = v;
    return true;
}

int main(int argc, char *argv[]) {
    long baud = DEFAULT_BAUD, start = DEFAULT_START, end = DEFAULT_END;
    long reg = DEFAULT_REGISTER, func = DEFAULT_FUNC;
    long timeout_ms = DEFAULT_TIMEOUT, retries = DEFAULT_RETRIES;

    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-j") || !strcmp(argv[i], "--json")) g_json = true;

    static struct option long_opts[] = {
        {"baud",     required_argument, 0, 'b'},
        {"start",    required_argument, 0, 's'},
        {"end",      required_argument, 0, 'e'},
        {"register", required_argument, 0, 'r'},
        {"function", required_argument, 0, 'f'},
        {"timeout",  required_argument, 0, 't'},
        {"retries",  required_argument, 0, 'R'},
        {"json",     no_argument,       0, 'j'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    char msg[192];
    while ((opt = getopt_long(argc, argv, "b:s:e:r:f:t:R:jh", long_opts, NULL)) != -1) {
        bool ok = true;
        switch (opt) {
            case 'b': ok = parse_num(optarg, 1, 4000000, &baud) && baud_to_speed(baud);
                      if (!ok) snprintf(msg, sizeof(msg), "Nicht unterstuetzte Baudrate: %s (erlaubt: 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200)", optarg);
                      break;
            case 's': ok = parse_num(optarg, 1, 247, &start);
                      if (!ok) snprintf(msg, sizeof(msg), "Ungueltige Startadresse: %s (erlaubt: 1-247)", optarg);
                      break;
            case 'e': ok = parse_num(optarg, 1, 247, &end);
                      if (!ok) snprintf(msg, sizeof(msg), "Ungueltige Endadresse: %s (erlaubt: 1-247)", optarg);
                      break;
            case 'r': ok = parse_num(optarg, 0, 0xFFFF, &reg);
                      if (!ok) snprintf(msg, sizeof(msg), "Ungueltiges Register: %s (0-0xFFFF)", optarg);
                      break;
            case 'f': ok = parse_num(optarg, 3, 4, &func);
                      if (!ok) snprintf(msg, sizeof(msg), "Ungueltige Funktion: %s (erlaubt: 0x03, 0x04)", optarg);
                      break;
            case 't': ok = parse_num(optarg, 10, 60000, &timeout_ms);
                      if (!ok) snprintf(msg, sizeof(msg), "Ungueltiger Timeout: %s (erlaubt: 10-60000 ms)", optarg);
                      break;
            case 'R': ok = parse_num(optarg, 1, 10, &retries);
                      if (!ok) snprintf(msg, sizeof(msg), "Ungueltige Anzahl Versuche: %s (erlaubt: 1-10)", optarg);
                      break;
            case 'j': break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]);
                      return fail("Unbekannte Option", "INVALID_ARGS", false);
        }
        if (!ok) return fail(msg, "INVALID_ARGS", false);
    }
    if (start > end) {
        snprintf(msg, sizeof(msg), "Startadresse %ld liegt hinter der Endadresse %ld", start, end);
        return fail(msg, "INVALID_ARGS", false);
    }

    int fd = serial_open(SERIAL_DEV, baud_to_speed(baud));
    if (fd < 0) return 1;

    struct gpiod_chip *chip = NULL;
    struct gpiod_line *rts = gpio_setup(&chip);
    if (!rts) {
        close(fd);
        return 1;
    }

    static char list[32768];
    size_t pos = 0;
    int found = 0;

    if (!g_json)
        printf("Scanne Adressen %ld-%ld, %ld Baud 8N1, %s Register %ld ...\n",
               start, end, baud, func == 3 ? "Holding" : "Input", reg);

    for (long addr = start; addr <= end; addr++) {
        uint8_t resp[MAX_RESP_LEN];
        size_t n = probe(fd, rts, (uint8_t)addr, (uint8_t)func, (uint16_t)reg,
                         (int)timeout_ms, (int)retries, resp);
        if (n == 0) continue;
        found++;

        bool exc = (resp[1] & 0x80) != 0;
        int value = (!exc && n >= 5) ? (int16_t)((resp[3] << 8) | resp[4]) : 0;
        int code = exc ? resp[2] : 0;

        if (g_json) {
            int w = exc
                ? snprintf(list + pos, sizeof(list) - pos, "%s{\"address\": %ld, \"exception\": %d}",
                           found > 1 ? ", " : "", addr, code)
                : snprintf(list + pos, sizeof(list) - pos, "%s{\"address\": %ld, \"value\": %d}",
                           found > 1 ? ", " : "", addr, value);
            if (w > 0 && pos + (size_t)w < sizeof(list)) pos += (size_t)w;
        } else if (exc) {
            printf("Adresse %ld: Geraet antwortet (Modbus-Exception %d)\n", addr, code);
            fflush(stdout);
        } else {
            printf("Adresse %ld: Geraet gefunden, Register %ld = %d\n", addr, reg, value);
            fflush(stdout);
        }
    }

    gpiod_line_release(rts);
    gpiod_chip_close(chip);
    close(fd);

    if (g_json) {
        static char result[32768 + 256];
        snprintf(result, sizeof(result),
                 "{\"baud\": %ld, \"start\": %ld, \"end\": %ld, \"register\": %ld, "
                 "\"function\": %ld, \"found\": %d, \"devices\": [%s]}",
                 baud, start, end, reg, func, found, list);
        JSON_SUCCESS_RESULT(TOOL_NAME, result);
    } else {
        printf("Scan abgeschlossen: %d Geraet(e) gefunden.\n", found);
    }
    return found > 0 ? 0 : 2;
}

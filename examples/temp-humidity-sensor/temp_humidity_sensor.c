/* temp_humidity_sensor.c
 * Liest den RS485-Temperatur/Feuchte-Sensor ueber Modbus-RTU.
 * RTS/DE-Umschaltung erfolgt ueber GPIO (libgpiod), Bus liegt an /dev/ttyTHS1.
 *
 * Register:
 *   0x0000  Luftfeuchtigkeit  0-100 %RH,     0.1 %RH Aufloesung
 *   0x0001  Temperatur        -30..+80 C,    0.1 C Aufloesung, signed
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

#define SERIAL_DEV      "/dev/ttyTHS1"
#define GPIO_CHIP       "/dev/gpiochip0"
#define GPIO_LINE_OFFS  112          /* PR.04, UART1_RTS_PR4 (Pin 207) */
#define BAUD            B4800
#define BAUD_VALUE      4800
#define DEFAULT_ADDR    1
#define REG_START       0x0000
#define REG_COUNT       2            /* Feuchte, Temperatur */
#define EXPECTED_LEN    9            /* addr+func+bytecount+4data+2crc */
#define MAX_RETRIES     5
#define READ_TIMEOUT_MS 300

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

static size_t build_read_frame(uint8_t *out, uint8_t addr, uint8_t func,
                                uint16_t reg, uint16_t count) {
    out[0] = addr;
    out[1] = func;
    out[2] = (reg >> 8) & 0xFF;
    out[3] = reg & 0xFF;
    out[4] = (count >> 8) & 0xFF;
    out[5] = count & 0xFF;
    uint16_t crc = crc16_modbus(out, 6);
    out[6] = crc & 0xFF;
    out[7] = (crc >> 8) & 0xFF;
    return 8;
}

static bool check_crc(const uint8_t *resp, size_t len) {
    if (len < 3) return false;
    uint16_t recv = resp[len - 2] | (resp[len - 1] << 8);
    uint16_t calc = crc16_modbus(resp, len - 2);
    return recv == calc;
}

static int serial_open(const char *dev) {
    int fd = open(dev, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        fprintf(stderr, "Fehler beim Oeffnen von %s: %s\n", dev, strerror(errno));
        return -1;
    }

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        fprintf(stderr, "tcgetattr fehlgeschlagen: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    cfmakeraw(&tty);
    cfsetispeed(&tty, BAUD);
    cfsetospeed(&tty, BAUD);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;   /* keine Paritaet */
    tty.c_cflag &= ~CSTOPB;   /* 1 Stopbit */
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;       /* 8 Datenbits */
    tty.c_cflag &= ~CRTSCTS;  /* kein Hardware-Flow-Control */

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        fprintf(stderr, "tcsetattr fehlgeschlagen: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    tcflush(fd, TCIOFLUSH);
    return fd;
}

/* Liest bis zu expected_len Bytes oder bis Timeout ablaeuft */
static ssize_t serial_read_timeout(int fd, uint8_t *buf, size_t expected_len,
                                    int timeout_ms) {
    size_t total = 0;
    while (total < expected_len) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        int rv = select(fd + 1, &rfds, NULL, NULL, &tv);
        if (rv <= 0) break; /* Timeout oder Fehler */

        ssize_t n = read(fd, buf + total, expected_len - total);
        if (n <= 0) break;
        total += (size_t)n;
    }
    return (ssize_t)total;
}

static struct gpiod_line *gpio_setup(struct gpiod_chip **chip_out) {
    struct gpiod_chip *chip = gpiod_chip_open(GPIO_CHIP);
    if (!chip) {
        fprintf(stderr, "GPIO-Chip konnte nicht geoeffnet werden: %s\n", strerror(errno));
        return NULL;
    }
    struct gpiod_line *line = gpiod_chip_get_line(chip, GPIO_LINE_OFFS);
    if (!line) {
        fprintf(stderr, "GPIO-Line %d konnte nicht geholt werden: %s\n",
                GPIO_LINE_OFFS, strerror(errno));
        gpiod_chip_close(chip);
        return NULL;
    }
    if (gpiod_line_request_output(line, "temp-humidity-sensor", 0) < 0) {
        fprintf(stderr, "GPIO-Line konnte nicht als Ausgang angefordert werden: %s\n",
                strerror(errno));
        gpiod_chip_close(chip);
        return NULL;
    }
    *chip_out = chip;
    return line;
}

static bool query_sensor(int fd, struct gpiod_line *rts, uint8_t addr,
                          uint8_t *resp, size_t expected_len) {
    uint8_t req[8];
    size_t req_len = build_read_frame(req, addr, 0x03, REG_START, REG_COUNT);

    for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
        tcflush(fd, TCIFLUSH);

        gpiod_line_set_value(rts, 1);   /* Senden */
        ssize_t written = write(fd, req, req_len);
        (void)written;
        tcdrain(fd);                     /* wartet bis Hardware fertig gesendet hat */
        gpiod_line_set_value(rts, 0);   /* Empfangen */

        ssize_t n = serial_read_timeout(fd, resp, expected_len, READ_TIMEOUT_MS);
        if (n == (ssize_t)expected_len && check_crc(resp, expected_len)) {
            return true;
        }
        usleep(50000); /* 50ms vor erneutem Versuch */
    }
    return false;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Verwendung: %s [-a ADRESSE] [-j] [-h]\n"
        "  -a, --address ADRESSE   Modbus-Slave-Adresse (Default: %d)\n"
        "  -j, --json              Ausgabe als JSON statt Klartext\n"
        "  -h, --help              Diese Hilfe anzeigen\n",
        prog, DEFAULT_ADDR);
}

int main(int argc, char *argv[]) {
    int address = DEFAULT_ADDR;
    bool json_output = false;

    static struct option long_opts[] = {
        {"address", required_argument, 0, 'a'},
        {"json",    no_argument,       0, 'j'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "a:jh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'a':
                address = atoi(optarg);
                if (address < 1 || address > 247) {
                    fprintf(stderr, "Ungueltige Adresse: %s (1-247 erlaubt)\n", optarg);
                    return 1;
                }
                break;
            case 'j':
                json_output = true;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
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

    uint8_t resp[EXPECTED_LEN];
    bool ok = query_sensor(fd, rts, (uint8_t)address, resp, EXPECTED_LEN);

    int ret = 0;
    if (ok) {
        uint16_t humidity_raw = (resp[3] << 8) | resp[4];
        int16_t  temp_raw     = (int16_t)((resp[5] << 8) | resp[6]);

        double humidity = humidity_raw / 10.0;
        double temp     = temp_raw / 10.0;

        if (json_output) {
            printf("{\"address\":%d,\"humidity_percent\":%.1f,"
                   "\"temperature_c\":%.1f}\n",
                   address, humidity, temp);
        } else {
            printf("Adresse: %d\n", address);
            printf("Feuchte: %.1f %%RH\n", humidity);
            printf("Temperatur: %.1f C\n", temp);
        }
    } else {
        if (json_output) {
            printf("{\"address\":%d,\"error\":\"no valid response\"}\n", address);
        } else {
            fprintf(stderr, "Keine gueltige Antwort von Adresse %d erhalten.\n", address);
        }
        ret = 2;
    }

    gpiod_line_release(rts);
    gpiod_chip_close(chip);
    close(fd);
    return ret;
}

/* modbus_tool.c
 * Hilfswerkzeug fuer RS485-Modbus-RTU-Sensoren: Scannt Slave-Adressen und
 * kann die Slave-Adresse eines Sensors per Write-Single-Register (0x06) neu
 * setzen. RTS/DE-Umschaltung erfolgt ueber GPIO (libgpiod), Bus liegt an
 * /dev/ttyTHS1 -- Beschaltung identisch zu soil_sensor/solar_sensor.
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
#define DEFAULT_REGISTER 1
#define MAX_RESP_LEN    8            /* groesste erwartete Antwort (Write-Echo) */
#define DEFAULT_TIMEOUT_MS   300
#define DEFAULT_RETRIES_SCAN 1
#define DEFAULT_RETRIES_SET  5

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

/* Baut ein 6-Byte-Modbus-Kommando (addr,func,reg_hi,reg_lo,x_hi,x_lo) + CRC.
 * Fuer Function 0x03 (Read Holding Registers) ist x = Registeranzahl,
 * fuer Function 0x06 (Write Single Register) ist x = zu schreibender Wert. */
static size_t build_frame(uint8_t *out, uint8_t addr, uint8_t func,
                           uint16_t reg, uint16_t x) {
    out[0] = addr;
    out[1] = func;
    out[2] = (reg >> 8) & 0xFF;
    out[3] = reg & 0xFF;
    out[4] = (x >> 8) & 0xFF;
    out[5] = x & 0xFF;
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

/* Parst eine Registerangabe dezimal ("2000") oder hexadezimal ("0x7D0"). */
static bool parse_register(const char *s, long *out) {
    char *end;
    int base = (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 16 : 10;
    const char *num = (base == 16) ? s + 2 : s;
    long val = strtol(num, &end, base);
    if (end == num || *end != '\0') return false;
    *out = val;
    return true;
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

/* Liest bis zu max_len Bytes; kehrt zurueck, sobald max_len erreicht ist
 * oder timeout_ms lang keine weiteren Daten mehr eintreffen. Dadurch werden
 * auch kuerzere Antworten (z.B. 5-Byte Modbus-Exception) sauber erkannt. */
static ssize_t serial_read_timeout(int fd, uint8_t *buf, size_t max_len,
                                    int timeout_ms) {
    size_t total = 0;
    while (total < max_len) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        int rv = select(fd + 1, &rfds, NULL, NULL, &tv);
        if (rv <= 0) break; /* Timeout oder Fehler */

        ssize_t n = read(fd, buf + total, max_len - total);
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
    if (gpiod_line_request_output(line, "modbus-tool", 0) < 0) {
        fprintf(stderr, "GPIO-Line konnte nicht als Ausgang angefordert werden: %s\n",
                strerror(errno));
        gpiod_chip_close(chip);
        return NULL;
    }
    *chip_out = chip;
    return line;
}

/* Sendet eine Anfrage und liefert die Laenge einer CRC-gueltigen Antwort
 * zurueck (0 = keine/ungueltige Antwort). Erkennt dabei auch verkuerzte
 * Modbus-Exception-Antworten (5 Byte). */
static size_t query_once(int fd, struct gpiod_line *rts, const uint8_t *req,
                          size_t req_len, uint8_t *resp, int timeout_ms,
                          int retries) {
    for (int attempt = 0; attempt < retries; attempt++) {
        tcflush(fd, TCIFLUSH);

        gpiod_line_set_value(rts, 1);   /* Senden */
        ssize_t written = write(fd, req, req_len);
        (void)written;
        tcdrain(fd);                     /* wartet bis Hardware fertig gesendet hat */
        gpiod_line_set_value(rts, 0);   /* Empfangen */

        ssize_t n = serial_read_timeout(fd, resp, MAX_RESP_LEN, timeout_ms);
        if (n >= 5 && check_crc(resp, (size_t)n)) {
            return (size_t)n;
        }
        if (attempt + 1 < retries) usleep(50000); /* 50ms vor erneutem Versuch */
    }
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Verwendung:\n"
        "  %s --scan [-r REGISTER] [-t TIMEOUT_MS] [-R VERSUCHE] [-j]\n"
        "  %s --set-address -n NEUE_ADRESSE -r REGISTER [-c AKTUELLE_ADRESSE]\n"
        "                    [-t TIMEOUT_MS] [-R VERSUCHE] [-j]\n"
        "\n"
        "  -s, --scan                 Alle Adressen 1-247 nach Sensoren absuchen\n"
        "  -S, --set-address          Slave-Adresse eines Sensors neu setzen\n"
        "  -c, --current ADRESSE      Aktuelle Adresse (Default: %d)\n"
        "  -n, --new-address ADRESSE  Neue Adresse (Pflicht bei --set-address)\n"
        "  -r, --register REG         Holding-Register, dezimal oder 0x-hex\n"
        "                             (Scan-Default: %d; bei --set-address\n"
        "                             Pflichtangabe, siehe Datenblatt des Sensors)\n"
        "  -t, --timeout MS           Timeout pro Anfrage in ms (Default: %d)\n"
        "  -R, --retries N            Versuche pro Adresse/Anfrage\n"
        "                             (Default: %d beim Scan, %d bei --set-address)\n"
        "  -j, --json                 Ausgabe als JSON statt Klartext\n"
        "  -h, --help                 Diese Hilfe anzeigen\n",
        prog, prog, DEFAULT_ADDR, DEFAULT_REGISTER, DEFAULT_TIMEOUT_MS,
        DEFAULT_RETRIES_SCAN, DEFAULT_RETRIES_SET);
}

static int do_scan(int fd, struct gpiod_line *rts, int reg, int timeout_ms,
                    int retries, bool json_output) {
    uint8_t resp[MAX_RESP_LEN];
    int found = 0;
    bool first = true;

    if (json_output) {
        printf("{\"register\":%d,\"results\":[", reg);
    } else {
        printf("Scanne Adressen 1-247 auf Register %d ...\n", reg);
    }

    for (int addr = 1; addr <= 247; addr++) {
        uint8_t req[8];
        size_t req_len = build_frame(req, (uint8_t)addr, 0x03, (uint16_t)reg, 1);
        size_t n = query_once(fd, rts, req, req_len, resp, timeout_ms, retries);
        if (n == 0) continue;

        found++;
        if (resp[1] == 0x83) {
            int exc = resp[2];
            if (json_output) {
                printf("%s{\"address\":%d,\"exception\":%d}", first ? "" : ",", addr, exc);
            } else {
                printf("Adresse %d: Modbus-Exception %d\n", addr, exc);
            }
        } else if (resp[1] == 0x03 && n >= 5) {
            int16_t value = (int16_t)((resp[3] << 8) | resp[4]);
            if (json_output) {
                printf("%s{\"address\":%d,\"value\":%d}", first ? "" : ",", addr, value);
            } else {
                printf("Adresse %d: Register %d = %d\n", addr, reg, value);
            }
        } else {
            continue; /* unerwartete Antwort ignorieren */
        }
        first = false;
    }

    if (json_output) {
        printf("]}\n");
    } else {
        printf("Scan abgeschlossen: %d Sensor(en) gefunden.\n", found);
    }
    return found > 0 ? 0 : 2;
}

static int do_set_address(int fd, struct gpiod_line *rts, int current, int new_addr,
                           int reg, int timeout_ms, int retries, bool json_output) {
    uint8_t resp[MAX_RESP_LEN];
    uint8_t req[8];
    size_t req_len = build_frame(req, (uint8_t)current, 0x06, (uint16_t)reg,
                                  (uint16_t)new_addr);
    size_t n = query_once(fd, rts, req, req_len, resp, timeout_ms, retries);

    if (n == 0) {
        if (json_output) {
            printf("{\"current_address\":%d,\"new_address\":%d,\"register\":%d,"
                   "\"success\":false,\"error\":\"no valid response\"}\n",
                   current, new_addr, reg);
        } else {
            fprintf(stderr, "Keine gueltige Antwort von Adresse %d erhalten.\n", current);
        }
        return 2;
    }

    if (resp[1] == 0x86) {
        int exc = resp[2];
        if (json_output) {
            printf("{\"current_address\":%d,\"new_address\":%d,\"register\":%d,"
                   "\"success\":false,\"exception\":%d}\n",
                   current, new_addr, reg, exc);
        } else {
            fprintf(stderr, "Modbus-Exception %d beim Schreiben von Register %d.\n",
                    exc, reg);
        }
        return 2;
    }

    bool echoed = (n == req_len) && (memcmp(resp, req, req_len) == 0);
    if (json_output) {
        printf("{\"current_address\":%d,\"new_address\":%d,\"register\":%d,"
               "\"success\":%s}\n",
               current, new_addr, reg, echoed ? "true" : "false");
    } else {
        if (echoed) {
            printf("Adresse geaendert: %d -> %d (Register %d)\n",
                   current, new_addr, reg);
        } else {
            fprintf(stderr, "Unerwartete Antwort beim Schreiben von Register %d.\n", reg);
        }
    }
    return echoed ? 0 : 2;
}

int main(int argc, char *argv[]) {
    bool scan_mode = false;
    bool set_mode = false;
    bool json_output = false;
    int current = DEFAULT_ADDR;
    int new_addr = -1;
    int reg = -1;
    int timeout_ms = DEFAULT_TIMEOUT_MS;
    int retries = -1;

    static struct option long_opts[] = {
        {"scan",        no_argument,       0, 's'},
        {"set-address", no_argument,       0, 'S'},
        {"current",     required_argument, 0, 'c'},
        {"new-address", required_argument, 0, 'n'},
        {"register",    required_argument, 0, 'r'},
        {"timeout",     required_argument, 0, 't'},
        {"retries",     required_argument, 0, 'R'},
        {"json",        no_argument,       0, 'j'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "sSc:n:r:t:R:jh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 's': scan_mode = true; break;
            case 'S': set_mode = true; break;
            case 'c':
                current = atoi(optarg);
                if (current < 1 || current > 247) {
                    fprintf(stderr, "Ungueltige Adresse: %s (1-247 erlaubt)\n", optarg);
                    return 1;
                }
                break;
            case 'n':
                new_addr = atoi(optarg);
                if (new_addr < 1 || new_addr > 247) {
                    fprintf(stderr, "Ungueltige Adresse: %s (1-247 erlaubt)\n", optarg);
                    return 1;
                }
                break;
            case 'r': {
                long val;
                if (!parse_register(optarg, &val) || val < 0 || val > 0xFFFF) {
                    fprintf(stderr,
                        "Ungueltiges Register: %s (dezimal oder 0x-hex, 0-0xFFFF)\n",
                        optarg);
                    return 1;
                }
                reg = (int)val;
                break;
            }
            case 't':
                timeout_ms = atoi(optarg);
                if (timeout_ms < 1) {
                    fprintf(stderr, "Ungueltiger Timeout: %s\n", optarg);
                    return 1;
                }
                break;
            case 'R':
                retries = atoi(optarg);
                if (retries < 1) {
                    fprintf(stderr, "Ungueltige Anzahl Versuche: %s\n", optarg);
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

    if (scan_mode == set_mode) { /* keiner oder beide gesetzt */
        fprintf(stderr, "Bitte genau einen Modus angeben: --scan oder --set-address\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (set_mode) {
        if (new_addr < 0) {
            fprintf(stderr, "--set-address erfordert -n/--new-address\n");
            return 1;
        }
        if (reg < 0) {
            fprintf(stderr,
                "--set-address erfordert -r/--register (Konfigurationsregister\n"
                "laut Datenblatt des Sensors -- nicht raten!)\n");
            return 1;
        }
    }
    if (scan_mode && reg < 0) reg = DEFAULT_REGISTER;
    if (retries < 0) retries = set_mode ? DEFAULT_RETRIES_SET : DEFAULT_RETRIES_SCAN;

    int fd = serial_open(SERIAL_DEV);
    if (fd < 0) return 1;

    struct gpiod_chip *chip = NULL;
    struct gpiod_line *rts = gpio_setup(&chip);
    if (!rts) {
        close(fd);
        return 1;
    }

    int ret;
    if (scan_mode) {
        ret = do_scan(fd, rts, reg, timeout_ms, retries, json_output);
    } else {
        ret = do_set_address(fd, rts, current, new_addr, reg, timeout_ms, retries,
                              json_output);
    }

    gpiod_line_release(rts);
    gpiod_chip_close(chip);
    close(fd);
    return ret;
}

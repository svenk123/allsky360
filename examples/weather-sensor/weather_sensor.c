/* weather_sensor.c
 * Liest den RS485 Ultraschall-Wettersensor Halisense HLS-UWS-SDTHA ueber
 * Modbus-RTU. RTS/DE-Umschaltung erfolgt ueber GPIO (libgpiod), Bus liegt an
 * /dev/ttyTHS1.
 *
 * Registerbelegung (Holding-Register, Funktion 0x03). Entspricht der
 * DFRobot-SEN0658-Referenz; die Adressen 0x01F8-0x01FD wurden am echten Sensor
 * gegengeprueft (Feuchte, Temperatur, Noise, Druck lieferten plausible Werte):
 *   0x01F4  Windgeschwindigkeit    Wert / 10 = m/s
 *   0x01F6  Windrichtung (Index)   0-7 (0 = Nord, im Uhrzeigersinn)
 *   0x01F7  Windrichtung           0-360 Grad
 *   0x01F8  Luftfeuchtigkeit       Wert / 10 = %RH
 *   0x01F9  Temperatur             signed, Wert / 10 = Grad C
 *   0x01FA  Noise                  Wert / 10 = dB
 *   0x01FB  PM2.5                  ug/m3
 *   0x01FC  PM10                   ug/m3
 *   0x01FD  Luftdruck              Wert = kPa*10 = hPa
 *   0x01FE  Beleuchtung High Word  } 32 Bit, Lux (Wortreihenfolge laut
 *   0x01FF  Beleuchtung Low Word   } Referenz nicht spezifiziert: High zuerst)
 *   0x0201  Regen                  Wert / 10 (mm)
 *   0x0202  Solarstrahlung         UNVERIFIZIERT (Adresse aus Reihenfolge
 *                                  geraten), wird nur roh ausgegeben
 *
 * Nicht am Sensor bestaetigt: Windgeschwindigkeit/-richtung, PM, Beleuchtung,
 * Regen und Solarstrahlung (zum Testzeitpunkt alle 0 bzw. kein Wind).
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
#define DEFAULT_FUNC    0x03  /* 0x03 Holding Register (4xxx), 0x04 Input Register (3xxx) */

/* Registeradressen (Basisadresse des Messwertbereichs: 0x01F4) */
#define REG_BASE            0x01F4
#define REG_WIND_SPEED      0x01F4
#define REG_WIND_DIR_IDX    0x01F6
#define REG_WIND_DIR_DEG    0x01F7
#define REG_HUMIDITY        0x01F8
#define REG_TEMPERATURE     0x01F9
#define REG_NOISE           0x01FA
#define REG_PM25            0x01FB
#define REG_PM10            0x01FC
#define REG_PRESSURE        0x01FD
#define REG_ILLUMINANCE     0x01FE   /* + 0x01FF, 32-Bit High/Low */
#define REG_RAINFALL        0x0201
#define REG_SOLAR           0x0202   /* unverifiziert */
#define REG_LAST            0x0202
#define BLOCK_COUNT         (REG_LAST - REG_BASE + 1)   /* 15 */

#define TIMEOUT_MS        1200
#define RETRIES           3

#define MAX_REG_COUNT   BLOCK_COUNT
#define MAX_RESP_LEN    (3 + MAX_REG_COUNT * 2 + 2)

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
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;

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
        if (rv <= 0) break;

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
    if (gpiod_line_request_output(line, "weather-sensor", 0) < 0) {
        fprintf(stderr, "GPIO-Line konnte nicht als Ausgang angefordert werden: %s\n",
                strerror(errno));
        gpiod_chip_close(chip);
        return NULL;
    }
    *chip_out = chip;
    return line;
}

/* Fragt genau ein Register-Paket (reg..reg+count-1) ab, mit auf den Kanal
 * abgestimmtem Timeout/Retries. Liefert die Rohwerte in values[0..count-1]. */
static bool query_channel(int fd, struct gpiod_line *rts, uint8_t addr,
                           uint8_t func, uint16_t reg, uint16_t count,
                           int timeout_ms, int retries, uint16_t *values) {
    uint8_t req[8];
    size_t req_len = build_read_frame(req, addr, func, reg, count);
    size_t expected_len = 3 + count * 2 + 2;
    uint8_t resp[MAX_RESP_LEN];

    for (int attempt = 0; attempt < retries; attempt++) {
        tcflush(fd, TCIFLUSH);

        gpiod_line_set_value(rts, 1);
        ssize_t written = write(fd, req, req_len);
        (void)written;
        tcdrain(fd);
        gpiod_line_set_value(rts, 0);

        ssize_t n = serial_read_timeout(fd, resp, expected_len, timeout_ms);
        if (n == (ssize_t)expected_len && check_crc(resp, expected_len) &&
            resp[0] == addr && resp[1] == func) {
            for (uint16_t i = 0; i < count; i++) {
                values[i] = (resp[3 + i * 2] << 8) | resp[3 + i * 2 + 1];
            }
            return true;
        }
        if (attempt + 1 < retries) usleep(50000);
    }
    return false;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Verwendung: %s [-a ADRESSE] [-f FUNKTION] [-j] [-h]\n"
        "  -a, --address ADRESSE   Modbus-Slave-Adresse (Default: %d)\n"
        "  -f, --function FUNKTION Read-Function-Code: 0x03 Holding Register\n"
        "                          (4xxx) oder 0x04 Input Register (3xxx)\n"
        "                          (Default: 0x%02X)\n"
        "  -j, --json              Ausgabe als JSON statt Klartext\n"
        "  -h, --help              Diese Hilfe anzeigen\n"
        "\n"
        "Liest alle Messwerte des Halisense HLS-UWS-SDTHA (Register ab 0x%04X).\n"
        "Exit-Code: 0 = alles gelesen, 2 = Sensor antwortet nicht, 1 = Argument-/\n"
        "Hardwarefehler. Die Solarstrahlung ist unverifiziert und nur roh.\n",
        prog, DEFAULT_ADDR, DEFAULT_FUNC, REG_BASE);
}

typedef struct {
    const char *key;    /* JSON-Feldname */
    const char *label;  /* Klartext-Label */
    bool ok;
    long value;         /* ganzzahliger Wert */
    double dvalue;      /* skalierter Wert */
    bool is_double;
    int decimals;       /* Nachkommastellen bei is_double */
} result_t;

int main(int argc, char *argv[]) {
    int address = DEFAULT_ADDR;
    unsigned int func = DEFAULT_FUNC;
    bool json_output = false;

    static struct option long_opts[] = {
        {"address",  required_argument, 0, 'a'},
        {"function", required_argument, 0, 'f'},
        {"json",     no_argument,       0, 'j'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "a:f:jh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'a':
                address = atoi(optarg);
                if (address < 1 || address > 247) {
                    fprintf(stderr, "Ungueltige Adresse: %s (1-247 erlaubt)\n", optarg);
                    return 1;
                }
                break;
            case 'f':
                func = (unsigned int)strtoul(optarg, NULL, 0); /* akzeptiert 0x-Praefix */
                if (func > 0xFF) {
                    fprintf(stderr, "Ungueltige Function: %s\n", optarg);
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

    result_t results[16];
    int n_results = 0;
    uint16_t v[BLOCK_COUNT];
    bool block_ok = query_channel(fd, rts, (uint8_t)address, (uint8_t)func, REG_BASE,
                                  BLOCK_COUNT, TIMEOUT_MS, RETRIES, v);
    bool any_failed = false;

    /* Fallback: Blocklesen schlug fehl -> jedes Register einzeln versuchen */
    #define VAL(reg, idx_out) \
        (block_ok ? (*(idx_out) = v[(reg) - REG_BASE], true) \
                  : query_channel(fd, rts, (uint8_t)address, (uint8_t)func, (reg), 1, \
                                  TIMEOUT_MS, RETRIES, (idx_out)))
    #define ADD_LONG(KEY, LABEL, REG, EXPR) do { \
        uint16_t w; result_t *r = &results[n_results++]; \
        r->key = KEY; r->label = LABEL; r->is_double = false; r->decimals = 0; \
        r->ok = VAL(REG, &w); \
        if (r->ok) r->value = (EXPR); else any_failed = true; } while (0)
    #define ADD_SCALED(KEY, LABEL, REG, EXPR, DEC) do { \
        uint16_t w; result_t *r = &results[n_results++]; \
        r->key = KEY; r->label = LABEL; r->is_double = true; r->decimals = (DEC); \
        r->ok = VAL(REG, &w); \
        if (r->ok) r->dvalue = (EXPR); else any_failed = true; } while (0)

    ADD_SCALED("wind_speed_ms",        "Windgeschwindigkeit (m/s)", REG_WIND_SPEED, w / 10.0, 1);
    ADD_LONG  ("wind_direction_index", "Windrichtung (Index 0-7)",  REG_WIND_DIR_IDX, w);
    ADD_LONG  ("wind_direction_deg",   "Windrichtung (Grad)",       REG_WIND_DIR_DEG, w);
    ADD_SCALED("humidity_percent",     "Luftfeuchtigkeit (%RH)",    REG_HUMIDITY, w / 10.0, 1);
    ADD_SCALED("temperature_c",        "Temperatur (C)",            REG_TEMPERATURE, (int16_t)w / 10.0, 1);
    ADD_SCALED("noise_db",             "Noise (dB)",                REG_NOISE, w / 10.0, 1);
    ADD_LONG  ("pm25_ug_m3",           "PM2.5 (ug/m3)",             REG_PM25, w);
    ADD_LONG  ("pm10_ug_m3",           "PM10 (ug/m3)",              REG_PM10, w);
    ADD_SCALED("pressure_hpa",         "Luftdruck (hPa)",           REG_PRESSURE, w, 1);
    /* Beleuchtung: 32 Bit aus 0x01FE (High) und 0x01FF (Low) */
    {
        uint16_t hi, lo;
        result_t *r = &results[n_results++];
        r->key = "illuminance_lux"; r->label = "Beleuchtung (Lux)";
        r->is_double = false; r->decimals = 0;
        r->ok = VAL(REG_ILLUMINANCE, &hi) && VAL(REG_ILLUMINANCE + 1, &lo);
        if (r->ok) r->value = ((long)hi << 16) | lo; else any_failed = true;
    }
    ADD_SCALED("rainfall_mm",          "Regen (mm)",                REG_RAINFALL, w / 10.0, 1);
    ADD_LONG  ("solar_irradiance_raw", "Solarstrahlung (roh, unverifiziert)", REG_SOLAR, w);

    if (json_output) {
        printf("{\"address\":%d", address);
        for (int i = 0; i < n_results; i++) {
            result_t *r = &results[i];
            if (r->ok) {
                if (r->is_double) {
                    printf(",\"%s\":%.*f", r->key, r->decimals, r->dvalue);
                } else {
                    printf(",\"%s\":%ld", r->key, r->value);
                }
            } else {
                printf(",\"%s\":null,\"%s_error\":\"no valid response\"", r->key, r->key);
            }
        }
        printf("}\n");
    } else {
        printf("Adresse: %d\n", address);
        for (int i = 0; i < n_results; i++) {
            result_t *r = &results[i];
            if (r->ok) {
                if (r->is_double) {
                    printf("%s: %.*f\n", r->label, r->decimals, r->dvalue);
                } else {
                    printf("%s: %ld\n", r->label, r->value);
                }
            } else {
                fprintf(stderr, "%s: keine gueltige Antwort\n", r->label);
            }
        }
    }

    gpiod_line_release(rts);
    gpiod_chip_close(chip);
    close(fd);
    return any_failed ? 2 : 0;
}

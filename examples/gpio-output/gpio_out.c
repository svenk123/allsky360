/* gpio_out.c
 * Schaltet die 4 Ausgaenge OUT1-OUT4 des I/O-Boards (PCA9534-I2C-GPIO-Expander,
 * gpiochip2) manuell dauerhaft ein oder aus und fragt ihren Zustand ab.
 * Kein RS485 -- nur GPIO. Basis: magnetventil_steuern (gleiche Hardware).
 *
 * WICHTIG (empirisch bei magnetventil_steuern verifiziert): Auf diesem
 * PCA9534 faellt eine auf HIGH getriebene Line SOFORT auf LOW zurueck, sobald
 * der anfordernde Prozess sie freigibt oder beendet wird. Ein "dauerhaft an"
 * geht daher nur, wenn ein Prozess die Line weiter belegt. `--action ein`
 * startet deshalb einen detached Halteprozess (setsid), der die Line auf HIGH
 * haelt, bis `--action aus` ihn per SIGTERM beendet. Es gibt KEINE Zeitsteuerung
 * und keinen Watchdog: der Ausgang bleibt an, bis er ausgeschaltet wird.
 * Wird der Halteprozess hart beendet (SIGKILL, Reboot), faellt der Ausgang auf
 * LOW zurueck (Fail-Safe).
 *
 * Der Halteprozess benennt sich per prctl in "gpio_out_N" um; so findet ihn das
 * Tool wieder (Scan von /proc), ohne PID-Dateien in /run zu brauchen.
 *
 * ANNAHME POLARITAET: active-HIGH (HIGH = Ausgang aktiv/unter Strom), wie bei
 * magnetventil_steuern.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <dirent.h>
#include <ctype.h>
#include <time.h>
#include <sys/types.h>
#include <sys/prctl.h>
#include <gpiod.h>
#include "json_output.h"

#define TOOL_NAME     "gpio-out"
#define CONSUMER_NAME "gpio_out"
#define GPIO_CHIP_NAME "gpiochip2"
#define NUM_OUTPUTS   4

static bool g_json = false;

/* OUT1 -> Offset 3, OUT2 -> 2, OUT3 -> 1, OUT4 -> 0 */
static int offset_for_output(int out) { return NUM_OUTPUTS - out; }

static int fail(const char *msg, const char *code, bool retry_safe) {
    if (g_json) JSON_ERROR(TOOL_NAME, msg, code, retry_safe);
    else fprintf(stderr, "Fehler (%s): %s\n", code, msg);
    return 1;
}

static const char *code_for_errno(int e, const char *fallback) {
    return (e == EACCES || e == EPERM) ? "PERMISSION_DENIED" : fallback;
}

/* ---- Halteprozess finden / beenden ---------------------------------- */

static void holder_comm(int out, char *buf, size_t n) {
    snprintf(buf, n, "gpio_out_%d", out);
}

/* PID des Halteprozesses fuer OUT<out> oder 0 */
static pid_t find_holder(int out) {
    char want[32];
    holder_comm(out, want, sizeof(want));
    DIR *d = opendir("/proc");
    if (!d) return 0;
    pid_t found = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        pid_t pid = (pid_t)atol(e->d_name);
        if (pid == getpid()) continue;
        char path[300], comm[32] = {0};
        snprintf(path, sizeof(path), "/proc/%s/comm", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        size_t n = fread(comm, 1, sizeof(comm) - 1, f);
        fclose(f);
        while (n > 0 && comm[n - 1] == '\n') comm[--n] = '\0';
        if (strcmp(comm, want) == 0) { found = pid; break; }
    }
    closedir(d);
    return found;
}

/* SIGTERM, bis zu 1 s warten, sonst SIGKILL. true wenn der Halter weg ist. */
static bool stop_holder(int out) {
    pid_t pid = find_holder(out);
    if (!pid) return true;
    kill(pid, SIGTERM);
    for (int i = 0; i < 100 && find_holder(out); i++) {
        struct timespec ts = { 0, 10L * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    if (!find_holder(out)) return true;
    kill(pid, SIGKILL);
    for (int i = 0; i < 50 && find_holder(out); i++) {
        struct timespec ts = { 0, 10L * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    return find_holder(out) == 0;
}

/* ---- GPIO ------------------------------------------------------------ */

static struct gpiod_line *open_line(struct gpiod_chip **chip_out, int offset, int *err_rc) {
    struct gpiod_chip *chip = gpiod_chip_open_by_name(GPIO_CHIP_NAME);
    char msg[256];
    if (!chip) {
        snprintf(msg, sizeof(msg), "GPIO-Chip %s konnte nicht geoeffnet werden: %s",
                 GPIO_CHIP_NAME, strerror(errno));
        *err_rc = fail(msg, code_for_errno(errno, "HARDWARE_NOT_FOUND"), false);
        return NULL;
    }
    struct gpiod_line *line = gpiod_chip_get_line(chip, offset);
    if (!line) {
        snprintf(msg, sizeof(msg), "GPIO-Line %d konnte nicht geholt werden: %s",
                 offset, strerror(errno));
        *err_rc = fail(msg, "HARDWARE_NOT_FOUND", false);
        gpiod_chip_close(chip);
        return NULL;
    }
    *chip_out = chip;
    return line;
}

/* ---- Kommandos ------------------------------------------------------- */

static volatile sig_atomic_t g_stop = 0;
static void on_term(int sig) { (void)sig; g_stop = 1; }

static void print_state(int out, const char *state, bool already) {
    if (g_json) {
        char result[128];
        snprintf(result, sizeof(result), "{\"output\": %d, \"state\": \"%s\"%s}",
                 out, state, already ? ", \"changed\": false" : "");
        JSON_SUCCESS_RESULT(TOOL_NAME, result);
    } else {
        printf("OUT%d: %s%s\n", out, state, already ? " (war bereits so)" : "");
    }
}

static int cmd_ein(int out) {
    if (find_holder(out)) {
        print_state(out, "ein", true);
        return 0;
    }
    int offset = offset_for_output(out);

    int pipefd[2];
    if (pipe(pipefd) < 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "Halteprozess konnte nicht vorbereitet werden (pipe): %s", strerror(errno));
        return fail(msg, "HOLDER_INIT_FAILED", false);
    }
    pid_t child = fork();
    if (child < 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "Halteprozess konnte nicht gestartet werden (fork): %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        return fail(msg, "HOLDER_INIT_FAILED", false);
    }

    if (child == 0) {
        /* Halteprozess: fordert die Line selbst an und haelt sie auf HIGH */
        close(pipefd[0]);
        setsid();
        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO) close(devnull);
        }
        char comm[32];
        holder_comm(out, comm, sizeof(comm));
        prctl(PR_SET_NAME, comm, 0, 0, 0);

        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = on_term;
        sigaction(SIGTERM, &sa, NULL);
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGHUP, &sa, NULL);

        char buf[320];
        struct gpiod_chip *chip = gpiod_chip_open_by_name(GPIO_CHIP_NAME);
        if (!chip) {
            snprintf(buf, sizeof(buf), "%s\tGPIO-Chip %s konnte nicht geoeffnet werden: %s\n",
                     code_for_errno(errno, "HARDWARE_NOT_FOUND"), GPIO_CHIP_NAME, strerror(errno));
            ssize_t w = write(pipefd[1], buf, strlen(buf)); (void)w;
            _exit(1);
        }
        struct gpiod_line *line = gpiod_chip_get_line(chip, offset);
        if (!line) {
            snprintf(buf, sizeof(buf), "HARDWARE_NOT_FOUND\tGPIO-Line %d konnte nicht geholt werden: %s\n",
                     offset, strerror(errno));
            ssize_t w = write(pipefd[1], buf, strlen(buf)); (void)w;
            _exit(1);
        }
        if (gpiod_line_request_output(line, CONSUMER_NAME, 1) < 0) {
            snprintf(buf, sizeof(buf), "%s\tGPIO-Line konnte nicht auf 1 gesetzt werden: %s\n",
                     code_for_errno(errno, "GPIO_WRITE_FAILED"), strerror(errno));
            ssize_t w = write(pipefd[1], buf, strlen(buf)); (void)w;
            _exit(1);
        }
        ssize_t w = write(pipefd[1], "OK\n", 3); (void)w;
        close(pipefd[1]);

        while (!g_stop) pause();

        gpiod_line_set_value(line, 0);
        gpiod_line_release(line);
        gpiod_chip_close(chip);
        _exit(0);
    }

    /* Elternprozess: wartet nur auf die Rueckmeldung des Halteprozesses */
    close(pipefd[1]);
    char buf[512];
    ssize_t n;
    do { n = read(pipefd[0], buf, sizeof(buf) - 1); } while (n < 0 && errno == EINTR);
    close(pipefd[0]);

    if (n <= 0) {
        kill(child, SIGTERM);
        return fail("Halteprozess hat keine Rueckmeldung gesendet", "GPIO_WRITE_FAILED", true);
    }
    buf[n] = '\0';
    char *nl = strchr(buf, '\n');
    if (nl) *nl = '\0';
    if (strcmp(buf, "OK") == 0) {
        print_state(out, "ein", false);
        return 0;
    }

    static const char *valid[] = { "HARDWARE_NOT_FOUND", "PERMISSION_DENIED", "GPIO_WRITE_FAILED" };
    const char *code = "GPIO_WRITE_FAILED", *msg = "Halteprozess meldete einen unbekannten Fehler";
    char *tab = strchr(buf, '\t');
    if (tab) {
        *tab = '\0';
        for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); i++)
            if (strcmp(buf, valid[i]) == 0) { code = valid[i]; msg = tab + 1; break; }
    }
    return fail(msg, code, strcmp(code, "GPIO_WRITE_FAILED") == 0);
}

static int cmd_aus(int out) {
    bool was_on = find_holder(out) != 0;
    if (!stop_holder(out))
        return fail("Halteprozess laesst sich nicht beenden, Line vermutlich noch belegt",
                    "GPIO_WRITE_FAILED", true);

    /* Explizit LOW setzen und freigeben (auch wenn kein Halter lief) */
    int rc = 1;
    struct gpiod_chip *chip = NULL;
    struct gpiod_line *line = open_line(&chip, offset_for_output(out), &rc);
    if (!line) return rc;
    if (gpiod_line_request_output(line, CONSUMER_NAME, 0) < 0) {
        char msg[256];
        int e = errno;
        snprintf(msg, sizeof(msg), "GPIO-Line konnte nicht auf 0 gesetzt werden: %s", strerror(e));
        gpiod_chip_close(chip);
        return fail(msg, code_for_errno(e, "GPIO_WRITE_FAILED"), e == EBUSY);
    }
    gpiod_line_release(line);
    gpiod_chip_close(chip);
    print_state(out, "aus", !was_on);
    return 0;
}

/* Liefert 1 = ein, 0 = aus, -1 = unbekannt (Line von fremdem Prozess belegt);
 * bei hartem Fehler -2 (Meldung bereits ausgegeben). */
static int read_state(int out, char *consumer, size_t consumer_len, int *err_rc) {
    consumer[0] = '\0';
    if (find_holder(out)) return 1;

    struct gpiod_chip *chip = NULL;
    struct gpiod_line *line = open_line(&chip, offset_for_output(out), err_rc);
    if (!line) return -2;

    struct gpiod_line_request_config cfg = {
        .consumer = CONSUMER_NAME,
        .request_type = GPIOD_LINE_REQUEST_DIRECTION_AS_IS,
    };
    if (gpiod_line_request(line, &cfg, 0) < 0) {
        int e = errno;
        if (e == EBUSY) {
            const char *c = gpiod_line_consumer(line);
            snprintf(consumer, consumer_len, "%s", c ? c : "unbekannt");
            gpiod_chip_close(chip);
            return -1;
        }
        char msg[256];
        snprintf(msg, sizeof(msg), "GPIO-Line konnte nicht fuer Status-Lesen angefordert werden: %s", strerror(e));
        *err_rc = fail(msg, code_for_errno(e, "GPIO_WRITE_FAILED"), false);
        gpiod_chip_close(chip);
        return -2;
    }
    int val = gpiod_line_get_value(line);
    gpiod_line_release(line);
    gpiod_chip_close(chip);
    if (val < 0) {
        *err_rc = fail("GPIO-Wert konnte nicht gelesen werden", "GPIO_WRITE_FAILED", true);
        return -2;
    }
    return val;
}

/* out == 0: alle vier Ausgaenge */
static int cmd_status(int out) {
    int first = out ? out : 1, last = out ? out : NUM_OUTPUTS;
    int states[NUM_OUTPUTS + 1];
    char consumers[NUM_OUTPUTS + 1][32];
    for (int o = first; o <= last; o++) {
        int rc = 1;
        states[o] = read_state(o, consumers[o], sizeof(consumers[o]), &rc);
        if (states[o] == -2) return rc;
    }

    if (g_json) {
        static char result[1024];
        size_t pos = 0;
        pos += (size_t)snprintf(result + pos, sizeof(result) - pos, "{\"outputs\": [");
        for (int o = first; o <= last; o++) {
            const char *s = states[o] == 1 ? "ein" : states[o] == 0 ? "aus" : "unbekannt";
            pos += (size_t)snprintf(result + pos, sizeof(result) - pos,
                                    "%s{\"output\": %d, \"state\": \"%s\"", o > first ? ", " : "", o, s);
            if (states[o] == -1) {
                char esc[64];
                json_escape(consumers[o], esc, sizeof(esc));
                pos += (size_t)snprintf(result + pos, sizeof(result) - pos, ", \"busy_by\": \"%s\"", esc);
            }
            pos += (size_t)snprintf(result + pos, sizeof(result) - pos, "}");
        }
        snprintf(result + pos, sizeof(result) - pos, "]}");
        JSON_SUCCESS_RESULT(TOOL_NAME, result);
    } else {
        for (int o = first; o <= last; o++) {
            if (states[o] == -1) printf("OUT%d: unbekannt (Line belegt von '%s')\n", o, consumers[o]);
            else printf("OUT%d: %s\n", o, states[o] ? "ein" : "aus");
        }
    }
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Verwendung:\n"
        "  %s -o <1-4> -a <ein|aus> [-j]\n"
        "  %s -s [-o <1-4>] [-j]\n"
        "\n"
        "  -o, --output N      Ausgang 1-4 (OUT1-OUT4). Bei --status optional (Default: alle)\n"
        "  -a, --action A      ein oder aus (auch: an, on, off, 1, 0). Der Ausgang bleibt\n"
        "                      dauerhaft in diesem Zustand, bis er wieder geschaltet wird\n"
        "  -s, --status        Zustand abfragen, keine Aenderung\n"
        "  -j, --json          Ausgabe als JSON statt Klartext\n"
        "  -h, --help          Diese Hilfe anzeigen\n"
        "\n"
        "Exit-Code 0 bei Erfolg, 1 bei Fehler.\n",
        prog, prog);
}

int main(int argc, char *argv[]) {
    long out = 0;
    const char *action = NULL;
    bool status = false;

    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-j") || !strcmp(argv[i], "--json")) g_json = true;

    static struct option long_opts[] = {
        {"output", required_argument, 0, 'o'},
        {"action", required_argument, 0, 'a'},
        {"status", no_argument,       0, 's'},
        {"json",   no_argument,       0, 'j'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    opterr = 0;
    int opt;
    char msg[160];
    while ((opt = getopt_long(argc, argv, "o:a:sjh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'o': {
                char *end;
                errno = 0;
                out = strtol(optarg, &end, 10);
                if (errno || end == optarg || *end != '\0' || out < 1 || out > NUM_OUTPUTS) {
                    snprintf(msg, sizeof(msg), "Ungueltiger Ausgang: %s (erlaubt: 1-%d)", optarg, NUM_OUTPUTS);
                    return fail(msg, "INVALID_ARGS", false);
                }
                break;
            }
            case 'a': action = optarg; break;
            case 's': status = true; break;
            case 'j': break;
            case 'h': print_usage(argv[0]); return 0;
            default:
                return fail("Unbekannte oder fehlerhafte Kommandozeilenoption", "INVALID_ARGS", false);
        }
    }

    if (status) {
        if (action) return fail("--status kann nicht mit --action kombiniert werden", "INVALID_ARGS", false);
        return cmd_status((int)out);
    }
    if (!out) return fail("--output ist Pflicht (1-4)", "INVALID_ARGS", false);
    if (!action) return fail("--action ist Pflicht (ein|aus), ausser bei --status", "INVALID_ARGS", false);

    if (!strcmp(action, "ein") || !strcmp(action, "an") || !strcmp(action, "on") || !strcmp(action, "1"))
        return cmd_ein((int)out);
    if (!strcmp(action, "aus") || !strcmp(action, "off") || !strcmp(action, "0"))
        return cmd_aus((int)out);

    snprintf(msg, sizeof(msg), "Unbekannte --action: %s (erlaubt: ein, aus)", action);
    return fail(msg, "INVALID_ARGS", false);
}

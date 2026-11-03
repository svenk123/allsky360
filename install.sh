#!/bin/bash
#
# install.sh - one-step installation of allsky360
#
#   sudo ./install.sh              detect platform, install everything,
#                                  run the configuration assistant
#   ./install.sh --configure       only the configuration assistant
#                                  (location, INDIGO, camera, mount, token)
#   ./install.sh --help            all options
#
# The script only orchestrates the Makefile targets; everything it does can
# also be done by hand, see INSTALL.md.
#
# Copyright (c) 2025 Sven Kreiensen
#
# You can use this software under the terms of the MIT software license
# (see LICENSE.md)
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="allskyuser"
CONFIG_FILE="/home/${SERVICE_USER}/config.ini"
INDIGO_ROOT="${INDIGO_ROOT:-$(cd "$REPO_DIR/.." && pwd)/indigo}"
START_UNITS="allsky360-camera.service allsky360-aurora.timer allsky360-meteo.timer allsky360-mount-position.timer allsky360-sensors.timer"
MIN_FREE_GB=5

# Options
USE_CUDA=auto
WITH_AI=ask
ASSUME_YES=0
SKIP_DEPS=0
MODE=install

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    C_B=$'\e[1m'; C_G=$'\e[32m'; C_Y=$'\e[33m'; C_R=$'\e[31m'; C_0=$'\e[0m'
else
    C_B=""; C_G=""; C_Y=""; C_R=""; C_0=""
fi
step()  { echo; echo "${C_B}==> $*${C_0}"; }
ok()    { echo "  ${C_G}OK${C_0}   $*"; }
warn()  { echo "  ${C_Y}WARN${C_0} $*"; WARNINGS=$((WARNINGS + 1)); }
fail()  { echo "  ${C_R}FAIL${C_0} $*"; FAILURES=$((FAILURES + 1)); }
die()   { echo "${C_R}ERROR:${C_0} $*" >&2; exit 1; }
WARNINGS=0
FAILURES=0

usage() {
    cat <<EOF
Usage: sudo ./install.sh [options]
       ./install.sh --configure [--config FILE]

Installation (run with sudo from the repository directory):
  --cuda            build with NVIDIA CUDA (default: auto-detect)
  --cpu             CPU-only build
  --with-ai         also install the Ultralytics YOLO venv (several GB)
  --no-ai           do not ask about the YOLO venv
  --skip-deps       do not install apt/Node.js packages
  --indigo-root DIR INDIGO source tree with build/ (default: $INDIGO_ROOT)
  --yes             no questions: defaults everywhere, no configuration
                    assistant, services are not started

Configuration only:
  --configure       run only the configuration assistant
  --config FILE     config.ini to edit (default: $CONFIG_FILE)

The camera service and the data timers are never started without asking:
config.ini has to contain the location and the INDIGO devices first.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cuda)        USE_CUDA=1 ;;
        --cpu)         USE_CUDA=0 ;;
        --with-ai)     WITH_AI=1 ;;
        --no-ai)       WITH_AI=0 ;;
        --skip-deps)   SKIP_DEPS=1 ;;
        --indigo-root) INDIGO_ROOT="$2"; shift ;;
        --yes|-y)      ASSUME_YES=1 ;;
        --configure)   MODE=configure ;;
        --config)      CONFIG_FILE="$2"; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             usage; die "unknown option: $1" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Prompts (all answers have a default, --yes takes it)
# ---------------------------------------------------------------------------
ask() {   # ask VAR "question" "default"
    local __var="$1" __q="$2" __def="$3" __ans
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then
        printf -v "$__var" '%s' "$__def"
        return
    fi
    read -r -p "  $__q [$__def]: " __ans || true
    printf -v "$__var" '%s' "${__ans:-$__def}"
}

confirm() {   # confirm "question" y|n  -> exit status
    local __ans __def="$2"
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then
        [ "$__def" = y ]
        return
    fi
    read -r -p "  $1 [$([ "$__def" = y ] && echo Y/n || echo y/N)]: " __ans || true
    __ans="${__ans:-$__def}"
    [[ "$__ans" =~ ^[YyJj] ]]
}

# ---------------------------------------------------------------------------
# config.ini helpers
# ---------------------------------------------------------------------------
cfg_get() {   # cfg_get KEY -> value without quotes ("" if missing)
    sed -n -E "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$CONFIG_FILE" \
        | tail -n 1 | sed -E 's/[[:space:]]+$//; s/^"(.*)"$/\1/'
}

cfg_set() {   # cfg_set KEY VALUE  (quotes values with blanks, keeps the line position)
    local key="$1" val="$2" line tmp
    if [[ "$val" =~ [[:space:]] ]]; then line="$key=\"$val\""; else line="$key=$val"; fi
    tmp="$(mktemp "$(dirname "$CONFIG_FILE")/.config.ini.XXXXXX")"
    awk -v key="$key" -v line="$line" '
        BEGIN { done = 0 }
        $0 ~ "^[[:space:]]*" key "[[:space:]]*=" { if (!done) { print line; done = 1 }; next }
        { print }
        END { if (!done) print line }
    ' "$CONFIG_FILE" > "$tmp"
    # keep owner and mode of the original (config.ini holds the api-token)
    chmod --reference="$CONFIG_FILE" "$tmp"
    chown --reference="$CONFIG_FILE" "$tmp" 2>/dev/null || true
    mv "$tmp" "$CONFIG_FILE"
}

# Same rules as parse_lat_lon() in src/coordinates.h
valid_coordinate() {   # valid_coordinate VALUE lat|lon
    local v="$1" axis="$2" pos neg max num
    if [ "$axis" = lat ]; then pos=N; neg=S; max=90; else pos=E; neg=W; max=180; fi
    if [[ "$v" =~ ^[+-]?[0-9]+(\.[0-9]+)?$ ]]; then
        num="$v"
    elif [[ "$v" =~ ^([0-9]+(\.[0-9]+)?)[[:space:]]*([$pos${pos,,}$neg${neg,,}])$ ]]; then
        num="${BASH_REMATCH[1]}"
    else
        return 1
    fi
    awk -v n="$num" -v m="$max" 'BEGIN { if (n < 0) n = -n; exit !(n <= m) }'
}

# ---------------------------------------------------------------------------
# Configuration assistant
# ---------------------------------------------------------------------------
configure() {
    [ -f "$CONFIG_FILE" ] || die "$CONFIG_FILE not found (run the installation first)"
    [ -w "$CONFIG_FILE" ] || die "$CONFIG_FILE is not writable - run with sudo"

    step "Configuration assistant ($CONFIG_FILE)"
    echo "  Press Enter to keep the value shown in brackets."

    echo; echo "  ${C_B}Location${C_0} (decimal degrees: 51.2349N / 7.4479E, or signed: -33.9249)"
    local v
    while :; do
        ask v "Latitude " "$(cfg_get location_latitude)"
        valid_coordinate "$v" lat && break
        echo "  ${C_R}invalid latitude${C_0} - e.g. 51.2349N, 33.9249S or -33.9249"
    done
    cfg_set location_latitude "$v"
    while :; do
        ask v "Longitude" "$(cfg_get location_longitude)"
        valid_coordinate "$v" lon && break
        echo "  ${C_R}invalid longitude${C_0} - e.g. 7.4479E, 74.0060W or -74.0060"
    done
    cfg_set location_longitude "$v"
    while :; do
        ask v "Altitude above sea level (m)" "$(cfg_get location_altitude)"
        [[ "$v" =~ ^-?[0-9]+$ ]] && break
        echo "  ${C_R}whole metres please${C_0}"
    done
    cfg_set location_altitude "$v"
    ask v "City (shown in the overlay)" "$(cfg_get location_city)"; cfg_set location_city "$v"
    ask v "Owner (shown in the overlay)" "$(cfg_get location_owner)"; cfg_set location_owner "$v"
    # Display time zone; storage is always UTC. Default: configured value,
    # else the system zone (if it is not plain UTC), else Europe/Berlin.
    local tz_def tz_sys
    tz_def="$(cfg_get location_timezone)"
    tz_sys="$(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null)"
    if [ -z "$tz_def" ]; then
        case "$tz_sys" in ""|UTC|Etc/UTC|Etc/Universal|Universal) tz_def="Europe/Berlin" ;; *) tz_def="$tz_sys" ;; esac
    fi
    while :; do
        ask v "Time zone for local time display (IANA, e.g. Europe/Berlin)" "$tz_def"
        [ -f "/usr/share/zoneinfo/$v" ] && [[ "$v" != /* && "$v" != *..* ]] && break
        echo "  ${C_R}unknown time zone${C_0} - see: timedatectl list-timezones"
    done
    cfg_set location_timezone "$v"

    echo; echo "  ${C_B}INDIGO server${C_0}"
    local host port name
    ask host "Server address" "$(cfg_get indigo_server)"; cfg_set indigo_server "$host"
    ask port "Server port" "$(cfg_get indigo_port)"; cfg_set indigo_port "$port"
    name="$(cfg_get indigo_servername)"; name="${name:-indigo_server}"
    cfg_set indigo_servername "$name"

    # Devices are reported as "<device> @ <server name>"; the listing tools
    # connect as "indigo_server", so swap the suffix if another name is used.
    local lister="$REPO_DIR/tools/indigo-cameras/bin/indigo_cameras"
    local mountlister="$REPO_DIR/tools/indigo-mount-position/bin/indigo_mount_position"
    local -a cams=() mounts=()
    if [ -x "$lister" ]; then
        echo "  Asking $host:$port for cameras..."
        mapfile -t cams < <(timeout 30 "$lister" -h "$host" -p "$port" -t 10 -l -j 2>/dev/null \
            | python3 -c 'import json,sys; [print(c if isinstance(c, str) else c["name"]) for c in json.load(sys.stdin).get("cameras", [])]' 2>/dev/null \
            | sed "s/ @ indigo_server\$/ @ $name/" || true)
    fi
    choose_device v "Camera" "$(cfg_get indigo_camera)" "${cams[@]+"${cams[@]}"}"
    cfg_set indigo_camera "$v"

    if [ -x "$mountlister" ]; then
        local tmpcfg; tmpcfg="$(mktemp)"
        printf 'indigo_server=%s\nindigo_port=%s\nindigo_servername=%s\n' "$host" "$port" "$name" > "$tmpcfg"
        echo "  Asking $host:$port for mounts..."
        mapfile -t mounts < <(timeout 30 "$mountlister" -c "$tmpcfg" -t 5 -l 2>/dev/null | sed -n 's/^  //p' || true)
        rm -f "$tmpcfg"
    fi
    choose_device v "Mount (empty = no mount)" "$(cfg_get indigo_mount)" "${mounts[@]+"${mounts[@]}"}"
    cfg_set indigo_mount "$v"

    echo; echo "  ${C_B}RS485/Modbus sensors${C_0} (slave address, 0 = not connected)"
    local key label
    for key in weather:"Weather sensor" wind_speed:"Anemometer" wind_direction:"Wind vane" cabinet:"Control cabinet temp/humidity"; do
        label="${key#*:}"; key="modbus_${key%%:*}_address"
        while :; do
            ask v "$label" "$(cfg_get "$key")"; v="${v:-0}"
            [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -le 247 ] && break
            echo "  ${C_R}0-247 please${C_0}"
        done
        cfg_set "$key" "$v"
    done

    echo; echo "  ${C_B}Web UI service control${C_0}"
    if [ -z "$(cfg_get api-token)" ]; then
        local token
        token="$(openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')"
        cfg_set api-token "$token"
        ok "api-token generated - you need it for the Start/Stop/Restart buttons:"
        echo "       $token"
        echo "       (it is stored in $CONFIG_FILE)"
    else
        ok "api-token already set (see $CONFIG_FILE)"
    fi
}

# choose_device VAR "label" "current" [found devices...]
choose_device() {
    local __var="$1" label="$2" current="$3"; shift 3
    local -a found=("$@")
    if [ ${#found[@]} -gt 0 ] && [ "$ASSUME_YES" != 1 ] && [ -t 0 ]; then
        echo "  Found:"
        local i
        for i in "${!found[@]}"; do echo "    $((i + 1))) ${found[$i]}"; done
        local ans
        read -r -p "  $label: number, name, or Enter for [$current]: " ans || true
        if [[ "$ans" =~ ^[0-9]+$ ]] && [ "$ans" -ge 1 ] && [ "$ans" -le ${#found[@]} ]; then
            printf -v "$__var" '%s' "${found[$((ans - 1))]}"
        else
            printf -v "$__var" '%s' "${ans:-$current}"
        fi
    else
        [ ${#found[@]} -eq 0 ] && echo "  (no device found on the server - enter the name by hand)"
        ask "$__var" "$label" "$current"
    fi
}

# ---------------------------------------------------------------------------
# Installation steps
# ---------------------------------------------------------------------------
preflight() {
    step "Checking the system"
    [ "$(id -u)" -eq 0 ] || die "run with sudo: sudo ./install.sh"
    BUILD_USER="${SUDO_USER:-}"
    [ -n "$BUILD_USER" ] && [ "$BUILD_USER" != root ] \
        || die "start it with sudo from your normal user account, the build must not run as root"
    ok "build user: $BUILD_USER"

    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        ok "system: ${PRETTY_NAME:-unknown} ($(uname -m))"
    fi

    if [ -r /etc/nv_tegra_release ]; then
        ok "NVIDIA Jetson: $(head -1 /etc/nv_tegra_release | sed 's/^# //; s/, GCID.*//')"
        if dpkg-query -W -f='${Version}' nvidia-jetpack >/dev/null 2>&1; then
            ok "JetPack $(dpkg-query -W -f='${Version}' nvidia-jetpack)"
        fi
    fi

    if [ "$USE_CUDA" = auto ]; then
        if [ -x /usr/local/cuda/bin/nvcc ] || command -v nvcc >/dev/null 2>&1; then USE_CUDA=1; else USE_CUDA=0; fi
    fi
    if [ "$USE_CUDA" = 1 ]; then
        local nvcc=/usr/local/cuda/bin/nvcc
        [ -x "$nvcc" ] || nvcc="$(command -v nvcc || true)"
        [ -n "$nvcc" ] || die "--cuda requested but nvcc was not found (install the CUDA toolkit / JetPack)"
        ok "build variant: CUDA ($("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))"
    else
        ok "build variant: CPU only"
    fi

    local free_gb
    free_gb=$(df -P -BG /opt | awk 'NR==2 { gsub("G", "", $4); print $4 }')
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
        warn "only ${free_gb} GB free on /opt (images and videos need much more)"
    else
        ok "free disk space: ${free_gb} GB"
    fi

    if [ -d "$INDIGO_ROOT/indigo_libs" ] && [ -f "$INDIGO_ROOT/build/lib/libjpeg.a" ]; then
        ok "INDIGO source tree: $INDIGO_ROOT"
    else
        echo
        echo "  The INDIGO source tree with a finished build is needed (headers,"
        echo "  libindigo and the bundled libjpeg.a). Not found in: $INDIGO_ROOT"
        echo "  Build INDIGO first (INSTALL.md, Part 1), then run:"
        echo "    sudo ./install.sh --indigo-root /path/to/indigo"
        die "INDIGO not found"
    fi
}

install_deps() {
    [ "$SKIP_DEPS" = 1 ] && { step "Skipping packages (--skip-deps)"; return; }
    step "Installing packages (apt, CUDA libraries, Node.js)"
    make -C "$REPO_DIR" deps USE_CUDA="$USE_CUDA"
}

build() {
    step "Building as $BUILD_USER"
    sudo -u "$BUILD_USER" make -C "$REPO_DIR" USE_CUDA="$USE_CUDA" INDIGO_ROOT="$INDIGO_ROOT"
    sudo -u "$BUILD_USER" make -C "$REPO_DIR" tools USE_CUDA="$USE_CUDA" INDIGO_ROOT="$INDIGO_ROOT"
}

install_files() {
    step "Installing to /opt/allsky360"
    make -C "$REPO_DIR" install USE_CUDA="$USE_CUDA" INDIGO_ROOT="$INDIGO_ROOT"
}

install_ai() {
    case "$WITH_AI" in
        1) ;;
        0) return ;;
        *) step "Optional: AI sky classification (Ultralytics YOLO26, several GB download)"
           confirm "Install the YOLO venv now? (can be done later with: sudo make yolo)" n || return ;;
    esac
    step "Installing the Ultralytics YOLO venv"
    make -C "$REPO_DIR" yolo
}

selftest() {
    step "Self-test"
    local bin=/opt/allsky360/bin
    for b in allsky360-camera allsky360-timelapse allsky360-meteo allsky360-aurora allsky360-sensors indigo_mount_position indigo_cameras; do
        if [ -x "$bin/$b" ]; then ok "$b installed"; else fail "$b missing in $bin"; fi
    done
    if "$bin/allsky360-camera" --help >/dev/null 2>&1; then ok "allsky360-camera starts (all libraries found)"
    else fail "allsky360-camera does not start: $("$bin/allsky360-camera" --help 2>&1 | head -1)"; fi

    if make -s -C "$REPO_DIR" node-check >/dev/null 2>&1; then ok "Node.js $(node -v)"; else fail "Node.js too old or missing (sudo make node-deps)"; fi

    local port
    port="$(cfg_get api-port)"; port="${port:-3000}"
    if systemctl is-active --quiet allsky360-api.service; then
        local code=000 i
        for i in 1 2 3 4 5; do
            code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/" || true)
            [ "$code" = 200 ] && break; sleep 1
        done
        if [ "$code" = 200 ]; then ok "web UI answers on port $port"; else fail "web UI on port $port answers $code"; fi
    else
        fail "allsky360-api.service is not running (journalctl -u allsky360-api)"
    fi

    if [ -f /etc/polkit-1/rules.d/50-allsky360.rules ]; then ok "polkit rule for service control installed"
    else fail "polkit rule missing (sudo make polkit)"; fi
    [ -n "$(cfg_get api-token)" ] && ok "api-token set" || warn "api-token empty - web UI service buttons stay disabled"

    local host iport cam
    host="$(cfg_get indigo_server)"; iport="$(cfg_get indigo_port)"; cam="$(cfg_get indigo_camera)"
    if timeout 5 bash -c "exec 3<>/dev/tcp/$host/${iport:-7624}" 2>/dev/null; then
        ok "INDIGO server $host:${iport:-7624} reachable"
    else
        warn "INDIGO server $host:${iport:-7624} not reachable - is it running?"
    fi
    [ -n "$cam" ] && ok "camera: $cam" || warn "no camera configured (indigo_camera)"

    if [ -x "/home/$SERVICE_USER/venvs/yolo/bin/python" ]; then
        if make -s -C "$REPO_DIR" yolo-check >/dev/null 2>&1; then ok "YOLO venv works (GPU)"; else warn "YOLO venv present but 'make yolo-check' fails"; fi
    fi
}

start_services() {
    step "Starting the services"
    local disabled="" u
    for u in $START_UNITS; do systemctl is-enabled --quiet "$u" || disabled="$disabled $u"; done
    if [ -z "$disabled" ]; then
        ok "camera and timers are already enabled"
        return
    fi
    echo "  Not started yet:$disabled"
    if [ "$FAILURES" -eq 0 ] && confirm "Check config.ini done - enable and start them now?" n; then
        # shellcheck disable=SC2086
        systemctl enable --now $disabled
        ok "started - camera log: tail -f /var/log/allsky360/camera.log"
    else
        echo "  Start them later with:"
        echo "    sudo systemctl enable --now$disabled"
    fi
}

# ---------------------------------------------------------------------------
if [ "$MODE" = configure ]; then
    configure
    echo
    echo "Done. A running camera picks up the new values with its next image;"
    echo "the INDIGO settings need: sudo systemctl restart allsky360-camera"
    exit 0
fi

preflight
install_deps
build
install_files
if [ "$ASSUME_YES" != 1 ] && [ -t 0 ]; then configure; else step "Skipping the configuration assistant (--yes): edit $CONFIG_FILE"; fi
install_ai
selftest
start_services

echo
if [ "$FAILURES" -gt 0 ]; then
    echo "${C_R}Installation finished with $FAILURES error(s) and $WARNINGS warning(s) - see above.${C_0}"
    exit 1
fi
echo "${C_G}Installation finished${C_0} ($WARNINGS warning(s))."
api_port="$(cfg_get api-port)"
echo "  Web UI:        http://$(hostname -I 2>/dev/null | awk '{print $1}'):${api_port:-3000}/"
echo "  Configuration: $CONFIG_FILE   (assistant again: sudo ./install.sh --configure)"

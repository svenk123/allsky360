#!/bin/bash
#
# postprocessing.sh
# Hook that the allsky360 camera daemon runs after every captured image.
#
# Usage:
#   postprocessing.sh /path/to/images/YYYYMMDD/image-YYYYMMDDHHMMSS.jpg
#
# The daemon spawns this detached and as the same user it runs as itself, with
# stdout/stderr on /dev/null - everything this script wants to report has to go
# into $LOG. Besides $1 the daemon exports:
#
#   ALLSKY360_IMAGE_PATH        same as $1
#   ALLSKY360_IMAGE_DIRECTORY   base image directory
#   ALLSKY360_TODAY_STR         YYYYMMDD (UTC)
#   ALLSKY360_NOW_STR           HHMMSS (UTC)
#   ALLSKY360_IS_NIGHT          1 during night mode, 0 during the day
#
# This is the operator's extension point: run an AI model, upload the image,
# trigger an alert - whatever is needed. The AI part below is an example and
# skips itself cleanly when the model or the venv is not installed.
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

# ========== CONFIG ==========
LOG="${ALLSKY360_POSTPROCESSING_LOG:-/var/log/allsky360/postprocessing.log}"
LOCKFILE="/run/allsky360/postprocessing.lock"

AI_MODEL_DIR="/opt/allsky360/ai/models"
MODEL="$AI_MODEL_DIR/weather.pt"
CLASSES_FILE="$AI_MODEL_DIR/classes.txt"
MASK_IMAGE="/opt/allsky360/scripts/mask.jpg"
IMGSZ=640
# Ultralytics venv created by "sudo make yolo"
YOLO="${HOME:-/home/allskyuser}/venvs/yolo/bin/yolo"

# =============================================
mkdir -p "$(dirname "$LOG")" 2>/dev/null || LOG=/dev/null
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [POSTPROCESSING] $*" >> "$LOG"; }

IMAGE="${1:-${ALLSKY360_IMAGE_PATH:-}}"
if [ -z "$IMAGE" ]; then
    log "ERROR: no image path given (argument 1 or ALLSKY360_IMAGE_PATH)"
    exit 1
fi
if [ ! -f "$IMAGE" ]; then
    log "ERROR: image '$IMAGE' does not exist"
    exit 1
fi

# Only one instance at a time. A capture every 120 s must never pile up jobs
# if the model is slow; skipping one image is the right trade-off.
if command -v flock >/dev/null 2>&1; then
    mkdir -p "$(dirname "$LOCKFILE")" 2>/dev/null || LOCKFILE="/tmp/allsky360-postprocessing.lock"
    exec 9>"$LOCKFILE"
    if ! flock -n 9; then
        log "still busy with a previous image, skipping $IMAGE"
        exit 0
    fi
fi

log "start: $IMAGE"

# The AI example needs these; without them the hook is a no-op and that is fine.
for tool in jq; do
    command -v "$tool" >/dev/null 2>&1 || { log "note: '$tool' not installed, nothing to do"; exit 0; }
done
if [ ! -x "$YOLO" ] || [ ! -f "$MODEL" ] || [ ! -f "$CLASSES_FILE" ]; then
    log "note: AI backend not installed (yolo=$YOLO model=$MODEL), nothing to do"
    exit 0
fi

# 1) Optional masking
###################################################
TMP_IMG="$(mktemp -t allsky360-pp.XXXXXXXX)"
mv -f "$TMP_IMG" "$TMP_IMG.jpg" 2>/dev/null || true
TMP_IMG="$TMP_IMG.jpg"
cleanup() { rm -f "$TMP_IMG"; }
trap cleanup EXIT

INFER_SRC="$IMAGE"
if [ -f "$MASK_IMAGE" ] && command -v convert >/dev/null 2>&1; then
    if convert "$IMAGE" "$MASK_IMAGE" -compose multiply -composite "$TMP_IMG"; then
        INFER_SRC="$TMP_IMG"
    else
        log "WARNING: masking failed, continuing with the unmasked image"
    fi
fi

# 2) Inference
###################################################
set +e
OUT="$("$YOLO" predict task=classify model="$MODEL" imgsz="$IMGSZ" \
        source="$INFER_SRC" save=False save_txt=False show=False save_conf=False 2>&1)"
RC=$?
set -e
if [ $RC -ne 0 ]; then
    log "ERROR: yolo predict failed rc=$RC"
    exit 1
fi

RAW="$(printf '%s\n' "$OUT"   | sed -n -E 's/.*: [0-9]+x[0-9]+ ([^ ,]+).*/\1/p'        | head -n 1)"
SCORE="$(printf '%s\n' "$OUT" | sed -n -E 's/.*: [0-9]+x[0-9]+ [^ ]+ ([0-9.]+).*/\1/p' | head -n 1)"
if [ -z "${RAW:-}" ]; then
    log "ERROR: could not parse a class from the yolo output"
    exit 1
fi

# Models without embedded class names print "class<N>" - map the index via
# classes.txt. Current Ultralytics releases print the class name stored in
# the model directly (e.g. "clear"); use it as it is.
IDX="$(printf '%s' "$RAW" | sed -n -E 's/^class([0-9]+)$/\1/p')"
if [ -n "${IDX:-}" ]; then
    WEATHER="$(awk -v n="$((IDX+1))" 'NR==n{print; exit}' "$CLASSES_FILE")"
    if [ -z "${WEATHER:-}" ]; then
        log "ERROR: no class name for index $IDX in $CLASSES_FILE"
        exit 1
    fi
else
    WEATHER="$RAW"
fi
[ -n "${SCORE:-}" ] || SCORE=0

# 3) Patch the image metadata - never destroy it
###################################################
JSON="${IMAGE%.jpg}.json"
if [ ! -f "$JSON" ]; then
    log "WARNING: $JSON does not exist, result not stored (weather=$WEATHER score=$SCORE)"
    exit 0
fi

ORIG_UID="$(stat -c %u "$JSON")"
ORIG_GID="$(stat -c %g "$JSON")"
ORIG_MODE="$(stat -c %a "$JSON")"

TMP_JSON="$(mktemp -t allsky360-json.XXXXXXXX)"
if ! jq --arg weather "$WEATHER" --arg score "$SCORE" \
        '.weather=$weather | .weather_score=($score|tonumber)' \
        "$JSON" > "$TMP_JSON"; then
    log "ERROR: jq failed on $JSON - metadata left untouched"
    rm -f "$TMP_JSON"
    exit 1
fi
if [ ! -s "$TMP_JSON" ]; then
    log "ERROR: jq produced an empty result - metadata left untouched"
    rm -f "$TMP_JSON"
    exit 1
fi

chown "$ORIG_UID:$ORIG_GID" "$TMP_JSON" 2>/dev/null || true
chmod "$ORIG_MODE" "$TMP_JSON" 2>/dev/null || true
mv -f "$TMP_JSON" "$JSON"

log "done: $JSON weather=$WEATHER score=$SCORE"
exit 0

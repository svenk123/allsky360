#!/bin/bash
#
# postprocessing.sh
# Wrapper script to run AI image analysis for Allsky360.
#
# Usage:
#   postprocessing.sh /path/to/image.jpg
#
# This script is designed to be executed via systemd-run.
#
# Copyright (c) 2025 Sven Kreiensen
#
# You can use this software under the terms of the MIT software license
# (see LICENSE.md)
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#


# Configuration
LOG="/var/log/allsky360-postprocessing.log"

CURRENT_USER=$(whoami)
# or
# CURRENT_USER=$(id -un)

USER_HOME=$HOME
# or
# USER_HOME=$(getent passwd "$CURRENT_USER" | cut -d: -f6)

# ========== CONFIG ==========
AI_MODEL_DIR="/opt/allsky360/ai/models"
AI_BACKEND="yolov11"         # Possible: tensorrt | onnx | yolov11

IMGSZ=640
CLASSES_FILE="${AI_MODEL_DIR}/classes.txt"

# Model paths (modify according to your structure)
TENSORRT_MODEL="$AI_MODEL_DIR/weather.engine"
#ONNX_MODEL="$AI_MODEL_DIR/weather.onnx"
#YOLO_MODEL="$AI_MODEL_DIR/weather.pt"

# =============================================
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [POSTPROCESSING] $*" >> "$LOG"
}

DATE=`jq -r '.date' ${ALLSKY360_IMAGE_DIRECTORY}/latest_image.json`
TIME=`jq -r '.image' ${ALLSKY360_IMAGE_DIRECTORY}/latest_image.json`
IMAGE=${ALLSKY360_IMAGE_DIRECTORY}/${DATE}/image-${DATE}${TIME}.jpg

log "Starting postprocessing on: $IMAGE (backend=$AI_BACKEND)"

if [[ ! -f "$IMAGE" ]]; then
    log "ERROR: Image '$IMAGE' does not exist."
    exit 1
fi

# 0) Mask image
###################################################
TMP_IMG=$(mktemp --dry-run --suffix=.jpg)
MASK_IMAGE=/opt/allsky360/scripts/mask.jpg
TMP_DIR=/tmp

if [ ! -w $TMP_DIR ]; then
    log "ERROR: TMP_DIR '$TMP_DIR' is not writable."
    exit 1
fi

convert $IMAGE $MASK_IMAGE -compose multiply -composite $TMP_IMG
if [ $? -ne 0 ]; then
    log "ERROR: convert failed."
    rm $TMP_IMG
    exit 1
fi

# 1) Inference
################################

YOLO="${USER_HOME}/venvs/yolo11/bin/yolo"
OUT="$(
    $YOLO predict task=classify model="$TENSORRT_MODEL" imgsz="$IMGSZ" source="${TMP_IMG}" save=False save_txt=False show=False save_conf=False 2>&1
)"
RC=$?

# Remove temporary image
rm $TMP_IMG

#log "$OUT"
if [[ $RC -ne 0 ]]; then
  log "ERROR: yolo predict failed rc=$RC"
  exit 1
fi

# 2) Extract top1 label (class0, class1, ...)
###################################################
RAW="$(printf '%s\n' "$OUT" | sed -n 's/.*: [0-9]\+x[0-9]\+ \([^ ,]\+\).*/\1/p' | head -n 1)"
#log "$RAW"
SCORE="$(printf '%s\n' "$OUT" | sed -n 's/.*: [0-9]\+x[0-9]\+ [^ ]\+ \([0-9.]\+\).*/\1/p' | head -n 1)"
#log "$SCORE"

IDX="$(printf '%s' "$RAW" | sed -n 's/^class\([0-9]\+\)$/\1/p')"
if [[ -z "${IDX:-}" ]]; then
  echo "Could not parse class index from: $RAW" >&2
  exit 1
fi

WEATHER="$(awk -v n="$((IDX+1))" 'NR==n{print; exit}' "$CLASSES_FILE")"
if [[ -z "${WEATHER:-}" ]]; then
  echo "No class name for index $IDX in $CLASSES_FILE" >&2
  exit 1
fi

# 3) JSON updaten (atomic)
###################################################
JSON="${IMAGE%.jpg}.json"

ORIG_UID="$(stat -c %u "$JSON")"
ORIG_GID="$(stat -c %g "$JSON")"
ORIG_MODE="$(stat -c %a "$JSON")"

TMP_JSON="$(mktemp)"
jq --arg weather "$WEATHER" --arg score "$SCORE" \
   '.weather=$weather | .weather_score=($score|tonumber)' \
   "$JSON" > "$TMP_JSON"

mv -f "$TMP_JSON" "$JSON"
chown "$ORIG_UID:$ORIG_GID" "$JSON"
chmod "$ORIG_MODE" "$JSON"
log "Updated $JSON: weather=$WEATHER score=$SCORE"

log "Postprocessing finished: $IMAGE.json"

exit 0

#!/bin/bash
#
# run_ai_processing.sh
# Wrapper script to run AI image analysis for Allsky360.
# Supports: TensorRT engine, ONNX, Ultralytics YOLO (YOLO26 / YOLO11 models).
#
# Usage:
#   run_ai_processing.sh /path/to/image.jpg
#
# This script is designed to be executed via systemd-run.
#

IMAGE="$1"
LOG="/var/log/allsky360/ai.log"

# ========== CONFIG ==========
AI_MODEL_DIR="/opt/allsky360/ai/models"
AI_BACKEND="yolo"            # Possible: tensorrt | onnx | yolo

# Model paths (modify according to your structure)
TENSORRT_ENGINE="$AI_MODEL_DIR/weather.engine"
ONNX_MODEL="$AI_MODEL_DIR/weather.onnx"
YOLO_MODEL="$AI_MODEL_DIR/weather.pt"

# Ultralytics venv created by "sudo make yolo"
PYTHON="${HOME:-/home/allskyuser}/venvs/yolo/bin/python"
AI_SCRIPTS="/opt/allsky360/ai"

# =============================================
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [AI] $*" >> "$LOG"
}

if [[ ! -f "$IMAGE" ]]; then
    log "ERROR: Image '$IMAGE' does not exist."
    exit 1
fi

log "Starting AI processing on: $IMAGE (backend=$AI_BACKEND)"

# =============================================
# BACKEND: TENSORRT (.engine)
# =============================================

run_tensorrt() {
    "$PYTHON" "$AI_SCRIPTS/trt_infer.py" \
        --engine "$TENSORRT_ENGINE" \
        --input "$IMAGE" \
        --output "$IMAGE.json"
}

# =============================================
# BACKEND: ONNX
# =============================================

run_onnx() {
    "$PYTHON" "$AI_SCRIPTS/onnx_infer.py" \
        --model "$ONNX_MODEL" \
        --input "$IMAGE" \
        --output "$IMAGE.json"
}

# =============================================
# BACKEND: Ultralytics YOLO
# =============================================

run_yolo() {
    "$PYTHON" "$AI_SCRIPTS/yolo_infer.py" \
        --model "$YOLO_MODEL" \
        --source "$IMAGE" \
        --save_json "$IMAGE.json"
}

# =============================================
# SELECT BACKEND
# =============================================

case "$AI_BACKEND" in
    tensorrt)   run_tensorrt   ;;
    onnx)       run_onnx       ;;
    yolo)       run_yolo       ;;
    *)
        log "ERROR: Unknown backend: $AI_BACKEND"
        exit 2
        ;;
esac

log "AI processing finished: $IMAGE.json"
exit 0

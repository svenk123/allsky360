#!/usr/bin/env bash
#
# Export Ultralytics YOLO classification model to ONNX.
# Usage:
#   ./export_onnx.sh /path/to/best.pt 320
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

MODEL="${1:-}"
IMGSZ="${2:-320}"
DEVICE="${3:-cpu}"
# Ultralytics CLI from the venv created by "sudo make yolo", override with YOLO=
YOLO="${YOLO:-${HOME:-/home/allskyuser}/venvs/yolo/bin/yolo}"
[[ -x "$YOLO" ]] || YOLO="yolo"

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 /path/to/best.pt [imgsz] [device]"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL"
  exit 1
fi

echo "Exporting to ONNX:"
echo "  model : $MODEL"
echo "  imgsz : $IMGSZ"
echo "  device: $DEVICE"

"$YOLO" export model="$MODEL" format=onnx imgsz="$IMGSZ" device="$DEVICE"

echo "Done."
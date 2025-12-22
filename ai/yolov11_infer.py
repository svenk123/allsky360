#!/usr/bin/env python3
# YOLOv11 inference script for Allsky360
#
# This script is designed to be executed via systemd-run.
#
# Usage:
#   yolov11_infer.py --model /path/to/yolov11.pt --source /path/to/image.jpg --save_json /path/to/output.json
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
import argparse
from ultralytics import YOLO
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--source")
parser.add_argument("--save_json")
args = parser.parse_args()

model = YOLO(args.model)

res = model.predict(args.source, verbose=False, imgsz=640, device=0)

# YOLOv11 result serialization
objects = []
for r in res:
    for box in r.boxes:
        objects.append({
            "cls": int(box.cls),
            "name": model.names[int(box.cls)],
            "conf": float(box.conf),
            "x1": float(box.xyxy[0][0]),
            "y1": float(box.xyxy[0][1]),
            "x2": float(box.xyxy[0][2]),
            "y2": float(box.xyxy[0][3]),
        })

with open(args.save_json, "w") as f:
    json.dump({"image": args.source, "objects": objects}, f, indent=2)

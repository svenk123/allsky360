#!/usr/bin/env python3
# ONNX inference script for Allsky360
#
# Usage:
#   onnx_infer.py --model /path/to/model.onnx --input /path/to/image.jpg --output /path/to/output.json
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
import argparse
import onnxruntime as ort
import cv2
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

# Load model
session = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

img = cv2.imread(args.input)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[None, ...]

inputs = {session.get_inputs()[0].name: img}
outputs = session.run(None, inputs)[0]

result = outputs[0].tolist()

with open(args.output, "w") as f:
    json.dump({"image": args.input, "result": result}, f, indent=2)

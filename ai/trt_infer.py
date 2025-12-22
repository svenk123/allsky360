#!/usr/bin/env python3
# Simple TensorRT inference wrapper for Allsky360
#
# Usage:
#   trt_infer.py --engine /path/to/engine.plan --input /path/to/image.jpg --output /path/to/output.json
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
import tensorrt as trt
import numpy as np
import cv2
import json
import pycuda.driver as cuda
import pycuda.autoinit

parser = argparse.ArgumentParser()
parser.add_argument("--engine")
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

# Load engine
logger = trt.Logger(trt.Logger.WARNING)
with open(args.engine, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Load image
img = cv2.imread(args.input)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[None, ...]

# Allocate buffers
bindings = []
inputs = []
outputs = []

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host = cuda.pagelocked_empty(size, dtype)
    dev = cuda.mem_alloc(host.nbytes)
    bindings.append(int(dev))

    if engine.binding_is_input(binding):
        inputs.append((host, dev))
    else:
        outputs.append((host, dev))

np.copyto(inputs[0][0], img.flatten())
cuda.memcpy_htod(inputs[0][1], inputs[0][0])
context.execute_v2(bindings)
cuda.memcpy_dtoh(outputs[0][0], outputs[0][1])

# Example: output is probabilities
result = outputs[0][0].tolist()

with open(args.output, "w") as f:
    json.dump({"image": args.input, "result": result}, f, indent=2)

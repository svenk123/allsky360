###############################################################################
# 
# Makefile for Timelapse cuda nv
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
###############################################################################

CC = gcc

CFLAGS = -Wall -O2 -fopenmp -I./src
# Find libjpeg.so.9 explicitly and link it first to avoid conflicts
# with libjpeg.so.8 required by libgdk_pixbuf (indirectly loaded)
JPEG_LIB_PATH = $(shell find /usr/lib* -name "libjpeg.so.9" 2>/dev/null | head -1)
JPEG_LIB_DIR = $(if $(JPEG_LIB_PATH),$(dir $(JPEG_LIB_PATH)),)
# Link libjpeg first, before other libraries that might pull in libjpeg.so.8
JPEG_LIBS = $(if $(JPEG_LIB_DIR),-L$(JPEG_LIB_DIR),) -ljpeg
LDFLAGS = $(JPEG_LIBS) -lavformat -lavcodec -lavutil -lswscale -lpng -lm -fopenmp

# Include path for allsky360 headers
CFLAGS += -I../../include

USE_CUDA ?= 0

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TARGET = $(BIN_DIR)/allsky360-timelapse

C_SRC = $(SRC_DIR)/main.c $(SRC_DIR)/video_encoder.c $(SRC_DIR)/image_check.c $(SRC_DIR)/stacker.c $(SRC_DIR)/png_to_rgb8.c
C_OBJS = $(C_SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

ifeq ($(USE_CUDA),1)
	USE_NVJPEG = 1

# Intelligent CUDA-path detection
	CUDA_PATH = $(shell \
        if command -v nvcc >/dev/null 2>&1; then \
            dirname $$(dirname $$(which nvcc)); \
        elif [ -d "/usr/local/cuda" ]; then \
            echo "/usr/local/cuda"; \
        elif [ -d "/usr/local/cuda-12.6" ]; then \
            echo "/usr/local/cuda-12.6"; \
        elif [ -d "/usr/local/cuda-12.0" ]; then \
            echo "/usr/local/cuda-12.0"; \
        elif [ -d "/usr/local/cuda-11.8" ]; then \
            echo "/usr/local/cuda-11.8"; \
        else \
            echo "/usr/local/cuda"; \
        fi)

# CUDA nutzt standardmäßig gcc-9 oder gcc-10, während das System GCC 11 verwendet.
# nvcc mischt unterschiedliche Versionen der Standardbibliothek, was zur Redefinition von std::integral_constant führt.
# Der Fehler tritt oft bei CUDA 11+ mit neuen GCC-Versionen auf.
# Lösung:
    NVCC = nvcc --compiler-bindir=/usr/bin/gcc-11 -I$(CUDA_PATH)/include
	NVCCFLAGS = -arch=sm_87 -O2 --compiler-options '-fPIC -std=c++17'
# Jetson nano
#    NVCC = nvcc --compiler-bindir=/usr/bin/gcc-7 -I$(CUDA_PATH)/include

#LDFLAGS += -L/usr/local/cuda-12.6/targets/aarch64-linux/lib

	CFLAGS +=  -I/usr/include/aarch64-linux-gnu -I$(CUDA_PATH)/include
	LDFLAGS += -L/usr/lib/aarch64-linux-gnu -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib
	LDFLAGS += -lnppig -lcuda -lcudart -lnvjpeg
	CUDA_INCLUDE = -I$(CUDA_PATH)/include -I./src -I../../include

    CU_SRC = $(SRC_DIR)/image_loader_nvjpeg.cu
	CU_OBJS = $(CU_SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
else
	C_SRC += $(SRC_DIR)/image_loader_libjpeg.c
    CU_SRCS =
    CU_OBJS =
endif

all: $(TARGET)

ifneq ($(shell id -u), 0)
deps:
	echo This must be run with root permissions.
	echo Please run 'sudo make deps'
else
deps:
	@echo `date +%F\ %R:%S` Installing build dependencies...
	@apt update
	@apt -y install libavcodec-dev libavformat-dev libswscale-dev
	@apt -y install cuda-cudart-dev-12-6 libnvjpeg-dev-12-6 cuda-nvcc-12-6 libnpp-dev-12-6
endif

$(TARGET): $(C_OBJS) $(CU_OBJS)
	@mkdir -p $(BIN_DIR)
	# Use --as-needed and suppress the libjpeg version mismatch warning
	# The warning occurs because libgdk_pixbuf (indirectly loaded via FFmpeg)
	# requires libjpeg.so.8, while we link against libjpeg.so.9.
	# Both versions are usually compatible, so the warning can be safely ignored.
	$(CC) -Wl,--as-needed -Wl,--no-warn-mismatch -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA sources
ifeq ($(USE_CUDA),1)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $< -o $@ -Xcompiler -fPIC -I./src
endif

clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)


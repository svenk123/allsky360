###############################################################################
# 
# Makefile for Allsky360
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
CFLAGS = -Wall -Wextra -O2 -DINDIGO_LINUX -std=gnu11

# Cairo + FreeType + libjpeg Flags
CAIRO_CFLAGS = $(shell pkg-config --cflags cairo)
CAIRO_LIBS = $(shell pkg-config --libs cairo)
JPEG_LIBS = -ljpeg

FREETYPE_CFLAGS = $(shell pkg-config --cflags freetype2)
FREETYPE_LIBS = $(shell pkg-config --libs freetype2)

# FITS
#CFITSIO_LIBS = -lcfitsio
CFITSIO_LIBS = $(shell pkg-config --exists cfitsio && pkg-config --cflags --libs cfitsio)

# indigo server
INDIGO_ROOT = ../indigo
INDIGO_CFLAGS = -I$(INDIGO_ROOT)/indigo_libs
INDIGO_LIBS = -L$(INDIGO_ROOT)/build/lib -lindigo

# Libnova
ASTRO_CFLAGS = -I/usr/include/libnova
ASTRO_LIBS = -L/usr/lib -lnova

# PNG Library
PNG_LIBS = -lpng

INCLUDE = $(CAIRO_CFLAGS) $(FREETYPE_CFLAGS) $(INDIGOI_CFLAGS) $(ASTRO_CFLAGS) -I./include
LDFLAGS += $(CAIRO_LIBS) $(FREETYPE_LIBS) $(INDIGO_LIBS) $(ASTRO_LIBS)  $(JPEG_LIBS) $(PNG_LIBS) $(CFITSIO_LIBS) -lm

# OpenMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp

# Json-c
LDFLAGS += -ljson-c

# Sqlite
LDFLAGS += -lsqlite3

# NVidia Cuda support
# Toggle CUDA Support (Set to 1 to enable, 0 to disable)
USE_CUDA ?= 0
USE_GPUJPEG ?= 0

ifeq ($(USE_CUDA), 1)
    # Intelligent CUDA-path detection
    CUDA_PATH ?= $(shell \
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
    
    CUDA_LIBS = -L$(CUDA_PATH)/lib64
# NVidia Orin
    NVCC = nvcc --compiler-bindir=/usr/bin/gcc-11
    NVCCFLAGS = -arch=sm_87 -O2 --compiler-options '-fPIC -std=c++17'
    CUDA_LIBS += -L$(CUDA_PATH)/targets/aarch64-linux/lib
## Jetson Nano
#    NVCC = nvcc --compiler-bindir=/usr/bin/gcc-7
#    NVCCFLAGS = -arch=sm_53 -O2 --compiler-options '-fPIC -std=c++11'
# Or other systems (multi-arch)
#    NVCC = nvcc
#    NVCCFLAGS = -arch=sm_53 -code=sm_53,sm_87 -O2 --compiler-options '-fPIC -std=c++11'
    CUDA_LIBS += -lcudart -lnvjpeg
    CFLAGS += -DUSE_CUDA -pthread 
    INCLUDE += -I$(CUDA_PATH)/include -I/usr/include/cairo -I./include
    CUDA_INCLUDE = -I$(CUDA_PATH)/include -I./include
    LDFLAGS += $(CUDA_LIBS)
    CUDAFLAGS += -DCUDA_FORCE_CXX03 -L$(CUDA_PATH)/lib64 -lcudart -lfreetype
    CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
    CU_OBJS = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CU_SRCS))

ifeq ($(USE_GPUJPEG), 1)
    CUDA_INCLUDE += -I/usr/local/include -DUSE_GPUJPEG
    CUDA_LIBS += -lgpujpeg
endif

    # Cuda wrappers are only used when CUDA is active
    CUDA_GPU_FUNCTIONS_WRAPPER_SRC = $(SRCDIR)/gpu_functions_cuda_wrapper.c
    CUDA_GPU_FUNCTIONS_WRAPPER_OBJ = $(OBJDIR)/gpu_functions_cuda_wrapper.o
    CUDA_ACDNR_FILTER__WRAPPER_SRC = $(SRCDIR)/acdnr_filter_cuda_wrapper.c
    CUDA_ACDNR_FILTER_WRAPPER_OBJ = $(OBJDIR)/acdnr_filter_cuda_wrapper.o
	CUDA_DEHAZE_FILTER_WRAPPER_SRC = $(SRCDIR)/dehaze_filter_cuda_wrapper.c
	CUDA_DEHAZE_FILTER_WRAPPER_OBJ = $(OBJDIR)/dehaze_filter_cuda_wrapper.o
	CUDA_CLARITY_FILTER_WRAPPER_SRC = $(SRCDIR)/clarity_filter_cuda_wrapper.c
	CUDA_CLARITY_FILTER_WRAPPER_OBJ = $(OBJDIR)/clarity_filter_cuda_wrapper.o
    CUDA_COLOR_CALIBRATION_WRAPPER_SRC = $(SRCDIR)/color_calibration_cuda_wrapper.c
    CUDA_COLOR_CALIBRATION_WRAPPER_OBJ = $(OBJDIR)/color_calibration_cuda_wrapper.o
	CUDA_HDR_MERGE_WRAPPER_SRC = $(SRCDIR)/hdr_merge_cuda_wrapper.c
	CUDA_HDR_MERGE_WRAPPER_OBJ = $(OBJDIR)/hdr_merge_cuda_wrapper.o
	CUDA_JPEG_FUNCTIONS_WRAPPER_SRC = $(SRCDIR)/jpeg_functions_cuda_wrapper.c
	CUDA_JPEG_FUNCTIONS_WRAPPER_OBJ = $(OBJDIR)/jpeg_functions_cuda_wrapper.o
	CUDA_LIGHTS_AND_SHADOWS_WRAPPER_SRC = $(SRCDIR)/lights_and_shadows_cuda_wrapper.c
    CUDA_LIGHTS_AND_SHADOWS_WRAPPER_OBJ = $(OBJDIR)/lights_and_shadows_cuda_wrapper.o
    CUDA_MEDIAN_FILTER_WRAPPER_SRC = $(SRCDIR)/median_filter_cuda_wrapper.c
    CUDA_MEDIAN_FILTER_WRAPPER_OBJ = $(OBJDIR)/median_filter_cuda_wrapper.o
	CUDA_WAVELET_SHARPEN_WRAPPER_SRC = $(SRCDIR)/wavelet_sharpen_cuda_wrapper.c
	CUDA_WAVELET_SHARPEN_WRAPPER_OBJ = $(OBJDIR)/wavelet_sharpen_cuda_wrapper.o
    CUDA_WHITE_BALANCE_WRAPPER_SRC = $(SRCDIR)/white_balance_cuda_wrapper.c
    CUDA_WHITE_BALANCE_WRAPPER_OBJ = $(OBJDIR)/white_balance_cuda_wrapper.o
else
    CU_SRCS =
    CU_OBJS =
    CUDA_GPU_FUNCTIONS_WRAPPER_SRC =
    CUDA_GPU_FUNCTIONS_WRAPPER_OBJ =
    CUDA_ACDNR_FILTER_WRAPPER_SRC =
    CUDA_ACDNR_FILTER_WRAPPER_OBJ =
	CUDA_DEHAZE_FILTER_WRAPPER_SRC =
	CUDA_DEHAZE_FILTER_WRAPPER_OBJ =
	CUDA_CLARITY_FILTER_WRAPPER_SRC =
	CUDA_CLARITY_FILTER_WRAPPER_OBJ =
    CUDA_COLOR_CALIBRATION_WRAPPER_SRC =
    CUDA_COLOR_CALIBRATION_WRAPPER_OBJ =
    CUDA_HDR_MERGE_WRAPPER_SRC =
    CUDA_HDR_MERGE_WRAPPER_OBJ =
	CUDA_JPEG_FUNCTIONS_WRAPPER_SRC =
	CUDA_JPEG_FUNCTIONS_WRAPPER_OBJ =
    CUDA_LIGHTS_AND_SHADOWS_WRAPPER_SRC =
    CUDA_LIGHTS_AND_SHADOWS_WRAPPER_OBJ =
    CUDA_MEDIAN_FILTER_WRAPPER_SRC =
    CUDA_MEDIAN_FILTER_WRAPPER_OBJ =
	CUDA_WAVELET_SHARPEN_WRAPPER_SRC =
	CUDA_WAVELET_SHARPEN_WRAPPER_OBJ =
    CUDA_WHITE_BALANCE_WRAPPER_SRC =
    CUDA_WHITE_BALANCE_WRAPPER_OBJ =
endif

SRCDIR = src
OBJDIR = obj
BINDIR = bin
TESTDIR = test
TESTBIN = $(BINDIR)/test_sun_calc 
UTILSDIR = utils
UTILSBIN = $(UTILSDIR)/keogram
NODEDIR = allsky-api

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
TARGET = $(BINDIR)/allsky360-camera

all: $(TARGET)

ifneq ($(shell id -u), 0)
deps:
	echo This must be run with root permissions.
	echo Please run 'sudo make deps'
else
deps:
	@echo `date +%F\ %R:%S` Installing build dependencies...
	@apt update
	@apt -y install libjpeg9-dev libcfitsio-dev libpng-dev libcairo-dev libnova-dev libjson-c-dev
	@apt -y install sqlite3 libsqlite3-dev
	@apt -y install libcurl4-openssl-dev
	@apt -y install cuda-nvcc-12-6 libnvjpeg-dev-12-6
	@echo "Adding NodeSource repo (Node 20 LTS)…"
	curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
	apt-get update
	apt-get install -y nodejs
	# optional: build tools falls native addons
	# apt-get install -y build-essential
endif

node-build:
	@echo "Installing npm deps (ci)…"
	cd $(NODEDIR) && npm ci
	@echo "Ensuring dev tool (esbuild) is available…"
	cd $(NODEDIR) && npm pkg get devDependencies.esbuild >/dev/null 2>&1 || npm i -D esbuild
	@echo "Creating bundle entry…"
	cd $(NODEDIR) && mkdir -p public/js
	cd $(NODEDIR) && printf "%s\n%s\n" \
		"import * as astronomia from 'astronomia';" \
		"export default astronomia;" > astro_bundle.js
	@echo "Bundling astronomia for browser…"
	cd $(NODEDIR) && npm install astronomia --save
	cd $(NODEDIR) && npx --yes esbuild astro_bundle.js \
		--bundle \
		--minify \
		--format=iife \
		--global-name=Astronomia \
		--outfile=public/js/astronomia.bundle.min.js

.PHONY : deps

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CAIRO_CFLAGS) $(INCLUDE) $(CFLAGS) -c $< -o $@

# Compile CUDA-Wrappers, when CUDA is active
ifeq ($(USE_CUDA), 1)
$(OBJDIR)/gpu_functions_cuda_wrapper.o: $(SRCDIR)/gpu_functions_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/acdnr_filter_cuda_wrapper.o: $(SRCDIR)/acdnr_filter_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/dehaze_filter_cuda_wrapper.o: $(SRCDIR)/dehaze_filter_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/clarity_filter_cuda_wrapper.o: $(SRCDIR)/clarity_filter_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/color_calibration_cuda_wrapper.o: $(SRCDIR)/color_calibration_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/jpeg_functions_cuda_wrapper.o: $(SRCDIR)/jpeg_functions_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/wavelet_sharpen_cuda_wrapper.o: $(SRCDIR)/wavelet_sharpen_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/white_balance_cuda_wrapper.o: $(SRCDIR)/white_balance_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)/lights_and_shadows_cuda_wrapper.o: $(SRCDIR)/lights_and_shadows_cuda_wrapper.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

endif

$(TARGET): $(OBJS) $(CU_OBJS) $(CUDA_GPU_FUNCTIONS_WRAPPER_OBJ) $(CUDA_ACDNR_FILTER_WRAPPER_OBJ) $(CUDA_DEHAZE_FILTER_WRAPPER_OBJ) $(CUDA_CLARITY_FILTER_WRAPPER_OBJ) $(COLOR_CALIBRATION_WRAPPER_OBJ) $(CUDA_JPEG_FUNCTIONS_WRAPPER_OBJ) $(CUDA_HDR_MERGE_WRAPPER_OBJ) $(CUDA_LIGHTS_AND_SHADOWS_WRAPPER_OBJ) $(CUDA_MEDIAN_FILTER_WRAPPER_OBJ) $(CUDA_WAVELET_SHARPEN_WRAPPER_OBJ) $(CUDA_WHITE_BALANCE_WRAPPER_OBJ)
	@mkdir -p $(BINDIR)
ifeq ($(USE_CUDA), 1)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(CU_OBJS) $(LDFLAGS)
else
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)
endif

# Compile CUDA sources
ifeq ($(USE_CUDA), 1)
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCLUDE) -c $< -o $@
endif

# Build for utils
tools: timelapse meteo aurora location list-cameras panorama mount-status

timelapse:
	@echo "Building timelapse tool"
ifeq ($(USE_CUDA), 1)
	@$(MAKE) -C tools/timelapse USE_CUDA=1 USE_NVJPEG=1
else
	@$(MAKE) -C tools/timelapse
endif

meteo:
	@echo "Bulding meteo tool"
	@$(MAKE) -C tools/meteo

aurora:
	@echo "Building aurora tool"
	@$(MAKE) -C tools/aurora

location:
	@echo "Building location tool"
	@$(MAKE) -C tools/location

list-cameras:
	@echo "Building list-cameras tool"
	@$(MAKE) -C tools/indigo-list-cameras

mount-status:
	@echo "Building mount-status tool"
	@$(MAKE) -C tools/indigo-mount-status

panorama:
	@echo "Building panorama tool"
	@$(MAKE) -C tools/panorama

# Build for unit tests
tests: $(TESTBIN)

$(TESTBIN): $(TESTDIR)/%.c
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -o $(TESTBIN) $(TESTDIR)/test_sun_calc.c $(SRCDIR)/sun_calc.c $(LDFLAGS)

##################### Clean Build

clean_databases:
	rm -rf ./database
	mkdir ./database

clean_images:
	rm -rf ./images
	mkdir ./images

clean_node:
	cd $(NODEDIR) && rm -rf node_modules package-lock.json

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(UTILSDIR)

##################### Installation

USERNAME          := allskyuser
GROUPNAME         := allskygroup

INSTALL           := install
SYSTEMCTL         := systemctl

APP_DIR           := /opt/allsky360/bin
SYSTEMD_DIR       := /etc/systemd/system
CRON_DIR          := /etc/cron.d

RUN_DIR           := /run/allsky360
LOG_DIR           := /var/log/allsky360
CONFIG_DIR        := /home/allskyuser
DATA_DIR          := /opt/allsky360
HOME_DIR          := /home/allskyuser

user:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Ensuring group $(GROUPNAME) exists…"
	@getent group $(GROUPNAME) >/dev/null 2>&1 || groupadd --system $(GROUPNAME)
	@echo "Creating user $(USERNAME) and adding to group $(GROUPNAME)..."
	@id -u $(USERNAME) >/dev/null 2>&1 || useradd -r -s /bin/bash $(USERNAME)
	usermod -a -G $(GROUPNAME) $(USERNAME)
	usermod -a -G sudo $(USERNAME)
	# Must be added to these groups to use the GPU
	usermod -a -G video $(USERNAME)
	usermod -a -G render $(USERNAME)
	@echo "Creating home directory $(HOME_DIR)..."
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 775 $(HOME_DIR)
	@echo "Setting up bash environment for $(USERNAME)..."
	@echo "# Allsky360 environment" > $(HOME_DIR)/.bashrc
	@echo "export PATH=\$$PATH:/opt/allsky360/bin" >> $(HOME_DIR)/.bashrc
	chown $(USERNAME):$(GROUPNAME) $(HOME_DIR)/.bashrc
	chmod 644 $(HOME_DIR)/.bashrc

directories:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "This Makefile must be run as root! Please use: sudo make install"; \
		exit 1; \
	fi
	@echo "Creating directory structure..."
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/bin
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/allsky-api/public
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/allsky-api/public/js
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 775 $(HOME_DIR)/database
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 775 $(HOME_DIR)/images
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 775 $(HOME_DIR)/videos
	ln -sf $(HOME_DIR)/database $(DATA_DIR)/allsky-api
	ln -sf $(HOME_DIR)/images $(DATA_DIR)/allsky-api/public/images
	ln -sf $(HOME_DIR)/videos $(DATA_DIR)/allsky-api/public/videos
	cp -a allsky-api/public/. $(DATA_DIR)/allsky-api/public
	# Copy other files
	$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) allsky-api/index.js $(DATA_DIR)/allsky-api/index.js
	# Optional: package.json etc.
	@if [ -f allsky-api/package.json ]; then \
		$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) allsky-api/package.json $(DATA_DIR)/allsky-api/; \
	fi
	@if ls allsky-api/package-lock.json >/dev/null 2>&1; then \
		$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) allsky-api/package-lock.json $(DATA_DIR)/allsky-api/; \
	fi
	chown -R $(USERNAME):$(GROUPNAME) $(DATA_DIR)

shell-scripts:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Installing shell scripts to $(DATA_DIR)/scripts..."
	# Shell scripts
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/scripts
	$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) scripts/endofnight.sh $(DATA_DIR)/scripts/
	$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) scripts/run_ai_processing.sh $(DATA_DIR)/scripts/
	# AI scripts
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/ai/models
	$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) ai/trt_infer.py $(DATA_DIR)/ai/
	$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) ai/onnx_infer.py $(DATA_DIR)/ai/
	$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) ai/yolov11_infer.py $(DATA_DIR)/ai/
	chown -R $(USERNAME):$(GROUPNAME) $(DATA_DIR)

binaries:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Installing binaries to $(APP_DIR)..."
	$(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 755 bin/allsky360-camera $(APP_DIR)
	$(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 755 tools/timelapse/bin/allsky360-timelapse $(APP_DIR)
	$(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 755 tools/meteo/bin/allsky360-meteo $(APP_DIR)
	$(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 755 tools/aurora/bin/allsky360-aurora $(APP_DIR)
	$(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 755 tools/location/bin/allsky360-location $(APP_DIR)
	$(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 755 tools/panorama/bin/panorama $(APP_DIR)
config:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Installing configuration to $(CONFIG_DIR)..."
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(CONFIG_DIR)
	$(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 644 config.ini.default $(CONFIG_DIR)/config.ini

AURORA_SERVICE   = allsky360-aurora.service
SUDOERS_FILE = /etc/sudoers.d/allsky360
METEO_SERVICE   = allsky360-meteo.service
CAMERA_SERVICE   = allsky360-camera.service
API_SERVICE   = allsky360-api.service
LOCATION_SERVICE   = allsky360-location.service
END_OF_NIGHT_SERVICE   = allsky360-endofnight.service

sudoers:
	@echo "Installing sudoers entry for $(USERNAME)"
	@echo '$(USERNAME) ALL=(root) NOPASSWD: /usr/bin/systemctl start $(AURORA_SERVICE), /usr/bin/systemctl stop $(AURORA_SERVICE), /usr/bin/systemctl restart $(AURORA_SERVICE), /usr/bin/systemctl is-active $(AURORA_SERVICE), /usr/bin/systemctl show $(AURORA_SERVICE)' > $(SUDOERS_FILE)
	@echo '$(USERNAME) ALL=(root) NOPASSWD: /usr/bin/systemctl start $(METEO_SERVICE), /usr/bin/systemctl stop $(METEO_SERVICE), /usr/bin/systemctl restart $(METEO_SERVICE), /usr/bin/systemctl is-active $(METEO_SERVICE), /usr/bin/systemctl show $(METEO_SERVICE)' >> $(SUDOERS_FILE)
	@echo '$(USERNAME) ALL=(root) NOPASSWD: /usr/bin/systemctl start $(CAMERA_SERVICE), /usr/bin/systemctl stop $(CAMERA_SERVICE), /usr/bin/systemctl restart $(CAMERA_SERVICE), /usr/bin/systemctl is-active $(CAMERA_SERVICE), /usr/bin/systemctl show $(CAMERA_SERVICE)' >> $(SUDOERS_FILE)
	@echo '$(USERNAME) ALL=(root) NOPASSWD: /usr/bin/systemctl start $(API_SERVICE), /usr/bin/systemctl stop $(API_SERVICE), /usr/bin/systemctl restart $(API_SERVICE), /usr/bin/systemctl is-active $(API_SERVICE), /usr/bin/systemctl show $(API_SERVICE)' >> $(SUDOERS_FILE)
	@echo '$(USERNAME) ALL=(root) NOPASSWD: /usr/bin/systemctl start $(LOCATION_SERVICE), /usr/bin/systemctl stop $(LOCATION_SERVICE), /usr/bin/systemctl restart $(LOCATION_SERVICE), /usr/bin/systemctl is-active $(LOCATION_SERVICE), /usr/bin/systemctl show $(LOCATION_SERVICE)' >> $(SUDOERS_FILE)
	@echo '$(USERNAME) ALL=(root) NOPASSWD: /usr/bin/systemd-run' >> $(SUDOERS_FILE)
	@sudo chmod 440 $(SUDOERS_FILE)
	@echo "Sudoers entries installed at $(SUDOERS_FILE)"
	# Check with sudo -l -U allskyuser and see if the entries are there
	sudo -l -U $(USERNAME)
	sudo -l -U $(USERNAME) | grep "aurora"
	sudo -l -U $(USERNAME) | grep "meteo"
	sudo -l -U $(USERNAME) | grep "location"
	sudo -l -U $(USERNAME) | grep "camera"
	sudo -l -U $(USERNAME) | grep "api"

systemd:
	@echo "Installing systemd service..."
	$(INSTALL) -m 644 etc/systemd/system/allsky360-camera.service $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-api.service $(SYSTEMD_DIR)/
	# Tools
	$(INSTALL) -m 644 etc/systemd/system/allsky360-aurora.service $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-meteo.service $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-location.service $(SYSTEMD_DIR)/
	# Timer
	$(INSTALL) -m 644 etc/systemd/system/allsky360-aurora.timer $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-meteo.timer $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-location.timer $(SYSTEMD_DIR)/

	# Reload daemons and activate timers
	$(SYSTEMCTL) daemon-reexec
	$(SYSTEMCTL) enable --now allsky360-camera.service
	$(SYSTEMCTL) enable --now allsky360-api.service
	$(SYSTEMCTL) enable --now allsky360-aurora.timer
	$(SYSTEMCTL) enable --now allsky360-meteo.timer
	$(SYSTEMCTL) enable --now allsky360-location.timer

cron:
	@echo "Installing cronjobs..."
	$(INSTALL) -m 644 etc/cron/allsky360-tools.cron $(CRON_DIR)/allsky360-tools.cron
	$(SYSTEMCTL) restart cron || $(SYSTEMCTL) restart crond || true

node-install:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "This Makefile must be run as root! Please use: sudo make install"; \
		exit 1; \
	fi
	@echo "Installing Node app to $(DATA_DIR)/allsky-api/…"
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/allsky-api
	$(INSTALL) -m 644 -o $(USERNAME) -g $(GROUPNAME) $(NODEDIR)/index.js $(DATA_DIR)/allsky-api/
	@test -f $(NODEDIR)/package.json && $(INSTALL) -m 644 -o $(USERNAME) -g $(GROUPNAME) $(NODEDIR)/package.json $(DATA_DIR)/allsky-api/ || true
	@test -f $(NODEDIR)/package-lock.json && $(INSTALL) -m 644 -o $(USERNAME) -g $(GROUPNAME) $(NODEDIR)/package-lock.json $(DATA_DIR)/allsky-api/ || true
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 0755 $(DATA_DIR)/allsky-api/public/js
	cp -a $(NODEDIR)/public/. $(DATA_DIR)/allsky-api/public/
	chown -R $(USERNAME):$(GROUPNAME) $(DATA_DIR)/allsky-api
	@echo "Installing npm deps (ci)…"
	cd $(DATA_DIR)/allsky-api && npm ci
	@echo "Ensuring dev tool (esbuild) is available…"
	cd $(DATA_DIR)/allsky-api && npm pkg get devDependencies.esbuild >/dev/null 2>&1 || npm i -D esbuild
	@echo "Creating bundle entry…"
	cd $(DATA_DIR)/allsky-api && mkdir -p public/js
	cd $(DATA_DIR)/allsky-api && printf "%s\n%s\n" \
		"import * as astronomia from 'astronomia';" \
		"export default astronomia;" > astro_bundle.js
	@echo "Bundling astronomia for browser…"
	cd $(DATA_DIR)/allsky-api && npm install astronomia --save
	cd $(DATA_DIR)/allsky-api && npx --yes esbuild astro_bundle.js \
		--bundle \
		--minify \
		--format=iife \
		--global-name=Astronomia \
		--outfile=public/js/astronomia.bundle.min.js


install:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "This Makefile must be run as root! Please use: sudo make install"; \
		exit 1; \
	fi
	@$(MAKE) user directories config binaries sudoers node-install systemd cron
	@echo "Installation complete"

uninstall:
	@if [ "$$(id -u)" -ne 0 ]; then \
    	    echo "This Makefile must be run as root! Please use: sudo make uninstall"; \
    	    exit 1; \
	fi
	@echo "Removing binaries..."
	rm -f $(APP_DIR)/allsky360-camera
	rm -f $(APP_DIR)/allsky360-timelapse
	rm -f $(APP_DIR)/allsky360-meteo
	rm -f $(APP_DIR)/allsky360-aurora

	@echo "Removing crond scripts..."
	rm -f $(CRON_DIR)/allsky360-tools.cron
	$(SYSTEMCTL) restart cron || $(SYSTEMCTL) restart crond || true

	@echo "Removing systemd service..."
	$(SYSTEMCTL) stop allsky360-camera.service || true
	$(SYSTEMCTL) disable allsky360-camera.service || true
	rm -f $(SYSTEMD_DIR)/allsky360-camera.service
	$(SYSTEMCTL) stop allsky360-api.service || true
	$(SYSTEMCTL) disable allsky360-api.service || true
	rm -f $(SYSTEMD_DIR)/allsky360-api.service
	$(SYSTEMCTL) daemon-reexec
	$(SYSTEMCTL) disable --now allsky360-aurora.timer
	$(SYSTEMCTL) disable --now allsky360-meteo.timer
	$(SYSTEMCTL) disable --now allsky360-location.timer
	rm -f $(UNIT_DIR)/allsky360-aurora.{service,timer}
	rm -f $(UNIT_DIR)/allsky360-meteo.{service,timer}
	rm -f $(UNIT_DIR)/allsky360-location.{service,timer}
	$(SYSTEMCTL) daemon-reload

	@echo "Removing cronjob..."
	rm -f $(CRON_DIR)/allsky360

	@echo "Removing configuration..."
	rm -rf $(CONFIG_DIR)/*

	@echo "Attention: User $(USERNAME) and home directory $(HOME_DIR) not deleted."
	@echo "Uninstall complete"

log:
	journalctl -u allsky360-api.service -f

node-run:
	@echo "Starting Allsky API Server"
	cd $(NODEDIR) && sudo -u $(USERNAME) node index.js

camera-run:
	@echo "Starting Allsky Camera"
	sudo -u $(USERNAME) bin/allsky360-camera -c config.ini.uranus-c -vvv -e 0.002 -g 0.0

help:
	@echo "Allsky360 Makefile - Available targets:"
	@echo ""
	@echo "Installation targets:"
	@echo "  install          - Complete installation (user, binaries, config, systemd, cron)"
	@echo "  user             - Create allskyuser and allskygroup with bash shell and PATH setup"
	@echo "  binaries         - Install allsky360 binaries to /opt/allsky360/bin"
	@echo "  config           - Install configuration files to user home directory"
	@echo "  systemd          - Install and enable systemd services and timers"
	@echo "  cron             - Install cron jobs for automated tasks"
	@echo ""
	@echo "Development targets:"
	@echo "  build            - Build allsky360-camera binary"
	@echo "  clean            - Clean build artifacts and temporary files"
	@echo "  tests            - Run test suite"
	@echo ""
	@echo "Runtime targets:"
	@echo "  node-run         - Start Allsky API Server (development mode)"
	@echo "  camera-run       - Start Allsky Camera (development mode)"
	@echo "  log              - Show live logs from allsky360-api.service"
	@echo ""
	@echo "Maintenance targets:"
	@echo "  uninstall        - Remove allsky360 installation (keeps user and home directory)"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  sudo make install     - Full installation"
	@echo "  sudo make uninstall   - Remove installation"
	@echo "  make help            - Show this help"
	@echo "  make log             - View live logs"

.PHONY: all clean install uninstall tests help


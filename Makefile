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

FREETYPE_CFLAGS = $(shell pkg-config --cflags freetype2)
FREETYPE_LIBS = $(shell pkg-config --libs freetype2)

# FITS
#CFITSIO_LIBS = -lcfitsio
CFITSIO_LIBS = $(shell pkg-config --exists cfitsio && pkg-config --cflags --libs cfitsio)

# CUDA package version for 'make deps' (JetPack 7.2 = 13-2, JetPack 6.2 = 12-6).
# Taken from the installed CUDA toolkit, 13-2 if none is installed yet.
CUDA_PKG_VERSION ?= $(or $(shell sed -n '/"cuda"/,/}/s/.*"version" *: *"\([0-9]*\)\.\([0-9]*\).*/\1-\2/p' /usr/local/cuda/version.json 2>/dev/null | head -1),13-2)

# Node.js major version for the API (must match "engines" in allsky-api/package.json)
NODE_MAJOR ?= 24
NODESOURCE_KEYRING_DIR = /usr/share/keyrings
NODESOURCE_KEYRING = $(NODESOURCE_KEYRING_DIR)/nodesource.gpg

# indigo server
INDIGO_ROOT ?= ../indigo
ifeq ($(wildcard $(INDIGO_ROOT)/indigo_libs),)
    $(warning INDIGO source tree not found in '$(INDIGO_ROOT)'. Clone and build INDIGO there or pass INDIGO_ROOT=/path/to/indigo)
endif
INDIGO_CFLAGS = -I$(INDIGO_ROOT)/indigo_libs
INDIGO_LIBS = -L$(INDIGO_ROOT)/build/lib -lindigo

# libjpeg Flags
#JPEG_CFLAGS := $(shell pkg-config --cflags libjpeg 2>/dev/null)
#JPEG_LIBS   := $(shell pkg-config --libs   libjpeg 2>/dev/null)
JPEG_CFLAGS = -I$(INDIGO_ROOT)/indigo_libs/externals/libjpeg
JPEG_LIBS = $(INDIGO_ROOT)/build/lib/libjpeg.a

# Libnova
ASTRO_CFLAGS = -I/usr/include/libnova
ASTRO_LIBS = -L/usr/lib -lnova

# PNG Library
PNG_LIBS = -lpng

INCLUDE = $(CAIRO_CFLAGS) $(FREETYPE_CFLAGS) $(INDIGO_CFLAGS) $(ASTRO_CFLAGS) $(JPEG_CFLAGS) -I./include
LDFLAGS += $(CAIRO_LIBS) $(FREETYPE_LIBS) $(INDIGO_LIBS) $(ASTRO_LIBS) $(PNG_LIBS) $(CFITSIO_LIBS) $(JPEG_LIBS) -lm

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
# Host compiler for nvcc: prefer gcc-11 if installed, otherwise default gcc
    NVCC_HOST_CC ?= $(shell command -v gcc-11 2>/dev/null || command -v gcc)
    NVCC = $(CUDA_PATH)/bin/nvcc --compiler-bindir=$(NVCC_HOST_CC)
# GPU architecture: 'native' compiles for the GPU of the build machine
# (Orin = sm_87). Cross builds: e.g. make USE_CUDA=1 CUDA_ARCH=sm_87
    CUDA_ARCH ?= native
    NVCCFLAGS = -arch=$(CUDA_ARCH) -O2 -DUSE_CUDA --compiler-options '-fPIC -std=c++17'
    CUDA_LIBS += -L$(CUDA_PATH)/targets/aarch64-linux/lib
## Jetson Nano
#    NVCC = nvcc --compiler-bindir=/usr/bin/gcc-7
#    NVCCFLAGS = -arch=sm_53 -O2 --compiler-options '-fPIC -std=c++11'
# Or other systems (multi-arch)
#    NVCC = nvcc
#    NVCCFLAGS = -arch=sm_53 -code=sm_53,sm_87 -O2 --compiler-options '-fPIC -std=c++11'
    CUDA_LIBS += -lcudart -lnvjpeg -lstdc++
    CFLAGS += -DUSE_CUDA -pthread 
    INCLUDE += -I$(CUDA_PATH)/include -I/usr/include/cairo -I./include
    # JPEG_CFLAGS comes first in the nvcc rule: the .cu files must see the
    # jpeglib.h of the linked INDIGO libjpeg (v9), not the system one (v8)
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
# Separate object directories for the CPU and the CUDA build. Without this,
# switching between 'make' and 'make USE_CUDA=1' mixes objects compiled with
# and without -DUSE_CUDA (duplicate symbols at link time or a wrong binary).
BUILD_VARIANT := $(if $(filter 1,$(USE_CUDA)),cuda,cpu)
OBJDIR = obj/$(BUILD_VARIANT)
BINDIR = bin
# Records the variant of the current binary, forces a relink on a switch
VARIANT_STAMP = $(BINDIR)/.build-variant
TESTDIR = test
TEST_SUN_CALC_BIN = $(BINDIR)/test_sun_calc
TEST_RAW_DUMP_BIN = $(BINDIR)/test_raw_dump
TEST_COORDINATES_BIN = $(BINDIR)/test_coordinates
TESTBINS = $(TEST_SUN_CALC_BIN) $(TEST_RAW_DUMP_BIN) $(TEST_COORDINATES_BIN)
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
	@apt -y install libjpeg-dev libcfitsio-dev libpng-dev libcairo-dev libnova-dev libjson-c-dev
	@apt -y install sqlite3 libsqlite3-dev
	@apt -y install libcurl4-openssl-dev
# dependencies for timelapse tool
	@apt -y install libavcodec-dev libavformat-dev libswscale-dev
ifeq ($(USE_CUDA), 1)
	@apt -y install cuda-nvcc-$(CUDA_PKG_VERSION) libnvjpeg-dev-$(CUDA_PKG_VERSION)
	@apt -y install cuda-cudart-dev-$(CUDA_PKG_VERSION) libnpp-dev-$(CUDA_PKG_VERSION)
endif
	@$(MAKE) node-deps
endif

# Node.js from the NodeSource apt repository. The repository is configured
# directly (signed-by keyring) instead of piping a setup script into bash.
# Replaces an existing NodeSource entry, e.g. the old node_20.x one.
node-deps:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "This must be run with root permissions: sudo make node-deps"; \
		exit 1; \
	fi
	@echo "Configuring NodeSource repository for Node.js $(NODE_MAJOR).x…"
	apt-get install -y ca-certificates curl gnupg build-essential python3
	install -d -m 0755 $(NODESOURCE_KEYRING_DIR)
	curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
		| gpg --dearmor --yes -o $(NODESOURCE_KEYRING)
	rm -f /etc/apt/sources.list.d/nodesource.list
	printf '%s\n' \
		'Types: deb' \
		'URIs: https://deb.nodesource.com/node_$(NODE_MAJOR).x' \
		'Suites: nodistro' \
		'Components: main' \
		"Architectures: $$(dpkg --print-architecture)" \
		'Signed-By: $(NODESOURCE_KEYRING)' \
		> /etc/apt/sources.list.d/nodesource.sources
	apt-get update
	apt-get install -y nodejs
	@$(MAKE) --no-print-directory node-check

# Fails if the installed Node.js is older than NODE_MAJOR
node-check:
	@command -v node >/dev/null 2>&1 || { echo "ERROR: node not found. Run: sudo make node-deps"; exit 1; }
	@v=$$(node -p 'process.versions.node.split(".")[0]'); \
	if [ "$$v" -lt $(NODE_MAJOR) ]; then \
		echo "ERROR: Node.js $$(node -v) found, $(NODE_MAJOR) or newer required. Run: sudo make node-deps"; \
		exit 1; \
	fi; \
	echo "Node.js $$(node -v), npm $$(npm -v)"

node-build: node-check
	@echo "Installing npm deps from package-lock.json…"
	cd $(NODEDIR) && npm ci
	@echo "Bundling astronomia for the browser…"
	cd $(NODEDIR) && npm run build

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

$(VARIANT_STAMP): FORCE
	@mkdir -p $(BINDIR)
	@echo $(BUILD_VARIANT) | cmp -s - $@ || echo $(BUILD_VARIANT) > $@

FORCE:

$(TARGET): $(VARIANT_STAMP) $(OBJS) $(CU_OBJS) $(CUDA_GPU_FUNCTIONS_WRAPPER_OBJ) $(CUDA_ACDNR_FILTER_WRAPPER_OBJ) $(CUDA_DEHAZE_FILTER_WRAPPER_OBJ) $(CUDA_CLARITY_FILTER_WRAPPER_OBJ) $(COLOR_CALIBRATION_WRAPPER_OBJ) $(CUDA_JPEG_FUNCTIONS_WRAPPER_OBJ) $(CUDA_HDR_MERGE_WRAPPER_OBJ) $(CUDA_LIGHTS_AND_SHADOWS_WRAPPER_OBJ) $(CUDA_MEDIAN_FILTER_WRAPPER_OBJ) $(CUDA_WAVELET_SHARPEN_WRAPPER_OBJ) $(CUDA_WHITE_BALANCE_WRAPPER_OBJ)
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
	$(NVCC) $(NVCCFLAGS) $(JPEG_CFLAGS) $(CUDA_INCLUDE) -c $< -o $@
endif

# Build for utils
tools: timelapse meteo aurora location mount-position cameras panorama list-cameras mount-status

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

mount-position:
	@echo "Building mount-position tool"
	@$(MAKE) -C tools/indigo-mount-position

cameras:
	@echo "Building cameras tool"
	@$(MAKE) -C tools/indigo-cameras

panorama:
	@echo "Building panorama tool"
	@$(MAKE) -C tools/panorama

list-cameras:
	@echo "Building list-cameras tool"
	@$(MAKE) -C tools/indigo-list-cameras

mount-status:
	@echo "Building mount-status tool"
	@$(MAKE) -C tools/indigo-mount-status

# Build for unit tests
tests: $(TESTBINS)

$(TEST_SUN_CALC_BIN): $(TESTDIR)/test_sun_calc.c $(SRCDIR)/sun_calc.c
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -o $(TEST_SUN_CALC_BIN) $(TESTDIR)/test_sun_calc.c $(SRCDIR)/sun_calc.c $(LDFLAGS)

$(TEST_RAW_DUMP_BIN): $(TESTDIR)/test_raw_dump.c $(SRCDIR)/raw_dump.c
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -o $(TEST_RAW_DUMP_BIN) $(TESTDIR)/test_raw_dump.c $(SRCDIR)/raw_dump.c

$(TEST_COORDINATES_BIN): $(TESTDIR)/test_coordinates.c $(SRCDIR)/coordinates.h
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $(TEST_COORDINATES_BIN) $(TESTDIR)/test_coordinates.c -lm

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
	rm -rf obj $(BINDIR) $(UTILSDIR)

##################### Installation

USERNAME          := allskyuser
GROUPNAME         := allskygroup

INSTALL           := install
SYSTEMCTL         := systemctl
# Units that need a configured config.ini and are started by the user
START_MANUALLY_UNITS := allsky360-camera.service allsky360-aurora.timer \
                        allsky360-meteo.timer allsky360-location.timer \
                        allsky360-mount-position.timer

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
	@id -u $(USERNAME) >/dev/null 2>&1 || useradd -r -s /usr/sbin/nologin $(USERNAME)
	@# Service account: no login shell (review.md, SEC-09). Also fixes accounts
	@# created by older versions with /bin/bash. For debugging use
	@# "sudo -u $(USERNAME) bash" - that still works.
	usermod -s /usr/sbin/nologin $(USERNAME)
	usermod -a -G $(GROUPNAME) $(USERNAME)
	# NOTE: $(USERNAME) is deliberately NOT added to the 'sudo' group. The service
	# account only needs the narrow polkit rule from the 'polkit' target below.
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
	$(INSTALL) -d -o root -g root -m 755 $(DATA_DIR)
	$(INSTALL) -d -o root -g root -m 755 $(DATA_DIR)/bin
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/allsky-api
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/allsky-api/public
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/allsky-api/public/js
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 775 $(HOME_DIR)/database
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 775 $(HOME_DIR)/images
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 775 $(HOME_DIR)/videos
	# Log- und Lockverzeichnis: bisher nur als Variable definiert, nie angelegt
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(LOG_DIR)
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(RUN_DIR)
	@for file in allsky-api/*; do \
		if [ -f $$file ]; then \
			$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) $$file $(DATA_DIR)/allsky-api/; \
		fi; \
	done
	@for file in allsky-api/public/*; do \
		if [ -f $$file ]; then \
			$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) $$file $(DATA_DIR)/allsky-api/public/; \
		fi; \
	done
	@for file in allsky-api/public/js/*; do \
		if [ -f $$file ]; then \
			$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) $$file $(DATA_DIR)/allsky-api/public/js/; \
		fi; \
	done

	ln -sf $(HOME_DIR)/database $(DATA_DIR)/allsky-api/database
	ln -sf $(HOME_DIR)/images $(DATA_DIR)/allsky-api/public/images
	ln -sf $(HOME_DIR)/videos $(DATA_DIR)/allsky-api/public/videos
	# NOTE: no recursive chown of $(DATA_DIR) here - $(DATA_DIR)/bin, /scripts and /ai
	# are owned by root on purpose (see 'binaries' and 'shell-scripts').

	# Optional: package.json etc.
	@if [ -f allsky-api/package.json ]; then \
		$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) allsky-api/package.json $(DATA_DIR)/allsky-api/; \
	fi
	@if ls allsky-api/package-lock.json >/dev/null 2>&1; then \
		$(INSTALL) -m 0644 -o $(USERNAME) -g $(GROUPNAME) allsky-api/package-lock.json $(DATA_DIR)/allsky-api/; \
	fi
	chown -R $(USERNAME):$(GROUPNAME) $(DATA_DIR)/allsky-api

shell-scripts:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Installing shell scripts to $(DATA_DIR)/scripts..."
	# Shell scripts. Owned by root and read-only for the service account: these are
	# started with elevated privileges, so $(USERNAME) must not be able to rewrite them.
	$(INSTALL) -d -o root -g root -m 755 $(DATA_DIR)/scripts
	$(INSTALL) -m 0755 -o root -g root scripts/endofnight.sh $(DATA_DIR)/scripts/
	$(INSTALL) -m 0755 -o root -g root scripts/postprocessing.sh $(DATA_DIR)/scripts/
	@if [ -f scripts/run_ai_processing.sh ]; then \
	    $(INSTALL) -m 0755 -o root -g root scripts/run_ai_processing.sh $(DATA_DIR)/scripts/; \
	fi
	# AI scripts
	$(INSTALL) -d -o root -g root -m 755 $(DATA_DIR)/ai
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(DATA_DIR)/ai/models
	$(INSTALL) -m 0644 -o root -g root ai/trt_infer.py $(DATA_DIR)/ai/
	$(INSTALL) -m 0644 -o root -g root ai/onnx_infer.py $(DATA_DIR)/ai/
	$(INSTALL) -m 0644 -o root -g root ai/yolo_infer.py $(DATA_DIR)/ai/
	rm -f $(DATA_DIR)/ai/yolov11_infer.py

binaries:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Installing binaries to $(APP_DIR)..."
	# Owned by root: the service account runs these, it must not be able to replace them.
	$(INSTALL) -o root -g root -m 755 bin/allsky360-camera $(APP_DIR)
	$(INSTALL) -o root -g root -m 755 tools/timelapse/bin/allsky360-timelapse $(APP_DIR)
	$(INSTALL) -o root -g root -m 755 tools/meteo/bin/allsky360-meteo $(APP_DIR)
	$(INSTALL) -o root -g root -m 755 tools/aurora/bin/allsky360-aurora $(APP_DIR)
	$(INSTALL) -o root -g root -m 755 tools/location/bin/allsky360-location $(APP_DIR)
	$(INSTALL) -o root -g root -m 755 tools/indigo-mount-position/bin/indigo_mount_position $(APP_DIR)
	$(INSTALL) -o root -g root -m 755 tools/indigo-cameras/bin/indigo_cameras $(APP_DIR)
config:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Installing configuration to $(CONFIG_DIR)..."
	$(INSTALL) -d -o $(USERNAME) -g $(GROUPNAME) -m 755 $(CONFIG_DIR)
	@# Never overwrite an existing config.ini - an update would otherwise wipe
	@# all settings including the api-token. The new template is put next to it.
	@# Mode 640: config.ini holds the api-token, all services run as $(USERNAME).
	@if [ -f $(CONFIG_DIR)/config.ini ]; then \
	    $(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 640 config.ini.default $(CONFIG_DIR)/config.ini.new; \
	    chmod 640 $(CONFIG_DIR)/config.ini; \
	    echo "Keeping existing $(CONFIG_DIR)/config.ini, new template: $(CONFIG_DIR)/config.ini.new"; \
	else \
	    $(INSTALL) -o $(USERNAME) -g $(GROUPNAME) -m 640 config.ini.default $(CONFIG_DIR)/config.ini; \
	    echo "Installed new $(CONFIG_DIR)/config.ini from config.ini.default"; \
	fi

AURORA_SERVICE   = allsky360-aurora.service
# Obsolete (replaced by the polkit rule), removed by the polkit target
SUDOERS_FILE = /etc/sudoers.d/allsky360
POLKIT_RULE  = /etc/polkit-1/rules.d/50-allsky360.rules
METEO_SERVICE   = allsky360-meteo.service
CAMERA_SERVICE   = allsky360-camera.service
API_SERVICE   = allsky360-api.service
LOCATION_SERVICE   = allsky360-location.service
MOUNT_POSITION_SERVICE = allsky360-mount-position.service
END_OF_NIGHT_SERVICE   = allsky360-endofnight.service

# Service control from the web UI (POST /api/services/...). The API runs
# "systemctl start|stop|restart <unit>" as $(USERNAME); systemd asks polkit,
# and this rule allows exactly these verbs on the allsky360 units.
# Replaces the former sudoers rules: sudo is setuid and cannot work inside the
# hardened API unit (NoNewPrivileges) - review.md, SEC-09.
polkit:
	@if [ "$$(id -u)" -ne 0 ]; then \
	    echo "This Makefile must be run as root! Please use: sudo make install"; \
	    exit 1; \
	fi
	@echo "Installing polkit rule $(POLKIT_RULE)"
	$(INSTALL) -d -m 755 $(dir $(POLKIT_RULE))
	$(INSTALL) -m 644 -o root -g root etc/polkit-1/rules.d/50-allsky360.rules $(POLKIT_RULE)
	@# polkitd picks up new rule files automatically
	@if [ -f $(SUDOERS_FILE) ]; then \
	    echo "Removing obsolete sudoers rules $(SUDOERS_FILE)"; \
	    rm -f $(SUDOERS_FILE); \
	fi

systemd:
	@echo "Installing systemd service..."
	$(INSTALL) -m 644 etc/systemd/system/allsky360-camera.service $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-api.service $(SYSTEMD_DIR)/
	# Tools
	$(INSTALL) -m 644 etc/systemd/system/allsky360-aurora.service $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-meteo.service $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-location.service $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-mount-position.service $(SYSTEMD_DIR)/
	# Timer
	$(INSTALL) -m 644 etc/systemd/system/allsky360-aurora.timer $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-meteo.timer $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-location.timer $(SYSTEMD_DIR)/
	$(INSTALL) -m 644 etc/systemd/system/allsky360-mount-position.timer $(SYSTEMD_DIR)/

	# Reload unit files. Camera and timers are only enabled by the user.
	$(SYSTEMCTL) daemon-reload
	@# Camera and timers are NOT enabled on a fresh install: config.ini still
	@# holds the template values (location, INDIGO server, camera, mount).
	@# The camera would fail in a loop, the timers would store weather/aurora
	@# data for the template location. The user edits config.ini and starts
	@# them manually - see the note at the end of 'make install'.
	@# On an update (unit already enabled) a running camera is restarted so it
	@# picks up the new binary; a stopped one stays stopped. Enabled timers
	@# keep running and use the new tool binaries on their next run.
	@for unit in $(START_MANUALLY_UNITS); do \
		if $(SYSTEMCTL) is-enabled --quiet $$unit; then \
			echo "$$unit is enabled, restarting it if it is running"; \
			$(SYSTEMCTL) try-restart $$unit; \
		else \
			echo "$$unit installed but not enabled (edit config.ini first)"; \
		fi; \
	done
	@# The API works with the template config and is needed to look at the
	@# station. 'restart' (not 'enable --now') so an update loads the new code.
	$(SYSTEMCTL) enable allsky360-api.service
	$(SYSTEMCTL) restart allsky360-api.service

node-install: node-check
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
	@# images/ and videos/ are symlinks to the runtime data in $(HOME_DIR)
	@# (see 'directories'). Never copy over them: cp -a fails when the source
	@# tree contains such a directory (e.g. a leftover from 'make node-run'),
	@# and it must never write into the image archive either.
	tar -C $(NODEDIR)/public --exclude=./images --exclude=./videos -cf - . \
		| tar -C $(DATA_DIR)/allsky-api/public -xf -
	chown -R $(USERNAME):$(GROUPNAME) $(DATA_DIR)/allsky-api
	$(INSTALL) -m 644 -o $(USERNAME) -g $(GROUPNAME) $(NODEDIR)/astro_bundle.js $(DATA_DIR)/allsky-api/
	@echo "Installing npm deps from package-lock.json…"
	cd $(DATA_DIR)/allsky-api && npm ci
	@echo "Bundling astronomia for the browser…"
	cd $(DATA_DIR)/allsky-api && npm run build
	@echo "Removing build-only packages (esbuild, astronomia)…"
	cd $(DATA_DIR)/allsky-api && npm prune --omit=dev
	chown -R $(USERNAME):$(GROUPNAME) $(DATA_DIR)/allsky-api

# ===== Ultralytics YOLO (YOLO26) venv for allskyuser =====
# Follows https://docs.ultralytics.com/quickstart (pip, PyTorch installed
# first) and the JetPack 7.2 section of
# https://docs.ultralytics.com/guides/nvidia-jetson/ :
#   - torch/torchvision from the PyTorch CUDA 13 index (the PyPI wheels do not
#     work with CUDA 13 on JetPack 7.2)
#   - pip install -U ultralytics
#   - onnxruntime-gpu 1.24.0 for Python 3.12
#   - TensorRT Python bindings from JetPack (TensorRT 10.x, not 11.x)
# A venv is used because Ubuntu 24.04 does not allow pip installs into the
# system Python (PEP 668).
#
# Run as root:  sudo make yolo
YOLO_VENV           ?= $(HOME_DIR)/venvs/yolo
YOLO_PYTHON         ?= /usr/bin/python3
TORCH_INDEX_URL     ?= https://download.pytorch.org/whl/cu130
ONNXRUNTIME_GPU_WHL ?= https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
# System site-packages of the JetPack Python with the TensorRT bindings
TENSORRT_DIST_DIR   ?= /usr/lib/python3.12/dist-packages

yolo:
	@if [ "$$(id -u)" -ne 0 ]; then echo "Run as root: sudo make yolo"; exit 1; fi
	@id $(USERNAME) >/dev/null 2>&1 || { echo "User $(USERNAME) does not exist, run 'sudo make user' first"; exit 1; }
	@echo "[YOLO] Installing Ultralytics venv for '$(USERNAME)' in '$(YOLO_VENV)'"

	@echo "[YOLO] APT packages (pip, venv, TensorRT Python bindings)"
	apt-get update
	apt-get install -y python3-pip python3-venv python3-libnvinfer

	@echo "[YOLO] Create venv"
	sudo -u $(USERNAME) -H mkdir -p "$(dir $(YOLO_VENV))"
	sudo -u $(USERNAME) -H "$(YOLO_PYTHON)" -m venv "$(YOLO_VENV)"
	sudo -u $(USERNAME) -H "$(YOLO_VENV)/bin/python" -m pip install -U pip

	@echo "[YOLO] PyTorch for CUDA 13 (first, so ultralytics does not pull the PyPI build)"
	sudo -u $(USERNAME) -H "$(YOLO_VENV)/bin/python" -m pip install -U torch torchvision --index-url "$(TORCH_INDEX_URL)"

	@echo "[YOLO] Ultralytics"
	sudo -u $(USERNAME) -H "$(YOLO_VENV)/bin/python" -m pip install -U ultralytics

	@echo "[YOLO] onnxruntime-gpu"
	sudo -u $(USERNAME) -H "$(YOLO_VENV)/bin/python" -m pip install "$(ONNXRUNTIME_GPU_WHL)"

	@echo "[YOLO] Make the JetPack TensorRT bindings visible in the venv"
	@site=$$(sudo -u $(USERNAME) -H "$(YOLO_VENV)/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
	for d in $(TENSORRT_DIST_DIR)/tensorrt $(TENSORRT_DIST_DIR)/tensorrt_lean $(TENSORRT_DIST_DIR)/tensorrt_dispatch \
	         $(TENSORRT_DIST_DIR)/tensorrt-*.dist-info $(TENSORRT_DIST_DIR)/tensorrt_lean-*.dist-info $(TENSORRT_DIST_DIR)/tensorrt_dispatch-*.dist-info; do \
		[ -e "$$d" ] && sudo -u $(USERNAME) ln -sfn "$$d" "$$site/$$(basename $$d)"; \
	done; true

	@$(MAKE) --no-print-directory yolo-check

# Verifies the venv: CUDA visible to torch, TensorRT importable, Ultralytics OK
yolo-check:
	@test -x "$(YOLO_VENV)/bin/python" || { echo "No venv in $(YOLO_VENV), run: sudo make yolo"; exit 1; }
	@"$(YOLO_VENV)/bin/python" -c 'import torch, torchvision, tensorrt, onnxruntime, ultralytics; \
	print("torch", torch.__version__, "CUDA", torch.version.cuda, "available:", torch.cuda.is_available()); \
	print("torchvision", torchvision.__version__); \
	print("tensorrt", tensorrt.__version__); \
	print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers()); \
	print("ultralytics", ultralytics.__version__); \
	assert torch.cuda.is_available(), "torch does not see the GPU"; \
	print("tensor on GPU:", torch.tensor([1.0, 2.0], device="cuda"))'
	@echo "[YOLO] OK"

# Old name, kept so existing notes keep working
yolov11: yolo

install:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "This Makefile must be run as root! Please use: sudo make install"; \
		exit 1; \
	fi
	@$(MAKE) user directories config binaries shell-scripts polkit node-install systemd
	@echo "Installation complete"
	@if ! grep -qE '^[[:space:]]*api-token[[:space:]]*=[[:space:]]*[^[:space:]]' $(CONFIG_DIR)/config.ini; then \
	    echo ""; \
	    echo "NOTE: api-token in $(CONFIG_DIR)/config.ini is empty."; \
	    echo "      The Start/Stop/Restart buttons of the web UI stay disabled until you set one."; \
	    echo "      See INSTALL.md, 'Step 4: Set the API token'."; \
	fi
	@disabled=""; \
	for unit in $(START_MANUALLY_UNITS); do \
		$(SYSTEMCTL) is-enabled --quiet $$unit || disabled="$$disabled $$unit"; \
	done; \
	if [ -n "$$disabled" ]; then \
	    echo ""; \
	    echo "NEXT STEPS: camera and data timers are installed but NOT started."; \
	    echo "  1. Edit $(CONFIG_DIR)/config.ini - at least:"; \
	    echo "       location_latitude, location_longitude, location_altitude"; \
	    echo "       indigo_server, indigo_port, indigo_camera, indigo_mount"; \
	    echo "     sudo -u $(USERNAME) nano $(CONFIG_DIR)/config.ini"; \
	    echo "  2. Enable and start them:"; \
	    echo "     sudo systemctl enable --now$$disabled"; \
	    echo "  3. Follow the camera log:"; \
	    echo "     tail -f /var/log/allsky360/camera.log"; \
	fi

uninstall:
	@if [ "$$(id -u)" -ne 0 ]; then \
    	    echo "This Makefile must be run as root! Please use: sudo make uninstall"; \
    	    exit 1; \
	fi
	@echo "Removing polkit/sudoers rules..."
	rm -f $(POLKIT_RULE) $(SUDOERS_FILE)
	@echo "Removing binaries..."
	rm -f $(APP_DIR)/allsky360-camera
	rm -f $(APP_DIR)/allsky360-timelapse
	rm -f $(APP_DIR)/allsky360-meteo
	rm -f $(APP_DIR)/allsky360-aurora
	rm -f $(APP_DIR)/allsky360-mount-position

	@echo "Removing legacy cron files (replaced by systemd timers)..."
	rm -f $(CRON_DIR)/allsky360 $(CRON_DIR)/allsky360-tools $(CRON_DIR)/allsky360-tools.cron

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
	$(SYSTEMCTL) disable --now allsky360-mount-position.timer
	rm -f $(UNIT_DIR)/allsky360-aurora.{service,timer}
	rm -f $(UNIT_DIR)/allsky360-meteo.{service,timer}
	rm -f $(UNIT_DIR)/allsky360-location.{service,timer}
	rm -f $(UNIT_DIR)/allsky360-mount-position.{service,timer}
	$(SYSTEMCTL) daemon-reload

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
	@echo "  install          - Complete installation (user, binaries, config, systemd)"
	@echo "  user             - Create allskyuser and allskygroup (service account, no login shell)"
	@echo "  polkit           - Allow allskyuser to start/stop/restart the allsky360 units (web UI)"
	@echo "  binaries         - Install allsky360 binaries to /opt/allsky360/bin"
	@echo "  config           - Install configuration files to user home directory"
	@echo "  systemd          - Install and enable systemd services and timers"
	@echo ""
	@echo "Development targets:"
	@echo "  build            - Build allsky360-camera binary"
	@echo "  clean            - Clean build artifacts and temporary files"
	@echo "  tests            - Run test suite"
	@echo ""
	@echo "Runtime targets:"
	@echo "  node-run         - Start Allsky API Server (development mode)"
	@echo "  node-deps        - Install Node.js $(NODE_MAJOR) from NodeSource (root)"
	@echo "  node-check       - Verify that Node.js $(NODE_MAJOR) or newer is installed"
	@echo "  node-build       - npm ci + build the browser bundle (development)"
	@echo "  yolo             - Install Ultralytics YOLO26 venv for AI classification (root)"
	@echo "  yolo-check       - Verify the YOLO venv (torch CUDA, TensorRT, onnxruntime)"
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

.PHONY: all clean install uninstall tests help FORCE polkit node-deps node-check node-build node-install yolo yolo-check yolov11


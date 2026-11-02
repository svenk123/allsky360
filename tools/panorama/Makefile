###############################################################################
# 
# Makefile for panorama tool
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
###############################################################################

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=gnu11

# Check if OpenMP is available
# Create a temporary test file and try to compile it with OpenMP
OPENMP_AVAILABLE = $(shell echo 'int main(){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo "yes" || echo "no")
ifeq ($(OPENMP_AVAILABLE),yes)
    CFLAGS += -fopenmp
    LDFLAGS_OPENMP = -fopenmp
else
    LDFLAGS_OPENMP =
endif

# Directory structure
SRCDIR = src
OBJDIR = obj
BINDIR = bin
TARGET = panorama

# Source directories (relative to this Makefile)
SRC_ROOT = ../../src
INCLUDE_ROOT = ../../include

# Sources and objects
MAIN_SRC = $(SRCDIR)/main.c
MAIN_OBJ = $(OBJDIR)/main.o

# Source files from parent directory
ALLSKY_PANORAMA_SRC = $(SRC_ROOT)/allsky_panorama.c
ALLSKY_PANORAMA_OBJ = $(OBJDIR)/allsky_panorama.o

JPEG_TO_CAIRO_SRC = $(SRC_ROOT)/jpeg_to_cairo.c
JPEG_TO_CAIRO_OBJ = $(OBJDIR)/jpeg_to_cairo.o

PNG_TO_CAIRO_SRC = $(SRC_ROOT)/png_to_cairo.c
PNG_TO_CAIRO_OBJ = $(OBJDIR)/png_to_cairo.o

ALL_OBJS = $(MAIN_OBJ) $(ALLSKY_PANORAMA_OBJ) $(JPEG_TO_CAIRO_OBJ) $(PNG_TO_CAIRO_OBJ)

# Cairo + FreeType + libjpeg Flags
CAIRO_CFLAGS = $(shell pkg-config --cflags cairo)
CAIRO_LIBS = $(shell pkg-config --libs cairo)
JPEG_LIBS = -ljpeg
PNG_LIBS = -lpng

# Include paths
INCLUDE = $(CAIRO_CFLAGS) -I$(SRC_ROOT) -I$(INCLUDE_ROOT)

# Libraries
LIBS = $(CAIRO_LIBS) $(JPEG_LIBS) $(PNG_LIBS) -lm $(LDFLAGS_OPENMP)

.PHONY: all clean install

# Default target
all: $(BINDIR)/$(TARGET)

# Link object files to final binary
$(BINDIR)/$(TARGET): $(ALL_OBJS) | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile main.c
$(MAIN_OBJ): $(MAIN_SRC) | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Compile allsky_panorama.c
$(ALLSKY_PANORAMA_OBJ): $(ALLSKY_PANORAMA_SRC) | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Compile jpeg_to_cairo.c
$(JPEG_TO_CAIRO_OBJ): $(JPEG_TO_CAIRO_SRC) | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Compile png_to_cairo.c
$(PNG_TO_CAIRO_OBJ): $(PNG_TO_CAIRO_SRC) | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Create obj directory if missing
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Create bin directory if missing
$(BINDIR):
	mkdir -p $(BINDIR)

# Install binary to system
install: $(BINDIR)/$(TARGET)
	install -Dm755 $(BINDIR)/$(TARGET) /usr/local/bin/$(TARGET)

# Remove build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)


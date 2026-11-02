###############################################################################
#
# Makefile for Allsky360 indigo_cameras
#
# Copyright (c) 2026 Sven Kreiensen
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

CC = gcc
CFLAGS = -Wall -Wextra -O2 -DINDIGO_LINUX -std=gnu11

# indigo server
INDIGO_ROOT ?= ../../../indigo
INDIGO_CFLAGS = -I$(INDIGO_ROOT)/indigo_libs
INDIGO_LIBS = -L$(INDIGO_ROOT)/build/lib -lindigo

LDFLAGS ?=
LDLIBS  ?=

BINDIR = bin
TARGET := indigo_cameras

CFLAGS  += -I. $(INDIGO_CFLAGS)
LDFLAGS += $(INDIGO_LIBS) -lpthread -lm

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(TARGET).c
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BINDIR)

.PHONY: all clean

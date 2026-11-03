###############################################################################
#
# Makefile for allsky360-sensors (RS485/Modbus sensors -> daily database)
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

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=gnu11 -D_GNU_SOURCE

SRCDIR  = src
OBJDIR  = obj
BINDIR  = bin
TARGET  = allsky360-sensors

# Shared sources from the main allsky360 project (atomic_file.c)
MAIN_SRCDIR = ../../src
CFLAGS += -I$(MAIN_SRCDIR)

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES)) $(OBJDIR)/atomic_file.o

# libgpiod v1 API (gpiod_chip_get_line); Ubuntu 24.04 / JetPack 7 ship 1.6
LIBS    = -lgpiod -lsqlite3 -ljson-c -lm

.PHONY: all clean

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/atomic_file.o: $(MAIN_SRCDIR)/atomic_file.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR) $(BINDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)
